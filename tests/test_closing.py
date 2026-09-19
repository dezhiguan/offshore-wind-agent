# -*- coding: utf-8 -*-
"""步数用尽收口。

这条路只在撞步数上限时才走（26 条用例里 3 次），但它是**唯一**一次非流式调用，
也是唯一一次曾经把工具定义从请求里拿掉的调用——而那会让前缀缓存整段作废。
"""
from types import SimpleNamespace

from agent.loop import _force_answer, _unusable_answer
from agent.prompts import OUT_OF_SCOPE_MARK
from agent.tracing import DEGRADED, OK, Trace

# 一份长度像样的三段式正文。此前几处桩用的是「正文」「## 结论\n有正文」这种占位串，
# 而成文可用性现在带字数下限（见 _unusable_answer）—— 占位串会被判成不可用并触发重写，
# 那是桩不真实，不是判据太严：三段标题本身就有三十多字。
DRAFT = ("## 结论\nT03 的 24002 构成重复故障，工单 WO-260703 仍为 NORMAL，应升为 HIGH。\n\n"
         "## 依据\n第 3.1 条：24 小时内 4 次。\n\n## 现有资料无法确认\n- 无")


def _resp(content="", tool_calls=None, prompt=6000, cached=3072, completion=200):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


class _StubClient:
    """记下每次请求的参数，按顺序吐预设响应。"""

    def __init__(self, *responses):
        self.calls = []
        self._queue = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._queue.pop(0)


def _messages():
    return [{"role": "system", "content": "…"}, {"role": "user", "content": "问题"}]


class TestToolsStayInTheRequest:
    def test_tools_are_sent_so_the_prefix_cache_survives(self):
        """把 tools 拿掉等于换了前缀：实测 prompt 6005 只命中 1024，而带着工具的
        同类调用命中 3072——多算约 2000 token 的 prefill，且按未缓存价计费。"""
        client = _StubClient(_resp(content=DRAFT))
        trace = Trace()
        answer = _force_answer(client, "qwen3.8-flash", _messages(), trace)
        assert answer.startswith("## 结论")
        assert len(client.calls) == 1
        assert client.calls[0]["tools"], "收口调用必须带上 tools，否则前缀缓存作废"

    def test_single_call_records_one_span(self):
        client = _StubClient(_resp(content=DRAFT))
        trace = Trace()
        _force_answer(client, "qwen3.8-flash", _messages(), trace)
        spans = trace.as_list()
        assert [s["name"] for s in spans] == ["步数用尽收口"]
        assert spans[0]["status"] == OK
        assert spans[0]["cached_tokens"] == 3072


class TestFallbackWhenItStillCallsTools:
    """不让它调工具靠的是那句用户消息，不是靠把工具藏起来——万一不听，得有兜底。"""

    def _run(self):
        client = _StubClient(
            _resp(content="", tool_calls=[SimpleNamespace(id="x")]),   # 不听劝，正文是空的
            _resp(content=DRAFT),                              # 撤下工具再问
        )
        trace = Trace()
        answer = _force_answer(client, "qwen3.8-flash", _messages(), trace)
        return client, trace, answer

    def test_retries_without_tools_and_answers(self):
        client, _, answer = self._run()
        assert answer == DRAFT
        assert len(client.calls) == 2
        assert "tools" not in client.calls[1], "重试这次不给工具，它就没有别的选择"

    def test_both_calls_are_recorded_separately(self):
        """两次调用各自烧了 token，合成一条会让模型调用次数少算一次、成本漏计。"""
        _, trace, _ = self._run()
        spans = trace.as_list()
        assert [s["name"] for s in spans] == ["步数用尽收口", "步数用尽收口（撤下工具重试）"]
        assert trace.summary()["model_calls"] == 2
        assert trace.summary()["completion_tokens"] == 400

    def test_the_wasted_round_is_flagged_as_degraded(self):
        """收口轮没产出可用正文，得在异常清单里看得见，不能悄悄重试完就算了。"""
        _, trace, _ = self._run()
        first = trace.as_list()[0]
        assert first["status"] == DEGRADED
        assert "仍发起工具调用" in first["output"]


class TestToolCallsWithContentAreNotRetried:
    def test_content_present_wins_even_if_tools_were_called(self):
        """既写了正文又顺手发了工具调用：正文可用就不必再烧一次。"""
        client = _StubClient(_resp(content=DRAFT, tool_calls=[SimpleNamespace(id="x")]))
        _force_answer(client, "qwen3.8-flash", _messages(), Trace())
        assert len(client.calls) == 1


class TestUnusableAnswerIsRewritten:
    """成文那一次吐了垃圾，不能当答案发出去。

    起因是 2026-09-19 回归用例 Q6：规则引擎已经把 WO-260708 / 观察 15 分钟不足 120 分钟 /
    不符合关闭要求原样交回来了，成文那次却只回了 `<previous_tool_call>\n\n</previous_tool_call>`
    共 11 个 token。链路照单全收 —— format_parsed 记了 false，但没有任何地方消费它，
    残片进了成文、进了指标、进了页面，最后是用例断言把它拦下来的。
    """

    JUNK = "<previous_tool_call>\n\n</previous_tool_call>"

    def test_the_exact_junk_from_Q6_is_rejected(self):
        assert _unusable_answer(self.JUNK)

    def test_empty_and_whitespace_are_rejected(self):
        assert _unusable_answer("")
        assert _unusable_answer("   \n  ")

    def test_too_short_to_be_an_answer_is_rejected(self):
        assert _unusable_answer("## 结论\n是")

    def test_a_real_draft_passes(self):
        assert _unusable_answer(DRAFT) is None

    def test_out_of_scope_declaration_passes_even_though_it_is_short(self):
        """范围外声明是护栏的合规出口，天然简短，不能被字数下限误杀。"""
        assert _unusable_answer("## 结论\n%s" % OUT_OF_SCOPE_MARK) is None

    def test_closing_call_retries_when_the_body_is_junk(self):
        """原判据是「正文为空且又发了工具调用」—— 残片 .strip() 非空，会被直接放行。"""
        client = _StubClient(_resp(content=self.JUNK), _resp(content=DRAFT))
        trace = Trace()
        answer = _force_answer(client, "qwen3.8-flash", _messages(), trace)
        assert answer == DRAFT
        assert len(client.calls) == 2
        assert "tools" not in client.calls[1]

    def test_sampling_is_deterministic(self):
        """同一条用例跑两次不该是两份不同的答案：温度必须显式传 0，不吃厂商默认。"""
        client = _StubClient(_resp(content=DRAFT))
        _force_answer(client, "qwen3.8-flash", _messages(), trace=Trace())
        assert client.calls[0]["temperature"] == 0
