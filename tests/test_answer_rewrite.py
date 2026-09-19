# -*- coding: utf-8 -*-
"""成文不可用时的重写。

起因是 2026-09-19 回归用例 Q6：`check_rule` 已经把标准答案原样交回来了
（WO-260708 / 观察 15 分钟不足 120 分钟 / 不符合关闭要求），成文那一次却只回了
`<previous_tool_call>\n\n</previous_tool_call>` 共 11 个 token。链路对此没有任何判据 ——
只要这一轮没发工具调用，吐出来的东西就是最终答案，残片进了成文、进了指标、
进了页面，最后是用例断言把它拦下来的。meta.format_parsed 记了 false，但没人消费。

判据只认确定性形态（控制残片 / 空 / 短到不可能是一份三段式答案），不做语义判断。
"""
from types import SimpleNamespace

from agent import loop

JUNK = "<previous_tool_call>\n\n</previous_tool_call>"
DRAFT = ("## 结论\nT09 的 WO-260708 不符合关闭要求：观察 15 分钟，不足第 6.4 条要求的 120 分钟。\n\n"
         "## 依据\n第 6.4 条 · 观察时间不少于 120 分钟。\n\n## 现有资料无法确认\n- 无")


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(finish_reason=None,
                                 delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _usage_chunk(prompt=100, completion=20):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        choices=[])


def _tool_chunk(name, arguments):
    return _chunk(tool_calls=[SimpleNamespace(
        index=0, id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments))])


def _resp(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop",
                                 message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)))


class _StubClient:
    """流式与非流式各走各的队列：重写用的是非流式调用。"""

    def __init__(self, *, streamed, plain=()):
        self.calls = []
        self._streamed = list(streamed)
        self._plain = list(plain)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._streamed.pop(0))
        return self._plain.pop(0)


def _run(monkeypatch, client, question="WO-260708 是否符合关闭要求？"):
    monkeypatch.setattr(loop, "_client", lambda: client)
    result = {}
    for event in loop.run_agent_stream(question):
        if event["type"] == "done":
            result = event["result"]
    return result


def _client_with(*plain_answers):
    """先老老实实查一次库（证据要有），再吐残片；重写走非流式。"""
    return _StubClient(
        streamed=[
            [_tool_chunk("query_db",
                         '{"sql": "SELECT work_order_id FROM maintenance_records LIMIT 1"}'),
             _usage_chunk()],
            [_chunk(JUNK), _usage_chunk()],
        ],
        plain=list(plain_answers))


class TestJunkBodyIsRewritten:
    def test_rewrite_replaces_the_junk(self, monkeypatch):
        client = _client_with(_resp(DRAFT))
        run = _run(monkeypatch, client)
        assert run["draft"] == DRAFT
        assert run["stop_reason"] == "completed"

    def test_rewrite_is_a_separate_model_call_without_tools(self, monkeypatch):
        """重写是货真价实的一次调用，得单独记；证据已经齐了，这一步不该再给工具。"""
        client = _client_with(_resp(DRAFT))
        run = _run(monkeypatch, client)
        assert len(client.calls) == 3
        assert "tools" not in client.calls[2]
        assert "成文重写" in [s["name"] for s in run["spans"] if s["kind"] == "MODEL"]

    def test_the_wasted_round_is_visible(self, monkeypatch):
        """悄悄重写完就算了的话，后台看到的是一条全绿链路。"""
        client = _client_with(_resp(DRAFT))
        run = _run(monkeypatch, client)
        guards = [s for s in run["spans"] if s["kind"] == "GUARD"]
        assert [g["name"] for g in guards] == ["成文不可用"]
        assert guards[0]["status"] == "DEGRADED"

    def test_evidence_collected_before_the_junk_is_kept(self, monkeypatch):
        """重写的前提就是证据还在：丢了证据等于把这一轮白跑。"""
        client = _client_with(_resp(DRAFT))
        run = _run(monkeypatch, client)
        assert run["evidence"]["tables"], "查过的表不能因为成文失败而丢掉"


class TestRewriteAlsoFails:
    def test_no_usable_body_is_stated_not_faked(self, monkeypatch):
        """重写仍是残片：如实说没产出，不拿残片冒充答案。"""
        client = _client_with(_resp(JUNK))
        run = _run(monkeypatch, client)
        assert run["stop_reason"] == "answer_unusable"
        assert JUNK not in run["draft"]
        assert "未能生成可用正文" in run["draft"]

    def test_it_is_recorded_as_an_error(self, monkeypatch):
        client = _client_with(_resp(JUNK))
        run = _run(monkeypatch, client)
        guards = [s for s in run["spans"] if s["kind"] == "GUARD"]
        assert guards[-1]["status"] == "ERROR"

    def test_not_counted_as_a_usable_answer(self, monkeypatch):
        """正文非空但不是有效回答 —— 与未取证拒答同一条口径，三处共用 NO_ANSWER_STOPS。"""
        from agent.tracing import NO_ANSWER_STOPS
        client = _client_with(_resp(JUNK))
        run = _run(monkeypatch, client)
        assert run["stop_reason"] in NO_ANSWER_STOPS


class TestGoodBodyIsNotTouched:
    def test_no_rewrite_when_the_body_is_fine(self, monkeypatch):
        client = _StubClient(streamed=[
            [_tool_chunk("query_db",
                         '{"sql": "SELECT work_order_id FROM maintenance_records LIMIT 1"}'),
             _usage_chunk()],
            [_chunk(DRAFT), _usage_chunk()],
        ])
        run = _run(monkeypatch, client)
        assert run["draft"] == DRAFT
        assert len(client.calls) == 2, "正文可用就不该再烧一次调用"
        assert not [s for s in run["spans"] if s["kind"] == "GUARD"]

    def test_sampling_is_deterministic_on_every_call(self, monkeypatch):
        """评测要可比：温度显式传 0，决策轮和成文轮都不吃厂商默认。"""
        client = _client_with(_resp(DRAFT))
        _run(monkeypatch, client)
        assert all(c["temperature"] == 0 for c in client.calls)
