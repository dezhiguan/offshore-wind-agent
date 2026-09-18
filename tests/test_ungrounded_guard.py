# -*- coding: utf-8 -*-
"""未取证护栏的两条出口。

红线是「一次工具都没调就作答」要拦。但拦得太死会长出另一个毛病：问到四类资料
根本不涉及的事（问模型是谁、问天气、问外部标准），没有任何工具能给出依据，
打回只是逼模型去查点无关的东西凑数——对抗性测试里它就是这么干的，而凑出来的
那段自由发挥里混着没人核对的数字（工单总数被自己加成 16，真值 15）。

所以护栏要有一个**合规出口**，且判据必须确定：模型写下固定标记才算数。
这两条用例守的就是"出口开着"和"红线仍在"这两件事，缺一不可。
"""
from types import SimpleNamespace

import pytest

from agent import loop
from agent.prompts import OUT_OF_SCOPE_MARK


def _chunk(content=None):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))])


def _usage_chunk(prompt=100, completion=20):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        choices=[])


class _StubClient:
    """按顺序吐预设的流式响应，并记下每次请求。"""

    def __init__(self, *texts):
        self.calls = []
        self._queue = [[_chunk(t), _usage_chunk()] for t in texts]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._queue.pop(0))


def _run(monkeypatch, client, question="你用的什么模型？"):
    monkeypatch.setattr(loop, "_client", lambda: client)
    result = {}
    for event in loop.run_agent_stream(question):
        if event["type"] == "done":
            result = event["result"]
    return result


class TestOutOfScopeExit:
    def test_declared_out_of_scope_is_let_through_without_tools(self, monkeypatch):
        """带固定标记的拒答直接放行，不打回、不逼它凑工具。"""
        answer = "## 结论\n%s，无法回答。\n\n## 现有资料无法确认\n- 全部事项" % OUT_OF_SCOPE_MARK
        client = _StubClient(answer)
        run = _run(monkeypatch, client)

        assert run["stop_reason"] == "out_of_scope"
        assert run["draft"] == answer
        assert run["trace"] == [], "范围外拒答不应产生任何工具调用"
        assert len(client.calls) == 1, "放行的路径上不该有第二次模型调用"

    def test_exit_is_recorded_as_a_guard_span(self, monkeypatch):
        """出口要留痕：后台得分得清「合规拒答」和「凑工具答对了」。"""
        client = _StubClient("## 结论\n%s。" % OUT_OF_SCOPE_MARK)
        run = _run(monkeypatch, client)
        guards = [s for s in run["spans"] if s["kind"] == "GUARD"]
        assert [g["name"] for g in guards] == ["范围外声明"]


class TestRedLineStillHolds:
    def test_ungrounded_answer_without_the_mark_is_still_refused(self, monkeypatch):
        """没有标记就不是范围外，是凭自带知识作答——打回一次，仍不改则拒答。

        出口的判据是固定串而不是语义猜测，正是为了让这条红线还在：
        否则"第二次仍不调工具就放行"等于把护栏拆了。
        """
        client = _StubClient("24002 一般是通讯线松动，紧固即可。",
                             "我确认就是通讯线松动。")
        run = _run(monkeypatch, client, question="24002 怎么修？")

        assert run["stop_reason"] == "refused_ungrounded"
        assert "无法回答" in run["draft"]
        assert len(client.calls) == 2, "应当打回一次后才拒答"

    def test_retry_prompt_offers_the_exit(self, monkeypatch):
        """打回时要把出口告诉模型，否则它只能靠凑工具满足要求。"""
        client = _StubClient("凭经验答一句。", "还是凭经验。")
        _run(monkeypatch, client, question="24002 怎么修？")
        retry = client.calls[1]["messages"][-1]
        assert retry["role"] == "user"
        assert OUT_OF_SCOPE_MARK in retry["content"]
