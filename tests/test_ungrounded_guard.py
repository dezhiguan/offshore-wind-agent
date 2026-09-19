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


class TestExitNeedsEvidenceWhenTheQuestionNamesAnEntity:
    """问题点名了库内对象时，范围外出口要先核实一次。

    2026-09-19 边界用例 B8 实测：「WO-260708 这张工单是谁关闭的？处理人是谁？」——
    工单就在库里，只是没有人员字段。模型打出范围外标记，0 步直接作答；话是对的，
    但一次证据都没取，「使用的数据源」整个是空的。
    「资料里没有这个字段」与「问题不落在四类资料范围内」被共用了一个出口，
    而前者必须查过那张表才说得出口。
    """

    ANSWER = "## 结论\n%s\n\n## 现有资料无法确认\n- 处理人" % OUT_OF_SCOPE_MARK

    def test_first_declaration_is_challenged(self, monkeypatch):
        client = _StubClient(self.ANSWER, self.ANSWER)
        run = _run(monkeypatch, client, question="WO-260708 这张工单是谁关闭的？处理人是谁？")
        guards = [s["name"] for s in run["spans"] if s["kind"] == "GUARD"]
        assert guards[0] == "范围外声明待核实"
        assert len(client.calls) == 2, "应当先要求它取证一次"

    def test_second_declaration_is_let_through(self, monkeypatch):
        """只拦一次：真·范围外的问题捎带一个风机号，不能被逼着去查无关的东西凑数。"""
        client = _StubClient(self.ANSWER, self.ANSWER)
        run = _run(monkeypatch, client, question="WO-260708 这张工单是谁关闭的？处理人是谁？")
        assert run["stop_reason"] == "out_of_scope"
        assert run["draft"] == self.ANSWER

    def test_question_without_any_entity_is_untouched(self, monkeypatch):
        """问天气、问模型是谁：没有任何对象可查，出口照旧当场放行。"""
        client = _StubClient(self.ANSWER)
        run = _run(monkeypatch, client, question="你用的什么模型？")
        assert run["stop_reason"] == "out_of_scope"
        assert len(client.calls) == 1

    def test_entity_shapes_recognised(self):
        from agent.loop import _names_db_entity
        assert _names_db_entity("WO-260708 是谁关的")
        assert _names_db_entity("T08 现在什么状态")
        assert _names_db_entity("24010 怎么处理")
        # 中文在 Unicode 下也算 \w：用 \b 划边界的话，这两条没有空格的中文问句
        # 一条都匹配不上，规则等于没写
        assert _names_db_entity("24011故障怎么处理")
        assert _names_db_entity("查一下T08现在的状态")
        assert not _names_db_entity("你用的什么模型？")
        assert not _names_db_entity("观察时间要满 120 分钟吗")
