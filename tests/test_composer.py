# -*- coding: utf-8 -*-
from agent.composer import compose

DRAFT = """## 结论
T03 在窗口内发生 4 次，构成重复故障，工单应升为 HIGH。

## 依据
- alarm_records：4 条记录，时间为 07-18 08:15 等
- 规程 第 3.1 条：连续 24 小时内 3 次及以上即为重复故障

## 现有资料无法确认
- 现场安全条件
- 责任归属
"""

RUN = {
    "draft": DRAFT,
    "trace": [{"step": 1, "tool": "query_db", "summary": "命中 4 行"}],
    "evidence": {
        "tables": [{"sql": "SELECT * FROM alarm_records WHERE turbine_id='T03'", "rows": []}],
        "docs": [{"doc": "海上风电机组检修作业与安全管理规程.md", "section_id": "3.1"}],
        "rules": [],
    },
    "stop_reason": "completed",
    "elapsed_ms": 1234,
}


def test_three_sections_parsed():
    out = compose("T03 重复故障？", RUN)
    assert "4 次" in out["answer"]
    assert len(out["basis"]) == 2
    assert out["unverifiable"] == ["现场安全条件", "责任归属"]
    assert out["meta"]["format_parsed"] is True


def test_sources_derived_from_actual_calls():
    out = compose("q", RUN)
    assert "运行告警记录（alarm_records）" in out["sources"]
    # 规程带条款号：只说"用了规程"，看的人没法对着条款去查
    assert "检修作业与安全管理规程（第 3.1 条）" in out["sources"]
    assert "维检工单记录（maintenance_records）" not in out["sources"]


def test_rule_verdict_counts_as_regulation_source():
    """T05 那条：结论全部来自规程第 2.1 / 2.4 条，模型一次文档都没取。

    只统计模型自己发的 SQL 和自己取回的文档，规程就会整条消失，
    而「结论依据」那一栏还明明白白引着条款号。
    """
    run = dict(RUN, evidence={
        "tables": [], "docs": [],
        "rules": [{"rule": "priority_required", "result": {
            "ok": True,
            "sources": ["alarm_records", "maintenance_records", "safety_regulation"],
            "clauses": ["2.4", "2.1"],
        }}],
    })
    assert compose("q", run)["sources"] == [
        "运行告警记录（alarm_records）",
        "维检工单记录（maintenance_records）",
        "检修作业与安全管理规程（第 2.1、2.4 条）",
    ]


def test_source_order_is_stable_regardless_of_call_order():
    run = dict(RUN, evidence={
        "tables": [{"sql": "SELECT * FROM maintenance_records", "rows": []}],
        "docs": [{"doc": "故障处理手册.md", "section_id": "24005"}],
        "rules": [{"rule": "repeat_fault",
                   "result": {"ok": True, "sources": ["alarm_records"], "clauses": ["3.1"]}}],
    })
    assert compose("q", run)["sources"] == [
        "运行告警记录（alarm_records）",
        "维检工单记录（maintenance_records）",
        "故障处理手册",
        "检修作业与安全管理规程（第 3.1 条）",
    ]


def test_clauses_sorted_numerically_not_lexically():
    run = dict(RUN, evidence={"tables": [], "docs": [], "rules": [
        {"rule": "close_compliance",
         "result": {"ok": True, "sources": ["safety_regulation"],
                    "clauses": ["6.10", "6.2", "6.1"]}}]})
    assert compose("q", run)["sources"] == ["检修作业与安全管理规程（第 6.1、6.2、6.10 条）"]


def test_no_records_no_clauses_means_no_regulation_chip():
    """查无记录、没得判的规则不能顺手把规程也记成数据源。"""
    run = dict(RUN, evidence={"tables": [], "docs": [], "rules": [
        {"rule": "priority_required",
         "result": {"ok": True, "clauses": [], "verdict": "数据库中没有 T99 的告警记录。"}}]})
    assert compose("q", run)["sources"] == []


def test_none_marker_means_empty_not_missing():
    run = dict(RUN, draft="## 结论\nA\n\n## 依据\n- x\n\n## 现有资料无法确认\n无\n")
    assert compose("q", run)["unverifiable"] == []


def test_unparsable_draft_degrades_and_flags_it():
    run = dict(RUN, draft="模型没按格式输出的一段自由文本")
    out = compose("q", run)
    assert out["answer"] == "模型没按格式输出的一段自由文本"
    assert out["meta"]["format_parsed"] is False


class TestBasisRenderedFromEvidence:
    """「依据」由代码从证据渲染（2026-09-18 改）。

    原先这一段由模型复述，约 300 字符、3~4 秒。它复述的恰恰是 evidence 里已有的
    东西——规则的 verdict 是规则引擎算出来的原话，命中行数是执行结果。
    让模型再说一遍，既花时间，又多一次转述漂移的机会。
    """

    def test_basis_ignores_what_the_model_wrote(self):
        """草稿里就算还留着「依据」段，也以证据为准——两者不一致时，编的是前者。"""
        run = dict(RUN, draft=DRAFT.replace("alarm_records：4 条记录", "alarm_records：400 条记录"))
        assert not any("400 条" in line for line in compose("q", run)["basis"])

    def test_rule_verdict_is_used_verbatim_with_clause_prefix(self):
        run = dict(RUN, evidence={"tables": [], "docs": [], "rules": [
            {"rule": "repeat_fault", "result": {
                "ok": True, "clauses": ["3.2", "3.1"],
                "verdict": "构成重复故障：最大连续 24 小时窗口内发生 4 次。"}}]})
        assert compose("q", run)["basis"] == [
            "第 3.1、3.2 条 · 构成重复故障：最大连续 24 小时窗口内发生 4 次。"]

    def test_clause_inlined_by_rule_is_not_listed_twice(self):
        """同一条款既被规则内联、又被单独取回时只出现一次，否则读起来像两项证据。"""
        run = dict(RUN, evidence={
            "tables": [],
            "docs": [{"doc": "海上风电机组检修作业与安全管理规程.md", "section_id": "3.1",
                      "title": "第 3.1 条 重复故障", "text": "### 第 3.1 条\n连续 24 小时内 3 次及以上。"}],
            "rules": [{"rule": "repeat_fault", "result": {
                "ok": True, "clauses": ["3.1"], "verdict": "构成重复故障。"}}]})
        assert compose("q", run)["basis"] == ["第 3.1 条 · 构成重复故障。"]

    def test_empty_result_shows_the_condition_it_queried(self):
        """查空时「查了什么没查到」才是依据，只说「无匹配」等于没说。"""
        run = dict(RUN, evidence={"rules": [], "docs": [], "tables": [
            {"sql": "SELECT * FROM alarm_records WHERE turbine_id='T20' ORDER BY occurred_at",
             "rows": [], "row_count": 0}]})
        line = compose("q", run)["basis"][0]
        assert "turbine_id='T20'" in line and "无匹配记录" in line
        assert "order by" not in line.lower()

    def test_single_cell_aggregate_shows_the_number(self):
        run = dict(RUN, evidence={"rules": [], "docs": [], "tables": [
            {"sql": "SELECT COUNT(*) AS n FROM alarm_records", "columns": ["n"],
             "rows": [{"n": 47}], "row_count": 1}]})
        assert compose("q", run)["basis"] == ["`alarm_records` → n = 47"]

    def test_overflow_is_announced_not_silently_dropped(self):
        run = dict(RUN, evidence={"tables": [], "docs": [], "rules": [
            {"rule": "r%d" % i, "result": {"ok": True, "clauses": [], "verdict": "判定 %d。" % i}}
            for i in range(9)]})
        basis = compose("q", run)["basis"]
        assert len(basis) == 7                      # 6 行 + 1 行说明
        assert "另有 3 项依据" in basis[-1]

    def test_falls_back_to_the_draft_when_there_is_no_evidence(self):
        """未取证拒答、或复放老链路时不能整段空掉。"""
        run = dict(RUN, evidence={"tables": [], "docs": [], "rules": []})
        assert len(compose("q", run)["basis"]) == 2  # 回落到草稿里那两行

    def test_general_rule_keeps_its_clause_sections(self):
        """通则调用的 verdict 是一句免责说明，压掉条款行就只剩这句话了。"""
        run = dict(RUN, evidence={
            "tables": [],
            "docs": [{"doc": "海上风电机组检修作业与安全管理规程.md", "section_id": "4.1",
                      "title": "第 4.1 条 禁止远程复位", "text": "### 第 4.1 条\n以下情形禁止远程复位。"}],
            "rules": [{"rule": "remote_reset_ban", "result": {
                "ok": True, "is_general": True, "clauses": ["4.1"],
                "verdict": "未指定 turbine_id，以下是通则要求，未针对任何具体工单作出判定。"}}]})
        basis = compose("q", run)["basis"]
        assert len(basis) == 2
        assert "第 4.1 条 禁止远程复位" in basis[1]


class TestUnverifiableConsistency:
    """结论里说了"无法确认"，第二段却空着。

    对抗性测试里模型把 7 项未确认条件全写进了结论，第二段写「无」——界面那块
    面板读的是第二段，显示为空，等于把模型自己说出来的缺口藏了起来。
    这里只标记不回填：从自由文本里切句子，切错比空着更糟。
    """

    def _run(self, draft):
        return compose("问题", {"draft": draft, "trace": [], "evidence": {}})

    def test_inconsistent_when_conclusion_says_unconfirmed_but_section_empty(self):
        out = self._run("## 结论\n母线电压现有资料无法确认，需要现场核实。\n\n"
                        "## 现有资料无法确认\n无")
        assert out["unverifiable"] == []
        assert out["meta"]["unverifiable_inconsistent"] is True

    def test_consistent_when_items_are_listed_in_the_section(self):
        out = self._run("## 结论\n母线电压现有资料无法确认。\n\n"
                        "## 现有资料无法确认\n- 母线电压实测值，需要现场核实。")
        assert out["unverifiable"]
        assert out["meta"]["unverifiable_inconsistent"] is False

    def test_clean_answer_is_not_flagged(self):
        out = self._run("## 结论\nT01 共 5 条告警。\n\n## 现有资料无法确认\n无")
        assert out["meta"]["unverifiable_inconsistent"] is False
