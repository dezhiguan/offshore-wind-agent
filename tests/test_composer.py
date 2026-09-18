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

    def test_completed_run_has_no_truncation_block(self):
        """跑完的链路不该挂"没查完"的牌子。"""
        assert compose("q", RUN)["truncation"] is None

    def test_max_steps_run_reports_what_was_never_read(self):
        """步数用尽：把「本轮没查完」和没读过的表作为事实摆出来。

        不摆出来的话，模型写进「现有资料无法确认」的那几条会被读成
        「库里没有」，而实际是「系统没去查」。
        """
        run = dict(RUN, stop_reason="max_steps")
        t = compose("q", run)["truncation"]
        assert t["reason"] == "max_steps"
        assert t["steps"] == 1
        # 本轮只查了 alarm_records，工单表一次没读
        assert t["unread_sources"] == ["维检工单记录（maintenance_records）"]

    def test_truncation_does_not_rewrite_the_models_own_items(self):
        """只陈述事实，不改判模型写的那几条——改判属于会误伤的语义判定。"""
        run = dict(RUN, stop_reason="max_steps")
        out = compose("q", run)
        assert out["unverifiable"] == ["现场安全条件", "责任归属"]

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


# --------------------------------------------------------- 标题容错与「无」归一化
# 2026-09-18 跑测发现：模型偶发把标题写成 `**## 结论**`（K1 三次复跑中一次），
# 原先的严格式正则匹配不上，整篇原文落进 answer、「现有资料无法确认」整段丢失。

def test_heading_tolerates_bold_wrappers():
    from agent.composer import _split_sections
    for draft in (
        "## 结论\nA\n\n## 现有资料无法确认\n无\n",
        "**## 结论**\nA\n\n**## 现有资料无法确认**\n无。\n",
        "**结论**\nA\n\n**现有资料无法确认**\n- x\n",
        "##**结论**\nA\n\n### 现有资料无法确认：\n无\n",
    ):
        parts = _split_sections(draft)
        assert "结论" in parts, draft
        assert "现有资料无法确认" in parts, draft


def test_bold_heading_draft_is_parsed_not_dumped():
    """加粗标题的整篇原文不应再原样落进 answer。"""
    run = {"draft": "**## 结论**\nT06 型号 OWT-5.0A。\n\n**## 现有资料无法确认**\n无。\n",
           "evidence": {}, "trace": []}
    out = compose("T06 的型号？", run)
    assert out["meta"]["format_parsed"] is True
    assert "##" not in out["answer"]
    assert out["answer"] == "T06 型号 OWT-5.0A。"
    assert out["unverifiable"] == []


def test_none_item_with_parenthetical_is_empty():
    """「无（……）。」是有依据的空，不是一条真实的无法确认项。"""
    run = {"draft": "## 结论\nA\n\n## 现有资料无法确认\n无（本题为手册条文查询，答案完整）。\n",
           "evidence": {}, "trace": []}
    assert compose("Q", run)["unverifiable"] == []


def test_real_unverifiable_item_survives():
    """不能过度归一化：真实的无法确认项必须留下。"""
    run = {"draft": "## 结论\nA\n\n## 现有资料无法确认\n- 母线电压——现有资料无法确认。\n",
           "evidence": {}, "trace": []}
    assert compose("Q", run)["unverifiable"] == ["母线电压——现有资料无法确认。"]


def test_prompt_meta_wording_leak_is_dropped():
    """提示词里的格式纪律被模型抄进列表，属元指令泄漏，不是业务结论。"""
    draft = ("## 结论\nA\n\n## 现有资料无法确认\n"
             "- 母线电压是否降至约 20 V——需现场核实。\n"
             "- （依规程第 1.2 条措辞：现有资料无法确认，需要现场核实。）\n")
    out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
    assert out["unverifiable"] == ["母线电压是否降至约 20 V——需现场核实。"]


def test_meta_filter_keeps_substantive_clause_mention():
    """只剔「措辞」那一类；正常提到第 1.2 条的业务内容要保留。"""
    draft = ("## 结论\nA\n\n## 现有资料无法确认\n"
             "- 规程第 1.2 条要求现场留痕，本次无记录。\n")
    out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
    assert out["unverifiable"] == ["规程第 1.2 条要求现场留痕，本次无记录。"]
