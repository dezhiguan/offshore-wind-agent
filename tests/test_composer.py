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


def _run(draft, *, stop_reason="completed", rules=None):
    return {
        "draft": draft,
        "trace": [{"step": 1, "tool": "check_rule", "summary": "ok"}],
        "evidence": {"tables": [], "docs": [], "rules": rules or []},
        "stop_reason": stop_reason,
        "elapsed_ms": 1,
    }


def _sweep(rule):
    """一次 scope="all" 的判定：coverage.完整 是 sweep 写下的机器事实。"""
    return {"args": {"rule": rule, "scope": "all"},
            "result": {"rule": rule, "coverage": {"判定对象总数": 25, "已判定": 25, "完整": True}}}


def _per_subject(rule):
    return {"args": {"rule": rule, "turbine_id": "T03"}, "result": {"rule": rule}}


class TestNoneItemNormalization:
    """「无」写成两行时，两行都逃过了归一化，面板上凭空多出两条待核实项。"""

    def test_none_with_trailing_annotation_line(self):
        out = compose("q", _run("""## 结论
T03 有 2 条 WARNING 告警。

## 现有资料无法确认
无。
（本次未涉及规程判定，也无遗漏对象。）
"""))
        assert out["unverifiable"] == []

    def test_annotation_kept_when_real_items_present(self):
        # 注解行与真实条目并列时可能是上一条的续行，丢了就是删内容
        out = compose("q", _run("""## 结论
结论正文。

## 现有资料无法确认
- 现场安全条件
- （母线电压未记录）
"""))
        assert len(out["unverifiable"]) == 2


class TestSectionRerouting:
    """「本次未完成」被写进「现有资料无法确认」——重跑就能消的缺口
    被包装成派人上岛也未必消得掉的缺口。"""

    def test_not_run_item_moves_to_incomplete(self):
        out = compose("q", _run("""## 结论
已判 T03。

## 现有资料无法确认
- WO-260708 关闭过程是否满足第 6.1 条——本轮未完成查询：close_compliance 判定需补查。
- 现场安全条件（挂牌上锁、验电）——现有资料无法确认，需要现场核实。
"""))
        assert len(out["unverifiable"]) == 1
        assert "现场安全条件" in out["unverifiable"][0]
        assert any("close_compliance" in i for i in out["incomplete"])
        assert len(out["meta"]["section_rerouted"]) == 1

    def test_tool_did_not_return_also_moves(self):
        # 实测形态：判定工具本轮没返回，条目却写在「现有资料无法确认」里。
        # 工具没返回是重跑就能变的，不是资料里没有。
        out = compose("q", _run("""## 结论
已判一部分。

## 现有资料无法确认
- 第 5.1 条九项前置条件的逐项判定结果——现有资料无法确认（判定工具本轮未能返回）。
"""))
        assert out["unverifiable"] == []
        assert len(out["incomplete"]) == 1

    def test_item_needing_site_check_stays(self):
        # 两个标记同时命中时不搬：搬走会把现场核实那一半一起带走
        out = compose("q", _run("""## 结论
已判 T03。

## 现有资料无法确认
- 母线电压实际值——本次未查询，且现有资料无法确认，需要现场核实。
"""))
        assert len(out["unverifiable"]) == 1
        assert out["incomplete"] == []

    def test_no_duplicate_when_model_wrote_both_sections(self):
        line = "全部 COMPLETED 工单的关闭合规性判定——本轮未完成查询，需补查。"
        out = compose("q", _run("""## 结论
已判一部分。

## 现有资料无法确认
- %s

## 本次未完成
- %s
""" % (line, line)))
        assert out["incomplete"].count(line) == 1


class TestCoverageIsPerRule:
    """一条完整 sweep 只证明那一项判定判完了，证明不了整个回答覆盖完整。"""

    def test_complete_sweep_alone_suppresses_notice(self):
        out = compose("q", _run("""## 结论
全场 25 个组合已全部判完，仅 T03/24002 构成重复故障。
""", stop_reason="max_steps", rules=[_sweep("repeat_fault")]))
        assert "步数用尽" not in out["answer"]
        assert out["meta"]["coverage_complete"] is True
        assert out["truncation"] is None

    def test_model_declared_unfinished_beats_one_complete_sweep(self):
        out = compose("q", _run("""## 结论
重复故障全场判完；工单合规只判了一部分。

## 本次未完成
- 四台机组的工单五要素核对——需补查 work_order_assessment。
""", stop_reason="max_steps",
            rules=[_sweep("repeat_fault"), _per_subject("work_order_assessment")]))
        assert "步数用尽" in out["answer"]
        assert out["meta"]["coverage_complete"] is False
        assert out["truncation"] is not None

    def test_coverage_rules_accounted_by_rule(self):
        out = compose("q", _run("""## 结论
正文。
""", rules=[_sweep("repeat_fault"), _per_subject("close_compliance")]))
        assert out["meta"]["coverage_rules"] == {
            "swept": ["repeat_fault"], "per_subject": ["close_compliance"]}


class TestToolNamesScrubbed:
    """内部工具名不进给人看的正文。

    提示词已经写了禁止，但实测同一道题两轮跑，一轮干净、一轮写出
    「check_rule 判定命中第 4.1 条」「工具判定不构成重复故障」——措辞类要求靠
    提示词只能压低频率，压不到零，所以再加一道确定性的收口。
    """

    def test_tool_name_replaced_in_answer(self):
        draft = "## 结论\ncheck_rule 判定命中第 4.1 条第 2 款。\n"
        out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
        assert "check_rule" not in out["answer"]
        # 不能洗成「规程判定 判定命中」：紧跟的那个名词要一起吃掉
        assert out["answer"].strip() == "规程判定命中第 4.1 条第 2 款。"

    def test_tool_verdict_phrase_replaced(self):
        draft = "## 结论\n工具判定不构成重复故障。\n"
        out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
        assert out["answer"].strip() == "规程判定不构成重复故障。"

    def test_scrub_applies_to_unverifiable_items(self):
        draft = ("## 结论\nA\n\n## 现有资料无法确认\n"
                 "- query_db 未记录母线电压，需现场核实。\n")
        out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
        assert out["unverifiable"] == ["数据库查询未记录母线电压，需现场核实。"]

    def test_table_names_and_content_untouched(self):
        """洗的是工具名，不是表名——表名是资料的名字，现场看得懂也该看见。

        「工具」在现场多指扳手这类实物，同样不能泛洗。
        """
        draft = ("## 结论\n按 alarm_records 与 maintenance_records 核对，"
                 "作业前须备齐力矩扳手等工具。\n")
        out = compose("Q", {"draft": draft, "evidence": {}, "trace": []})
        assert "alarm_records" in out["answer"]
        assert "maintenance_records" in out["answer"]
        assert "力矩扳手等工具" in out["answer"]

    def test_basis_keeps_tool_names(self):
        """依据面板与链路追踪要照原样展示 SQL 和工具名，洗掉就没法追执行过程。"""
        run = {"draft": "## 结论\nA\n", "trace": [], "evidence": {
            "tables": [{"sql": "SELECT * FROM alarm_records WHERE turbine_id='T03'",
                        "rows": []}]}}
        out = compose("Q", run)
        assert any("alarm_records" in b for b in out["basis"])
