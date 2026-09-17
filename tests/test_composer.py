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
