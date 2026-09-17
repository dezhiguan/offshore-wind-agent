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
    assert "检修作业与安全管理规程" in out["sources"]
    assert "维检工单记录（maintenance_records）" not in out["sources"]


def test_none_marker_means_empty_not_missing():
    run = dict(RUN, draft="## 结论\nA\n\n## 依据\n- x\n\n## 现有资料无法确认\n无\n")
    assert compose("q", run)["unverifiable"] == []


def test_unparsable_draft_degrades_and_flags_it():
    run = dict(RUN, draft="模型没按格式输出的一段自由文本")
    out = compose("q", run)
    assert out["answer"] == "模型没按格式输出的一段自由文本"
    assert out["meta"]["format_parsed"] is False
