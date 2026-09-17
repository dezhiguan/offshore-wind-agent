# -*- coding: utf-8 -*-
"""把循环产物整理成对外的回答契约。

刻意不再多调一次 LLM 做结构化：模型按固定三段标题输出，这里用确定性解析拆开。
更重要的是——evidence 和 sources 都由代码从**真实发生过的工具调用**汇总，
不经过模型转述，所以「依据展示」在结构上就不可能被编造。
"""
from __future__ import annotations

import re
from typing import Any

H_CONCLUSION = "结论"
H_BASIS = "依据"
H_UNVERIFIABLE = "现有资料无法确认"

_HEADING = re.compile(r"^#{1,4}\s*(结论|依据|现有资料无法确认)\s*$", re.M)

SOURCE_LABELS = {
    "alarm_records": "运行告警记录（alarm_records）",
    "maintenance_records": "维检工单记录（maintenance_records）",
    "故障处理手册.md": "故障处理手册",
    "海上风电机组检修作业与安全管理规程.md": "检修作业与安全管理规程",
}


def _split_sections(text: str) -> dict[str, str]:
    marks = [(m.start(), m.end(), m.group(1)) for m in _HEADING.finditer(text or "")]
    if not marks:
        return {}
    out: dict[str, str] = {}
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out[name] = text[end:stop].strip()
    return out


def _as_items(block: str) -> list[str]:
    items = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.、)]\s*", "", line)
        if line:
            items.append(line)
    return items


def _detect_sources(evidence: dict[str, Any]) -> list[str]:
    """从实际查过的东西反推用了哪些数据源，而不是问模型。"""
    found: list[str] = []
    for t in evidence.get("tables", []):
        sql = (t.get("sql") or "").lower()
        for table in ("alarm_records", "maintenance_records"):
            if table in sql and SOURCE_LABELS[table] not in found:
                found.append(SOURCE_LABELS[table])
    for d in evidence.get("docs", []):
        label = SOURCE_LABELS.get(d.get("doc"), d.get("doc"))
        if label and label not in found:
            found.append(label)
    return found


def compose(question: str, run: dict[str, Any]) -> dict[str, Any]:
    draft = run.get("draft") or ""
    parts = _split_sections(draft)

    unverifiable = _as_items(parts.get(H_UNVERIFIABLE, ""))
    # 「无」是有依据的空，不是漏答
    if len(unverifiable) == 1 and unverifiable[0] in {"无", "无。", "None", "-"}:
        unverifiable = []

    evidence = run.get("evidence", {})
    return {
        "question": question,
        "answer": parts.get(H_CONCLUSION, "").strip() or draft.strip(),
        "basis": _as_items(parts.get(H_BASIS, "")),
        "unverifiable": unverifiable,
        "sources": _detect_sources(evidence),
        "evidence": evidence,
        "trace": run.get("trace", []),
        # 链路追踪：逐段可复核的执行明细与汇总
        "spans": run.get("spans", []),
        "trace_summary": run.get("trace_summary") or {},
        "meta": {
            "stop_reason": run.get("stop_reason"),
            "elapsed_ms": run.get("elapsed_ms"),
            "steps": len(run.get("trace", [])),
            # 三段标题没解析出来时降级为原文，同时把降级标出来，不假装成功
            "format_parsed": bool(parts),
            # 本次会话实际进入上下文的语料占比，供界面展示与事后审计
            "budget": run.get("budget"),
            "model": run.get("model"),
        },
    }
