# -*- coding: utf-8 -*-
"""把循环产物整理成对外的回答契约。

刻意不再多调一次 LLM 做结构化：模型按固定三段标题输出，这里用确定性解析拆开。
更重要的是——evidence 和 sources 都由代码从**真实发生过的工具调用**汇总，
不经过模型转述，所以「依据展示」在结构上就不可能被编造。
"""
from __future__ import annotations

import re
import time
from typing import Any

from agent.checklist import build as build_checklists
from tools.pricing import basis as price_basis

H_CONCLUSION = "结论"
H_BASIS = "依据"
H_UNVERIFIABLE = "现有资料无法确认"

_HEADING = re.compile(r"^#{1,4}\s*(结论|依据|现有资料无法确认)\s*$", re.M)

# 数据源统一用这四个 key 表示；回归用例的 sources: 字段写的也是它们。
SOURCE_LABELS = {
    "alarm_records": "运行告警记录（alarm_records）",
    "maintenance_records": "维检工单记录（maintenance_records）",
    "fault_manual": "故障处理手册",
    "safety_regulation": "检修作业与安全管理规程",
}
# 展示顺序固定：先两张表，再手册，最后规程。同一个问题问两次，chip 的顺序不该变。
SOURCE_ORDER = ("alarm_records", "maintenance_records", "fault_manual", "safety_regulation")
# 文档检索层回传的是文件名，这里换回 key
DOC_KEYS = {
    "故障处理手册.md": "fault_manual",
    "海上风电机组检修作业与安全管理规程.md": "safety_regulation",
}
_TABLES = ("alarm_records", "maintenance_records")
_CLAUSE_ID = re.compile(r"^\d+(?:\.\d+)?$")


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


def _clause_key(clause: str) -> tuple:
    try:
        return tuple(int(part) for part in str(clause).split("."))
    except ValueError:
        return (9999,)


def detect_sources(evidence: dict[str, Any]) -> list[str]:
    """从实际用过的东西反推数据源，而不是问模型。

    三条来源缺一不可：模型自己发的 SQL、模型自己取回的文档，以及**规则判定**。
    规程是以代码化条款的形式参与判定的，整轮问答可以一次 get_doc_section 都不调；
    只统计前两条，就会出现「结论依据白纸黑字引了第 2.1 条、数据源里却没有规程」
    这种自相矛盾 —— 而且规则引擎替模型查掉的表也会跟着一起漏。
    """
    keys: list[str] = []
    clauses: set[str] = set()

    def add(key: str | None) -> None:
        if key and key not in keys:
            keys.append(key)

    for t in evidence.get("tables", []):
        sql = (t.get("sql") or "").lower()
        for table in _TABLES:
            if table in sql:
                add(table)

    for r in evidence.get("rules", []):
        result = r.get("result") or {}
        for key in result.get("sources") or ():
            add(key)
        clauses.update(result.get("clauses") or ())

    for d in evidence.get("docs", []):
        key = DOC_KEYS.get(d.get("doc"))
        add(key or d.get("doc"))
        section_id = str(d.get("section_id") or "")
        if key == "safety_regulation" and _CLAUSE_ID.match(section_id):
            clauses.add(section_id)

    # 引用了条款就是用了规程。规则结果里的 sources 是后加的字段，
    # 早先留存的链路只有 clauses —— 拿条款兜底，老记录也不会漏报。
    if clauses:
        add("safety_regulation")

    ordered = ([k for k in SOURCE_ORDER if k in keys]
               + [k for k in keys if k not in SOURCE_ORDER])
    found = []
    for key in ordered:
        label = SOURCE_LABELS.get(key, key)
        # 规程带上条款号：只说"用了规程"等于没说，看的人要能对着条款去查
        if key == "safety_regulation" and clauses:
            label += "（第 %s 条）" % "、".join(sorted(clauses, key=_clause_key))
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
        # 把规则判定转成现场核实清单：同一份数据，从「我不知道」变成「你要去办的几件事」
        "checklists": build_checklists(evidence),
        "sources": detect_sources(evidence),
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
            # 成本是按核对过的官方单价算的，还是落到兜底价的估算
            "price_basis": price_basis(run.get("model") or ""),
        },
    }


def stamp_served(result: dict[str, Any], started: float) -> dict[str, Any]:
    """补上端到端服务耗时。

    ``meta.elapsed_ms`` 只量到链路收口那一刻，解析三段、生成核实清单、接地校验
    都在表外。差值实测只有几毫秒，但"总耗时"这个词说的是用户等了多久，
    差一点也不该由读的人自己去猜哪一段没算进去。
    """
    result.setdefault("meta", {})["served_ms"] = int((time.monotonic() - started) * 1000)
    return result
