# -*- coding: utf-8 -*-
"""把循环产物整理成对外的回答契约。

刻意不再多调一次 LLM 做结构化：模型按固定三段标题输出，这里用确定性解析拆开。
更重要的是——evidence 和 sources 都由代码从**真实发生过的工具调用**汇总，
不经过模型转述，所以「依据展示」在结构上就不可能被编造。

**「依据」也由代码渲染**（2026-09-18 改）：这一段原先由模型复述一遍，约 300 字符。
实测生成速率 84~110 字符/秒，光这一段就是 3~4 秒，占端到端的一成有余。而它复述的
恰恰是 evidence 里已有的东西——规则判定的 verdict 是规则引擎算出来的原话，SQL 与
命中行数是执行结果，条款号来自真实取回的章节。让模型再说一遍，既花时间，又多一次
转述漂移的机会。改由代码从 evidence 渲染后，这一段与执行轨迹在结构上完全一致。

模型仍然写「结论」与「现有资料无法确认」两段：前者是要回答的问题本身，后者需要
判断"资料里到底有没有"，都不是代码能替它做的。
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

# 依据行里回显查询条件用。取 WHERE 到下一个子句之间那段——空结果时，
# "查了什么条件没查到"比"没查到"有用得多。
_WHERE = re.compile(r"\bwhere\b(.*?)(?=\border\s+by\b|\bgroup\s+by\b|\blimit\b|$)",
                    re.I | re.S)
# 行里用来指认"这批是谁的数据"的列，按这个顺序取
_ID_COLUMNS = ("turbine_id", "fault_code", "work_order_id")
# 依据段的行数上限。证据面板里是全量，这里是摘要；超出要**说出来**，
# 不能静默截断——截断后看上去就像"依据只有这几条"。
BASIS_MAX_LINES = 6
# 单行上限。规则 verdict 最长的一条有 400 多字符（关单合规逐项核对），
# 整条摊进依据段就没法一眼扫完；全文在证据面板里点得开，这里按句截。
BASIS_LINE_CHARS = 110
_SENTENCE_END = re.compile(r"[。；;！!？?\n]")


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


def _clip(text: str, limit: int = BASIS_LINE_CHARS) -> str:
    """按句截断，截不到句末再硬截。省略号是给人看的信号，不是装饰。"""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cuts = [m.end() for m in _SENTENCE_END.finditer(text[:limit]) if m.end() > limit * 0.4]
    return (text[:cuts[-1]] if cuts else text[:limit]) + "…"


def _table_of(sql: str) -> str | None:
    low = (sql or "").lower()
    return next((t for t in _TABLES if t in low), None)


def _condition(sql: str) -> str:
    match = _WHERE.search(sql or "")
    if not match:
        return ""
    cond = " ".join(match.group(1).split())
    return cond[:58] + "…" if len(cond) > 58 else cond


def _table_basis(entry: dict[str, Any]) -> str:
    sql = entry.get("sql") or ""
    rows = entry.get("rows") or []
    count = entry.get("row_count", len(rows))
    head = "`%s`" % (_table_of(sql) or "数据表")
    cond = _condition(sql)
    if cond:
        head += "：WHERE %s" % cond
    if not count:
        return head + " → 无匹配记录"
    columns = entry.get("columns") or (list(rows[0]) if rows else [])
    # 聚合查询（COUNT / MAX）的单格结果直接给数：这才是它被查出来的原因
    if count == 1 and len(columns) == 1 and rows:
        return "%s → %s = %s" % (head, columns[0], rows[0].get(columns[0]))
    marks = []
    for col in _ID_COLUMNS:
        seen: list[str] = []
        for row in rows:
            value = row.get(col)
            if value is not None and str(value) not in seen:
                seen.append(str(value))
        if seen:
            marks.append("、".join(seen[:3]) + ("…" if len(seen) > 3 else ""))
    return "%s → 命中 %d 行%s" % (head, count, "（%s）" % " / ".join(marks) if marks else "")


def _doc_basis(section: dict[str, Any]) -> str:
    title = section.get("title") or section.get("section_id") or ""
    body = [line.strip() for line in (section.get("text") or "").splitlines()
            if line.strip() and not line.strip().startswith("#")]
    point = _SENTENCE_END.split(body[0])[0].strip() if body else ""
    if len(point) > 40:
        point = point[:40] + "…"
    return "《%s》%s%s" % (section.get("doc", ""), title, "：%s" % point if point else "")


def _rule_basis(entry: dict[str, Any]) -> str | None:
    """规则判定这一行直接用 verdict 原话。

    它是规则引擎按条款算出来的结论，不是模型的转述——让模型复述一遍，只会在
    「4 次」「3 次门限」这种数字上多一次出错的机会。
    """
    result = entry.get("result") or {}
    verdict = (result.get("verdict") or "").strip()
    if not verdict:
        return None
    clauses = result.get("clauses") or []
    prefix = "第 %s 条 · " % "、".join(sorted(clauses, key=_clause_key)) if clauses else ""
    return prefix + _clip(verdict)


def render_basis(evidence: dict[str, Any]) -> list[str]:
    """从真实发生过的工具调用渲染「依据」。

    顺序是**按信息量**排的，不是按调用顺序：规则判定带着结论与条款号，信息量最大；
    其次是查库拿到的事实；最后是引用到的章节。超过上限的不静默丢掉，最后一行明说
    还有多少项——证据面板里是全量，但读依据的人未必会往下翻。
    """
    lines: list[str] = []
    covered_clauses: set[str] = set()

    for entry in evidence.get("rules", []) or []:
        line = _rule_basis(entry)
        result = entry.get("result") or {}
        if line:
            lines.append(line)
        # 通则调用（没传 id）的 verdict 只是一句"未针对具体对象作出判定"的说明，
        # 真正的依据是那几条条款原文本身 —— 这种情况不能把条款行压掉，
        # 压掉后依据里就只剩一句免责声明。
        if not result.get("is_general"):
            covered_clauses.update(result.get("clauses") or ())

    for entry in evidence.get("tables", []) or []:
        lines.append(_table_basis(entry))

    for section in evidence.get("docs", []) or []:
        # 规则已经内联过的条款不再单列：同一条款在依据里出现两次，读的人会以为是两项证据
        if (DOC_KEYS.get(section.get("doc")) == "safety_regulation"
                and str(section.get("section_id") or "") in covered_clauses):
            continue
        lines.append(_doc_basis(section))

    # 同一条 SQL 被查两次、同一节被两条路径带出来时，渲染出的文字会完全相同
    unique: list[str] = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    if len(unique) <= BASIS_MAX_LINES:
        return unique
    return unique[:BASIS_MAX_LINES] + [
        "（另有 %d 项依据，见下方证据面板）" % (len(unique) - BASIS_MAX_LINES)]


def compose(question: str, run: dict[str, Any]) -> dict[str, Any]:
    draft = run.get("draft") or ""
    parts = _split_sections(draft)

    unverifiable = _as_items(parts.get(H_UNVERIFIABLE, ""))
    # 「无」是有依据的空，不是漏答
    if len(unverifiable) == 1 and unverifiable[0] in {"无", "无。", "None", "-"}:
        unverifiable = []

    evidence = run.get("evidence", {})
    # 依据由代码从证据渲染。渲染不出东西时（未取证拒答、或复放 2026-09-18 之前的
    # 老链路）才回落到模型写的那一段——回落也要有东西可回落，不能整段空掉。
    basis = render_basis(evidence) or _as_items(parts.get(H_BASIS, ""))
    return {
        "question": question,
        "answer": parts.get(H_CONCLUSION, "").strip() or draft.strip(),
        "basis": basis,
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
            # 快路径：认没认出来、接没接管、影子档下与模型自己查的是否一致
            "fastpath": run.get("fastpath") or {},
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
