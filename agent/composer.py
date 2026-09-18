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

import os
import re
import time
from typing import Any

from agent.checklist import build as build_checklists
from tools.pricing import basis as price_basis

H_CONCLUSION = "结论"
H_BASIS = "依据"
H_UNVERIFIABLE = "现有资料无法确认"
# 「没算完」和「要现场看」是两件事，必须分开呈现。
#
# 起因是一次实测：问「全场哪几台构成重复故障」，步数用尽只判完 25 个组合里的 5 个，
# 未判的 20 个被写进「现有资料无法确认」，界面标题是「其他需现场核实的事项」——
# 于是"我还没算"被包装成"要你去现场核实"，而正文那句全称否定看不出任何残缺。
# 前者靠再跑一次就能消掉，后者派人上岛也未必消得掉，混在一起读者无从分辨。
H_INCOMPLETE = "本次未完成"

# 标题行的容错：模型偶发把标题写成 `**## 结论**`（三次复跑中一次），原先的严格式
# 匹配不上 → 整篇原文落进 answer，界面直接显示 `**## 结论**` 裸标记，
# 「现有资料无法确认」整段丢失，而 format_parsed 之外的指标照样全绿。
# 这里放宽到三种写法：`## 结论`、`**## 结论**`、`**结论**`，并容忍结尾冒号。
_HEADING = re.compile(
    r"^[ \t]*(?:"
    r"\**[ \t]*#{1,4}[ \t]*\**"   # ## 结论 / **## 结论** / ##**结论**
    r"|\*{2,3}"                    # **结论**
    r")[ \t]*(结论|依据|现有资料无法确认|本次未完成)[ \t]*[:：]?[ \t]*\**[ \t]*$",
    re.M)

# 全称结论的判据。**只用来标记，不改写正文**——措辞判据一定会有误伤，
# 而误伤一个正确结论的代价，比漏标一个残缺结论更难发现。
#
# 三档开关（COVERAGE_GUARD=off|shadow|enforce，默认 shadow）：
#   off     —— 不判
#   shadow  —— 判并记进 meta，正文不动。先用真实链路标定误伤形状，再决定要不要拦。
#   enforce —— 除记录外，在正文前加一行提示
# 步数用尽的事实提示（_TRUNCATION_NOTICE）不受这个开关管：stop_reason 是机器事实，
# 不是措辞判断，不存在误伤。
# 收紧过一次：最初写成「均不 / 都不」也算，拿 50 条真实链路一跑，5 次命中里 3 次是
# 误伤——「备件均不可用」「备件可用与否都不能免除…」都是正常句子，与覆盖面无关。
# 现在只认两种形态：光杆的「没有一台/无一张」，以及带全场范围词的「全场……均不」。
_UNIVERSAL = re.compile(
    r"没有一[台个张条项]|无一[台个张条项]|没有任何一[台个张条项]"
    r"|(?:全场|全部|所有|九台|各台)[^。；\n]{0,15}(?:均不|都不|全都不|均未|都未)")

def _rule_name(item: Any) -> str | None:
    result = item.get("result") if isinstance(item, dict) else None
    if not isinstance(result, dict):
        return None
    return result.get("rule") or (item.get("args") or {}).get("rule")


def swept_rules(evidence: dict[str, Any]) -> list[str]:
    """本次有哪几条判定是「覆盖全部对象」跑完的。

    只看 stop_reason 判不了覆盖面。实测：scope="all" 一次就把 25 个组合判完了，
    模型拿到完整名单后又去查工单、查手册、判优先级，照样把 6 轮步数用光——
    这时挂「可能未覆盖全部对象」的提示是误报，名单本身是全的。
    coverage.完整 是 sweep 写下的机器事实，用它来分辨「没判完」和「判完了还在查别的」。

    **按 rule 记账，不再是一个布尔**（2026-09-18 改）：原先任一条 sweep 完整就
    整体判「覆盖完整」，多指标问题上这是错的——覆盖面是每项判定各自的属性。
    """
    out: list[str] = []
    for item in evidence.get("rules") or []:
        result = item.get("result") if isinstance(item, dict) else None
        if isinstance(result, dict) and (result.get("coverage") or {}).get("完整"):
            name = _rule_name(item)
            if name and name not in out:
                out.append(name)
    return out


def judged_rules(evidence: dict[str, Any]) -> list[str]:
    """本次实际跑过的全部判定，不论覆盖面。与 swept_rules 的差集即逐对象判的那些。"""
    out: list[str] = []
    for item in evidence.get("rules") or []:
        name = _rule_name(item)
        if name and name not in out:
            out.append(name)
    return out


_TRUNCATION_NOTICE = (
    "> ⚠ 本次因查询步数用尽提前收口，下列结论**可能未覆盖全部对象**，"
    "不要当作全场完整名单。\n\n")

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


# 「无」的各种写法。原先只认四个字面量，模型写「无（本题为手册条文查询，答案完整）。」
# 就被当成一条真实的"无法确认"项渲染出去——给正确答案挂了个假的未知项，
# 后台「无法确认」的计数也跟着虚高。
_NONE_ITEM = re.compile(r"^(?:无|none|n/?a|不适用|-|—)\s*(?:[（(].*[)）])?\s*[。.．、]?$", re.I)

# 提示词里给模型看的格式纪律（第 1.2 条的固定措辞），被模型当成一条业务结论写进
# 「现有资料无法确认」列表。这是元指令泄漏，不是业务内容，确定性剔掉。
# 提示词侧也已改写（见 agent/prompts.py），两头都堵。
_META_ITEM = re.compile(r"第\s*1\.2\s*条.*措辞|措辞.*第\s*1\.2\s*条")


# 结论段里出现这些措辞，说明模型确实认定有事项无法确认。若此时「现有资料无法确认」
# 段是空的，两处就自相矛盾：界面那块面板读的是第二段，会显示为空，
# 等于把模型自己说出来的缺口藏了起来。实测 S3 就是这个形状——正文列了 7 项，
# 第二段写「无」。这里只**标记**不回填：回填要从自由文本里切句子，切错比空着更糟。
_UNVERIFIABLE_HINT = re.compile(r"现有资料无法确认|需要现场核实|无法确认|资料中?均?无记录")

# 「本次未完成」被写进「现有资料无法确认」的形状。两段的分界在 H_INCOMPLETE 那段
# 注释里写得很清楚，提示词里也写了，模型仍然串——2026-09-18 的 25 条真实链路里
# 4 条串了，形态一致：条目自己明说「本轮未完成查询 / 本次未查询 / 需补查」，
# 却挂在「其他需现场核实的事项」面板下。
# 后果不对称：重跑就能消的缺口被包装成派人上岛也未必消得掉的缺口，运维会被指去
# 现场核实一件根本不用去现场的事；反过来真正的现场事项被这些条目稀释。
_NOT_RUN = re.compile(r"本[轮次][^。；\n]{0,8}"
                      r"(?:未查询|未完成|未判定|未做|未跑|没查|未能返回|未能取回|未能执行"
                      r"|未执行|未返回|未取回)"
                      r"|需补查|尚未判定|未做该判定")
# 真·资料缺失的标记，用的是第 1.2 条的固定措辞。两个标记同时出现时**不搬**：
# 这种条目既要重跑也要现场核实，搬走会把现场核实那一半一起带走——
# 漏掉一项现场事项，比多留一条重跑提示危险得多。
_NEEDS_SITE = re.compile(r"需要现场核实|需现场核实|需要现场确认|另行调查")


def _reroute_not_run(unverifiable: list[str], incomplete: list[str]) -> tuple[list[str], list[str], list[str]]:
    """把串到「现有资料无法确认」里的「本次未完成」搬回去。

    判据是确定性的字面标记，不做语义判断：条目自己写了"本轮未完成 / 需补查"，
    这件事就是"还没算"而不是"资料没有"。两个标记都命中时保持原状。
    """
    moved = [u for u in unverifiable if _NOT_RUN.search(u) and not _NEEDS_SITE.search(u)]
    if not moved:
        return unverifiable, incomplete, []
    rest = [u for u in unverifiable if u not in moved]
    # 去重：模型偶发两段都写（F1 实测同一件事同时出现在两个面板），搬过来不能再叠一遍
    merged = incomplete + [m for m in moved if m not in incomplete]
    return rest, merged, moved


# 纯注解行：整行是一对括号里的说明，没有业务内容。模型写「无」时常跟一行
# 「（本次未涉及规程判定，也无遗漏对象。）」解释为什么是无。
_ANNOTATION_ITEM = re.compile(r"^[（(][^)）]*[)）]\s*[。.．、]?$")


def _drop_none_items(items: list[str]) -> list[str]:
    """把「无」及其附带的注解行剔干净。

    原先只认**整段恰好一行**的「无」。模型写成两行——「无。」加一行括号说明——
    两行就都逃过归一化，面板上凭空多出两条"待核实项"，后台「无法确认」的计数
    跟着虚高；这正是 _NONE_ITEM 那个坑的第二形态。
    改为逐条剔：先去掉所有「无」类行，若剩下的全是注解行，这一段实为空段。
    注解行只在**没有任何真实条目**时才丢——与真实条目并列时它可能是上一条的
    续行，丢了就是删内容。
    """
    kept = [i for i in items if not _NONE_ITEM.match(i.strip())]
    if kept and all(_ANNOTATION_ITEM.match(i.strip()) for i in kept):
        return []
    return kept


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


def source_keys(evidence: dict[str, Any]) -> list[str]:
    """本轮真正触达过的数据源 key（不带条款号后缀）。

    与 detect_sources 同源：一份统计两处各写一遍，迟早会出现"数据源 chip 里有、
    未读清单里也有"这种自相矛盾。
    """
    return _scan(evidence)[0]


def _scan(evidence: dict[str, Any]) -> tuple[list[str], set[str]]:
    """从实际用过的东西反推数据源，而不是问模型。返回（有序 key，引用到的条款号）。

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
    return ordered, clauses


def detect_sources(evidence: dict[str, Any]) -> list[str]:
    """数据源 chip：key 换成展示名，规程再带上本次引用到的条款号。"""
    ordered, clauses = _scan(evidence)
    found = []
    for key in ordered:
        label = SOURCE_LABELS.get(key, key)
        # 规程带上条款号：只说"用了规程"等于没说，看的人要能对着条款去查
        if key == "safety_regulation" and clauses:
            label += "（第 %s 条）" % "、".join(sorted(clauses, key=_clause_key))
        found.append(label)
    return found


def build_truncation(run: dict[str, Any], evidence: dict[str, Any],
                     declared_incomplete: bool = False) -> dict[str, Any] | None:
    """步数用尽时，把「本轮没查完」这件事作为事实摆出来。

    起因是 2026-09-18 边界用例 M2 的一次实测：链路在第 6 步用尽预算，
    `maintenance_records` 一次都没查过，收口时却把工单优先级写进了
    「现有资料无法确认」。证据不足有三种成因——资料确实没有、检索没召回、
    执行预算耗尽——第三种被贴上了第一种的标签，看的人会以为"库里查不到"，
    而实际是"系统没去查"。

    这里只陈述可确定的事实：跑了几步、哪张表整轮没被碰过。**不去改判模型写的
    那几条**：判断某一条无法确认到底属于哪种成因需要语义判定，而会误伤的判定
    要先影子跑标定，不能直接上线拦人。事实摆出来，归类交给看的人。
    """
    if run.get("stop_reason") != "max_steps":
        return None
    if swept_rules(evidence) and not declared_incomplete:
        # 步数用尽 ≠ 覆盖不全。实测 scope="all" 一次判完 25 个组合后，模型又去查
        # 工单、查手册，照样把 6 轮用光；此时挂「本轮未完成」是误报，名单是全的。
        # coverage.完整 是 sweep 写下的机器事实，用它把两种情形分开。
        # 但模型自己写下「本次未完成」时不适用：那是它对同一个问题的另一项判定
        # 没跑完，一条 sweep 完整证明不了别的判定也完整。
        return None
    touched = set(source_keys(evidence))
    unread = [SOURCE_LABELS[k] for k in ("alarm_records", "maintenance_records")
              if k not in touched]
    return {
        "reason": "max_steps",
        "steps": len(run.get("trace", [])),
        "unread_sources": unread,
        "note": "本轮查询步数已用尽，链路未跑完即收口。下方「现有资料无法确认」中"
                "如涉及数据库字段，可能属于**本轮未查**而非资料中没有记录，"
                "重新提问或把问题拆细可继续核实。",
    }


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


# 逐项列举里漏掉对象的判据。取数捞回来 15 张工单、正文只列了 14 张却写着
# 「其余均无问题」——这种漏是静默的：读者没有第二份名单可对，看不出少了一张。
_ENTITY_PATTERNS = (
    ("work_order_id", re.compile(r"WO-\d{6}")),
    ("turbine_id", re.compile(r"(?<![0-9A-Za-z])T\d{2}(?![0-9])")),
)


_ID_SUFFIX = re.compile(r"^[A-Za-z]+-(\d{4,})$")


def _suffix_mentioned(entity: str, answer: str) -> bool:
    """「WO-260702」被缩写成「260702」时也算提到了。"""
    m = _ID_SUFFIX.match(entity)
    return bool(m) and re.search(r"(?<!\d)%s(?!\d)" % m.group(1), answer) is not None


def enumeration_gap(answer: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """正文逐项列举时，漏掉了取数结果里的哪些对象。

    只在「正文确实在逐项列举」时才判：提到的对象数要过半。否则一句
    「WO-260708 关得不合规」会被判成「另外 14 张全漏了」——那是正常的单点回答。
    """
    gaps: dict[str, Any] = {}
    for key, pattern in _ENTITY_PATTERNS:
        pool: set[str] = set()
        for table in evidence.get("tables") or []:
            for row in table.get("rows") or []:
                for value in row.values():
                    if isinstance(value, str):
                        pool.update(pattern.findall(value))
        if len(pool) < 3:
            continue
        # 正文常把工单号缩写成「WO-260701、260702、260704」——后面几个省掉了前缀。
        # 只按完整 id 匹配的话，这种正常写法会被判成「漏了 9 张」（实测 50 条里
        # 误伤 1 条）。所以带前缀的 id 再认一次它的数字尾部。
        mentioned = {e for e in pool if e in answer or _suffix_mentioned(e, answer)}
        missing = sorted(pool - mentioned)
        if missing and len(mentioned) >= max(2, len(pool) / 2):
            gaps[key] = {"证据内": len(pool), "正文提到": len(mentioned), "未提到": missing}
    return gaps


def compose(question: str, run: dict[str, Any]) -> dict[str, Any]:
    draft = run.get("draft") or ""
    parts = _split_sections(draft)

    unverifiable = _as_items(parts.get(H_UNVERIFIABLE, ""))
    # 元指令泄漏先剔掉，再判空：否则一条泄漏行会把「无」撑成"有内容"
    unverifiable = [u for u in unverifiable if not _META_ITEM.search(u)]
    # 「无」是有依据的空，不是漏答
    unverifiable = _drop_none_items(unverifiable)

    incomplete = _as_items(parts.get(H_INCOMPLETE, ""))
    incomplete = [i for i in incomplete if not _META_ITEM.search(i)]
    incomplete = _drop_none_items(incomplete)

    # 串段归位要在一致性判定之前：搬完才知道「无法确认」面板最终有没有内容
    unverifiable, incomplete, rerouted = _reroute_not_run(unverifiable, incomplete)

    # 两段式一致性：结论里说了"无法确认"，第二段却是空的
    conclusion = parts.get(H_CONCLUSION, "")
    unverifiable_inconsistent = bool(
        not unverifiable and conclusion and _UNVERIFIABLE_HINT.search(conclusion))

    evidence = run.get("evidence", {})
    answer = parts.get(H_CONCLUSION, "").strip() or draft.strip()
    truncated = run.get("stop_reason") == "max_steps"
    swept = swept_rules(evidence)
    per_subject = [r for r in judged_rules(evidence) if r not in swept]
    # 一条完整 sweep 只证明**那一项判定**判完了，证明不了整个回答覆盖完整。
    # 多指标问题上这点是致命的：实测「全场是不是都不构成重复故障、是不是都不用现场
    # 检修」——repeat_fault 走了 scope="all"，coverage_complete 就成了 true，步数用尽的
    # ⚠ 提示被抑制；而模型自己在「本次未完成」里写着另外两类判定没跑。
    # 模型自述没判完，是比"有一条 sweep 完整"更强的证据，以它为准。
    coverage_complete = bool(swept) and not incomplete
    guard = (os.getenv("COVERAGE_GUARD") or "shadow").strip().lower()
    universal = bool(_UNIVERSAL.search(answer))

    if truncated and not coverage_complete:
        # 步数用尽是机器事实，必须留痕，且不能只留在 meta 里——正文是主呈现面。
        answer = _TRUNCATION_NOTICE + answer
        if not incomplete:
            # 模型没自觉写「本次未完成」，代码补一条：这段为空等于宣称判完了。
            incomplete = ["本次因查询步数用尽提前收口，未能逐一判定全部对象；"
                          "重跑本问题或缩小范围（指定风机 / 工单）可得到完整结论。"]

    gaps = enumeration_gap(answer, evidence)
    if gaps and guard == "enforce":
        answer += "\n\n> ⚠ 取数结果中有 %s 未在上文列出。" % "；".join(
            "%d 个%s（%s）" % (len(g["未提到"]), k, "、".join(g["未提到"][:8]))
            for k, g in gaps.items())

    if universal and not truncated and guard == "enforce":
        answer = ("> ⚠ 本条含「全场均不…」这类全称结论，请核对下方依据是否已覆盖全部对象。"
                  "\n\n") + answer

    # 依据由代码从证据渲染。渲染不出东西时（未取证拒答、或复放 2026-09-18 之前的
    # 老链路）才回落到模型写的那一段——回落也要有东西可回落，不能整段空掉。
    basis = render_basis(evidence) or _as_items(parts.get(H_BASIS, ""))
    return {
        "question": question,
        "answer": answer,
        "basis": basis,
        "unverifiable": unverifiable,
        # 与 unverifiable 分开的第二个面。两者说的都不是「资料里没有」，但成因不同：
        #   incomplete —— 模型自己写下的「该判而未判」，重跑即可消除
        #   truncation —— 代码陈述的机器事实：跑了几步、哪张表整轮没被碰过
        # 界面合成一张卡展示，避免同一件事出现两块提示。
        "incomplete": incomplete,
        "truncation": build_truncation(run, evidence, bool(incomplete)),
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
            # 结论里写了"无法确认"、第二段却空着 —— 面板会是空的，与正文矛盾
            "unverifiable_inconsistent": unverifiable_inconsistent,
            # 步数用尽收口 —— 结论的覆盖面存疑，后台按它筛链路
            "truncated": truncated,
            # 证据里有没有一条覆盖全部对象的判定 —— 决定 truncated 要不要提示用户
            "coverage_complete": coverage_complete,
            # 按 rule 记的覆盖面：哪几项判定是全场 sweep 跑的，哪几项是逐对象跑的。
            # 一个布尔答不出"多指标问题里到底哪一项没判完"，标定时要的就是这张账。
            "coverage_rules": {"swept": swept, "per_subject": per_subject},
            # 从「现有资料无法确认」搬回「本次未完成」的条目，留痕供复核判据是否误伤
            "section_rerouted": rerouted,
            # 全称结论标记：shadow 档只记不改，供标定误伤形状
            "universal_claim": universal,
            # 逐项列举漏了哪些对象（shadow 档只记不改）
            "enumeration_gap": gaps,
            "coverage_guard": guard,
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
