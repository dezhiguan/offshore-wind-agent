# -*- coding: utf-8 -*-
"""上下文预算计量。

题面红线：不得将整个数据库或文档全部放入 LLM 上下文。

这里把"没超"从行为变成机制，并且可计量：累计每次会话进入上下文的数据行与文档字符，
超过语料占比阈值即拒绝继续取证，同时把用量随回答一并返回，供界面展示与事后审计。

**两个口径分开记，不要混成一个百分比**（2026-09-17 修）：

  · 覆盖（coverage）—— 去重后的语料占比。同一节被 search_docs 命中又被
    get_doc_section 取回、同一批行被两条 SQL 各查一次，只算一次。
    这一档回答的是"这次会话读到了语料的百分之多少"，也就是红线的字面要求。
  · 取回（drawn）—— 不去重的累计量。重复取回确实会在上下文里各占一份，
    所以它才是"上下文被压到哪"的那个数，也用来堵住"分多次重复取"绕过覆盖判据。

两档共用同一组阈值，任何一档超了都拒绝 —— 只用覆盖判据会比改造前更松，
只用取回判据则会把"反复引用同一条款"误判成"要把整份文档搬进去"。

另有一个纯展示量 ``context_chars``：工具返回真正注入上下文的字符总数（含 JSON
结构与结论文本）。它没有分母 —— 语料占比的分子只能是语料原文，把结论文本算进去，
分子分母就不是同一件东西了。实测一次 check_rule 返回 1260 字符，其中语料原文
只有 304：把 304 当作"上下文占用"会低估，把 1260 当作"语料占比"会虚高。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 超过语料的这个比例即视为接近"整体载入"，拒绝继续取证。
#
# 两侧取值不同，是因为正常用量分布不同：数据侧实测峰值 24%，余量充足；
# 文档侧一个复杂合规问题天然需要 5~8 个条款加一节手册，实测峰值已达 47%
# （P2「不存在的故障代码」检索面最广）。阈值定在 50% 只剩 1.06 倍余量，
# 会让正常问题误触发。护栏的职责是拦住"把整份文档搬进去"，不是卡住正常取证，
# 因此文档侧放宽到 70%——仍然明显低于"全部"，但不会误伤。
MAX_ROW_RATIO = 0.5
MAX_DOC_RATIO = 0.7

_totals: dict[str, int] | None = None
_CLAUSE_KEY = re.compile(r"(\d+(?:\.\d+)*)")


def corpus_totals() -> dict[str, int]:
    """语料总量。只在首次调用时统计一次。"""
    global _totals
    if _totals is None:
        from tools.db import query_db

        rows = 0
        for table in ("alarm_records", "maintenance_records"):
            probe = query_db("SELECT COUNT(*) AS n FROM %s" % table)
            rows += probe["rows"][0]["n"] if probe["ok"] else 0
        chars = sum(
            len(p.read_text(encoding="utf-8"))
            for p in DATA_DIR.glob("*.md")
            if p.name in ("故障处理手册.md", "海上风电机组检修作业与安全管理规程.md")
        )
        _totals = {"db_rows": rows, "doc_chars": chars}
    return _totals


class Draw(NamedTuple):
    """一次工具调用从语料里取走了什么。

    ``rows`` / ``units`` 带的是**去重键**而不只是数量：只记数量就无法区分
    "又读了 9 行新的"和"把刚才那 9 行换个投影再读一遍"。
    """

    rows: tuple[str, ...] = ()
    units: tuple[tuple[str, int], ...] = ()
    context_chars: int = 0

    @classmethod
    def raw(cls, rows: int = 0, chars: int = 0, tag: str = "raw") -> "Draw":
        """按数量直接造一次取数 —— 每一项都当作互不相同的新内容。

        给测试与没有行级明细的退化路径用：宁可当作全新（更早触发红线），
        也不要因为认不出来而当成重复（静默放行）。
        """
        keys = tuple("%s#%d" % (tag, i) for i in range(rows))
        units = ((("%s:chars" % tag), chars),) if chars else ()
        return cls(keys, units, chars)


class ContextBudget:
    """单次会话的用量计。每轮问答新建一个。"""

    def __init__(self) -> None:
        # 覆盖：去重后的语料占比
        self._rows_seen: set[str] = set()
        self._units_seen: dict[str, int] = {}
        # 取回：不去重的累计量
        self.drawn_rows = 0
        self.drawn_chars = 0
        # 展示量：工具返回实际注入上下文的字符总数
        self.context_chars = 0
        # 被红线拦下的取证：拦截前的水位 + 这一次想压到哪
        self.refusals: list[dict[str, Any]] = []
        totals = corpus_totals()
        self.max_rows = int(totals["db_rows"] * MAX_ROW_RATIO)
        self.max_chars = int(totals["doc_chars"] * MAX_DOC_RATIO)

    @property
    def db_rows(self) -> int:
        """覆盖到的语料行数（去重）。"""
        return len(self._rows_seen)

    @property
    def doc_chars(self) -> int:
        """覆盖到的语料字符数（去重；同一节按见过的最大篇幅计）。"""
        return sum(self._units_seen.values())

    def _new_rows(self, draw: Draw) -> int:
        return len({k for k in draw.rows if k not in self._rows_seen})

    def _new_chars(self, draw: Draw) -> int:
        """同一节先拿到 160 字摘要、后拿到全文时，只补记增量。"""
        best: dict[str, int] = {}
        for key, chars in draw.units:
            best[key] = max(best.get(key, 0), chars)
        return sum(max(0, chars - self._units_seen.get(key, 0)) for key, chars in best.items())

    def would_exceed(self, draw: Draw) -> str | None:
        """预检。返回拒绝原因，或 None 表示放行。

        两档分别判：覆盖档说"这次会话已经读到语料的多少"，取回档说"上下文里
        已经堆了多少"。文案里把是哪一档写明，否则模型收到拒绝却不知道该收窄
        查询范围还是别再重复取同一节。
        """
        totals = corpus_totals()
        # 拒绝文案里必须带上"这一次想引入多少"：只报已用量的话，会话刚开始就被拦下时
        # 界面上是"覆盖 0/3 行"，看起来像护栏自己出了问题。
        new_rows, new_chars = self._new_rows(draw), self._new_chars(draw)
        if self.db_rows + new_rows > self.max_rows:
            return ("本次取数会让上下文超过上限（本次引入 %d 行，会话已覆盖 %d/%d 行，"
                    "语料共 %d 行）。系统不允许把整张表载入上下文，"
                    "请缩小查询范围（加筛选条件或聚合）。"
                    % (new_rows, self.db_rows, self.max_rows, totals["db_rows"]))
        if self.drawn_rows + len(draw.rows) > self.max_rows:
            return ("本次取数会让上下文超过上限（本次取回 %d 行，会话累计取回 %d/%d 行，"
                    "含重复读取，语料共 %d 行）。系统不允许把整张表载入上下文，"
                    "请缩小查询范围（加筛选条件或聚合），并避免重复查询已经拿到的行。"
                    % (len(draw.rows), self.drawn_rows, self.max_rows, totals["db_rows"]))
        if self.doc_chars + new_chars > self.max_chars:
            return ("本次取回文档会让上下文超过上限（本次引入 %d 字符，会话已覆盖 %d/%d 字符，"
                    "语料共 %d 字符）。系统不允许把整份文档载入上下文，"
                    "请只取回确实需要引用的章节。"
                    % (new_chars, self.doc_chars, self.max_chars, totals["doc_chars"]))
        if self.drawn_chars + sum(c for _, c in draw.units) > self.max_chars:
            return ("本次取回文档会让上下文超过上限（本次取回 %d 字符，会话累计取回 %d/%d 字符，"
                    "含重复取回，语料共 %d 字符）。系统不允许把整份文档载入上下文，"
                    "请只取回确实需要引用的章节，已经拿到的不要再取第二次。"
                    % (sum(c for _, c in draw.units), self.drawn_chars, self.max_chars,
                       totals["doc_chars"]))
        return None

    def note_refusal(self, draw: Draw, reason: str) -> None:
        """记下一次被红线拒绝的取证 —— 拒绝**不 charge**，所以它不会留在用量里。

        只报已计费的量，红线水位就永远画不出撞线：被拒的那次不进账，覆盖率停在
        阈值以下，看板上写着"离红线还有余量"，而这条链路实际已经被拒过一次。
        真正该看的是**试图压到的水位**——它可以超过阈值，也可以超过 100%。

        两档共用同一个阈值、同一个单位，取两者较大的那个：它就是这一侧这次
        想压到的最高水位，与是哪一档触发的无关。
        """
        totals = corpus_totals()
        rows = max(self.db_rows + self._new_rows(draw),
                   self.drawn_rows + len(draw.rows))
        chars = max(self.doc_chars + self._new_chars(draw),
                    self.drawn_chars + sum(c for _, c in draw.units))
        pct = lambda num, den: round(num / den * 100, 1) if den else 0.0  # noqa: E731
        self.refusals.append({
            "reason": reason,
            "attempt_db_rows": rows,
            "attempt_doc_chars": chars,
            "attempt_db_rows_pct": pct(rows, totals["db_rows"]),
            "attempt_doc_chars_pct": pct(chars, totals["doc_chars"]),
        })

    def charge(self, draw: Draw) -> None:
        self.drawn_rows += len(draw.rows)
        self.drawn_chars += sum(c for _, c in draw.units)
        self.context_chars += draw.context_chars
        self._rows_seen.update(draw.rows)
        for key, chars in draw.units:
            self._units_seen[key] = max(self._units_seen.get(key, 0), chars)

    def report(self) -> dict[str, Any]:
        totals = corpus_totals()
        pct = lambda num, den: round(num / den * 100, 1) if den else 0.0  # noqa: E731
        return {
            # 覆盖档：去重后的语料占比，红线的字面口径
            "db_rows": self.db_rows,
            "db_rows_total": totals["db_rows"],
            "db_rows_pct": pct(self.db_rows, totals["db_rows"]),
            "doc_chars": self.doc_chars,
            "doc_chars_total": totals["doc_chars"],
            "doc_chars_pct": pct(self.doc_chars, totals["doc_chars"]),
            # 取回档：含重复，界面上与覆盖档并列显示，两者差值就是重复取回量
            "db_rows_drawn": self.drawn_rows,
            "doc_chars_drawn": self.drawn_chars,
            # 工具返回真正注入上下文的字符总数（含 JSON 结构与结论文本，无分母）
            "context_chars": self.context_chars,
            "limit_rows": self.max_rows,
            "limit_chars": self.max_chars,
            # 护栏拦截台账：没有拦截过就是 None，不是 0 —— 界面据此决定要不要出现这一行
            "refusals": len(self.refusals),
            "peak_attempt_db_rows_pct": max((r["attempt_db_rows_pct"] for r in self.refusals),
                                            default=None),
            "peak_attempt_doc_chars_pct": max((r["attempt_doc_chars_pct"] for r in self.refusals),
                                              default=None),
        }


def _fingerprint(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                   default=str).encode("utf-8")).hexdigest()[:12]


def _m_query_db(result: dict[str, Any]) -> Draw:
    """按行指纹计量。

    **计的是结果行，不是被扫到的库行**：`SELECT DISTINCT turbine_id` 扫全表 47 行
    只返回 9 行，就记 9 行；`COUNT(*)` 记 1 行。这是刻意的 —— 红线防的是"把整张表
    的内容搬进上下文"，聚合结果并没有把原始行搬进去。同理，不同投影的同一条记录
    指纹不同、各计一次，因为它们确实是两批不同的内容。
    """
    rows = result.get("rows")
    if rows is None:
        # 没有行级明细的退化路径：当作全新内容，宁可早拦不可漏计
        return Draw.raw(rows=result.get("row_count", 0), tag=_fingerprint(result))
    return Draw(tuple(_fingerprint(r) for r in rows))


def _m_get_doc_section(result: dict[str, Any]) -> Draw:
    text = result.get("text") or ""
    if not text:
        return Draw()
    key = "%s:%s" % (result.get("doc_key") or result.get("doc") or "?",
                     result.get("section_id") or "?")
    return Draw(units=((key, len(text)),))


def _m_search_docs(result: dict[str, Any]) -> Draw:
    units = []
    for hit in result.get("hits", []):
        key = "%s:%s" % (hit.get("doc_key") or hit.get("doc") or "?",
                         hit.get("section_id") or "?")
        units.append((key, len(hit.get("snippet") or "")))
    return Draw(units=tuple(units))


def _m_check_rule(result: dict[str, Any]) -> Draw:
    """规则结论里内联了条款原文，同样占上下文。

    去重键与 get_doc_section 对齐（safety_regulation:3.1）——同一条款先被规则
    内联、后又被单独取回，那是同一段原文，不该在覆盖率里算两遍。
    """
    units = []
    for label, text in (result.get("clause_texts") or {}).items():
        clause = _CLAUSE_KEY.search(str(label))
        key = "safety_regulation:%s" % (clause.group(1) if clause else label)
        units.append((key, len(text)))
    return Draw(units=tuple(units))


# 每个工具都必须在此登记计量规则。新增工具却忘了登记，会静默不计费——
# 那正是最难发现的一类缺陷，因此 tests/test_budget.py 有一条守卫：
# TOOL_REGISTRY 里的每个工具都必须在这里有对应项，否则测试失败。
METERS = {
    "query_db": _m_query_db,
    "get_doc_section": _m_get_doc_section,
    "search_docs": _m_search_docs,
    "check_rule": _m_check_rule,
}


def measure(name: str, result: dict[str, Any]) -> Draw:
    """一次工具调用从语料里取走了什么，又往上下文里塞了多少字符。"""
    if not result.get("ok", True):
        return Draw()
    payload = json.dumps(result, ensure_ascii=False, default=str)
    meter = METERS.get(name)
    if meter is None:
        # 未登记的工具按最保守处理：把它整个结果的字符数计入文档预算。
        # 宁可高估被拦，也不要静默漏计。
        return Draw(units=(("?%s" % _fingerprint(result), len(payload)),),
                    context_chars=len(payload))
    return meter(result)._replace(context_chars=len(payload))
