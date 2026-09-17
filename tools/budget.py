# -*- coding: utf-8 -*-
"""上下文预算计量。

题面红线：不得将整个数据库或文档全部放入 LLM 上下文。

实测 21 条链路，单次会话最多只用到全库 19% 的行、全部文档块的 6%——但那是**事后统计**，
不是运行时保证。架构上并不存在阻止 `SELECT * FROM alarm_records` 的机制（全表 47 行，
低于自动追加的 LIMIT 200）。

这里把"没超"从行为变成机制，并且可计量：累计每次会话进入上下文的数据行与文档字符，
超过语料占比阈值即拒绝继续取证，同时把用量随回答一并返回，供界面展示与事后审计。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

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


class ContextBudget:
    """单次会话的用量计。每轮问答新建一个。"""

    def __init__(self) -> None:
        self.db_rows = 0
        self.doc_chars = 0
        totals = corpus_totals()
        self.max_rows = int(totals["db_rows"] * MAX_ROW_RATIO)
        self.max_chars = int(totals["doc_chars"] * MAX_DOC_RATIO)

    def would_exceed(self, rows: int = 0, chars: int = 0) -> str | None:
        """预检。返回拒绝原因，或 None 表示放行。"""
        if self.db_rows + rows > self.max_rows:
            return ("本次会话累计取数已达上限（%d/%d 行，语料共 %d 行）。"
                    "系统不允许把整张表载入上下文，请缩小查询范围（加筛选条件或聚合）。"
                    % (self.db_rows, self.max_rows, corpus_totals()["db_rows"]))
        if self.doc_chars + chars > self.max_chars:
            return ("本次会话累计取回文档已达上限（%d/%d 字符，语料共 %d 字符）。"
                    "系统不允许把整份文档载入上下文，请只取回确实需要引用的章节。"
                    % (self.doc_chars, self.max_chars, corpus_totals()["doc_chars"]))
        return None

    def charge(self, rows: int = 0, chars: int = 0) -> None:
        self.db_rows += rows
        self.doc_chars += chars

    def report(self) -> dict[str, Any]:
        totals = corpus_totals()
        return {
            "db_rows": self.db_rows,
            "db_rows_total": totals["db_rows"],
            "db_rows_pct": round(self.db_rows / totals["db_rows"] * 100, 1) if totals["db_rows"] else 0.0,
            "doc_chars": self.doc_chars,
            "doc_chars_total": totals["doc_chars"],
            "doc_chars_pct": round(self.doc_chars / totals["doc_chars"] * 100, 1) if totals["doc_chars"] else 0.0,
            "limit_rows": self.max_rows,
            "limit_chars": self.max_chars,
        }


def _m_query_db(result: dict[str, Any]) -> tuple[int, int]:
    return result.get("row_count", 0), 0


def _m_get_doc_section(result: dict[str, Any]) -> tuple[int, int]:
    return 0, len(result.get("text") or "")


def _m_search_docs(result: dict[str, Any]) -> tuple[int, int]:
    return 0, sum(len(h.get("snippet") or "") for h in result.get("hits", []))


def _m_check_rule(result: dict[str, Any]) -> tuple[int, int]:
    # 规则结论里内联了条款原文，同样占上下文
    return 0, sum(len(t) for t in (result.get("clause_texts") or {}).values())


# 每个工具都必须在此登记计量规则。新增工具却忘了登记，会静默不计费——
# 那正是最难发现的一类缺陷，因此 tests/test_budget.py 有一条守卫：
# TOOL_REGISTRY 里的每个工具都必须在这里有对应项，否则测试失败。
METERS = {
    "query_db": _m_query_db,
    "get_doc_section": _m_get_doc_section,
    "search_docs": _m_search_docs,
    "check_rule": _m_check_rule,
}


def measure(name: str, result: dict[str, Any]) -> tuple[int, int]:
    """一次工具调用给上下文带来多少行、多少文档字符。"""
    if not result.get("ok", True):
        return 0, 0
    meter = METERS.get(name)
    if meter is None:
        # 未登记的工具按最保守处理：把它整个结果的字符数计入文档预算。
        # 宁可高估被拦，也不要静默漏计。
        return 0, len(str(result))
    return meter(result)
