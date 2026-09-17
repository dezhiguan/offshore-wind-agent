# -*- coding: utf-8 -*-
"""简单问题快路径：确定性路由掉第一次决策调用。

**为什么值得做**：26 条用例里 13 条只调一次工具，总耗时 P50 6.6 秒，其中第一次
「Agent 决策」P50 2.5 秒——占 38%。那一次调用做的事只是把「T06 的最新告警是什么」
翻译成一条 SQL，而这类问句的形状是固定的，代码认得出来。

**为什么必须保守**：认错了就不是慢一点，是拿另一个问题的答案去作答。因此：

  · 白名单从**真实跑测产物**里长出来，不是想出来的。当前两族对应 P1/Q1/S1（单台风机
    最新告警或型号）与 Q2（单台风机时间窗告警），SQL 模板照抄模型自己写过的那几条。
  · 每一族都带**否定门**：问句里只要出现故障码、工单、规程、条款、「几次」「是否」
    这类信号，就说明它要的不止一条 SQL，一律不接管。
  · 一族只认**恰好一个**风机编号。两个编号的问题是对比题，不是简单查询。
  · P5「哪些工单是 OPEN」、P9「哪些工单备件不可用」在用例里各只有一条。从单个样本
    泛化出模式，正是误判的来源，因此不收——覆盖率宁可低，不可错。

**兜底比判据更重要**：快路径**不是终局**。它只是替模型先发了第一次工具调用，循环照常
往下走——数据不够、或者压根查错了方向，模型自己会再调工具。最坏情况是上下文里多一份
没用上的查询结果（且它按正常口径计入预算），而不是答错。这条性质决定了这个改动的风险
上界，比任何判据都关键。

三档：off / shadow（默认）/ enforce。shadow 档照常匹配、照常执行 SQL，但**不注入、
不计费**，只把「如果接管了，结果与模型自己查的是否一致」记进 meta 供标定。
会误伤的判定层先影子跑再拦截——单文档红线档那次误伤（见 tools/budget.py）就是
没有先影子跑的代价。
"""
from __future__ import annotations

import os
import re
from typing import Any

MODE = os.getenv("FASTPATH_MODE", "shadow").lower()

_TURBINE = re.compile(r"\bT\d{2}\b")
_FAULT_CODE = re.compile(r"(?<!\d)\d{5}(?!\d)")
_WORK_ORDER = re.compile(r"WO-\w+")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")

# 出现任意一个就不接管：它们都意味着这道题不止一条 SQL。
# 「几次」「是否」「应该」这类是规则判定的信号——提示词明令那些必须走 check_rule，
# 快路径抢在前面发一条 SELECT，只会让模型拿着半截数据自己心算。
_STOP_WORDS = (
    "工单", "规程", "手册", "条款", "第", "排查", "原因", "怎么", "如何", "为什么",
    "复位", "关单", "关闭", "合规", "优先级", "升级", "备件", "更换", "注意事项",
    "几次", "多少次", "是否", "能否", "可以吗", "应该", "建议", "对比", "分别",
)

# 单条最新告警：字段照抄 P1 那条模型自己写的 SQL，少一列都会让「型号/状态/时间」
# 三问里有一问答不上来
_LATEST_COLUMNS = ("turbine_id, turbine_model, turbine_status, fault_code, fault_name, "
                   "severity, occurred_at, alarm_status")
_WINDOW_COLUMNS = ("occurred_at, fault_code, fault_name, severity, alarm_status, turbine_status")


def _sole_turbine(question: str) -> str | None:
    """恰好一个风机编号才认。两个编号是对比题，零个无从查起。"""
    found = set(_TURBINE.findall(question))
    return found.pop() if len(found) == 1 else None


def _blocked(question: str) -> str | None:
    if _FAULT_CODE.search(question):
        return "问句里有故障代码，多半要连规程或手册一起判"
    if _WORK_ORDER.search(question):
        return "问句里有工单编号"
    for word in _STOP_WORDS:
        if word in question:
            return "问句里出现「%s」，不是单条 SQL 能答完的" % word
    return None


def match(question: str) -> dict[str, Any] | None:
    """认出来就返回 {name, sql, max_rows, must_have}，认不出来返回 None。"""
    if MODE == "off":
        return None
    turbine = _sole_turbine(question)
    if not turbine:
        return None
    blocked = _blocked(question)
    if blocked:
        return None

    stamps = _TIMESTAMP.findall(question)
    if len(stamps) == 2 and ("告警" in question or "报警" in question):
        start, end = _normalize(stamps[0], False), _normalize(stamps[1], True)
        return {
            "name": "turbine_alarms_in_window",
            "why": "单台风机 + 明确起止时间的告警查询",
            "sql": ("SELECT %s FROM alarm_records WHERE turbine_id = '%s' "
                    "AND occurred_at BETWEEN '%s' AND '%s' ORDER BY occurred_at"
                    % (_WINDOW_COLUMNS, turbine, start, end)),
            "max_rows": 200,
            "must_have": ("occurred_at", "fault_code", "alarm_status"),
        }
    if stamps:
        # 只写了一个时间点、或写了三个以上：口径不明确，交回模型
        return None
    latest = ("最新" in question or "最近" in question) and ("告警" in question or "报警" in question)
    if latest or "型号" in question:
        return {
            "name": "turbine_latest_alarm",
            "why": "单台风机的最新一条告警 / 机型查询",
            "sql": ("SELECT %s FROM alarm_records WHERE turbine_id = '%s' "
                    "ORDER BY occurred_at DESC LIMIT 1" % (_LATEST_COLUMNS, turbine)),
            "max_rows": 1,
            "must_have": ("turbine_model", "turbine_status", "occurred_at"),
        }
    return None


def _normalize(stamp: str, is_end: bool) -> str:
    """只写到日期时补足时分秒，闭区间的右端补到当天末尾。

    不补的话 '2026-07-15' 会被当成 00:00:00，把当天整天漏在窗口外——
    而这正是提示词里反复交代「时间按闭区间比较」要防的那个错。
    """
    stamp = stamp.replace("T", " ")
    if len(stamp) == 10:
        return stamp + (" 23:59:59" if is_end else " 00:00:00")
    if len(stamp) == 16:
        return stamp + (":59" if is_end else ":00")
    return stamp


def shape_ok(plan: dict[str, Any], result: dict[str, Any]) -> str | None:
    """结果形状安全网。返回不通过的原因，或 None 表示可以用。

    **空结果是合格的**：查 T20 查不到，「数据库里没有这台风机的告警」就是正确答案。
    把零行当失败会让快路径在最该发挥作用的那一类问题上退回去。
    """
    if not result.get("ok", True):
        return "查询未成功：%s" % str(result.get("error"))[:60]
    count = result.get("row_count", 0)
    if count > plan["max_rows"]:
        return "返回 %d 行，超出该模板的上限 %d 行" % (count, plan["max_rows"])
    rows = result.get("rows") or []
    if rows:
        missing = [c for c in plan["must_have"] if c not in rows[0]]
        if missing:
            return "结果缺少字段 %s" % "、".join(missing)
    return None


def agrees(fast: dict[str, Any], model: dict[str, Any]) -> bool:
    """影子标定用：模型自己查到的每一行，快路径是否都查到了同样的内容。

    不要求列集相同——模型常只 SELECT 自己要的几列，而快路径取的是整族字段。
    判据是**包含**：模型那一行的每个字段取值，都能在快路径某一行里找到。
    """
    if fast.get("row_count") != model.get("row_count"):
        return False
    fast_rows = fast.get("rows") or []
    for row in model.get("rows") or []:
        if not any(all(k in candidate and candidate[k] == v for k, v in row.items())
                   for candidate in fast_rows):
            return False
    return True
