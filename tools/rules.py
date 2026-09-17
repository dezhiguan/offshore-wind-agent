# -*- coding: utf-8 -*-
"""确定性规则引擎。

《检修作业与安全管理规程》里的判定全部是可计算的：24 小时滑动窗口计数、
120 分钟观察阈值、关闭必备的七项记录、更换前置的九项条件。这些一律由代码算，
LLM 只负责引用结论。

为什么不交给模型：次数、时长、阈值比较这类算术，模型算错时错得很自然、
看不出来——比如把「任意连续 24 小时」当成自然日分组，T03 就会算成 3 次而不是
4 次，结论碰巧仍对但过程错。有确定性实现，现场被追问时才答得出。

每条规则都返回三段：
  verdict   一句话结论，供模型直接引用
  facts     支撑结论的事实（带原始记录，可回溯）
  unverifiable  资料确实没有记录的事项（规程第 1.2 条要求显式标注）
  clauses   本次引用到的条款号
"""
from __future__ import annotations

import re
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any

from tools.db import query_db

REPEAT_WINDOW = timedelta(hours=24)
REPEAT_THRESHOLD = 3          # 规程第 3.1 条
MIN_OBSERVATION_MINUTES = 120  # 规程第 6.4 条
TS_FMT = "%Y-%m-%d %H:%M:%S"

# 现场条件数据库一概不记录（规程第 1.2 条列举）
SITE_CONDITIONS = [
    "退出远程控制状态", "主电源隔离", "挂牌上锁", "验电结果",
    "母线电压是否降至约 20 V", "相关开关（如 Q6/Q13/Q7/Q8/Q41）状态",
    "现场人员资质", "海况、天气及作业环境",
]

_RE_TURBINE = re.compile(r"^T\d{2}$")
_RE_CODE = re.compile(r"^\d{4,6}$")
_RE_WO = re.compile(r"^WO-\w+$")


class RuleInputError(ValueError):
    pass


# 规则引擎自己查库、自己读手册、自己把规程条款代码化，这三件事都不经过模型的工具调用。
# 「使用的数据源」如果只统计模型点过什么，就会漏掉结论真正的出处 —— T05 的不合规
# 判定完全来自规程第 2.1 / 2.4 条，面板上却连规程都不显示。这里把每次规则执行实际
# 触达的数据源记下来，随结果一起交出去。
_TOUCHED: ContextVar[set[str] | None] = ContextVar("rule_sources", default=None)

_TABLES = ("alarm_records", "maintenance_records")


def _touch(source: str) -> None:
    touched = _TOUCHED.get()
    if touched is not None:
        touched.add(source)


def _lit(value: str, pattern: re.Pattern, name: str) -> str:
    """校验后再拼进 SQL。

    这些值来自模型生成的参数，不能直接拼接。规则引擎内部的查询是固定模板，
    只需把取值限制成白名单格式即可，比在这里再引一层参数化更简单也更好审。
    """
    v = (value or "").strip().upper() if name != "工单编号" else (value or "").strip()
    if not pattern.match(v):
        raise RuleInputError("%s 格式不正确：%r" % (name, value))
    return v


def _rows(sql: str) -> list[dict[str, Any]]:
    result = query_db(sql)
    if not result["ok"]:
        raise RuleInputError(result["error"])
    for table in _TABLES:
        if table in sql:
            _touch(table)
    return result["rows"]


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FMT)


def _alarms(turbine_id: str, fault_code: str) -> list[dict[str, Any]]:
    return _rows(
        "SELECT alarm_id, turbine_model, turbine_status, fault_code, fault_name, "
        "severity, occurred_at, alarm_status FROM alarm_records "
        "WHERE turbine_id='%s' AND fault_code='%s' ORDER BY occurred_at" % (turbine_id, fault_code)
    )


def _orders(turbine_id: str, fault_code: str) -> list[dict[str, Any]]:
    return _rows(
        "SELECT * FROM maintenance_records WHERE turbine_id='%s' AND fault_code='%s' "
        "ORDER BY created_at" % (turbine_id, fault_code)
    )


def _manual_text(fault_code: str) -> str:
    from tools.retriever import get_doc_section
    section = get_doc_section("fault_manual", fault_code)
    if not section.get("ok"):
        return ""
    # 安全链判定、4.1(3)、120 分钟门限是否适用都读了这一节，它是判定依据的一部分
    _touch("fault_manual")
    return section.get("text", "")


# ---------------------------------------------------------------- 1 重复故障

def repeat_fault(turbine_id: str, fault_code: str,
                 window_start: str | None = None, window_end: str | None = None) -> dict[str, Any]:
    """规程第 3.1 条：同一风机、同一故障代码在**任意连续 24 小时**内发生 3 次及以上。

    必须是滑动窗口，不能按自然日分组——第 3.1 条明确「不能把相隔超过 24 小时的
    记录简单累计到同一窗口」，反过来按日切分也会把跨零点的窗口拆开。
    """
    turbine_id = _lit(turbine_id, _RE_TURBINE, "风机编号")
    fault_code = _lit(fault_code, _RE_CODE, "故障代码")

    alarms = _alarms(turbine_id, fault_code)
    times = [_parse(a["occurred_at"]) for a in alarms]

    # 滑动窗口：以每条告警为起点，统计其后 24 小时内（闭区间）的条数
    best_count, best_span = 0, []
    for i, start in enumerate(times):
        inside = [t for t in times[i:] if t - start <= REPEAT_WINDOW]
        if len(inside) > best_count:
            best_count, best_span = len(inside), inside

    facts: dict[str, Any] = {
        "全部告警条数": len(alarms),
        "最大连续24小时窗口内次数": best_count,
        "该窗口": {
            "起": best_span[0].strftime(TS_FMT) if best_span else None,
            "止": best_span[-1].strftime(TS_FMT) if best_span else None,
            "各次时间": [t.strftime(TS_FMT) for t in best_span],
        },
    }

    # 题目常给定一个考察窗口，单独按闭区间统计一份
    if window_start and window_end:
        lo, hi = _parse(window_start), _parse(window_end)
        inside = [t for t in times if lo <= t <= hi]
        facts["指定窗口"] = {
            "起": window_start, "止": window_end, "闭区间": True,
            "次数": len(inside),
            "各次时间": [t.strftime(TS_FMT) for t in inside],
        }
        outside = [t.strftime(TS_FMT) for t in times if not (lo <= t <= hi)]
        if outside:
            facts["窗口外未计入的告警"] = outside

    is_repeat = best_count >= REPEAT_THRESHOLD
    verdict = (
        "构成重复故障：最大连续 24 小时窗口内发生 %d 次，达到第 3.1 条的 3 次门限。" % best_count
        if is_repeat else
        "不构成重复故障：最大连续 24 小时窗口内仅发生 %d 次，未达到第 3.1 条的 3 次门限。" % best_count
    )
    return {
        "ok": True, "rule": "repeat_fault", "is_repeat_fault": is_repeat,
        "verdict": verdict, "facts": facts, "unverifiable": [],
        "clauses": ["3.1"] + (["3.2", "3.3"] if is_repeat else []),
    }


# ---------------------------------------------------------------- 2 应有优先级

def _is_safety_chain(alarms: list[dict[str, Any]], manual: str) -> bool:
    """急停/安全链故障，按告警名称与手册原文判定，不硬编码故障码。"""
    names = " ".join(a.get("fault_name") or "" for a in alarms)
    return any(k in names for k in ("急停", "安全链")) or "安全链" in manual


def priority_required(turbine_id: str, fault_code: str) -> dict[str, Any]:
    """规程第 2.1~2.4 条：该故障当前应当建立什么优先级的工单。"""
    turbine_id = _lit(turbine_id, _RE_TURBINE, "风机编号")
    fault_code = _lit(fault_code, _RE_CODE, "故障代码")

    alarms = _alarms(turbine_id, fault_code)
    if not alarms:
        return {"ok": True, "rule": "priority_required", "required_priority": None,
                "verdict": "数据库中没有 %s / %s 的告警记录，无法判定应有优先级。" % (turbine_id, fault_code),
                "facts": {}, "unverifiable": [], "clauses": []}

    manual = _manual_text(fault_code)
    repeat = repeat_fault(turbine_id, fault_code)
    safety_chain = _is_safety_chain(alarms, manual)
    stopped = any(a["turbine_status"] == "STOPPED" for a in alarms)

    if safety_chain:
        required, clause = "EMERGENCY", "2.1"
        why = "属于急停 / 安全链类故障，第 2.1 条要求建立 EMERGENCY 工单，且不因告警暂时解除而降低。"
    elif repeat["is_repeat_fault"]:
        required, clause = "HIGH", "2.2"
        why = "满足重复故障条件，第 2.2 / 3.2 条要求建立或升级为 HIGH 工单。"
    elif stopped:
        required, clause = "HIGH", "2.2"
        why = "该故障已导致机组停机（STOPPED），第 2.2 条要求建立 HIGH 工单。"
    else:
        required, clause = "NORMAL", "2.3"
        why = "未见停机、未构成重复故障、非安全链类，第 2.3 条下可建立 NORMAL 工单。"

    orders = _orders(turbine_id, fault_code)
    current = [{"work_order_id": o["work_order_id"], "priority": o["priority"], "status": o["status"]}
               for o in orders]
    rank = {"NORMAL": 0, "HIGH": 1, "EMERGENCY": 2}
    needs_upgrade = [o for o in current if rank.get(o["priority"], 0) < rank[required]]

    verdict = "应为 %s 工单。%s" % (required, why)
    if needs_upgrade:
        verdict += " 现有工单 %s 优先级为 %s，按第 2.4 条应及时升级。" % (
            "、".join(o["work_order_id"] for o in needs_upgrade), needs_upgrade[0]["priority"])
    elif not current:
        verdict += " 数据库中尚无对应工单，应按该优先级建单。"

    return {
        "ok": True, "rule": "priority_required", "required_priority": required,
        "verdict": verdict,
        "facts": {"是否安全链类": safety_chain, "是否曾导致停机": stopped,
                  "是否重复故障": repeat["is_repeat_fault"],
                  "最大24小时窗口次数": repeat["facts"]["最大连续24小时窗口内次数"],
                  "现有工单": current},
        "unverifiable": [], "clauses": [clause, "2.4"] + (["3.2"] if repeat["is_repeat_fault"] else []),
    }


# ---------------------------------------------------------------- 3 工单安排五要素

def work_order_assessment(turbine_id: str, fault_code: str) -> dict[str, Any]:
    """机试题目说明第五节：判断工单安排时，应同时关注优先级、状态、处理记录、
    观察时间和备件五项，不能只看工单是否存在。"""
    turbine_id = _lit(turbine_id, _RE_TURBINE, "风机编号")
    fault_code = _lit(fault_code, _RE_CODE, "故障代码")

    orders = _orders(turbine_id, fault_code)
    if not orders:
        return {
            "ok": True, "rule": "work_order_assessment", "has_work_order": False,
            "verdict": "数据库中不存在 %s / %s 的维检工单记录。" % (turbine_id, fault_code),
            "facts": {"查询条件": {"turbine_id": turbine_id, "fault_code": fault_code}, "工单数": 0},
            "unverifiable": [], "clauses": [],
        }

    expected = priority_required(turbine_id, fault_code)
    items = []
    for o in orders:
        note = (o.get("resolution_note") or "").strip()
        obs = o.get("observation_minutes")
        part, avail = o.get("required_part"), o.get("part_available")
        items.append({
            "工单编号": o["work_order_id"],
            "① 优先级": {"值": o["priority"], "应为": expected["required_priority"],
                        "是否达标": o["priority"] == expected["required_priority"]},
            "② 状态": o["status"],
            "③ 处理记录": {"有无": bool(note), "原文": note or None},
            "④ 观察时间": {"分钟": obs, "是否达到120分钟门限": (obs is not None and obs >= MIN_OBSERVATION_MINUTES)},
            "⑤ 备件": ({"名称": part,
                       "可用性": {1: "可用", 0: "不可用"}.get(avail, "未记录备件需求")}
                      if part else {"名称": None, "可用性": "该工单不需要备件或未记录"}),
        })

    gaps = []
    for it in items:
        wo = it["工单编号"]
        if not it["① 优先级"]["是否达标"]:
            gaps.append("%s 优先级为 %s，应为 %s（第 2.4 条）" % (wo, it["① 优先级"]["值"], it["① 优先级"]["应为"]))
        if not it["③ 处理记录"]["有无"]:
            gaps.append("%s 无处理记录" % wo)
        if it["⑤ 备件"]["可用性"] == "不可用":
            gaps.append("%s 所需备件「%s」当前不可用（第 7.2 条）" % (wo, it["⑤ 备件"]["名称"]))

    verdict = ("工单安排存在以下问题：" + "；".join(gaps) + "。") if gaps else \
              "五项要素（优先级、状态、处理记录、观察时间、备件）均未发现与规程冲突之处。"

    return {
        "ok": True, "rule": "work_order_assessment", "has_work_order": True,
        "verdict": verdict, "facts": {"工单": items},
        "unverifiable": ["备件的实物库存、预留、型号兼容与现场领用状态（第 7.1 条：字段已简化）"],
        "clauses": ["2.4", "7.1", "7.2"],
    }


# ---------------------------------------------------------------- 4 禁止远程复位

_BAN_DEENERGIZE = ("断电", "停电", "隔离", "母线电压")
_BAN_REPLACE_BOARD = ("更换控制单板", "更换接口板", "更换通讯模块", "更换相应部件", "DSP控制单板")
_BAN_POWER_CIRCUIT = ("母线电压", "功率回路")


def remote_reset_ban(turbine_id: str, fault_code: str) -> dict[str, Any]:
    """规程第 4.1 条：五种情形任一成立即禁止远程强制复位。"""
    turbine_id = _lit(turbine_id, _RE_TURBINE, "风机编号")
    fault_code = _lit(fault_code, _RE_CODE, "故障代码")

    alarms = _alarms(turbine_id, fault_code)
    manual = _manual_text(fault_code)
    repeat = repeat_fault(turbine_id, fault_code)

    hits, checks = [], []

    c1 = _is_safety_chain(alarms, manual)
    checks.append({"款": "4.1(1) 急停或安全链故障", "结论": "成立" if c1 else "不成立"})
    if c1:
        hits.append("急停或安全链故障")

    c2 = repeat["is_repeat_fault"]
    checks.append({"款": "4.1(2) 连续 24 小时内 3 次及以上", "结论": "成立" if c2 else "不成立",
                   "依据": "最大窗口内 %d 次" % repeat["facts"]["最大连续24小时窗口内次数"]})
    if c2:
        hits.append("同一风机同一故障代码在连续 24 小时内发生 %d 次" % repeat["facts"]["最大连续24小时窗口内次数"])

    c3 = any(k in manual for k in _BAN_REPLACE_BOARD) or any(k in manual for k in _BAN_DEENERGIZE)
    checks.append({"款": "4.1(3) 手册要求断电检查或更换单板", "结论": "成立" if c3 else "不成立"})
    if c3:
        hits.append("故障手册要求断电检查或更换单板/模块")

    # 4.1(4) 设备安全状态——数据库不记录，按第 1.2/1.3 条标为无法确认而非自动成立
    checks.append({"款": "4.1(4) 无法确认设备安全状态", "结论": "资料无法确认",
                   "说明": "数据库不记录设备安全状态。按第 1.3 条保守处置原则，不得推定其已确认。"})

    c5 = any(k in manual for k in _BAN_POWER_CIRCUIT)
    checks.append({"款": "4.1(5) 涉及母线电压或功率回路", "结论": "成立" if c5 else "不成立"})
    if c5:
        hits.append("涉及母线电压或功率回路")

    banned = bool(hits)
    verdict = (
        "禁止远程强制复位。命中第 4.1 条：%s。按第 3.3 / 4.2 条，应结合手册排查根因并安排现场检查，"
        "不得以反复远程复位规避现场检查和安全隔离。" % "；".join(hits)
        if banned else
        "第 4.1 条列举的五种禁止情形中，可由现有资料判定的四款均不成立；"
        "但设备安全状态无法从资料确认，仍应现场核实后再决定。"
    )
    return {
        "ok": True, "rule": "remote_reset_ban", "reset_banned": banned,
        "verdict": verdict, "facts": {"逐款核对": checks},
        "unverifiable": ["设备安全状态（第 4.1 条第 4 款）——数据库无记录，需要现场核实"],
        "clauses": ["4.1", "4.2"] + (["3.3"] if c2 else []),
    }


# ---------------------------------------------------------------- 5 关单合规

# 第 6.1 条七项必备记录。关键词判定是启发式的，所以结果里一并返回备注原文，
# 便于人工复核——不假装这是精确判定。
_CLOSE_ITEMS = [
    ("① 实际故障原因", ("原因", "因为", "由于", "根因", "查明")),
    ("② 处理措施", ("处理", "清理", "紧固", "更换", "检查", "复核", "调整", "修复")),
    ("③ 更换部件（未更换应说明）", ("更换", "未更换", "无需更换", "不需要更换")),
    ("④ 参数恢复或固化（不适用应说明）", ("参数", "恢复", "固化", "不适用")),
    ("⑤ 故障复位结果", ("复位", "解除", "消除", "清除")),
    ("⑥ 完整复检结果", ("复检", "复核", "验证", "复测")),
    ("⑦ 处理后观察时间", ()),  # 由 observation_minutes 字段判定
]
_EEPROM_ITEMS = [
    ("开机设置", ("开机", "设置")),
    ("参数恢复与固化情况", ("参数", "固化")),
    ("故障复位结果", ("复位",)),
    ("参数恢复验证", ("验证", "复检", "复核")),
]
_NO_RESTART_ONLY = ("重新上电", "重启", "断电重启", "重新启动")


def close_compliance(work_order_id: str) -> dict[str, Any]:
    """规程第 6.1~6.5 条：已完成工单的关闭过程是否合规。

    第 6.5 条明确 status=COMPLETED 只代表数据库状态，不能自证关闭合规。
    """
    work_order_id = _lit(work_order_id, _RE_WO, "工单编号")
    orders = _rows("SELECT * FROM maintenance_records WHERE work_order_id='%s'" % work_order_id)
    if not orders:
        return {"ok": True, "rule": "close_compliance", "is_compliant": None,
                "verdict": "数据库中不存在工单 %s。" % work_order_id,
                "facts": {}, "unverifiable": [], "clauses": []}

    o = orders[0]
    note = (o.get("resolution_note") or "").strip()
    obs = o.get("observation_minutes")
    code = o.get("fault_code") or ""
    manual = _manual_text(code)
    is_eeprom = "EEPROM" in manual.upper() or "EEPROM" in (note or "").upper()
    repeat = repeat_fault(o["turbine_id"], code)
    needs_120 = is_eeprom or repeat["is_repeat_fault"] or any(
        k in manual for k in ("控制单板", "接口板", "通讯模块"))

    checks, missing = [], []
    for label, keys in _CLOSE_ITEMS[:-1]:
        present = bool(note) and any(k in note for k in keys)
        checks.append({"项": label, "是否记录": present})
        if not present:
            missing.append(label)

    obs_ok = obs is not None and (obs >= MIN_OBSERVATION_MINUTES if needs_120 else True)
    checks.append({"项": "⑦ 处理后观察时间", "是否记录": obs is not None,
                   "分钟": obs,
                   "门限": MIN_OBSERVATION_MINUTES if needs_120 else "本类故障规程未设最低门限",
                   "是否达标": obs_ok})
    if obs is None:
        missing.append("⑦ 处理后观察时间")

    restart_only = bool(note) and any(k in note for k in _NO_RESTART_ONLY) and len(missing) >= 3
    eeprom_missing = []
    if is_eeprom:
        for label, keys in _EEPROM_ITEMS:
            if not (note and any(k in note for k in keys)):
                eeprom_missing.append(label)

    reasons = []
    if missing:
        reasons.append("第 6.1 条必备记录缺失：%s" % "、".join(missing))
    if needs_120 and not obs_ok:
        reasons.append("第 6.4 条要求观察时间不少于 %d 分钟，实际为 %s 分钟" % (
            MIN_OBSERVATION_MINUTES, obs if obs is not None else "未记录"))
    if restart_only:
        reasons.append("第 6.2 条：不得只以「重新上电后故障消失」作为关闭依据")
    if eeprom_missing:
        reasons.append("第 6.3 条 EEPROM 参数异常还应记录：%s" % "、".join(eeprom_missing))

    compliant = not reasons
    verdict = (
        "工单 %s 不符合关闭要求。%s。注意第 6.5 条：status=%s 只是数据库状态，不能证明关闭过程合规。"
        % (work_order_id, "；".join(reasons), o["status"])
        if not compliant else
        "工单 %s 在现有记录范围内未发现与第 6.1~6.4 条冲突之处；但按第 6.5 条，"
        "数据库状态本身不能证明关闭过程合规。" % work_order_id
    )
    return {
        "ok": True, "rule": "close_compliance", "is_compliant": compliant,
        "verdict": verdict,
        "facts": {"工单": {k: o[k] for k in ("work_order_id", "turbine_id", "fault_code",
                                             "priority", "status", "observation_minutes")},
                  "处理备注原文": note or None,
                  "是否适用 120 分钟门限": needs_120,
                  "逐项核对": checks,
                  "缺失项": missing},
        "unverifiable": ["处理备注为自由文本，逐项判定基于关键词匹配，"
                         "最终应由人工对照备注原文复核（原文已随结果返回）"],
        "clauses": ["6.1", "6.4", "6.5"] + (["6.2"] if restart_only else []) + (["6.3"] if is_eeprom else []),
    }


# ---------------------------------------------------------------- 6 更换前置条件

def replace_precondition(work_order_id: str) -> dict[str, Any]:
    """规程第 5.1 条九项前置条件逐项核对。

    第 5.2 条：数据库不包含全部现场条件；即使机组 STOPPED 且备件可用，
    也不能默认其他条件已满足。所以这里九项里只有两项能从资料判定，
    其余七项一律标为「现有资料无法确认，需要现场核实」——这正是本题要考的地方。
    """
    work_order_id = _lit(work_order_id, _RE_WO, "工单编号")
    orders = _rows("SELECT * FROM maintenance_records WHERE work_order_id='%s'" % work_order_id)
    if not orders:
        return {"ok": True, "rule": "replace_precondition", "can_replace_now": None,
                "verdict": "数据库中不存在工单 %s。" % work_order_id,
                "facts": {}, "unverifiable": [], "clauses": []}

    o = orders[0]
    alarms = _alarms(o["turbine_id"], o["fault_code"] or "")
    latest = alarms[-1] if alarms else None
    stopped = bool(latest and latest["turbine_status"] == "STOPPED")
    part, avail = o.get("required_part"), o.get("part_available")

    checks = [
        {"项": "1. 机组已经停机", "结论": "满足" if stopped else "不满足",
         "依据": "最新告警记录的机组状态为 %s" % (latest["turbine_status"] if latest else "无记录")},
        {"项": "2. 已退出远程控制", "结论": "现有资料无法确认"},
        {"项": "3. 主电源已经隔离", "结论": "现有资料无法确认"},
        {"项": "4. 已执行挂牌上锁", "结论": "现有资料无法确认"},
        {"项": "5. 已完成验电", "结论": "现有资料无法确认"},
        {"项": "6. 母线电压降至约 20 V", "结论": "现有资料无法确认"},
        {"项": "7. 已按手册断开相关开关", "结论": "现有资料无法确认"},
        {"项": "8. 所需备件当前可用",
         "结论": {1: "满足", 0: "不满足"}.get(avail, "该工单未记录备件需求"),
         "依据": "required_part=%s, part_available=%s" % (part, avail)},
        {"项": "9. 现场人员与环境满足要求", "结论": "现有资料无法确认"},
    ]

    blocked = [c["项"] for c in checks if c["结论"] == "不满足"]
    unknown = [c["项"] for c in checks if c["结论"] == "现有资料无法确认"]

    verdict = "不能认定已经具备立即更换条件。"
    if blocked:
        verdict += "以下条件明确不满足：%s。" % "、".join(blocked)
    verdict += ("第 5.1 条九项中有 %d 项无法由现有资料确认（%s），"
                "按第 5.2 / 7.3 条应逐项现场核实后再作业。" % (len(unknown), "、".join(unknown)))

    return {
        "ok": True, "rule": "replace_precondition", "can_replace_now": False,
        "verdict": verdict,
        "facts": {"工单": {k: o[k] for k in ("work_order_id", "turbine_id", "fault_code",
                                             "status", "required_part", "part_available")},
                  "逐项核对": checks},
        "unverifiable": [c["项"] for c in checks if c["结论"] == "现有资料无法确认"] + SITE_CONDITIONS[:0],
        "clauses": ["5.1", "5.2", "7.3"] + (["7.2"] if avail == 0 else []),
    }


# 每条规则的必填参数，用于在模型漏传时给出可执行的提示，而不是抛一句原始 TypeError
REQUIRED_ARGS = {
    "repeat_fault": ("turbine_id", "fault_code"),
    "priority_required": ("turbine_id", "fault_code"),
    "work_order_assessment": ("turbine_id", "fault_code"),
    "remote_reset_ban": ("turbine_id", "fault_code"),
    "close_compliance": ("work_order_id",),
    "replace_precondition": ("work_order_id",),
}

RULES = {
    "repeat_fault": repeat_fault,
    "priority_required": priority_required,
    "work_order_assessment": work_order_assessment,
    "remote_reset_ban": remote_reset_ban,
    "close_compliance": close_compliance,
    "replace_precondition": replace_precondition,
}


def _attach_clause_texts(result: dict[str, Any]) -> dict[str, Any]:
    """把本次引用到的条款原文内联进结果。

    不这么做的话，模型拿到规则结论后还会逐条 get_doc_section 取回原文佐证——
    实测 Q5 的 12 次工具调用里有 4~6 次是这个，耗时翻倍。改提示词治不住这种行为，
    因为它的动机是合理的（要原文才敢引用）；把原文直接给它，动机就消失了。
    """
    from tools.retriever import get_doc_section

    texts = {}
    for clause in result.get("clauses", []):
        section = get_doc_section("safety_regulation", clause)
        if section.get("ok"):
            texts["第 %s 条" % clause] = section["text"]
    if texts:
        result["clause_texts"] = texts
    return result


def _run_tracking_sources(fn, args: dict[str, Any]) -> dict[str, Any]:
    """跑一条规则，并把它实际触达的数据源写进结果。

    嵌套调用（priority_required 里会调 repeat_fault）共用同一个集合，
    所以子规则查过的表也算在父结果头上 —— 依据面板要回答的是
    「这个结论建立在什么之上」，不是「哪一层函数发的 SQL」。
    """
    token = _TOUCHED.set(set())
    try:
        result = fn(**args)
        touched = set(_TOUCHED.get() or ())
    finally:
        _TOUCHED.reset(token)

    # 引用了条款，规程就是本次结论的数据源之一 —— 哪怕模型一次 get_doc_section 都没调。
    # 反过来，没有记录、没得判（clauses 为空）时不能顺手把规程也记上。
    if result.get("clauses"):
        touched.add("safety_regulation")
    if touched:
        result["sources"] = sorted(touched)
    return result


def clause_sections(result: dict[str, Any]) -> list[dict[str, Any]]:
    """本次判定引用到的规程条款原文，供证据留存。

    原文早就随 clause_texts 给了模型，但只到模型为止：界面上的「原始依据」与
    「使用的数据源」都只认 evidence.docs，于是规程成了唯一一份"参与了判定、
    却在依据里看不到"的资料。这里把它补齐，条款号与原文同源，不是贴个标签。
    """
    from tools.retriever import get_doc_section

    sections = []
    for clause in result.get("clauses") or []:
        section = get_doc_section("safety_regulation", clause)
        if not section.get("ok"):
            continue
        sections.append({
            "doc": section.get("doc"),
            "section_id": section.get("section_id"),
            "title": section.get("title"),
            "path": section.get("path"),
            "text": section.get("text"),
            # 标明这一节不是模型自己取的，是规则判定带出来的
            "via": "check_rule",
        })
    return sections


def check_rule(rule: str, **kwargs) -> dict[str, Any]:
    fn = RULES.get(rule)
    if fn is None:
        return {"ok": False, "error": "未知规则 %r，可用：%s" % (rule, "、".join(RULES))}
    accepted = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    args = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    missing = [a for a in REQUIRED_ARGS.get(rule, ()) if a not in args]
    if missing:
        return {"ok": False,
                "error": "规则 %s 缺少必填参数：%s。请先查出这些值再调用本规则。"
                         % (rule, "、".join(missing))}
    try:
        return _attach_clause_texts(_run_tracking_sources(fn, args))
    except RuleInputError as exc:
        return {"ok": False, "error": str(exc)}
    except TypeError as exc:
        return {"ok": False, "error": "规则 %s 参数不足：%s" % (rule, exc)}
