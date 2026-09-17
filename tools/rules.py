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

# 规则引擎发的 SQL 也要留痕，理由和上面那段是同一个，只是更进一步。
#
# `sources` 回答的是「读过哪张表」，回答不了「这个数字怎么来的」。实测一次链路：
# 答案里写「现有工单 WO-260703 为 NORMAL」，而 span 明细里一条查工单表的 SQL 都没有
# —— 工单号来自 priority_required 内部的查询，它不经过模型的工具调用，于是
# **确定性的那条路反而比模型那条路更不透明**：模型发的 query_db 有 SQL 原文和返回行，
# 可复现；规则引擎发的只剩一句结论。而规则结论是直接进判定的，权重更高，越该留痕。
#
# 只进 span 明细，不进模型上下文：见 _run_tracking_sources 的 _audit_ 前缀约定。
_QUERIES: ContextVar[list[dict[str, Any]] | None] = ContextVar("rule_queries", default=None)

# 规则引擎读过的文档小节，同理要留痕：chip 上说"用了故障处理手册"，
# 就得点得开是哪一节 —— 参考来源写的是「章节：24010_SC_主变流器RAM自检失败」，
# 精确到节才对得上。条款那边早就这么做了，手册这边原先没有。
_SECTIONS: ContextVar[list[dict[str, str]] | None] = ContextVar("rule_sections", default=None)

_TABLES = ("alarm_records", "maintenance_records")


def _touch(source: str) -> None:
    touched = _TOUCHED.get()
    if touched is not None:
        touched.add(source)


def _note_section(doc: str, section_id: str) -> None:
    log = _SECTIONS.get()
    entry = {"doc": doc, "section_id": section_id}
    if log is not None and entry not in log:
        log.append(entry)


def _note_query(sql: str, row_count: int) -> None:
    log = _QUERIES.get()
    if log is not None:
        log.append({"sql": " ".join(sql.split()), "row_count": row_count})


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
    _note_query(sql, len(result["rows"]))
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
    _note_section("fault_manual", section.get("section_id") or fault_code)
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
    # 一条记录都没有，和「有记录但没到 3 次」是两回事，结论同为「不构成」但成因不同。
    # 只说「仅发生 0 次」的话，模型会把它读成「不构成重复故障而已」——实测就是这么读的，
    # 它照着这个 0 接着往下答，没有一个字提到这台风机压根没有这条故障。
    if not alarms:
        verdict = ("数据库中没有 %s / %s 的告警记录，无从按第 3.1 条统计次数，"
                   "因此不构成重复故障。" % (turbine_id, fault_code))
    elif is_repeat:
        verdict = "构成重复故障：最大连续 24 小时窗口内发生 %d 次，达到第 3.1 条的 3 次门限。" % best_count
    else:
        verdict = "不构成重复故障：最大连续 24 小时窗口内仅发生 %d 次，未达到第 3.1 条的 3 次门限。" % best_count
    return {
        "ok": True, "rule": "repeat_fault", "is_repeat_fault": is_repeat,
        "no_records": not alarms,
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

# 第 4.1(3) 款「故障手册明确要求**断电检查或更换单板**」是两条腿，得分开判。
#
# 原先两条腿都退化成了整节的裸词命中，其中 _BAN_DEENERGIZE 收了「隔离」：
# 24014 冷却风扇那节写「更换前执行停机、隔离、挂牌上锁和验电」——讲的是换风扇的
# 作业前置，跟「要求断电检查」「要求更换单板」都不沾边，却照样判成立，于是给风扇
# 故障扣上了单板条款的帽子。结论也许还站得住，引的条款是错的，现场一追问就露馅。
#
# 顺带查出来一件更要紧的事：_BAN_REPLACE_BOARD 里的 "DSP控制单板" 从来没匹配上过，
# 因为手册原文是「更换 DSP 控制单板」带空格。24010 这个真该命中的故障码，一直是靠
# 上面那个「隔离」的误命中兜着的——两个 bug 抵消成了对的答案。所以先去空白再匹配。
#
# 改判据：按句判，且要求两个词在同一句里共现。整节命中管不住——一节里既有「检查
# 线缆」又有别处的「隔离」，裸词命中就成立了，而这两件事根本不在一句话里。
_DEENERGIZE = ("断电", "停电", "隔离", "母线")
_INSPECT = ("检查", "检测", "核对", "排查", "测量")
_BOARD_PARTS = ("控制单板", "接口板", "通讯模块", "单板")
_BAN_POWER_CIRCUIT = ("母线电压", "功率回路")
_SENT_SPLIT = re.compile(r"[。；\n]")


def _manual_requires_deenergize_or_board(manual: str) -> tuple[str | None, str | None]:
    """第 4.1(3) 款逐句判。返回（命中的那条腿, 手册原句）。

    把原句带出来，是因为条款引用必须落到手册的哪一句上——上一版只给「成立」，
    被追问「手册哪里要求换单板了」时答不出来，也就没人能发现它判错了。
    """
    for sentence in _SENT_SPLIT.split(re.sub(r"\s+", "", manual)):
        if "更换" in sentence and any(p in sentence for p in _BOARD_PARTS):
            return "故障手册要求更换单板 / 通讯模块", sentence
        if any(d in sentence for d in _DEENERGIZE) and any(i in sentence for i in _INSPECT):
            return "故障手册要求断电检查", sentence
    return None, None


def remote_reset_ban(turbine_id: str, fault_code: str) -> dict[str, Any]:
    """规程第 4.1 条：五种情形任一成立即禁止远程强制复位。"""
    turbine_id = _lit(turbine_id, _RE_TURBINE, "风机编号")
    fault_code = _lit(fault_code, _RE_CODE, "故障代码")

    alarms = _alarms(turbine_id, fault_code)
    # 没有告警记录就没有判定对象。第 4.1 条五款里有三款读的是手册，而手册只按故障码
    # 分节、与哪台风机无关——不挡住空集的话，问「T09 的 24005 能不能复位」会拿 24005
    # 那一节判出「禁止复位，命中第 4.1(1)(3)(5) 款」，而 T09 压根没有这条告警。
    # 模型拿到的是一份带条款号、带逐款核对表的确定性结论，不会去怀疑前提。
    # priority_required / work_order_assessment 早就这么挡了，这里是漏的。
    if not alarms:
        return {"ok": True, "rule": "remote_reset_ban", "reset_banned": None,
                "verdict": "数据库中没有 %s / %s 的告警记录，无法判定是否禁止远程复位。"
                           "手册按故障代码分节、与具体风机无关，不能据此对这台风机下结论。"
                           % (turbine_id, fault_code),
                "facts": {"查询条件": {"turbine_id": turbine_id, "fault_code": fault_code},
                          "告警条数": 0},
                "unverifiable": [], "clauses": []}

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

    c3_leg, c3_sentence = _manual_requires_deenergize_or_board(manual)
    c3 = c3_leg is not None
    checks.append({"款": "4.1(3) 手册要求断电检查或更换单板",
                   "结论": "成立" if c3 else "不成立",
                   "依据": ("手册原句：%s" % c3_sentence) if c3 else
                           "手册本节既未要求断电检查，也未要求更换单板 / 模块"})
    if c3:
        hits.append(c3_leg)

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
# 第 6.1 条前六项。判据分两类，这个区分是这条规则的全部要点：
#
#   关键词可判的：②③④⑤⑥ 要找的都是**具体动作词**——更换、复位、复检、参数、固化。
#     写没写这些动作，词面上基本能看出来，漏判的代价也只是提示人去对一眼原文。
#
#   关键词判不了的：① 实际故障原因。「原因」是个概念，可以用任何名词短语表达——
#     「确认风扇轴承卡滞」「确认短时电网波动」都是在写原因，一个关键词都不占。
#     上一版按「原因/因为/由于/根因/查明」判，11 张已完成工单 11 张判缺失，
#     其中写明了原因的照判不误；更糟的是模型不会质疑这个结论，会替它编出
#     「备注仅写…未构成对故障原因的完整记录判定」这种合理化说辞。
#     阳性率 100% 的检测器没有判别力，只是在稳定地说同一句错话。
#
# 所以 ① 不再猜，改为把备注原文交出去、标成「需对照原文认定」。规则层不假装
# 自己能做语义判断——这跟次数、时长、阈值不一样，那些才是代码该算的。
_CLOSE_ITEMS = [
    # 「核对」「排查」「检测」原先不在表里，WO-260711 的「核对电网波动记录和采样回路」
    # 因此被判成没写处理措施——同一档动作词漏收，跟 ① 是两回事，补齐即可。
    ("② 处理措施", ("处理", "清理", "紧固", "更换", "检查", "检测", "核对",
                   "复核", "排查", "调整", "修复")),
    ("③ 更换部件（未更换应说明）", ("更换", "未更换", "无需更换", "不需要更换")),
    # 「恢复」原本也在这一档，但它太泛了：「等待电网电压恢复」「柜温恢复正常」
    # 都会命中，把没写参数的单子判成写了——与 ① 相反方向的同一种错。
    ("④ 参数恢复或固化（不适用应说明）", ("参数", "固化", "不适用")),
    ("⑤ 故障复位结果", ("复位", "解除", "消除", "清除")),
    ("⑥ 完整复检结果", ("复检", "复核", "验证", "复测")),
]
_CAUSE_ITEM = "① 实际故障原因"

# 第 6.2 条与第 6.4 条的适用范围差一类，原先混成了一个判据。
#   第 6.2 条：重复性故障、EEPROM 参数异常、控制单板故障、**通讯模块故障**
#   第 6.4 条：重复性故障、EEPROM 异常、控制单板故障
_BOARD_FAULT_KEYS = ("控制单板", "接口板")
_COMM_FAULT_KEYS = ("通讯模块",)
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
    is_board = any(k in manual for k in _BOARD_FAULT_KEYS)
    is_comm = any(k in manual for k in _COMM_FAULT_KEYS)
    needs_120 = is_eeprom or repeat["is_repeat_fault"] or is_board            # 第 6.4 条
    in_622_scope = needs_120 or is_comm                                      # 第 6.2 条

    # 第 6.1~6.4 条管的是「关闭前应当记录什么」。工单还没关就拿关闭合规去判它，
    # 问出来的是个伪命题：WO-260703 状态 OPEN、备注为空，判「七项全缺、不合规」
    # 字面没错，却让人以为这张单关错了——它根本还没关。
    if o["status"] != "COMPLETED":
        return {
            "ok": True, "rule": "close_compliance", "is_compliant": None,
            "verdict_kind": "未关闭",
            "verdict": "工单 %s 当前状态为 %s，尚未关闭。第 6.1~6.4 条是**关闭前**的记录要求，"
                       "此时无所谓关闭是否合规。若要评估当前安排是否恰当，"
                       "请改用 work_order_assessment 规则。" % (work_order_id, o["status"]),
            "facts": {"工单": {k: o[k] for k in ("work_order_id", "turbine_id", "fault_code",
                                                 "priority", "status", "observation_minutes")},
                      "处理备注原文": note or None},
            "unverifiable": [], "clauses": [],
        }

    # —— 第 6.1 条前六项：能判的判，判不了的把原文交出去 ——
    checks, unrecorded = [], []
    if note:
        checks.append({"项": _CAUSE_ITEM, "状态": "需对照原文认定",
                       "备注原文": note,
                       "要求": "从上述原文中指出记载故障原因的片段；指不出即为未记录。"
                               "不要按关键词判——「轴承卡滞」「电网波动」都是在写原因。"})
    else:
        checks.append({"项": _CAUSE_ITEM, "状态": "未见", "备注原文": None,
                       "要求": "处理备注为空，无可认定的内容。"})
        unrecorded.append(_CAUSE_ITEM)

    for label, keys in _CLOSE_ITEMS:
        hit = [k for k in keys if note and k in note]
        checks.append({"项": label, "状态": "已记录" if hit else "未见",
                       "命中词": hit or None})
        if not hit:
            unrecorded.append(label)

    obs_ok = obs is not None and (obs >= MIN_OBSERVATION_MINUTES if needs_120 else True)
    checks.append({"项": "⑦ 处理后观察时间",
                   "状态": "已记录" if obs is not None else "未见",
                   "分钟": obs,
                   "门限": MIN_OBSERVATION_MINUTES if needs_120 else "本类故障规程未设最低门限",
                   "是否达标": obs_ok})
    if obs is None:
        unrecorded.append("⑦ 处理后观察时间")

    # 第 6.2 条判据改为直接依据条文，不再挂靠「缺了几项」这种间接量：
    # 属于该条列举的四类故障 + 备注提到重新上电 + 除此之外没有实质处理与复检记录。
    recorded_labels = {c["项"] for c in checks if c.get("状态") == "已记录"}
    restart_only = (
        bool(note) and any(k in note for k in _NO_RESTART_ONLY) and in_622_scope
        and not any(lbl.startswith(("②", "⑥")) for lbl in recorded_labels)
    )
    eeprom_missing = []
    if is_eeprom:
        for label, keys in _EEPROM_ITEMS:
            if not (note and any(k in note for k in keys)):
                eeprom_missing.append(label)

    # 硬性违规（可计算、无歧义）与记录未覆盖（需对照原文）分开报。
    # 混成一个「不合规」，WO-260708 这种三条硬性违规的单子，和只差两句说明的
    # WO-260712 就看不出轻重——值班的人分不清哪张要紧。
    blocking = []
    if needs_120 and not obs_ok:
        blocking.append("第 6.4 条要求观察时间不少于 %d 分钟，实际为 %s" % (
            MIN_OBSERVATION_MINUTES,
            "%d 分钟" % obs if obs is not None else "未记录"))
    if restart_only:
        blocking.append("第 6.2 条：本类故障不得只以「重新上电后故障消失」作为关闭依据")
    if eeprom_missing:
        blocking.append("第 6.3 条 EEPROM 参数异常还应记录：%s" % "、".join(eeprom_missing))

    tail = "按第 6.5 条，status=%s 只是数据库状态，不能证明关闭过程合规。" % o["status"]
    if blocking:
        kind = "硬性违规"
        verdict = "工单 %s 不符合关闭要求：%s。%s" % (work_order_id, "；".join(blocking), tail)
        if unrecorded:
            verdict += "另有第 6.1 条 %d 项未在处理备注中体现：%s。" % (
                len(unrecorded), "、".join(unrecorded))
    elif unrecorded:
        kind = "记录待补"
        verdict = ("工单 %s 未发现第 6.2~6.4 条的硬性违规；但第 6.1 条有 %d 项未在处理备注中体现："
                   "%s。%s" % (work_order_id, len(unrecorded), "、".join(unrecorded), tail))
    else:
        kind = "未见冲突"
        verdict = "工单 %s 在可判定范围内未发现与第 6.1~6.4 条冲突之处。%s" % (work_order_id, tail)
    if note:
        verdict += "第 6.1 条①「实际故障原因」不做关键词判定，请对照备注原文认定：「%s」" % note

    return {
        "ok": True, "rule": "close_compliance",
        # False 只给可计算的硬性违规；None = 尚有需对照原文认定的项，不代表合规
        "is_compliant": False if blocking else None,
        "verdict_kind": kind,
        "verdict": verdict,
        "facts": {"工单": {k: o[k] for k in ("work_order_id", "turbine_id", "fault_code",
                                             "priority", "status", "observation_minutes")},
                  "处理备注原文": note or None,
                  "是否适用 120 分钟门限": needs_120,
                  "第 6.2 条适用": in_622_scope,
                  "逐项核对": checks,
                  "硬性违规": blocking,
                  "未在备注中体现": unrecorded},
        "unverifiable": [
            "①「实际故障原因」是自由文本的语义判定，规则层不做——"
            "备注原文已随结果返回，须对照原文认定。",
            "②~⑥ 按动作词匹配，命中词已随结果返回；措辞特殊时仍应对照原文复核。",
        ],
        "clauses": (["6.1", "6.5"] + (["6.4"] if needs_120 else [])
                    + (["6.2"] if restart_only else []) + (["6.3"] if is_eeprom else [])),
    }


# ---------------------------------------------------------------- 6 更换前置条件

_RE_SWITCH = re.compile(r"Q\d+")


def _manual_switches(manual: str) -> list[str]:
    """手册里「按步骤断开 Q6、Q13…」那一句列的开关。

    只从含「断开」的那一句里取：手册别处（控制原理、恢复步骤）也会出现 Q 编号，
    整篇扫会把不该断的开关也列进核对项 —— 在电力作业里这种"多列一个"不是小事。
    """
    found: list[str] = []
    for line in manual.splitlines():
        if "断开" not in line:
            continue
        for switch in _RE_SWITCH.findall(line):
            if switch not in found:
                found.append(switch)
    return found


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
    code = o.get("fault_code") or ""
    alarms = _alarms(o["turbine_id"], code)
    latest = alarms[-1] if alarms else None
    stopped = bool(latest and latest["turbine_status"] == "STOPPED")
    part, avail = o.get("required_part"), o.get("part_available")
    # 第 5.1 条第 7 项写的是「已按**故障手册**断开相关开关」—— 条款本身就指向手册，
    # 手册里写明了是哪几个开关。不读手册，这一项就只剩一句"无法确认"：
    # 手册进不了参考来源，现场核实清单也说不出到底要去断什么。
    manual = _manual_text(code)
    switches = _manual_switches(manual)

    checks = [
        {"项": "1. 机组已经停机", "结论": "满足" if stopped else "不满足",
         "依据": "最新告警记录的机组状态为 %s" % (latest["turbine_status"] if latest else "无记录")},
        {"项": "2. 已退出远程控制", "结论": "现有资料无法确认"},
        {"项": "3. 主电源已经隔离", "结论": "现有资料无法确认"},
        {"项": "4. 已执行挂牌上锁", "结论": "现有资料无法确认"},
        {"项": "5. 已完成验电", "结论": "现有资料无法确认"},
        {"项": "6. 母线电压降至约 20 V", "结论": "现有资料无法确认"},
        {"项": "7. 已按手册断开相关开关", "结论": "现有资料无法确认",
         **({"依据": "故障手册 %s 节要求断开 %s；是否已执行，数据库无记录"
                     % (code, "、".join(switches))} if switches else {})},
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
        # 第 8 项拿 part_available 判备件，那个字段的语义正是第 7.1 条定义的，应一并援引
        "clauses": ["5.1", "5.2", "7.1", "7.3"] + (["7.2"] if avail == 0 else []),
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

# 泛问时该给什么：规则覆盖的条款全集。
#
# 起因是一次实测：用户问「仓库里有备件是不是就可以直接开工了」——这是**泛问**，
# 不绑定任何工单，于是 check_rule 报「缺少必填参数 work_order_id」直接缺席，
# 判定只能退回文档检索，而第 7.3 条（备件可用不等于允许作业）在 BM25 里排第 12，
# 没进上下文。同理「断电重启之后报警没了，这单子能结吗」漏掉第 6.2 条。
#
# 这两条恰恰是泛问的正确答案。规则引擎明明知道自己管哪几条，却因为缺一个 id
# 而一句话不说 —— 缺的是判定对象，不是条款。所以缺参数时不再报错，
# 改为给出通则条款原文，并明确声明「未针对具体对象判定」。
GENERAL_CLAUSES = {
    "repeat_fault": ["3.1", "3.2", "3.3"],
    "priority_required": ["2.1", "2.2", "2.3", "2.4"],
    "work_order_assessment": ["2.4", "7.1", "7.2"],
    "remote_reset_ban": ["4.1", "4.2", "3.3"],
    "close_compliance": ["6.1", "6.2", "6.3", "6.4", "6.5"],
    "replace_precondition": ["5.1", "5.2", "5.3", "7.1", "7.2", "7.3"],
}

GENERAL_TITLES = {
    "repeat_fault": "重复故障认定与升级",
    "priority_required": "工单优先级",
    "work_order_assessment": "工单安排复核",
    "remote_reset_ban": "禁止远程复位",
    "close_compliance": "工单关闭合规性",
    "replace_precondition": "单板 / 通讯模块更换",
}


def _general_answer(rule: str, missing: list[str]) -> dict[str, Any]:
    """缺判定对象时给通则，而不是给一句错误。"""
    return {
        "ok": True,
        "rule": rule,
        "is_general": True,
        "missing_args": missing,
        "verdict": (
            "未指定 %s，以下是规程对「%s」的通则要求，**未针对任何具体工单或风机作出判定**。"
            "需要判定具体对象时，请先查出 %s 再调用本规则。"
            % ("、".join(missing), GENERAL_TITLES.get(rule, rule), "、".join(missing))
        ),
        "facts": {},
        "unverifiable": ["本次只给出通则条款，未核对任何记录；"
                         "具体对象是否合规必须另行按 id 判定"],
        "clauses": list(GENERAL_CLAUSES.get(rule, ())),
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
    qtoken = _QUERIES.set([])
    stoken = _SECTIONS.set([])
    try:
        result = fn(**args)
        touched = set(_TOUCHED.get() or ())
        queries = list(_QUERIES.get() or ())
        sections = list(_SECTIONS.get() or ())
    finally:
        _TOUCHED.reset(token)
        _QUERIES.reset(qtoken)
        _SECTIONS.reset(stoken)

    # `_audit_` 前缀 = 只进链路留存，不进模型上下文（loop._for_model 按前缀剥离）。
    # 原本靠"记得别回写 result"这条口头纪律，写进前缀约定才守得住。
    if queries:
        result["_audit_queries"] = queries
    if sections:
        result["_audit_doc_sections"] = sections

    # 引用了条款，规程就是本次结论的数据源之一 —— 哪怕模型一次 get_doc_section 都没调。
    # 反过来，没有记录、没得判（clauses 为空）时不能顺手把规程也记上。
    if result.get("clauses"):
        touched.add("safety_regulation")
    if touched:
        result["sources"] = sorted(touched)
    return result


def cited_sections(result: dict[str, Any]) -> list[dict[str, Any]]:
    """本次判定读过或引用到的文档小节原文，供证据留存。

    两类：引用到的规程条款，以及规则引擎自己读过的手册小节（如第 5.1 条第 7 项
    「已按**故障手册**断开相关开关」，条款本身就指向手册）。

    原文早就随 clause_texts 给了模型，但只到模型为止：界面上的「原始依据」与
    「使用的数据源」都只认 evidence.docs，于是这两份成了"参与了判定、
    却在依据里看不到"的资料。这里把它补齐，条款号与原文同源，不是贴个标签。
    """
    from tools.retriever import get_doc_section

    wanted = [("safety_regulation", clause) for clause in result.get("clauses") or []]
    wanted += [(e["doc"], e["section_id"]) for e in result.get("_audit_doc_sections") or ()]

    sections = []
    for doc, section_id in wanted:
        section = get_doc_section(doc, section_id)
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


# 六条规则的判定对象分两种握法：四条认「风机 + 故障码」，两条认工单号。
# 但提问的人未必按这个分界提问——「WO-260705 的优先级对不对」给的是工单号，
# 要的却是 priority_required。
_BY_TURBINE_FAULT = ("repeat_fault", "priority_required",
                     "work_order_assessment", "remote_reset_ban")
_BY_WORK_ORDER = ("close_compliance", "replace_precondition")


def _resolve_subject(rule: str, kwargs: dict[str, Any]) -> tuple[dict[str, Any], str | None,
                                                                 dict[str, Any] | None]:
    """在工单号与「风机 + 故障码」之间互换，补齐规则真正需要的那一种。

    起因是工具描述承诺了「你只需要给出风机编号、故障代码**或工单编号**」，
    而 priority_required 的签名里没有 work_order_id —— 这个参数会被 accepted
    过滤掉，只给工单号拿不到判定。模型于是自己补了一个故障码：两次实测分别
    编了 24001 和 24003，都是问「T05 的 WO-260705 优先级对不对」。

    24001 那次尤其危险：T05 确实有 24001，规则照常返回了一个**有效**结论
    （应为 NORMAL），只是回答的不是被问的问题；模型靠后续查库自己纠正了，
    但它没有任何理由必须纠正。空集守卫挡得住编出来的不存在组合，挡不住
    这种「编中了」的——所以要消掉它编的动机，而不是只加一道拦。

    返回 (补齐后的参数, 定位说明, 提前返回的结果)。
    """
    args = dict(kwargs)
    wo = (args.get("work_order_id") or "").strip()
    tid = (args.get("turbine_id") or "").strip().upper()
    code = (args.get("fault_code") or "").strip()

    # 只要带了工单号就进来：既补齐缺的，也核对带来的——两者都传且对不上，
    # 正是编造的形态，比只缺一个更该说破。
    if rule in _BY_TURBINE_FAULT and wo:
        wo = _lit(wo, _RE_WO, "工单编号")
        rows = _rows("SELECT turbine_id, fault_code FROM maintenance_records "
                     "WHERE work_order_id='%s'" % wo)
        if not rows:
            return args, None, {
                "ok": True, "rule": rule,
                "verdict": "数据库中不存在工单 %s，无法据此定位风机与故障码。" % wo,
                "facts": {"查询条件": {"work_order_id": wo}},
                "unverifiable": [], "clauses": []}
        o = rows[0]
        # 传了工单号又自己带了一个对不上的故障码——正是编造的形态，必须挡下来说破，
        # 不能悄悄挑一个用：挑工单的会掩盖模型在瞎猜，挑传入的会答错问题。
        if (tid and tid != o["turbine_id"]) or (code and code != o["fault_code"]):
            return args, None, {
                "ok": False,
                "error": "工单 %s 对应的是 %s / %s，与传入的 turbine_id=%s、fault_code=%s 不一致。"
                         "请以工单记录为准，不要自行推断风机编号或故障代码。"
                         % (wo, o["turbine_id"], o["fault_code"], tid or "（未传）", code or "（未传）")}
        args["turbine_id"], args["fault_code"] = o["turbine_id"], o["fault_code"]
        return args, "由工单 %s 定位到 %s / %s" % (wo, o["turbine_id"], o["fault_code"]), None

    if rule in _BY_WORK_ORDER and not wo and tid and code:
        tid = _lit(tid, _RE_TURBINE, "风机编号")
        code = _lit(code, _RE_CODE, "故障代码")
        rows = _orders(tid, code)
        if not rows:
            return args, None, {
                "ok": True, "rule": rule,
                "verdict": "数据库中不存在 %s / %s 的维检工单，无从判定。" % (tid, code),
                "facts": {"查询条件": {"turbine_id": tid, "fault_code": code}},
                "unverifiable": [], "clauses": []}
        if len(rows) > 1:
            return args, None, {
                "ok": False,
                "error": "%s / %s 对应多张工单：%s。请指定 work_order_id 再调用本规则。"
                         % (tid, code, "、".join(r["work_order_id"] for r in rows))}
        args["work_order_id"] = rows[0]["work_order_id"]
        return args, "由 %s / %s 定位到工单 %s" % (tid, code, rows[0]["work_order_id"]), None

    return args, None, None


def check_rule(rule: str, **kwargs) -> dict[str, Any]:
    fn = RULES.get(rule)
    if fn is None:
        return {"ok": False, "error": "未知规则 %r，可用：%s" % (rule, "、".join(RULES))}
    try:
        kwargs, locator, early = _resolve_subject(rule, kwargs)
    except RuleInputError as exc:
        return {"ok": False, "error": str(exc)}
    if early is not None:
        return early
    accepted = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    args = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    missing = [a for a in REQUIRED_ARGS.get(rule, ()) if a not in args]
    if missing:
        # 缺的是判定对象，不是条款 —— 给通则，别让规则引擎在泛问上整个缺席
        result = _general_answer(rule, missing)
        result["sources"] = ["safety_regulation"]
        return _attach_clause_texts(result)
    try:
        result = _attach_clause_texts(_run_tracking_sources(fn, args))
        if locator:
            # 换算过就把换算过程写进结果：判定对象是怎么定下来的，必须可回溯
            result["判定对象来源"] = locator
        return result
    except RuleInputError as exc:
        return {"ok": False, "error": str(exc)}
    except TypeError as exc:
        return {"ok": False, "error": "规则 %s 参数不足：%s" % (rule, exc)}
