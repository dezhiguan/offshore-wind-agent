# -*- coding: utf-8 -*-
"""把规则判定转成现场核实清单。

产品上的一个转换：「现有资料无法确认，需要现场核实」对用户是终点——
系统说我不知道，对话就结束了。但规程第 5.1 条本来就是九项具体条件，
规则引擎已经逐项算出了状态，那就应该给出一张清单：
哪几项系统替你确认了、哪一项明确不满足、剩下几项你上塔之后要逐个核。

同一份数据，从「我不知道」变成「这是你要去办的几件事」。
不新增任何能力，只改变它到达用户的形式。

三种状态必须分开，这在电力安全场景里是性命攸关的区分：
  ok      资料明确支持（如告警记录显示机组已停机）
  blocked 资料明确不支持（如 part_available=0，备件不可用）
  pending 资料没有记录，需要现场核实（如挂牌上锁、验电）
「明确不满足」和「无法确认」不是一回事——前者是已知的坏消息，
后者是系统的边界。混在一起报，用户就分不清哪些已成定论、哪些还要去查。
"""
from __future__ import annotations

from typing import Any

UNKNOWN = "现有资料无法确认"


def _state(verdict: str) -> str:
    if verdict == "满足":
        return "ok"
    if verdict == "不满足":
        return "blocked"
    return "pending"


def _from_replace(result: dict[str, Any]) -> dict[str, Any] | None:
    rows = (result.get("facts") or {}).get("逐项核对")
    if not rows:
        return None
    wo = (result.get("facts") or {}).get("工单", {})
    items = [{
        "label": r.get("项", ""),
        "state": _state(r.get("结论", "")),
        "note": r.get("依据") or ("" if _state(r.get("结论", "")) != "pending" else "需现场核实并留痕"),
        "verdict": r.get("结论", ""),
    } for r in rows]
    return {
        "key": "replace_precondition",
        "title": "更换单板 / 通讯模块前置条件",
        "clause": "规程第 5.1 条",
        "subject": wo.get("work_order_id", ""),
        "items": items,
    }


def _from_close(result: dict[str, Any]) -> dict[str, Any] | None:
    facts = result.get("facts") or {}
    rows = facts.get("逐项核对")
    if not rows:
        return None
    wo = facts.get("工单", {})
    items = []
    for r in rows:
        state = r.get("状态")
        if not state:
            continue
        # 关单记录缺失是**已知的缺失**，不是"无法确认"——备注原文就在库里，
        # 是它没写，不是我们查不到。所以「未见」是 blocked 而非 pending。
        #
        # 但 ①「实际故障原因」是第三种情况：原文在，规则层判不了语义，
        # 得有人对着原文认定。这既不是"已记录"也不是"缺失"，压成任何一边
        # 都是在替人下结论——上一版压成"缺失"，11 张单全判缺，没一张对。
        if "分钟" in r:
            ok = bool(r.get("是否达标"))
            items.append({
                "label": r.get("项", ""),
                "state": "ok" if ok else "blocked",
                "note": "实际 %s 分钟，门限 %s" % (r.get("分钟"), r.get("门限")),
                "verdict": "达标" if ok else ("未记录" if r.get("分钟") is None else "不足"),
            })
        elif state == "需对照原文认定":
            items.append({
                "label": r.get("项", ""),
                "state": "pending",
                "note": "对照备注原文认定：%s" % (r.get("备注原文") or ""),
                "verdict": "待认定",
            })
        else:
            ok = state == "已记录"
            hit = r.get("命中词")
            items.append({
                "label": r.get("项", ""),
                "state": "ok" if ok else "blocked",
                "note": ("命中：%s" % "、".join(hit)) if hit else "处理备注中未见",
                "verdict": "已记录" if ok else "缺失",
            })
    return {
        "key": "close_compliance",
        "title": "工单关闭必备记录",
        "clause": "规程第 6.1 / 6.4 条",
        "subject": wo.get("work_order_id", ""),
        "items": items,
    }


def _from_work_order(result: dict[str, Any]) -> dict[str, Any] | None:
    orders = (result.get("facts") or {}).get("工单")
    if not isinstance(orders, list) or not orders:
        return None
    o = orders[0]
    pr, obs, part = o.get("① 优先级", {}), o.get("④ 观察时间", {}), o.get("⑤ 备件", {})
    avail = part.get("可用性")
    items = [
        {"label": "优先级", "state": "ok" if pr.get("是否达标") else "blocked",
         "note": "当前 %s，应为 %s" % (pr.get("值"), pr.get("应为")),
         "verdict": "达标" if pr.get("是否达标") else "不达标"},
        {"label": "状态", "state": "ok", "note": str(o.get("② 状态", "")), "verdict": str(o.get("② 状态", ""))},
        # 标签点明"仅核对有无"：内容是否充分是第 6.1 条在关单时判的事，
        # 不加这四个字，卡片上的「有」会被读成「处理记录已合规」。
        {"label": "处理记录（仅核对有无）",
         "state": "ok" if o.get("③ 处理记录", {}).get("有无") else "blocked",
         "note": (o.get("③ 处理记录", {}).get("原文") or "无记录"),
         "verdict": "有" if o.get("③ 处理记录", {}).get("有无") else "无"},
        {"label": "观察时间", "state": "ok" if obs.get("是否达到120分钟门限") else "blocked",
         "note": "%s 分钟" % obs.get("分钟") if obs.get("分钟") is not None else "未进入观察阶段",
         "verdict": "达标" if obs.get("是否达到120分钟门限") else "未达标"},
        # 三分法：不可用=blocked（明确的坏消息）、未记录=pending（系统边界）、
        # 可用=ok。原先把"未记录"也画成 ok，卡片与正文因此自相矛盾。
        {"label": "备件",
         "state": ("blocked" if avail == "不可用"
                   else "pending" if str(avail).startswith("未记录") else "ok"),
         "note": ("%s · %s" % (part.get("名称"), avail) if part.get("名称")
                  else str(avail)),
         "verdict": str(avail)},
    ]
    return {
        "key": "work_order_assessment",
        "title": "工单安排五要素",
        "clause": "题面第五节 · 规程第 2.4 / 7.1 条",
        "subject": o.get("工单编号", ""),
        "items": items,
    }


def _from_remote_reset(result: dict[str, Any]) -> dict[str, Any] | None:
    """第 4.1 条五款逐条摊开，不靠模型转述。

    实测同一道题跑两轮：一轮写「第 4.1 条已命中第 2 款（24 小时内 4 次）、第 3 款
    （手册要求断电检查/更换单板）、第 5 款（涉及母线电压与功率回路）」，另一轮只写
    「不能继续反复远程复位」，一款都没提。规则引擎两轮算出的 `逐款核对` 完全相同，
    差别只在模型这轮想不想写 —— 那就别让它决定。

    状态口径与其他清单一致，但语义相反要小心：这里「成立」是禁止情形成立，
    是明确的坏消息（blocked）；「不成立」才是 ok。
    """
    rows = (result.get("facts") or {}).get("逐款核对")
    if not rows:
        return None
    items = []
    for r in rows:
        verdict = r.get("结论", "")
        items.append({
            "label": r.get("款", ""),
            "state": "blocked" if verdict == "成立" else ("ok" if verdict == "不成立" else "pending"),
            "note": r.get("依据") or ("需现场核实并留痕" if verdict not in ("成立", "不成立") else ""),
            "verdict": verdict,
        })
    return {
        "key": "remote_reset_ban",
        "title": "禁止远程复位 · 第 4.1 条逐款核对",
        "clause": "规程第 4.1 条",
        "subject": "",
        "items": items,
    }


BUILDERS = {
    "replace_precondition": _from_replace,
    "close_compliance": _from_close,
    "work_order_assessment": _from_work_order,
    "remote_reset_ban": _from_remote_reset,
}


def build(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """从本次实际调用过的规则里生成清单。没调用过就没有清单，不凭空造。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in (evidence or {}).get("rules", []) or []:
        rule = entry.get("rule")
        builder = BUILDERS.get(rule)
        if builder is None or rule in seen:
            continue
        result = entry.get("result") or {}
        if not result.get("ok"):
            continue
        card = builder(result)
        if not card or not card["items"]:
            continue
        seen.add(rule)
        items = card["items"]
        card["counts"] = {
            "ok": sum(1 for i in items if i["state"] == "ok"),
            "blocked": sum(1 for i in items if i["state"] == "blocked"),
            "pending": sum(1 for i in items if i["state"] == "pending"),
            "total": len(items),
        }
        out.append(card)
    return out
