# -*- coding: utf-8 -*-
"""链路追踪的进程内留存。

只保留最近 N 条，**不落库**：本系统是随题演示，为它配一套带持久化的运维后台
会引入新的状态与故障模式，而题面明确「不要求实现生产级系统」。
进程重启即清空，这一点在页面上明确写出，不假装是持久化存储。

留存的目的是让执行过程可复核——单看一次回答判断不了 agent 的决策是否合理，
要能横向比较多条链路：哪些步骤是必要的、哪一步降级了、token 花在哪里。
"""
from __future__ import annotations

import math
import statistics
import threading
import time
import uuid
from datetime import datetime
from collections import deque
from typing import Any

from tools.budget import MAX_DOC_RATIO, MAX_ROW_RATIO

MAX_TRACES = 50

_lock = threading.Lock()
_traces: deque[dict[str, Any]] = deque(maxlen=MAX_TRACES)
_seq = 0


def _diagnose(spans: list[dict[str, Any]], meta: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """从真实发生过的 span 推出链路状态与异常清单。

    不引入"可信度"之类无法验证的合成分数——只报告确实发生过的事：
    工具调用失败、护栏降级、未正常收口。
    """
    notes: list[dict[str, str]] = []
    for s in spans:
        if s.get("status") in ("ERROR", "DEGRADED"):
            # 被护栏拦下的那次调用，异常清单里要报"上下文预算"而不是工具名——
            # 报工具名会把"红线生效"读成"query_db 出错了"。
            notes.append({"kind": s.get("guard") or s.get("name", ""),
                          "text": s.get("output", "")})
    stop = meta.get("stop_reason")
    # 三段标题没解析出来时 compose 会回落成整段原文：回答照样非空，于是"有效回答率"
    # 把它算成产出可用回答，而「现有资料无法确认」那一段也解析不出来，"标注存疑"跟着少算。
    # 一次降级同时抬高一个指标、压低另一个，且两处都不出声 —— 必须记一笔。
    if meta.get("format_parsed") is False and stop != "llm_error":
        notes.append({"kind": "格式降级",
                      "text": "模型未按三段标题输出，已回落为整段原文；"
                              "「依据」与「现有资料无法确认」未能拆分"})
    if stop == "max_steps":
        notes.append({"kind": "步数上限", "text": "已达工具调用步数上限，回答基于当时已取得的证据"})
    elif stop == "refused_ungrounded":
        notes.append({"kind": "未取证拒答", "text": "两次均未调用任何工具，已拒绝作答而非放行无依据回答"})

    if any(s.get("status") == "ERROR" for s in spans) or stop == "refused_ungrounded":
        return "DEGRADED", notes
    if notes:
        return "DEGRADED", notes
    return "SUCCESS", notes


def record(question: str, result: dict[str, Any], *, source: str = "online",
           replay: bool = False, error: str | None = None) -> int | None:
    """留存一条链路。error 非空表示这次运行是失败收场（模型超时、网关报错等）。

    失败也要进来：后台的成功率如果只统计跑通的链路，分母里就没有失败，
    指标会恒为 100%，看不出任何问题。

    source 区分这条链路是谁跑出来的：``online`` 是有人在问答页真实提问，
    ``eval`` 是后台点「运行全部」跑回归。两者必须分开统计——回归用例是
    挑好的题目，把它混进线上指标里，线上成功率会被这批题目抬着走，
    而线上质量要回答的恰恰是"真实用户问出来的东西，系统接得住吗"。
    """
    global _seq
    spans = result.get("spans") or []
    if not spans and not replay and not error:
        return None
    ts = result.get("trace_summary") or {}
    meta = result.get("meta") or {}
    status, notes = _diagnose(spans, meta)
    if error:
        # 与 DEGRADED 区分开：DEGRADED 是降级但仍给出了回答，FAILED 是没有回答
        status = "FAILED"
        if not any(s.get("status") == "ERROR" for s in spans):
            notes = notes + [{"kind": "运行中断", "text": error}]
    with _lock:
        _seq += 1
        _traces.appendleft({
            "id": _seq,
            # 短 id 仅用于展示与检索，不参与任何逻辑
            "short_id": uuid.uuid4().hex[:12],
            "at": datetime.now().strftime("%m-%d %H:%M:%S"),
            # 排序用的数值时间。离线评测的产物要和留存的链路并进同一个列表，
            # "%m-%d %H:%M:%S" 这种展示串跨年就排不对，比较也慢。
            "at_ts": time.time(),
            "status": status,
            "notes": notes,
            "question": question,
            "source": source,
            "replay": replay,
            "model": meta.get("model"),
            "elapsed_ms": meta.get("elapsed_ms"),
            "stop_reason": meta.get("stop_reason"),
            "steps": meta.get("steps"),
            # elapsed_ms 只量到链路收口；served_ms 含解析、核实清单与接地校验
            "served_ms": meta.get("served_ms"),
            "price_basis": meta.get("price_basis") or {},
            # 接地校验（shadow 档只标注、不改回答）。不留存的话影子跑等于白跑：
            # 判定层要先标定再切 enforce，而标定靠的就是这批结果。
            "grounding": meta.get("grounding") or {},
            # 三段标题解析成功与否，决定上面两个回答类指标该不该被照字面读
            "format_parsed": meta.get("format_parsed"),
            "budget": meta.get("budget") or {},
            "sources": result.get("sources") or [],
            "answer": result.get("answer") or "",
            "unverifiable": result.get("unverifiable") or [],
            "spans": spans,
            "summary": ts,
        })
        return _seq


def _pick(source: str | None) -> list[dict[str, Any]]:
    """按来源取留存的链路。source 为空表示全要（链路追踪页要看全部）。"""
    with _lock:
        return [t for t in _traces if source is None or t.get("source", "online") == source]


def listing(source: str | None = None) -> list[dict[str, Any]]:
    """列表视图。剥掉 spans / answer / unverifiable 三个大字段，
    但把「存疑项条数」折算成一个数字带出来——列表上要显示它，
    为此把整段正文拉回前端不划算。"""
    return [dict({k: v for k, v in t.items() if k not in ("spans", "answer", "unverifiable")},
                 unverifiable_count=len(t.get("unverifiable") or []),
                 answered=bool((t.get("answer") or "").strip())
                          and t.get("stop_reason") != "refused_ungrounded")
            for t in _pick(source)]


def get(trace_id: int) -> dict[str, Any] | None:
    with _lock:
        return next((t for t in _traces if t["id"] == trace_id), None)


def _p95(values: list[int]) -> int | None:
    """最近秩（nearest-rank）：至少 95% 的样本不超过它。

    原先取 ``round(0.95 * (n - 1))``，n=39 时落在第 37 位，把最慢的两条整个排除在外，
    报出来的数比真实的第 95 百分位低一截。延迟指标宁可偏保守也不能偏乐观——
    偏乐观的那一版，恰恰在最该示警的时候最好看。
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def stats(source: str | None = None) -> dict[str, Any]:
    items = _pick(source)
    if not items:
        return {"count": 0}
    # 「端到端耗时」就按端到端取：served_ms 含收口后处理，老记录没有它才退回 elapsed_ms
    lat = [t.get("served_ms") or t["elapsed_ms"] for t in items if t.get("elapsed_ms")]
    toks = [(t["summary"].get("prompt_tokens", 0) or 0) + (t["summary"].get("completion_tokens", 0) or 0)
            for t in items]
    costs = [t["summary"].get("cost_cny", 0) or 0 for t in items]
    tool_calls = sum(t["summary"].get("tool_calls", 0) or 0 for t in items)
    tool_fail = sum(t["summary"].get("tool_failures", 0) or 0 for t in items)
    # 护栏按红线拒绝的调用：工具跑通了，结果没被放进上下文。既不算成功也不算失败，
    # 从成功率的分母里剔除 —— 否则红线越管用，"工具调用成功率"越难看。
    tool_refused = sum(t["summary"].get("tool_refused", 0) or 0 for t in items)
    tool_judged = tool_calls - tool_refused
    model_calls = sum(t["summary"].get("model_calls", 0) or 0 for t in items)
    model_fail = sum(t["summary"].get("model_failures", 0) or 0 for t in items)
    degraded = sum(t["summary"].get("degraded", 0) or 0 for t in items)
    normal = sum(1 for t in items if t.get("stop_reason") == "completed")
    degraded_traces = sum(1 for t in items if t.get("status") != "SUCCESS")

    # 以下几项是**线上口径**专用：线上没有标准答案，判不了对错，
    # 只能看系统自己留下的信号——收没收口、拒没拒答、有没有把
    # 「现有资料无法确认」标出来、上下文水位压到了哪。
    success = sum(1 for t in items if t.get("status") == "SUCCESS")
    # 非空还不够：未取证拒答写出来的正文同样非空（"无法回答：…"），
    # 只查非空会把"宁可不答"算成有效回答。与离线档同一条判据。
    answered = sum(1 for t in items if (t.get("answer") or "").strip()
                   and t.get("stop_reason") != "refused_ungrounded")
    refused = sum(1 for t in items if t.get("stop_reason") == "refused_ungrounded")
    max_steps = sum(1 for t in items if t.get("stop_reason") == "max_steps")
    flagged = sum(1 for t in items if t.get("unverifiable"))
    db_pcts = [t.get("budget", {}).get("db_rows_pct") for t in items]
    doc_pcts = [t.get("budget", {}).get("doc_chars_pct") for t in items]
    db_pcts = [v for v in db_pcts if v is not None]
    doc_pcts = [v for v in doc_pcts if v is not None]
    # 工具返回真正注入上下文的字符数（含重复与 JSON 结构），与覆盖率不是一回事
    ctx_chars = [t.get("budget", {}).get("context_chars") for t in items]
    ctx_chars = [v for v in ctx_chars if v is not None]
    unmetered = sum(t["summary"].get("usage_missing_calls", 0) or 0 for t in items)
    # 护栏按红线拒绝取证的次数，与"被拒那一刻试图压到的水位"。后者可以超过阈值，
    # 已计费的覆盖率永远不会 —— 看板上只画后者，撞线就永远看不见。
    guard_refusals = sum(t.get("budget", {}).get("refusals") or 0 for t in items)
    attempt_db = [t.get("budget", {}).get("peak_attempt_db_rows_pct") for t in items]
    attempt_doc = [t.get("budget", {}).get("peak_attempt_doc_chars_pct") for t in items]
    attempt_db = [v for v in attempt_db if v is not None]
    attempt_doc = [v for v in attempt_doc if v is not None]

    # 接地校验：回答里的数字与条款号，是否真在证据里出现过。
    # off 档不做检查，不能混进分母 —— 否则关掉校验会让"零存疑"看着像质量变好了。
    checked = [t for t in items if (t.get("grounding") or {}).get("mode") in ("shadow", "enforce")]
    g_flagged = sum(1 for t in checked if t["grounding"].get("flagged"))
    g_modes = sorted({(t.get("grounding") or {}).get("mode") for t in items
                      if (t.get("grounding") or {}).get("mode")})
    # 三段格式没解析出来的链路：回答类指标要照这个数打折看
    format_fallback = sum(1 for t in items if t.get("format_parsed") is False)
    return {
        "count": len(items),
        "capacity": MAX_TRACES,
        # 这 50 格是所有来源共用的：后台跑一轮回归就会挤掉同样多的线上链路。
        # 只报 count/capacity 的话，样本被顶掉是悄无声息的。
        "capacity_used": len(_pick(None)),
        # deque 是 appendleft，第 0 条是最新的一条
        "latest_at": items[0].get("at"),
        "earliest_at": items[-1].get("at"),
        "success": success,
        "success_rate": round(success / len(items) * 100, 1),
        "answered": answered,
        "answer_rate": round(answered / len(items) * 100, 1),
        "refused": refused,
        "refused_rate": round(refused / len(items) * 100, 1),
        "max_steps_hit": max_steps,
        "flagged_traces": flagged,
        "flagged_rate": round(flagged / len(items) * 100, 1),
        "peak_db_rows_pct": max(db_pcts) if db_pcts else None,
        "peak_doc_chars_pct": max(doc_pcts) if doc_pcts else None,
        "peak_context_chars": max(ctx_chars) if ctx_chars else None,
        # 界面上那条红线画在哪，跟着护栏的阈值走，不在前端另写一份
        "limit_row_pct": round(MAX_ROW_RATIO * 100, 1),
        "limit_doc_pct": round(MAX_DOC_RATIO * 100, 1),
        "guard_refusals": guard_refusals,
        "peak_attempt_db_rows_pct": max(attempt_db) if attempt_db else None,
        "peak_attempt_doc_chars_pct": max(attempt_doc) if attempt_doc else None,
        "usage_missing_calls": unmetered,
        "grounding_mode": "、".join(g_modes) if g_modes else None,
        "grounding_checked": len(checked),
        "grounding_flagged_traces": g_flagged,
        "grounding_flagged_rate": round(g_flagged / len(checked) * 100, 1) if checked else None,
        "grounding_flagged_items": sum(len(t["grounding"].get("flagged") or []) for t in checked),
        "format_fallback": format_fallback,
        # 延迟的样本数与链路条数不一定相等：没测到耗时的那些进不了分位数
        "latency_samples": len(lat),
        "p95_ms": _p95(lat),
        "p50_ms": int(statistics.median(lat)) if lat else None,
        "avg_tokens": round(statistics.mean(toks)) if toks else None,
        "avg_cost": round(statistics.mean(costs), 4) if costs else None,
        "total_cost": round(sum(costs), 4),
        "model_calls": model_calls,
        "model_failures": model_fail,
        "model_success_rate": round((model_calls - model_fail) / model_calls * 100, 1)
                              if model_calls else None,
        "failed_traces": sum(1 for t in items if t.get("status") == "FAILED"),
        "tool_calls": tool_calls,
        "tool_failures": tool_fail,
        "tool_refused": tool_refused,
        "tool_success_rate": round((tool_judged - tool_fail) / tool_judged * 100, 1)
                             if tool_judged else None,
        "chain_completion_rate": round(normal / len(items) * 100, 1),
        "degraded": degraded,
        "degraded_traces": degraded_traces,
    }
