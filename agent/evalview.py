# -*- coding: utf-8 -*-
"""离线评测只读视图（Agent 质量中心的离线那一档）。

刻意不加持久化：数据源就是 eval/run.py 已经落盘的产物，页面只做渲染。
不引入数据库——那会带来新的状态与故障模式，而本系统的定位是随题演示，
不是需要运维后台的生产服务。

线上那一档不在这里：它的样本是真实问答留下的链路，没有标准答案，判不了对错，
指标口径完全不同，见 agent/tracestore.py 的 stats(source="online")。
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.verdict import SUITES, check  # noqa: E402

OUT_DIR = ROOT / "eval" / "out"
EVAL_DIR = ROOT / "eval"

# 指标快照的流水账。**扩展名必须不是 .json**：load() 用 glob("*.json") 扫产物，
# 叫 _history.json 会被当成一条用例产物解析。
HISTORY = OUT_DIR / "_history.jsonl"


def _suite_of(case_id: str, stored: str | None) -> str:
    """产物里带 suite 就用它；早期产物没有，按 id 前缀回退。"""
    if stored in SUITES:
        return stored
    return {"P": "probes", "X": "pressure"}.get(case_id[:1].upper(), "cases")


def _case_specs() -> dict[str, dict[str, Any]]:
    """从用例文件读回断言，产物里存的那份可能已经过期。"""
    specs: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        path = EVAL_DIR / ("%s.yaml" % suite)
        if not path.exists():
            continue
        for case in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            specs[case["id"]] = dict(case, _suite=suite)
    return specs


def cases_by_id(ids: list[str]) -> list[dict[str, Any]]:
    """按 id 取用例；ids 为空表示全部。顺序按回归 → 探针 → 施压。"""
    specs = _case_specs()
    wanted = {i.strip().upper() for i in ids if i.strip()}
    rank = {"cases": 0, "probes": 1, "pressure": 2}
    picked = [c for cid, c in specs.items() if not wanted or cid.upper() in wanted]
    picked.sort(key=lambda c: (rank.get(c.get("_suite"), 9), len(c["id"]), c["id"]))
    return picked


def judge(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    """与命令行跑测器共用同一份判定。"""
    return check(case, result)


def persist(case: dict[str, Any], result: dict[str, Any]) -> None:
    """把本次结果落盘，与 eval/run.py 的产物格式一致，便于页面刷新后仍可见。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"case": {k: v for k, v in case.items() if not k.startswith("_")},
               "suite": case.get("_suite"), "result": result}
    (OUT_DIR / ("%s.json" % case["id"])).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def snapshot(ran: int | None = None) -> dict[str, Any]:
    """把这一轮跑完之后的指标记一条快照。

    「较上次」要有个上次。指标是从 eval/out 的产物现算的，而产物只留最新一份——
    不落快照的话，上一轮的数字在这一轮覆盖产物时就跟着消失了，趋势无从谈起。

    append-only 的一行一条：这是流水账不是状态。改写历史比丢失历史更糟——
    指标回退时，最该看的恰恰是"从哪一轮开始退的"。

    ran 记这一轮实际跑了几条。页面要用它说清楚："上次"是只跑了 1 条之后的状态，
    还是整套跑完之后的状态；两者的可比性不一样。
    """
    data = load()
    record = {
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ran": ran,
        "total": data["totals"]["total"],
        "passed": data["totals"]["passed"],
        "metrics": {k: v.get("value") for k, v in data["metrics"].items()},
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def history(limit: int = 30) -> list[dict[str, Any]]:
    """读快照流水账。坏行跳过而不是抛异常——一行手工编辑坏了，
    不该让整个评测页打不开。"""
    if not HISTORY.exists():
        return []
    records = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:]


def trend() -> dict[str, Any]:
    """本轮与上一轮的对照基准。

    流水账的最后一条就是当前这一轮（跑完即落），所以"上次"是倒数第二条。
    只有一条时没有可比对象——如实说"首轮"，不要拿空值算出一堆 0% 的涨跌。
    """
    records = history()
    if len(records) < 2:
        return {"baseline": None, "current": records[-1] if records else None,
                "reason": "首轮评测，尚无上一轮可比" if records else "尚未记录过指标快照"}
    return {"baseline": records[-2], "current": records[-1], "reason": None}


def load() -> dict[str, Any]:
    specs = _case_specs()
    items: list[dict[str, Any]] = []

    for path in sorted(OUT_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stored_case = blob.get("case", {})
        cid = stored_case.get("id") or path.stem
        result = blob.get("result", {})
        # 断言以用例文件为准，产物里那份可能过期
        spec = specs.get(cid, stored_case)
        ok, problems = check(spec, result)
        meta = result.get("meta", {}) or {}
        budget = meta.get("budget") or {}
        ts = result.get("trace_summary") or {}
        items.append({
            "id": cid,
            "suite": _suite_of(cid, blob.get("suite") or spec.get("_suite")),
            "question": spec.get("question") or stored_case.get("question", ""),
            "probe": spec.get("probe"),
            "ok": ok,
            "problems": problems,
            "elapsed_ms": meta.get("elapsed_ms"),
            "steps": meta.get("steps"),
            "stop_reason": meta.get("stop_reason"),
            "db_rows_pct": budget.get("db_rows_pct"),
            "doc_chars_pct": budget.get("doc_chars_pct"),
            "sources": result.get("sources", []),
            "model": meta.get("model"),
            "tool_calls": ts.get("tool_calls"),
            "tool_failures": ts.get("tool_failures"),
            "tool_success_rate": ts.get("tool_success_rate"),
            "model_calls": ts.get("model_calls"),
            "tokens": (ts.get("prompt_tokens", 0) or 0) + (ts.get("completion_tokens", 0) or 0),
            "cached_tokens": ts.get("cached_tokens"),
            "cost_cny": ts.get("cost_cny"),
            "spans": result.get("spans", []),
            "answer": result.get("answer", ""),
            "unverifiable": result.get("unverifiable", []),
            "trace": result.get("trace", []),
            "evidence": result.get("evidence", {}),
        })

    def order(item: dict[str, Any]) -> tuple:
        rank = {"cases": 0, "probes": 1, "pressure": 2}
        return (rank.get(item["suite"], 9), item["id"][:1], len(item["id"]), item["id"])

    items.sort(key=order)

    summary = []
    for suite, label in SUITES.items():
        group = [i for i in items if i["suite"] == suite]
        if not group:
            continue
        summary.append({
            "suite": suite,
            "label": label,
            "total": len(group),
            "passed": sum(1 for i in group if i["ok"]),
            "median_ms": sorted(i["elapsed_ms"] or 0 for i in group)[len(group) // 2],
        })

    pcts_row = [i["db_rows_pct"] for i in items if i["db_rows_pct"] is not None]
    pcts_doc = [i["doc_chars_pct"] for i in items if i["doc_chars_pct"] is not None]
    return {
        "items": items,
        "summary": summary,
        "metrics": metrics(items),
        "trend": trend(),
        "totals": {
            "total": len(items),
            "passed": sum(1 for i in items if i["ok"]),
            "peak_db_rows_pct": max(pcts_row) if pcts_row else None,
            "peak_doc_chars_pct": max(pcts_doc) if pcts_doc else None,
        },
    }


def _pct(num: int, den: int) -> float | None:
    return round(num / den * 100, 1) if den else None


def _p95(values: list[int]) -> int | None:
    """P95。样本很少时按最接近的序位取，不做插值——26 条样本上插值是假精度。"""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """关键指标。

    每个指标都标明口径与样本量——同一个"准确率"按不同口径能差出几十个点，
    只给数字不给口径，等于没给。
    """
    n = len(items)
    if not n:
        return {}

    # 结果准确率：断言全部命中才算对。断言写法约定见 eval/cases.yaml 头部。
    passed = sum(1 for i in items if i["ok"])

    # 任务完成率：产出了非空回答（未因未取证被拒、未因异常中断）
    completed = sum(1 for i in items if (i.get("answer") or "").strip())

    # 链路完成率：正常收口，未撞步数上限、未被拒答
    normal = sum(1 for i in items if i.get("stop_reason") == "completed")

    # 工具调用成功率：按调用次数加权，不是按用例平均——
    # 按用例平均会让只调一次工具的简单题和调十次的复杂题权重相同。
    tc = sum(i.get("tool_calls") or 0 for i in items)
    tf = sum(i.get("tool_failures") or 0 for i in items)

    lat = [i["elapsed_ms"] for i in items if i.get("elapsed_ms")]
    toks = [i["tokens"] for i in items if i.get("tokens")]
    costs = [i["cost_cny"] for i in items if i.get("cost_cny") is not None]

    return {
        "result_accuracy": {"value": _pct(passed, n), "note": "%d / %d 条断言全中" % (passed, n)},
        "task_completion": {"value": _pct(completed, n), "note": "%d / %d 条产出非空回答" % (completed, n)},
        "tool_success": {"value": _pct(tc - tf, tc) if tc else None,
                         "note": "%d 次调用 · %d 次失败" % (tc, tf) if tc else "无数据"},
        "chain_completion": {"value": _pct(normal, n),
                             "note": "%d / %d 条正常收口（未撞步数上限）" % (normal, n)},
        "p95_latency_ms": {"value": _p95(lat),
                           "note": ("P50 %.1fs · 样本 %d" % (statistics.median(lat) / 1000, len(lat)))
                                   if lat else "无数据"},
        "avg_tokens": {"value": round(statistics.mean(toks)) if toks else None,
                       "note": ("缓存命中均值 %d" % round(statistics.mean(
                           [i.get("cached_tokens") or 0 for i in items]))) if toks else "无数据"},
        "cost_per_task": {"value": round(statistics.mean(costs), 4) if costs else None,
                          "note": ("合计 ¥%.4f · %d 条" % (sum(costs), len(costs))) if costs else "无数据"},
    }
