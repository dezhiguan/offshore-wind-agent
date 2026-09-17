# -*- coding: utf-8 -*-
"""链路追踪：把一次问答拆成可逐段检查的 span。

目的不是监控告警，是**让执行过程可复核**——哪一步做了什么、花了多久、
吃了多少 token、有没有降级。题面要求展示「简要查询过程」，
但要判断 agent 的决策是否合理，摘要不够，得看到每一段的输入输出。

span 分五类，与它和模型的关系对应：
  MODEL   模型调用（决策或成文），带 token 与成本
  TOOL    工具执行（查库、取章节）
  RAG     检索召回
  RULE    规程判定（确定性，不经模型）
  GUARD   护栏判定（只读守卫、上下文预算、未取证拦截）

状态三档：OK / DEGRADED（降级但继续，例如预算拦截后改写查询）/ ERROR。

**DEGRADED 不是失败**（2026-09-17 修）：护栏按红线拒绝一次取数，工具本身跑通了，
是系统不让结果进上下文。把它记进 ``tool_failures`` 会让「工具调用成功率」被自家
护栏拉低 —— 那个指标就不再指示"工具靠不靠谱"。因此拒绝单独计数，且从成功率的
**分母里剔除**：既没成功也没失败的一次调用，放进分母的哪一边都是错的。
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Any

OK, DEGRADED, ERROR = "OK", "DEGRADED", "ERROR"

# span 类型 → 该工具属于哪一类
TOOL_KIND = {
    "query_db": "TOOL",
    "search_docs": "RAG",
    "get_doc_section": "RAG",
    "check_rule": "RULE",
}


class Span:
    __slots__ = ("kind", "name", "detail", "started", "elapsed_ms",
                 "input_summary", "output_summary", "status", "extra")

    def __init__(self, kind: str, name: str, detail: str | None = None) -> None:
        self.kind = kind
        self.name = name
        self.detail = detail
        self.started = time.monotonic()
        self.elapsed_ms = 0
        self.input_summary = ""
        self.output_summary = ""
        self.status = OK
        self.extra: dict[str, Any] = {}

    def close(self, *, input_summary: str = "", output_summary: str = "",
              status: str = OK, **extra: Any) -> "Span":
        self.elapsed_ms = int((time.monotonic() - self.started) * 1000)
        self.input_summary = input_summary
        self.output_summary = output_summary
        self.status = status
        self.extra.update({k: v for k, v in extra.items() if v is not None})
        return self

    def as_dict(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "kind": self.kind,
            "name": self.name,
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
            "input": self.input_summary,
            "output": self.output_summary,
            "status": self.status,
            **self.extra,
        }


class Trace:
    """一次问答的 span 收集器。"""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.started = time.monotonic()

    def start(self, kind: str, name: str, detail: str | None = None) -> Span:
        span = Span(kind, name, detail)
        self.spans.append(span)
        return span

    def record(self, kind: str, name: str, *, detail: str | None = None,
               input_summary: str = "", output_summary: str = "",
               status: str = OK, elapsed_ms: int = 0, **extra: Any) -> Span:
        """记录一个已经完成的瞬时 span（例如护栏判定）。"""
        span = Span(kind, name, detail)
        span.close(input_summary=input_summary, output_summary=output_summary,
                   status=status, **extra)
        span.elapsed_ms = elapsed_ms
        self.spans.append(span)
        return span

    def summary(self) -> dict[str, Any]:
        model = [s for s in self.spans if s.kind == "MODEL"]
        model_failed = [s for s in model if s.status == ERROR]
        # 用量没测到的调用（流式被异常打断，usage 还没到）。token 与成本那两格
        # 要据此标注"另有 N 次未计量"，否则缺失会被读成零。
        unmetered = [s for s in model if s.extra.get("usage_missing")]
        calls = [s for s in self.spans if s.kind in ("TOOL", "RAG", "RULE")]
        failed = [s for s in calls if s.status == ERROR]
        refused = [s for s in calls if s.status == DEGRADED]
        judged = len(calls) - len(refused)
        return {
            "total_ms": int((time.monotonic() - self.started) * 1000),
            "span_count": len(self.spans),
            "model_calls": len(model),
            "model_failures": len(model_failed),
            "model_success_rate": round((len(model) - len(model_failed)) / len(model) * 100, 1)
                                  if model else None,
            "usage_missing_calls": len(unmetered),
            "tool_calls": len(calls),
            "tool_failures": len(failed),
            # 护栏拒绝：工具跑通了，是红线不让结果进上下文
            "tool_refused": len(refused),
            "tool_success_rate": round((judged - len(failed)) / judged * 100, 1) if judged else None,
            "degraded": sum(1 for s in self.spans if s.status == DEGRADED),
            "prompt_tokens": sum(s.extra.get("prompt_tokens", 0) for s in model),
            "completion_tokens": sum(s.extra.get("completion_tokens", 0) for s in model),
            "cached_tokens": sum(s.extra.get("cached_tokens", 0) for s in model),
            "cost_cny": round(sum(s.extra.get("cost_cny", 0.0) for s in model), 6),
        }

    def as_list(self) -> list[dict[str, Any]]:
        return [s.as_dict(i + 1) for i, s in enumerate(self.spans)]


# ---- 耗时构成 ----
#
# 这几个函数由线上档（tracestore）与离线档（evalview）共用。两处各写一份，
# 同一个「P95 端到端耗时」就会在两个页面上算出不同的数 —— 而这正是已经发生过的：
# 离线档原先用 round(0.95 * (n - 1))，线上档的注释里已经把这个写法判为偏乐观。

# 不发起工具调用、只把答案写出来的那次模型调用。名字由 _call / _force_answer 决定。
ANSWER_SPANS = ("答案成文", "步数用尽收口")

# 按工具调用次数分桶。混在一起算分位数，得到的是**题目难度的分布**而不是系统性能：
# 一步就能答的题和要查六步的题放进同一个 P95，优化做没做出来会被题目组合的波动盖掉。
TOOL_BUCKETS = ("1 步", "2-3 步", "4 步以上")


def answer_ms(spans: list[dict[str, Any]]) -> int | None:
    """最后一次「写答案」的模型调用耗时。

    实测这一段是端到端耗时的大头（约 69%，见 设计说明 3.14）：它按字符数线性增长，
    而决策轮只有一两秒。把它单独摘出来，"这条链路慢在哪"才不用逐段去点。
    """
    ms = [s.get("elapsed_ms") or 0 for s in spans
          if s.get("kind") == "MODEL" and s.get("name") in ANSWER_SPANS]
    return ms[-1] if ms else None


def p95(values: list[int]) -> int | None:
    """最近秩（nearest-rank）：至少 95% 的样本不超过它。

    不用 ``round(0.95 * (n - 1))``——n=39 时它落在第 37 位，把最慢的两条整个排除，
    报出来的数比真实的第 95 百分位低一截。延迟指标宁可偏保守也不能偏乐观：
    偏乐观的那一版，恰恰在最该示警的时候最好看。
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def tool_bucket(calls: int) -> str:
    return TOOL_BUCKETS[0] if calls <= 1 else TOOL_BUCKETS[1] if calls <= 3 else TOOL_BUCKETS[2]


def latency_buckets(samples: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """按工具调用次数分桶的耗时分布。samples 为 (工具调用次数, 端到端毫秒)。"""
    grouped: dict[str, list[int]] = {}
    for calls, ms in samples:
        if ms:
            grouped.setdefault(tool_bucket(calls or 0), []).append(ms)
    out = []
    for bucket in TOOL_BUCKETS:
        values = grouped.get(bucket)
        if values:
            out.append({"bucket": bucket, "count": len(values),
                        "p50_ms": int(statistics.median(values)), "p95_ms": p95(values)})
    return out


def answer_share_pct(samples: list[tuple[int | None, int | None]]) -> float | None:
    """成文耗时占端到端的比例（中位数）。samples 为 (成文毫秒, 端到端毫秒)。

    取中位数不取均值：一条超时链路的比例会趋近于 0（时间全花在重试上），
    均值会被它拽下来，看起来像"成文不是瓶颈了"。
    """
    shares = [100 * a / t for a, t in samples if a and t]
    return round(statistics.median(shares), 1) if shares else None
