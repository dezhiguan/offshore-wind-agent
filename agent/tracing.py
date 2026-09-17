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
"""
from __future__ import annotations

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
        calls = [s for s in self.spans if s.kind in ("TOOL", "RAG", "RULE")]
        failed = [s for s in calls if s.status == ERROR]
        return {
            "total_ms": int((time.monotonic() - self.started) * 1000),
            "span_count": len(self.spans),
            "model_calls": len(model),
            "model_failures": len(model_failed),
            "model_success_rate": round((len(model) - len(model_failed)) / len(model) * 100, 1)
                                  if model else None,
            "tool_calls": len(calls),
            "tool_failures": len(failed),
            "tool_success_rate": round((len(calls) - len(failed)) / len(calls) * 100, 1) if calls else None,
            "degraded": sum(1 for s in self.spans if s.status == DEGRADED),
            "prompt_tokens": sum(s.extra.get("prompt_tokens", 0) for s in model),
            "completion_tokens": sum(s.extra.get("completion_tokens", 0) for s in model),
            "cached_tokens": sum(s.extra.get("cached_tokens", 0) for s in model),
            "cost_cny": round(sum(s.extra.get("cost_cny", 0.0) for s in model), 6),
        }

    def as_list(self) -> list[dict[str, Any]]:
        return [s.as_dict(i + 1) for i, s in enumerate(self.spans)]
