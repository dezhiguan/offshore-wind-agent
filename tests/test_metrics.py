# -*- coding: utf-8 -*-
"""后台指标的口径守卫。

2026-09-17 复核链路追踪那五格时发现：算术全对，错的都是口径 ——
护栏拦截被记成工具失败、价目表外的模型照样写"按官方单价"、
失败调用的用量缺失被当成零。这些都不会让某个数字算错，
只会让它在界面上被读成另一件事，因此必须由测试钉住。
"""
import os

import pytest

from agent.tracing import DEGRADED, ERROR, OK, Trace
from tools import pricing


def _chain(*statuses: str) -> Trace:
    tr = Trace()
    tr.record("MODEL", "Agent 决策", prompt_tokens=100, completion_tokens=10,
              cached_tokens=64, cost_cny=0.0001)
    for i, st in enumerate(statuses):
        tr.record("TOOL", "query_db", status=st,
                  guard="上下文预算" if st == DEGRADED else None)
    return tr


class TestGuardIsNotAFailure:
    def test_refused_call_is_not_counted_as_failure(self):
        s = _chain(OK, DEGRADED).summary()
        assert s["tool_calls"] == 2
        assert s["tool_failures"] == 0
        assert s["tool_refused"] == 1

    def test_refused_call_leaves_the_success_denominator(self):
        """护栏拒绝既不算成功也不算失败，放进分母的哪一边都是错的。"""
        assert _chain(OK, DEGRADED).summary()["tool_success_rate"] == 100.0

    def test_real_failure_still_lowers_the_rate(self):
        s = _chain(OK, ERROR, DEGRADED).summary()
        assert s["tool_failures"] == 1
        assert s["tool_success_rate"] == 50.0    # 分母是 2，被拒绝的那次不算

    def test_all_refused_leaves_no_rate_rather_than_zero(self):
        """全被护栏拒绝时没有可判的调用，报 None 而不是 0% —— 0% 会被读成全挂了。"""
        assert _chain(DEGRADED).summary()["tool_success_rate"] is None


class TestUnmeteredCalls:
    def test_missing_usage_is_reported_not_zeroed(self):
        tr = Trace()
        tr.record("MODEL", "Agent 决策", prompt_tokens=100, completion_tokens=10,
                  cached_tokens=0, cost_cny=0.0001)
        tr.record("MODEL", "Agent 决策", status=ERROR, usage_missing=True)
        s = tr.summary()
        assert s["model_calls"] == 2 and s["model_failures"] == 1
        assert s["usage_missing_calls"] == 1, "缺失必须能被界面标出来，不能当成零"
        assert s["prompt_tokens"] == 100


class TestPricingBasis:
    def test_known_model_is_marked_official(self, monkeypatch):
        for k in ("LLM_PRICE_INPUT_PER_1K", "LLM_PRICE_OUTPUT_PER_1K",
                  "LLM_PRICE_CACHED_INPUT_PER_1K"):
            monkeypatch.delenv(k, raising=False)
        b = pricing.basis("qwen3.8-flash")
        assert b["known"] and b["checked_at"] == pricing.PRICES_CHECKED_AT

    def test_unknown_model_is_marked_estimate(self, monkeypatch):
        for k in ("LLM_PRICE_INPUT_PER_1K", "LLM_PRICE_OUTPUT_PER_1K",
                  "LLM_PRICE_CACHED_INPUT_PER_1K"):
            monkeypatch.delenv(k, raising=False)
        b = pricing.basis("some-new-model-v9")
        assert not b["known"] and b["checked_at"] is None

    def test_overriding_input_price_also_moves_the_cached_price(self, monkeypatch):
        """只盖输入价、缓存价留在原模型上，会算出一张两个模型混起来的账。"""
        monkeypatch.setenv("LLM_PRICE_INPUT_PER_1K", "0.0024")
        monkeypatch.delenv("LLM_PRICE_CACHED_INPUT_PER_1K", raising=False)
        r = pricing.rates("qwen3.8-flash")
        assert r["cached_input"] == pytest.approx(0.0024 / 8)

    def test_explicit_cached_override_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_PRICE_INPUT_PER_1K", "0.0024")
        monkeypatch.setenv("LLM_PRICE_CACHED_INPUT_PER_1K", "0.0003")
        assert pricing.rates("qwen3.8-flash")["cached_input"] == 0.0003

    def test_override_is_no_longer_called_official(self, monkeypatch):
        monkeypatch.setenv("LLM_PRICE_INPUT_PER_1K", "0.0024")
        b = pricing.basis("qwen3.8-flash")
        assert not b["known"] and b["overridden"]

    def test_cost_formula_matches_the_published_rates(self, monkeypatch):
        """线上那条链路的实际数字：2870 输入（缓存 2048）+ 61 输出 = ¥0.001027。"""
        for k in ("LLM_PRICE_INPUT_PER_1K", "LLM_PRICE_OUTPUT_PER_1K",
                  "LLM_PRICE_CACHED_INPUT_PER_1K"):
            monkeypatch.delenv(k, raising=False)
        assert round(pricing.cost("qwen3.8-flash", 2870, 61, 2048), 6) == 0.001027
