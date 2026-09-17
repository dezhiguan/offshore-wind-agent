# -*- coding: utf-8 -*-
"""后台指标的口径守卫。

2026-09-17 复核链路追踪那五格时发现：算术全对，错的都是口径 ——
护栏拦截被记成工具失败、价目表外的模型照样写"按官方单价"、
失败调用的用量缺失被当成零。这些都不会让某个数字算错，
只会让它在界面上被读成另一件事，因此必须由测试钉住。
"""
import os

import pytest

from agent import tracestore
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


class TestPercentile:
    """P95 宁可偏保守，不能偏乐观 —— 偏乐观的那一版在最该示警时最好看。"""

    def test_p95_does_not_drop_the_slowest_samples(self):
        # 线上那 39 条的形状：原先取第 37 位（54.3s），把最慢的两条整个排除在外
        values = list(range(1, 40))
        assert tracestore._p95(values) == 38

    def test_p95_of_a_single_sample_is_itself(self):
        assert tracestore._p95([7]) == 7

    def test_p95_of_nothing_is_none(self):
        assert tracestore._p95([]) is None


def _trace(**meta):
    """造一条最小可留存链路：一次模型调用 + 一次工具调用。"""
    tr = Trace()
    tr.record("MODEL", "Agent 决策", prompt_tokens=100, completion_tokens=10, cost_cny=0.0001)
    tr.record("TOOL", "query_db", status=OK)
    base = {"stop_reason": "completed", "elapsed_ms": 1000, "steps": 1,
            "format_parsed": True, "model": "qwen3.8-flash"}
    base.update(meta)
    return {"answer": "## 结论\n有结论", "unverifiable": [], "spans": tr.as_list(),
            "trace_summary": tr.summary(), "meta": base}


@pytest.fixture
def store():
    """链路留存是模块级的 deque，用例之间必须清干净。"""
    tracestore._traces.clear()
    yield tracestore
    tracestore._traces.clear()


class TestGroundingIsKept:
    """接地校验默认跑 shadow：只标注、不改回答。结果不留存，影子跑就是白跑 ——
    判定层要先标定再切 enforce，而标定靠的正是这批结果。"""

    def test_shadow_result_survives_into_the_stats(self, store):
        store.record("Q1", _trace(grounding={"mode": "shadow", "flagged": [
            {"type": "number", "value": "480"}]}))
        store.record("Q2", _trace(grounding={"mode": "shadow", "flagged": []}))
        s = store.stats(source="online")
        assert s["grounding_checked"] == 2
        assert s["grounding_flagged_traces"] == 1
        assert s["grounding_flagged_items"] == 1
        assert s["grounding_flagged_rate"] == 50.0

    def test_off_mode_is_not_a_clean_bill(self, store):
        """关掉校验不等于零存疑：没检查过的不能进分母，否则关闭它像是质量变好了。"""
        store.record("Q", _trace(grounding={"mode": "off", "flagged": []}))
        s = store.stats(source="online")
        assert s["grounding_checked"] == 0
        assert s["grounding_flagged_rate"] is None


class TestFormatFallbackIsVisible:
    """三段标题没解析出来时回答回落成整段原文：非空、于是算"有效回答"，
    而「现有资料无法确认」也拆不出来、于是"自报存疑"少算。一次降级同时
    抬高一个指标、压低另一个，两处都不出声。"""

    def test_fallback_is_counted(self, store):
        store.record("Q", _trace(format_parsed=False))
        assert store.stats(source="online")["format_fallback"] == 1

    def test_fallback_degrades_the_chain(self, store):
        store.record("Q", _trace(format_parsed=False))
        item = store.listing(source="online")[0]
        assert item["status"] == "DEGRADED"
        assert any(n["kind"] == "格式降级" for n in item["notes"])

    def test_a_failed_run_is_not_also_a_format_problem(self, store):
        """链路中断时本来就没有正文，再报一条"格式降级"是噪音。"""
        store.record("Q", _trace(stop_reason="llm_error", format_parsed=False),
                     error="Timeout")
        item = store.listing(source="online")[0]
        assert item["status"] == "FAILED"
        assert not any(n["kind"] == "格式降级" for n in item["notes"])


class TestGuardRefusalIsVisible:
    def test_attempted_watermark_reaches_the_stats(self, store):
        store.record("Q", _trace(budget={"db_rows_pct": 46.8, "doc_chars_pct": 30.8,
                                         "refusals": 1, "peak_attempt_db_rows_pct": 93.5,
                                         "peak_attempt_doc_chars_pct": 30.8}))
        s = store.stats(source="online")
        assert s["guard_refusals"] == 1
        # 已计费的覆盖率在阈值以下，试图水位在阈值以上 —— 界面要能同时看到两个
        assert s["peak_db_rows_pct"] < s["limit_row_pct"]
        assert s["peak_attempt_db_rows_pct"] > s["limit_row_pct"]

    def test_threshold_follows_the_guard(self, store):
        """红线画在哪必须跟着护栏的阈值走，不能在前端另写一份。"""
        from tools import budget as budget_mod
        store.record("Q", _trace())
        s = store.stats(source="online")
        assert s["limit_row_pct"] == budget_mod.MAX_ROW_RATIO * 100
        assert s["limit_doc_pct"] == budget_mod.MAX_DOC_RATIO * 100


class TestSampleWindowIsShared:
    def test_eval_traces_occupy_the_same_50_slots(self, store):
        """留存是所有来源共用的一格 deque：跑一轮回归会挤掉同样多的线上链路。
        只报 count/capacity，样本被顶掉是悄无声息的。"""
        store.record("线上", _trace())
        store.record("回归", _trace(), source="eval")
        s = store.stats(source="online")
        assert s["count"] == 1 and s["capacity_used"] == 2
