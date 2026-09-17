# -*- coding: utf-8 -*-
"""上下文预算测试：把「没把整库塞进去」从行为变成机制。"""
import pytest

from tools.budget import ContextBudget, corpus_totals, measure


class TestCorpusTotals:
    def test_totals_match_actual_corpus(self):
        t = corpus_totals()
        assert t["db_rows"] == 62      # 47 条告警 + 15 张工单
        assert t["doc_chars"] > 6000   # 两份 Markdown


class TestBudgetEnforcement:
    def test_allows_normal_query(self):
        b = ContextBudget()
        assert b.would_exceed(rows=9) is None

    def test_blocks_full_table_dump(self):
        """全表 47 行告警超过阈值，必须拒绝——这正是题面红线要防的。"""
        b = ContextBudget()
        refusal = b.would_exceed(rows=47)
        assert refusal is not None
        assert "整张表" in refusal

    def test_blocks_cumulative_creep(self):
        """单次不超但累计超，同样要拦——否则分多次取就绕过去了。"""
        b = ContextBudget()
        for _ in range(3):
            if b.would_exceed(rows=12) is None:
                b.charge(rows=12)
        assert b.would_exceed(rows=12) is not None

    def test_blocks_whole_document_load(self):
        b = ContextBudget()
        refusal = b.would_exceed(chars=corpus_totals()["doc_chars"])
        assert refusal is not None
        assert "整份文档" in refusal

    def test_refusal_message_tells_model_what_to_do(self):
        """拒绝信息会回灌给模型，必须说清楚下一步怎么做，而不是只报错。"""
        b = ContextBudget()
        refusal = b.would_exceed(rows=47)
        assert "缩小查询范围" in refusal

    def test_report_is_auditable(self):
        b = ContextBudget()
        b.charge(rows=9, chars=500)
        r = b.report()
        assert r["db_rows"] == 9 and r["db_rows_total"] == 62
        assert 0 < r["db_rows_pct"] < 100
        assert r["doc_chars"] == 500


class TestMeasure:
    def test_counts_query_rows(self):
        assert measure("query_db", {"ok": True, "row_count": 7}) == (7, 0)

    def test_counts_section_text(self):
        assert measure("get_doc_section", {"ok": True, "text": "x" * 300}) == (0, 300)

    def test_counts_search_snippets(self):
        r = {"ok": True, "hits": [{"snippet": "a" * 100}, {"snippet": "b" * 60}]}
        assert measure("search_docs", r) == (0, 160)

    def test_counts_inlined_clause_texts(self):
        """check_rule 内联的条款原文同样占上下文，不能漏计。"""
        r = {"ok": True, "clause_texts": {"第 3.1 条": "y" * 200, "第 3.2 条": "z" * 150}}
        assert measure("check_rule", r) == (0, 350)

    def test_failed_call_costs_nothing(self):
        assert measure("query_db", {"ok": False, "error": "x"}) == (0, 0)


class TestThresholdHeadroom:
    def test_doc_threshold_above_observed_peak(self):
        """文档侧阈值必须明显高于实测峰值，否则正常问题会被误拦。

        实测峰值 47.2%（P2，检索面最广的一条）。阈值若定在 50%，余量只有 1.06 倍。
        """
        from tools.budget import MAX_DOC_RATIO
        observed_peak = 0.472
        assert MAX_DOC_RATIO >= observed_peak * 1.4

    def test_still_blocks_whole_document(self):
        """放宽阈值后仍必须拦住整份文档载入。"""
        b = ContextBudget()
        assert b.would_exceed(chars=corpus_totals()["doc_chars"]) is not None


class TestMeterCoverage:
    def test_every_tool_has_a_meter(self):
        """新增工具却忘了登记计量规则，会静默不计费。这条守卫就是为此存在的。"""
        from agent.tools_spec import TOOL_REGISTRY
        from tools.budget import METERS

        missing = set(TOOL_REGISTRY) - set(METERS)
        assert not missing, "以下工具未在 tools/budget.py 的 METERS 中登记计量规则：%s" % missing

    def test_unknown_tool_is_charged_conservatively(self):
        """未登记的工具不能按零计费——宁可高估被拦，也不要静默漏计。"""
        rows, chars = measure("some_future_tool", {"ok": True, "payload": "x" * 500})
        assert chars > 400
