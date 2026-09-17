# -*- coding: utf-8 -*-
"""上下文预算测试：把「没把整库塞进去」从行为变成机制。

2026-09-17 起分两档记：**覆盖**（去重后的语料占比）与**取回**（含重复的累计量）。
两档都要有守卫 —— 只测覆盖，重复取回绕过红线不会被发现；只测取回，
反复引用同一条款会被误判成"要把整份文档搬进去"。
"""
import pytest

from tools.budget import ContextBudget, Draw, corpus_totals, measure


def rows(n: int, tag: str = "r") -> Draw:
    return Draw.raw(rows=n, tag=tag)


def chars(n: int, tag: str = "c") -> Draw:
    return Draw.raw(chars=n, tag=tag)


class TestCorpusTotals:
    def test_totals_match_actual_corpus(self):
        t = corpus_totals()
        assert t["db_rows"] == 62      # 47 条告警 + 15 张工单
        assert t["doc_chars"] > 6000   # 两份 Markdown


class TestBudgetEnforcement:
    def test_allows_normal_query(self):
        b = ContextBudget()
        assert b.would_exceed(rows(9)) is None

    def test_blocks_full_table_dump(self):
        """全表 47 行告警超过阈值，必须拒绝——这正是题面红线要防的。"""
        b = ContextBudget()
        refusal = b.would_exceed(rows(47))
        assert refusal is not None
        assert "整张表" in refusal

    def test_blocks_cumulative_creep(self):
        """单次不超但累计超，同样要拦——否则分多次取就绕过去了。"""
        b = ContextBudget()
        for i in range(3):
            draw = rows(12, tag="batch%d" % i)
            if b.would_exceed(draw) is None:
                b.charge(draw)
        assert b.would_exceed(rows(12, tag="last")) is not None

    def test_blocks_whole_document_load(self):
        b = ContextBudget()
        refusal = b.would_exceed(chars(corpus_totals()["doc_chars"]))
        assert refusal is not None
        assert "整份文档" in refusal

    def test_refusal_message_tells_model_what_to_do(self):
        """拒绝信息会回灌给模型，必须说清楚下一步怎么做，而不是只报错。"""
        b = ContextBudget()
        refusal = b.would_exceed(rows(47))
        assert "缩小查询范围" in refusal

    def test_report_is_auditable(self):
        b = ContextBudget()
        b.charge(rows(9))
        b.charge(chars(500))
        r = b.report()
        assert r["db_rows"] == 9 and r["db_rows_total"] == 62
        assert 0 < r["db_rows_pct"] < 100
        assert r["doc_chars"] == 500


class TestCoverageVsDrawn:
    """覆盖与取回是两个数，报表上必须都在。"""

    def test_repeat_fetch_counts_once_in_coverage(self):
        b = ContextBudget()
        hit = {"ok": True, "hits": [{"doc_key": "fault_manual", "section_id": "24002",
                                     "snippet": "x" * 160}]}
        for _ in range(3):
            b.charge(measure("search_docs", hit))
        r = b.report()
        assert r["doc_chars"] == 160, "同一节取三次，覆盖只算一次"
        assert r["doc_chars_drawn"] == 480, "但上下文里确实堆了三份，取回档要如实记"

    def test_section_upgrade_charges_only_delta(self):
        """先拿 160 字摘要、后取回整节，覆盖按见过的最大篇幅算，不叠加。"""
        b = ContextBudget()
        b.charge(measure("search_docs", {"ok": True, "hits": [
            {"doc_key": "fault_manual", "section_id": "24002", "snippet": "x" * 160}]}))
        b.charge(measure("get_doc_section", {"ok": True, "doc_key": "fault_manual",
                                             "section_id": "24002", "text": "x" * 900}))
        assert b.report()["doc_chars"] == 900

    def test_clause_text_dedupes_against_section(self):
        """规则内联的条款原文与单独取回的同一节，是同一段语料。"""
        b = ContextBudget()
        b.charge(measure("check_rule", {"ok": True, "clause_texts": {"第 3.1 条": "y" * 300}}))
        b.charge(measure("get_doc_section", {"ok": True, "doc_key": "safety_regulation",
                                             "section_id": "3.1", "text": "y" * 300}))
        assert b.report()["doc_chars"] == 300

    def test_same_rows_requeried_counts_once(self):
        b = ContextBudget()
        result = {"ok": True, "row_count": 3,
                  "rows": [{"id": 1}, {"id": 2}, {"id": 3}]}
        b.charge(measure("query_db", result))
        b.charge(measure("query_db", result))
        r = b.report()
        assert r["db_rows"] == 3 and r["db_rows_drawn"] == 6

    def test_repeat_fetch_cannot_bypass_the_red_line(self):
        """去重是为了报表诚实，不是给绕过红线开的口子：取回档照样拦。"""
        b = ContextBudget()
        section = {"ok": True, "doc_key": "fault_manual", "section_id": "24002",
                   "text": "x" * 1600}
        blocked = False
        for _ in range(10):
            draw = measure("get_doc_section", section)
            if b.would_exceed(draw) is not None:
                blocked = True
                break
            b.charge(draw)
        assert blocked, "同一节反复取回，累计取回档必须在红线处拦住"

    def test_context_chars_counts_the_whole_payload(self):
        """注入上下文的是整个工具返回，不只是里面的语料原文。"""
        b = ContextBudget()
        draw = measure("check_rule", {"ok": True, "verdict": "不合规" * 50,
                                      "clause_texts": {"第 6.1 条": "y" * 300}})
        b.charge(draw)
        r = b.report()
        assert r["doc_chars"] == 300
        assert r["context_chars"] > 300, "结论文本也占上下文，只是不算语料覆盖"


class TestMeasure:
    def test_counts_query_rows(self):
        d = measure("query_db", {"ok": True, "row_count": 7,
                                 "rows": [{"i": i} for i in range(7)]})
        assert len(d.rows) == 7 and not d.units

    def test_counts_section_text(self):
        d = measure("get_doc_section", {"ok": True, "doc_key": "fault_manual",
                                        "section_id": "24002", "text": "x" * 300})
        assert sum(c for _, c in d.units) == 300

    def test_counts_search_snippets(self):
        r = {"ok": True, "hits": [{"doc_key": "d", "section_id": "1", "snippet": "a" * 100},
                                  {"doc_key": "d", "section_id": "2", "snippet": "b" * 60}]}
        assert sum(c for _, c in measure("search_docs", r).units) == 160

    def test_counts_inlined_clause_texts(self):
        """check_rule 内联的条款原文同样占上下文，不能漏计。"""
        r = {"ok": True, "clause_texts": {"第 3.1 条": "y" * 200, "第 3.2 条": "z" * 150}}
        assert sum(c for _, c in measure("check_rule", r).units) == 350

    def test_failed_call_costs_nothing(self):
        d = measure("query_db", {"ok": False, "error": "x"})
        assert not d.rows and not d.units and d.context_chars == 0

    def test_rows_without_detail_are_treated_as_new(self):
        """拿不到行级明细时当作全新内容 —— 宁可早拦，不可当成重复静默放行。"""
        b = ContextBudget()
        for _ in range(2):
            b.charge(measure("query_db", {"ok": True, "row_count": 5}))
        assert b.report()["db_rows"] == 5, "同一结果指纹相同，仍然去重"
        b.charge(measure("query_db", {"ok": True, "row_count": 5, "note": "另一批"}))
        assert b.report()["db_rows"] == 10


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
        assert b.would_exceed(chars(corpus_totals()["doc_chars"])) is not None


class TestMeterCoverage:
    def test_every_tool_has_a_meter(self):
        """新增工具却忘了登记计量规则，会静默不计费。这条守卫就是为此存在的。"""
        from agent.tools_spec import TOOL_REGISTRY
        from tools.budget import METERS

        missing = set(TOOL_REGISTRY) - set(METERS)
        assert not missing, "以下工具未在 tools/budget.py 的 METERS 中登记计量规则：%s" % missing

    def test_unknown_tool_is_charged_conservatively(self):
        """未登记的工具不能按零计费——宁可高估被拦，也不要静默漏计。"""
        d = measure("some_future_tool", {"ok": True, "payload": "x" * 500})
        assert sum(c for _, c in d.units) > 400
