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


class TestRefusalLedger:
    """被红线拒绝的那次取证不进用量 —— 于是覆盖率永远画不出撞线。

    2026-09-17 复核线上 39 条链路时发现：峰值 46.8% 对着 50% 的阈值，看板上是
    "还有 3.2pp 余量"，而那条链路（29/31 行）其实已经被护栏拒过一次。
    拒绝不 charge 是对的，漏的是**拒绝本身**没被记下来。
    """

    def test_refusal_is_not_charged_into_usage(self):
        b = ContextBudget()
        b.charge(rows(9))
        draw = rows(99, tag="huge")
        refusal = b.would_exceed(draw)
        assert refusal is not None
        b.note_refusal(draw, refusal)
        assert b.db_rows == 9, "被拒的取证不能进用量，否则红线自己把自己撑爆"

    def test_attempt_level_is_recorded_and_may_exceed_the_limit(self):
        b = ContextBudget()
        b.charge(rows(9))
        draw = rows(99, tag="huge")
        b.note_refusal(draw, b.would_exceed(draw))
        r = b.report()
        assert r["refusals"] == 1
        # 已计费的覆盖率停在阈值以下，试图水位越过阈值 —— 这才是该被看见的那个数
        assert r["db_rows_pct"] < 50
        assert r["peak_attempt_db_rows_pct"] > 50

    def test_attempt_takes_the_higher_of_both_gauges(self):
        """覆盖档与取回档共用同一个阈值和单位，报较高的那个才是"想压到哪"。"""
        b = ContextBudget()
        b.charge(rows(9))
        b.charge(rows(9))                       # 同一批行重复取回：覆盖 9，取回 18
        draw = rows(9, tag="new")
        b.note_refusal(draw, "重复取回触发")
        # 覆盖档试图到 18 行、取回档试图到 27 行 —— 取回档才是这次想压到的高度
        assert b.report()["peak_attempt_db_rows_pct"] == pytest.approx(round(27 / 62 * 100, 1))

    def test_no_refusal_reports_none_not_zero(self):
        """没被拦过时报 None：0% 会在界面上画出一条"试图压到 0%"的线。"""
        r = ContextBudget().report()
        assert r["refusals"] == 0
        assert r["peak_attempt_db_rows_pct"] is None
        assert r["peak_attempt_doc_chars_pct"] is None


class TestSingleDocGuard:
    """单文档档（2026-09-18 补）。

    起因：阈值的分母原来是两份文档合计 6538 字符，而故障处理手册全文只有 3845 —— 
    整本手册逐节取进上下文，合计口径只有 57%，低于 70% 的阈值，护栏一声不吭。
    红线说的是"整份文档"，合计档答不上这个问题。
    """

    def _manual_sections(self):
        from tools.retriever import get_index

        return [c for c in get_index().chunks if c.doc == "fault_manual"]

    def test_whole_document_is_blocked_before_it_is_whole(self):
        b = ContextBudget()
        blocked = None
        for c in self._manual_sections():
            draw = Draw(units=(("fault_manual:%s" % c.section_id, len(c.text)),))
            blocked = b.would_exceed(draw)
            if blocked:
                break
            b.charge(draw)
        assert blocked and "故障处理手册" in blocked
        # 拦下的那一刻合计档还远没到阈值 —— 正是旧口径放行的那条路
        assert b.report()["doc_chars_pct"] < 70

    def test_normal_compliance_question_is_not_hit(self):
        """标定峰值 34.9%（P8），正常取证不该被这一档拦住。"""
        b = ContextBudget()
        draw = Draw(units=tuple(("safety_regulation:%s" % n, 120)
                                for n in ("4.1", "4.2", "1.2", "3.1", "3.2", "6.1", "6.2")))
        assert b.would_exceed(draw) is None

    def test_unknown_unit_keys_skip_the_single_doc_gauge(self):
        """认不出文档的取数照常计入合计档，但不参与单文档判据。"""
        b = ContextBudget()
        assert b.would_exceed(chars(2000, tag="未登记工具")) is None
        b.charge(chars(2000, tag="未登记工具"))
        assert b.report()["peak_single_doc_pct"] == 0.0

    def test_refusal_records_the_single_doc_water_line(self):
        b = ContextBudget()
        draw = Draw(units=(("fault_manual:24002", 3000),))
        reason = b.would_exceed(draw)
        assert reason
        b.note_refusal(draw, reason)
        # 被拒不计费，已计费的单文档水位停在 0——只看它就永远画不出撞线
        r = b.report()
        assert r["peak_single_doc_pct"] == 0.0
        assert r["peak_attempt_single_doc_pct"] > 60

    def test_report_lists_every_document_even_untouched(self):
        """一份都没读到的文档也要出现在表里：缺行会被读成"这份没有分母"。"""
        rows_ = ContextBudget().report()["doc_chars_by_doc"]
        assert {d["doc"] for d in rows_} == {"fault_manual", "safety_regulation"}
        assert all(d["total"] > 0 and d["limit"] > 0 for d in rows_)

    def test_repeated_clause_inlining_is_not_mistaken_for_emptying_the_doc(self):
        """26 条跑测里误伤 3 次的那个形状（X2 / P8）：`check_rule` 的通则调用会内联
        同一批条款原文，一个合规问题连调两次，累计取回就顶到单文档上限——而去重后
        其实只读了规程的三分之一。红线问的是"整份文档是不是被搬进去了"，那是去重口径。
        """
        b = ContextBudget()
        clauses = tuple(("safety_regulation:%s" % n, 160)
                        for n in ("4.1", "4.2", "3.3", "5.1", "5.2"))
        for _ in range(4):                      # 同一批条款反复内联四次
            draw = Draw(units=clauses)
            assert b.would_exceed(draw) is None
            b.charge(draw)
        report = b.report()
        assert report["peak_single_doc_pct"] < 60          # 去重后水位不动
        assert report["doc_chars_drawn"] > report["doc_chars"]   # 取回档确实累计了
