# -*- coding: utf-8 -*-
"""检索器测试：切块粒度、码映射、条款定位、跨文档区分。"""
import pytest

from tools.retriever import get_doc_section, get_index, search_docs


class TestChunking:
    def test_chunk_counts(self):
        idx = get_index()
        manual = [c for c in idx.chunks if c.doc == "fault_manual"]
        regulation = [c for c in idx.chunks if c.doc == "safety_regulation"]
        assert len(manual) == 9          # 手册 9 个故障码
        assert len(regulation) == 25     # 规程 25 个条款

    def test_section_keeps_subheadings(self):
        sec = get_doc_section("fault_manual", "24002")
        assert sec["ok"] is True
        for sub in ("控制原理", "触发条件", "原因分析及解决方案", "故障处理注意事项"):
            assert sub in sec["text"]

    def test_chapter_recorded(self):
        sec = get_doc_section("safety_regulation", "4.1")
        assert "第四章" in sec["path"]


class TestCodeMapping:
    def test_bare_code_hits_manual_section(self):
        """库里是裸码 24002，手册标题是 24002_SC_变流器心跳——必须能对上。"""
        hits = search_docs("24002")["hits"]
        assert hits[0]["section_id"] == "24002_SC_变流器心跳"

    def test_full_code_name_also_hits(self):
        hits = search_docs("24002_SC_变流器心跳 的触发条件")["hits"]
        assert hits[0]["section_id"] == "24002_SC_变流器心跳"

    def test_get_section_by_bare_code(self):
        assert get_doc_section("fault_manual", "24002")["ok"] is True

    def test_all_nine_codes_reachable(self):
        """现场会换题，9 个码必须全覆盖，不能只对考察题里那几个生效。"""
        codes = ["24001", "24002", "24005", "24006", "24010", "24011", "24012", "24013", "24014"]
        for code in codes:
            assert get_doc_section("fault_manual", code)["ok"] is True, code
            assert search_docs(code)["hits"][0]["doc_key"] == "fault_manual", code


class TestClauseLookup:
    def test_clause_number_boosted_to_top(self):
        hits = search_docs("第 4.1 条")["hits"]
        assert hits[0]["section_id"] == "4.1"

    def test_all_25_clauses_reachable(self):
        idx = get_index()
        for c in [c for c in idx.chunks if c.doc == "safety_regulation"]:
            assert get_doc_section("safety_regulation", c.section_id)["ok"] is True, c.section_id

    def test_semantic_query_finds_reset_ban(self):
        hits = search_docs("哪些情形禁止远程强制复位", doc="规程")["hits"]
        assert "4.1" in [h["section_id"] for h in hits]

    def test_semantic_query_finds_observation_time(self):
        hits = search_docs("处理后最低观察时间要求", doc="规程")["hits"]
        assert "6.4" in [h["section_id"] for h in hits]


class TestDocFilter:
    def test_filter_restricts_to_one_doc(self):
        hits = search_docs("更换通讯模块", doc="手册")["hits"]
        assert all(h["doc_key"] == "fault_manual" for h in hits)

    def test_unknown_doc_falls_back_to_all(self):
        """文档名识别不了时全库搜，不要静默返空。"""
        assert search_docs("急停", doc="不存在的文档")["hits"]


class TestOutputShape:
    def test_search_returns_snippet_by_default(self):
        """命中有悬念时只回摘要，全文要显式 get_doc_section——防止上下文被撑爆。"""
        result = search_docs("变流器 温度 偏高 处理")
        assert result["inlined"] is None
        assert all("text" not in h for h in result["hits"])
        assert all(len(h["snippet"]) <= 161 for h in result["hits"])


class TestInlineFullText:
    """命中无悬念时 top-1 直接带回全文（2026-09-18 加），省掉一次 get_doc_section 往返。

    注入的是**一节**原文，与 get_doc_section 放进去的是同一段；红线针对的是
    "整份文档"，而手册只有 9 节 —— 所以只带 top-1，绝不给 top_k 全部带全文。
    """

    def test_exact_fault_code_inlines_top_hit(self):
        result = search_docs("24002 变流器心跳 常见原因")
        assert result["inlined"]["section_id"] == "24002_SC_变流器心跳"
        assert result["hits"][0]["text"].startswith("###")

    def test_exact_clause_number_inlines_top_hit(self):
        assert search_docs("第 4.1 条 禁止远程复位")["inlined"]["section_id"] == "4.1"

    def test_only_the_top_hit_ever_carries_full_text(self):
        """四节全带全文就是把大半本手册搬进上下文——那正是红线要拦的事。"""
        result = search_docs("24002 变流器心跳 常见原因")
        assert all("text" not in h for h in result["hits"][1:])

    def test_low_confidence_recall_never_inlines(self):
        """召回本身不可信时推全文，等于把一次漏召回放大成一整段像模像样的原文。"""
        result = search_docs("弄完之后要盯多久才算数")
        assert result["recall_check"]["status"] == "low_confidence"
        assert result["inlined"] is None

    def test_inlined_text_is_metered_as_full_section(self):
        """按摘要计量会凭空漏掉一节的量，而漏计的正是护栏要拦的那一侧。"""
        from tools.budget import measure

        result = search_docs("24002 变流器心跳 常见原因")
        full = len(result["hits"][0]["text"])
        drawn = dict(measure("search_docs", result).units)
        assert drawn["fault_manual:24002_SC_变流器心跳"] == full

    def test_missing_section_returns_error_not_exception(self):
        result = get_doc_section("fault_manual", "99999")
        assert result["ok"] is False
        assert "未找到章节" in result["error"]
