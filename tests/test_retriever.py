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
    def test_search_returns_snippet_not_full_text(self):
        """检索只回摘要，全文要显式 get_doc_section——防止上下文被撑爆。"""
        hit = search_docs("24002")["hits"][0]
        assert "text" not in hit
        assert len(hit["snippet"]) <= 161

    def test_missing_section_returns_error_not_exception(self):
        result = get_doc_section("fault_manual", "99999")
        assert result["ok"] is False
        assert "未找到章节" in result["error"]
