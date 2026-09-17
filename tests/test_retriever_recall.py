# -*- coding: utf-8 -*-
"""检索召回回归：把「漏召回」从静默变成会失败的测试。

单条查询自身分不出漏没漏（45 条真实链路检索词标定过：top_score、词项覆盖率、
未识别词数三个判据全部重叠不可分，见 retriever.Index.diagnose 的注释）。
所以召回退化只能靠固定探针集发现 —— 这份测试就是那个发现机制。

底线不是拍的，是实测基线：加停用词表前整体 81%，A/B 两组满分。
分数掉到底线以下就是退化，必须当回事，不许顺手改底线放过去。
"""
from pathlib import Path

import pytest
import yaml

from tools.retriever import get_index, search_docs

PROBES = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / "eval" / "retrieval_probes.yaml").read_text(encoding="utf-8")
)

# 实测基线（2026-09-17）：整体 81%，A 100%、B 100%、C 42%、D 100%、E 100%。
# 留一档余量防抖，但 A/B 不留 —— 术语原词和常见口语替换答不上就是硬伤。
FLOOR_OVERALL = 0.78
FLOOR_BY_GROUP = {"A": 1.0, "B": 1.0, "C": 0.33, "D": 0.75, "E": 0.75}
TOP_K = 4


def _rank(query: str, gold: list[str]) -> int | None:
    index = get_index()
    order = [h["section_id"] for h in index.search(query, top_k=len(index.chunks))]
    for i, section_id in enumerate(order, 1):
        if section_id in gold:
            return i
    return None


def _recall(probes) -> float:
    if not probes:
        return 1.0
    hit = sum(1 for p in probes if (lambda r: r is not None and r <= TOP_K)(_rank(p["query"], p["gold"])))
    return hit / len(probes)


def test_overall_recall_not_regressed():
    recall = _recall(PROBES)
    assert recall >= FLOOR_OVERALL, (
        "整体 top%d 召回跌到 %.0f%%，低于基线底线 %.0f%%。"
        "改了分词/停用词/打分就要看这个数，别改底线。" % (TOP_K, recall * 100, FLOOR_OVERALL * 100)
    )


@pytest.mark.parametrize("group", sorted(FLOOR_BY_GROUP))
def test_recall_by_group_not_regressed(group):
    probes = [p for p in PROBES if p["group"] == group]
    assert probes, "探针集里没有 %s 组用例" % group
    recall = _recall(probes)
    assert recall >= FLOOR_BY_GROUP[group], (
        "%s 组 top%d 召回 %.0f%%，低于底线 %.0f%%" % (group, TOP_K, recall * 100, FLOOR_BY_GROUP[group] * 100)
    )


class TestRecallCheckIsUsable:
    """诊断信号必须既报得准、又不刷屏 —— 两头都要守。"""

    def test_flags_the_unambiguous_miss(self):
        """全是语料里没有的词 → 无命中，必须标出来，不能装作正常。"""
        check = search_docs("弄完之后要盯多久才算数")["recall_check"]
        assert check["status"] == "low_confidence"
        assert "无命中" in check["reasons"]

    def test_stays_quiet_on_good_queries(self):
        """术语查询不许报警。判据放宽到 unknown_terms 会标掉 73% 的真实检索词，
        全黄等于全不黄 —— 这条测试就是防那次回潮。"""
        for query in ("24002 的触发条件", "第 4.1 条 禁止情形",
                      "更换单板 前置条件", "处理后观察时间 关闭工单 要求"):
            check = search_docs(query)["recall_check"]
            assert check["status"] == "ok", "%s 被误报为低置信：%s" % (query, check["reasons"])

    def test_unknown_terms_is_informational_only(self):
        """未识别词要如实列出（供排查），但不得单独触发报警。"""
        check = search_docs("断电重启之后报警没了，这单子能结吗")["recall_check"]
        assert "报警" in check["unknown_terms"]      # 语料写「告警」
        assert check["status"] == "ok"               # 但有命中，不报警


class TestStopwords:
    def test_idf_inversion_fixed(self):
        """虚词曾经权重高过术语：才/就/说明 idf=3.15 > 告警 0.95。"""
        index = get_index()
        for word in ("才", "就", "说明", "的", "情况"):
            assert index.df.get(word, 0) == 0, "%s 仍进了索引" % word

    def test_stopword_no_longer_drives_ranking(self):
        """这一题原本 top1 是「24006 低穿激活」，唯一贡献词是「才」。"""
        hits = search_docs("弄完之后要盯多久才算数")["hits"]
        assert not hits or hits[0]["section_id"] != "24006_SC_变流器低穿激活"
