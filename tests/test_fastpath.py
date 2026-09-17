# -*- coding: utf-8 -*-
"""快路径判据与安全网。

认错一道题不是慢一点，是拿另一个问题的答案去作答——所以这里的守卫重点不在
"该接管的接管了"，而在**不该接管的一律没接管**。
"""
import pytest

from tools import fastpath


class TestMatching:
    @pytest.mark.parametrize("question,expect", [
        ("T20 的最新一条告警是什么？", "turbine_latest_alarm"),
        ("T01 的风机型号是什么？", "turbine_latest_alarm"),
        ("请查询 T06，并回答：1. 风机型号是什么？2. 最新一条告警记录对应的运行状态是什么？",
         "turbine_latest_alarm"),
        ("请查询 T04 在 2026-07-10 00:00:00 至 2026-07-15 23:59:59 期间的告警记录。",
         "turbine_alarms_in_window"),
    ])
    def test_recognized_shapes(self, question, expect):
        assert fastpath.match(question)["name"] == expect

    @pytest.mark.parametrize("question,why", [
        ("T03 的 24002 告警发生了几次？", "带故障码且要计数，应走 check_rule"),
        ("T09 的 WO-260708 工单关闭得合规吗？", "工单合规判定"),
        ("T05 和 T07 哪个告警更多？", "两个风机编号是对比题"),
        ("24002 应该怎么排查？", "没有风机编号，且是文档题"),
        ("目前还有哪些工单处于 OPEN 状态？", "孤例，未纳入白名单"),
        ("T07 最近的告警需要更换单板吗？", "涉及更换判定"),
        ("T02 在 2026-07-12 之后还有告警吗？", "只给了一个时间点，口径不明确"),
    ])
    def test_refuses_everything_else(self, question, why):
        assert fastpath.match(question) is None, why


class TestClosedInterval:
    def test_end_of_day_is_included(self):
        """只写到日期时右端补到 23:59:59。补成 00:00:00 会把当天整天漏在窗口外——
        提示词里反复交代「时间按闭区间比较」防的就是这个。"""
        sql = fastpath.match("T04 在 2026-07-10 至 2026-07-15 的告警记录")["sql"]
        assert "'2026-07-10 00:00:00' AND '2026-07-15 23:59:59'" in sql


class TestShapeGuard:
    def _plan(self):
        return fastpath.match("T06 的最新一条告警是什么？")

    def test_empty_result_is_acceptable(self):
        """查 T20 查不到，「库里没有这台风机的告警」就是正确答案。
        把零行当失败，快路径会在最该发挥作用的那一类问题上退回去。"""
        assert fastpath.shape_ok(self._plan(), {"ok": True, "row_count": 0, "rows": []}) is None

    def test_too_many_rows_falls_back(self):
        bad = fastpath.shape_ok(self._plan(), {"ok": True, "row_count": 9, "rows": [{}] * 9})
        assert bad and "超出" in bad

    def test_missing_column_falls_back(self):
        bad = fastpath.shape_ok(self._plan(),
                                {"ok": True, "row_count": 1, "rows": [{"turbine_model": "X"}]})
        assert bad and "缺少字段" in bad

    def test_failed_query_falls_back(self):
        assert fastpath.shape_ok(self._plan(), {"ok": False, "error": "语法错误"})


class TestAgreement:
    def test_model_selecting_fewer_columns_still_agrees(self):
        """模型常只 SELECT 自己要的几列，快路径取的是整族字段——判据是包含，不是相等。"""
        fast = {"row_count": 1, "rows": [{"turbine_id": "T06", "turbine_model": "WT-5000",
                                          "occurred_at": "2026-07-18 08:15:00"}]}
        model = {"row_count": 1, "rows": [{"turbine_model": "WT-5000"}]}
        assert fastpath.agrees(fast, model)

    def test_different_values_do_not_agree(self):
        fast = {"row_count": 1, "rows": [{"turbine_model": "WT-5000"}]}
        model = {"row_count": 1, "rows": [{"turbine_model": "WT-3000"}]}
        assert not fastpath.agrees(fast, model)

    def test_row_count_mismatch_does_not_agree(self):
        assert not fastpath.agrees({"row_count": 1, "rows": [{"a": 1}]},
                                   {"row_count": 2, "rows": [{"a": 1}, {"a": 2}]})


class TestDefaultMode:
    def test_defaults_to_shadow(self, monkeypatch):
        """会误伤的判定层先影子跑再拦截。单文档红线档那次误伤就是没影子跑的代价。

        断言打在"环境变量缺省时的档位"上，不是打在当前进程的 MODE 上——后者会随
        跑测时的环境变量变，FASTPATH_MODE=enforce 跑一遍这条就红了，而那不是缺陷。
        """
        import importlib

        monkeypatch.delenv("FASTPATH_MODE", raising=False)
        try:
            assert importlib.reload(fastpath).MODE == "shadow"
        finally:
            monkeypatch.undo()
            importlib.reload(fastpath)
