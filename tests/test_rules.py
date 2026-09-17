# -*- coding: utf-8 -*-
"""规则引擎测试。断言的是规程条款的落地结果，不是模型输出。"""
import pytest

from tools.rules import (RuleInputError, check_rule, clause_sections,
                         close_compliance, repeat_fault)


class TestRepeatFault:
    def test_sliding_window_not_calendar_day(self):
        """T03/24002：窗口内 4 次。按自然日分组会算成 3+1，结论碰巧仍对但过程错。"""
        r = repeat_fault("T03", "24002")
        assert r["facts"]["最大连续24小时窗口内次数"] == 4
        assert r["is_repeat_fault"] is True
        assert r["facts"]["该窗口"]["起"] == "2026-07-18 08:15:00"
        assert r["facts"]["该窗口"]["止"] == "2026-07-19 06:30:00"

    def test_out_of_window_record_excluded_and_reported(self):
        """07-05 那条相隔超 24 小时，必须排除，且要显式说明排除了哪条。"""
        r = repeat_fault("T03", "24002",
                         window_start="2026-07-18 08:00:00", window_end="2026-07-19 08:00:00")
        assert r["facts"]["指定窗口"]["次数"] == 4
        assert "2026-07-05 16:00:00" in r["facts"]["窗口外未计入的告警"]

    def test_below_threshold_is_not_repeat(self):
        """T07/24012 只有 2 次——反向陷阱，不能顺着答成立。"""
        r = repeat_fault("T07", "24012")
        assert r["facts"]["最大连续24小时窗口内次数"] == 2
        assert r["is_repeat_fault"] is False
        assert "不构成重复故障" in r["verdict"]

    def test_window_is_closed_interval(self):
        """恰好落在端点上的记录要算进去。"""
        r = repeat_fault("T03", "24002",
                         window_start="2026-07-18 08:15:00", window_end="2026-07-18 19:05:00")
        assert r["facts"]["指定窗口"]["次数"] == 3


class TestPriorityRequired:
    def test_safety_chain_requires_emergency(self):
        """T05/24005 急停：第 2.1 条要求 EMERGENCY，且现有 NORMAL 工单应升级。"""
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert r["required_priority"] == "EMERGENCY"
        assert "WO-260705" in r["verdict"]
        assert "2.1" in r["clauses"]

    def test_repeat_fault_requires_high(self):
        r = check_rule("priority_required", turbine_id="T03", fault_code="24002")
        assert r["required_priority"] == "HIGH"
        assert "3.2" in r["clauses"]

    def test_no_alarm_returns_explicit_none(self):
        r = check_rule("priority_required", turbine_id="T01", fault_code="24999")
        assert r["required_priority"] is None
        assert "没有" in r["verdict"]


class TestWorkOrderAssessment:
    def test_missing_work_order_is_explicit_not_silent(self):
        """T07/24012 数据库里没有工单——必须显式说「不存在」，不能让模型去编。"""
        r = check_rule("work_order_assessment", turbine_id="T07", fault_code="24012")
        assert r["has_work_order"] is False
        assert "不存在" in r["verdict"]

    def test_five_elements_all_present(self):
        r = check_rule("work_order_assessment", turbine_id="T05", fault_code="24005")
        item = r["facts"]["工单"][0]
        for key in ("① 优先级", "② 状态", "③ 处理记录", "④ 观察时间", "⑤ 备件"):
            assert key in item

    def test_unavailable_part_flagged(self):
        """T08/24010 备件不可用，第 7.2 条禁止回答「可以立即完成更换」。"""
        r = check_rule("work_order_assessment", turbine_id="T08", fault_code="24010")
        assert "不可用" in r["verdict"]
        assert "7.2" in r["clauses"]


class TestRemoteResetBan:
    def test_emergency_stop_banned(self):
        r = check_rule("remote_reset_ban", turbine_id="T05", fault_code="24005")
        assert r["reset_banned"] is True
        assert "急停" in r["verdict"]

    def test_repeat_fault_banned(self):
        r = check_rule("remote_reset_ban", turbine_id="T03", fault_code="24002")
        assert r["reset_banned"] is True
        assert "4 次" in r["verdict"]

    def test_safety_state_always_unverifiable(self):
        """第 4.1(4) 款不能自动判成立，也不能默认已确认——只能标无法确认。"""
        r = check_rule("remote_reset_ban", turbine_id="T03", fault_code="24002")
        rows = [c for c in r["facts"]["逐款核对"] if "4.1(4)" in c["款"]]
        assert rows and rows[0]["结论"] == "资料无法确认"


class TestCloseCompliance:
    def test_t09_eeprom_order_not_compliant(self):
        """WO-260708：备注只有「重新上电后故障消失」、观察 15 分钟。
        同时违反第 6.2、6.3、6.4 条。"""
        r = close_compliance("WO-260708")
        assert r["is_compliant"] is False
        assert "120" in r["verdict"]
        assert r["facts"]["工单"]["observation_minutes"] == 15
        assert {"6.2", "6.3", "6.4"} <= set(r["clauses"])

    def test_completed_status_does_not_prove_compliance(self):
        r = close_compliance("WO-260708")
        assert "6.5" in r["clauses"]
        assert "只是数据库状态" in r["verdict"]

    def test_note_original_text_returned_for_human_review(self):
        """逐项判定是关键词启发式，必须把备注原文一并返回供复核。"""
        r = close_compliance("WO-260708")
        assert "重新上电" in r["facts"]["处理备注原文"]
        assert r["unverifiable"]

    def test_unknown_order_returns_explicit_message(self):
        r = close_compliance("WO-999999")
        assert r["is_compliant"] is None
        assert "不存在" in r["verdict"]


class TestReplacePrecondition:
    def test_nine_items_checked_seven_unverifiable(self):
        r = check_rule("replace_precondition", work_order_id="WO-260707")
        checks = r["facts"]["逐项核对"]
        assert len(checks) == 9
        unknown = [c for c in checks if c["结论"] == "现有资料无法确认"]
        assert len(unknown) == 7

    def test_never_asserts_ready_to_replace(self):
        """第 5.2 条：即使停机且备件可用，也不得认定具备立即更换条件。"""
        r = check_rule("replace_precondition", work_order_id="WO-260707")
        assert r["can_replace_now"] is False
        assert "不能认定" in r["verdict"]

    def test_unavailable_part_cites_clause_7_2(self):
        r = check_rule("replace_precondition", work_order_id="WO-260707")
        assert "7.2" in r["clauses"]


class TestDispatchAndInputGuard:
    def test_unknown_rule(self):
        assert check_rule("no_such_rule")["ok"] is False

    @pytest.mark.parametrize("bad", ["T1'; DROP TABLE alarm_records; --", "X99", ""])
    def test_turbine_id_whitelist(self, bad):
        r = check_rule("repeat_fault", turbine_id=bad, fault_code="24002")
        assert r["ok"] is False
        assert "格式不正确" in r["error"]

    def test_extra_kwargs_ignored(self):
        """模型常会多塞参数，不能因此报错。"""
        r = check_rule("repeat_fault", turbine_id="T03", fault_code="24002", nonsense=1)
        assert r["ok"] is True


class TestDeclaredSources:
    """规则引擎替模型查掉的东西，必须自己报出来。

    T05 那条不合规结论完全来自规程第 2.1 / 2.4 条，两张表也都是规则引擎查的，
    模型一次 query_db / get_doc_section 都不调也能成立 —— 不自报，依据面板就全空。
    """

    def test_reports_tables_touched_by_nested_rule(self):
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert "alarm_records" in r["sources"]          # priority_required 自己查的
        assert "maintenance_records" in r["sources"]    # 比对现有工单时查的
        assert "safety_regulation" in r["sources"]      # 判定引用了第 2.1 / 2.4 条
        assert "fault_manual" in r["sources"]           # 安全链判定读了手册那一节

    def test_no_clause_no_regulation_claimed(self):
        """查无记录时没有任何条款可援引，不能顺手把规程记成数据源。"""
        r = check_rule("priority_required", turbine_id="T01", fault_code="24999")
        assert "safety_regulation" not in (r.get("sources") or [])

    def test_clause_sections_return_original_text(self):
        """chip 上的条款号要能点开原文，两者必须同源。"""
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        sections = clause_sections(r)
        ids = {s["section_id"] for s in sections}
        assert {"2.1", "2.4"} <= ids
        assert all(s["text"] and s["doc"].endswith(".md") for s in sections)

    def test_sources_do_not_leak_across_calls(self):
        """上一条规则查过的表，不能算到下一条头上。"""
        full = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert "maintenance_records" in full["sources"]
        # 查无告警时在比对工单之前就返回了，工单表这一次没碰过
        r = check_rule("priority_required", turbine_id="T01", fault_code="24999")
        assert "maintenance_records" not in (r.get("sources") or [])
