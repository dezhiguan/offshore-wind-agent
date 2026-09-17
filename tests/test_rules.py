# -*- coding: utf-8 -*-
"""规则引擎测试。断言的是规程条款的落地结果，不是模型输出。"""
import pytest

from tools.rules import (RuleInputError, check_rule, cited_sections,
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

    def test_no_records_is_not_the_same_as_below_threshold(self):
        """T01 没有 24002。结论同为「不构成」，但成因必须分得开。

        只说「仅发生 0 次」时，模型会读成「不构成重复故障而已」，接着拿这个 0
        往下答，全程不提这台风机压根没有这条故障。
        """
        r = repeat_fault("T01", "24002")
        assert r["is_repeat_fault"] is False
        assert r["no_records"] is True
        assert "没有" in r["verdict"] and "记录" in r["verdict"]

    def test_existing_records_below_threshold_not_flagged_as_missing(self):
        """反向：T07/24012 有 2 条记录，不能被误标成「无记录」。"""
        r = repeat_fault("T07", "24012")
        assert r["no_records"] is False


class TestPriorityRequired:
    def test_safety_chain_requires_emergency(self):
        """T05/24005 急停：第 2.1 条要求 EMERGENCY，且现有 NORMAL 工单应升级。"""
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert r["required_priority"] == "EMERGENCY"
        assert "WO-260705" in r["verdict"]
        assert "2.1" in r["clauses"]

    def test_safety_chain_also_cites_remote_reset_ban(self):
        """急停成立 = 第 4.1 条第 1 款成立，条款直接规定，不该等模型想起来再调一次规则。

        实测同一道题两次问：一次调了 remote_reset_ban 援引到第 4.1 条，一次没调就没有。
        而"急停禁止远程强制复位"恰恰是这类问题最不该看模型发挥的那一条。
        """
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert "4.1" in r["clauses"]
        assert "禁止远程强制复位" in r["verdict"]

    def test_non_safety_chain_does_not_claim_reset_ban(self):
        """反过来也要守住：不是安全链类就不能顺手挂上第 4.1 条。"""
        r = check_rule("priority_required", turbine_id="T03", fault_code="24002")
        assert "4.1" not in r["clauses"]
        assert "禁止远程强制复位" not in r["verdict"]

    def test_repeat_fault_requires_high(self):
        r = check_rule("priority_required", turbine_id="T03", fault_code="24002")
        assert r["required_priority"] == "HIGH"
        assert "3.2" in r["clauses"]

    def test_no_alarm_returns_explicit_none(self):
        r = check_rule("priority_required", turbine_id="T01", fault_code="24999")
        assert r["required_priority"] is None
        assert "没有" in r["verdict"]


class TestWorkOrderAssessment:
    def test_inherits_clauses_from_priority_judgement(self):
        """① 优先级整项来自 priority_required，它援引的条款不该因为隔了一层函数就消失。"""
        r = check_rule("work_order_assessment", turbine_id="T05", fault_code="24005")
        assert {"2.1", "4.1"} <= set(r["clauses"])


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

    def test_no_alarm_refuses_to_judge(self):
        """T09 没有 24005。五款里有三款读的是手册，而手册按故障代码分节、与风机无关。

        不挡空集的话，问「T09 的 24005 能不能复位」会拿 24005 那一节判出
        「禁止复位，命中第 4.1(1)(3)(5) 款」——一份带条款号和逐款核对表的
        确定性结论，模型不会去怀疑它的前提。
        """
        r = check_rule("remote_reset_ban", turbine_id="T09", fault_code="24005")
        assert r["reset_banned"] is None
        assert "没有" in r["verdict"] and "记录" in r["verdict"]
        assert r["facts"]["告警条数"] == 0
        assert r["clauses"] == []

    def test_fan_fault_does_not_hit_board_clause(self):
        """T06/24014 是冷却风扇故障，手册要换的是风扇，不是单板。

        上一版把「隔离」当第 4.1(3) 款的命中词，而 24014 那节写的是
        「更换前执行停机、隔离、挂牌上锁和验电」——讲的是换风扇的作业前置。
        """
        r = check_rule("remote_reset_ban", turbine_id="T06", fault_code="24014")
        row = [c for c in r["facts"]["逐款核对"] if "4.1(3)" in c["款"]][0]
        assert row["结论"] == "不成立"
        assert r["reset_banned"] is False

    def test_board_clause_survives_whitespace_in_manual(self):
        """T08/24010 手册原文是「更换 DSP 控制单板」——带空格。

        老词表写死 "DSP控制单板"，从来没匹配上过；这个该命中的故障码一直是靠
        「隔离」的误命中兜着的，两个 bug 抵消成了对的答案。去空白后才是真命中。
        """
        r = check_rule("remote_reset_ban", turbine_id="T08", fault_code="24010")
        row = [c for c in r["facts"]["逐款核对"] if "4.1(3)" in c["款"]][0]
        assert row["结论"] == "成立"
        assert "控制单板" in row["依据"]

    def test_deenergize_clause_needs_same_sentence(self):
        """第 4.1(3) 的另一条腿：断电与检查要在同一句里，整节裸词命中管不住。"""
        r = check_rule("remote_reset_ban", turbine_id="T05", fault_code="24005")
        row = [c for c in r["facts"]["逐款核对"] if "4.1(3)" in c["款"]][0]
        assert row["结论"] == "成立"
        assert "断电" in row["依据"] and "检查" in row["依据"]


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

    def test_cause_item_is_handed_over_not_guessed(self):
        """①「实际故障原因」不做关键词判定。

        WO-260712 的备注写着「确认风扇轴承卡滞」——轴承卡滞就是实际故障原因，
        但一个「原因/因为/由于/根因/查明」都不占。按关键词判会判成缺失，
        而且模型不会质疑它，会替它编出合理化说辞。
        """
        r = close_compliance("WO-260712")
        cause = [c for c in r["facts"]["逐项核对"] if c["项"].startswith("①")][0]
        assert cause["状态"] == "需对照原文认定"
        assert "轴承卡滞" in cause["备注原文"]
        assert not any(x.startswith("①") for x in r["facts"]["未在备注中体现"])

    def test_only_the_designed_bad_order_is_a_hard_violation(self):
        """11 张已完成工单里，硬性违规只有 WO-260708 一张。

        上一版 11 张全判「不合规」——阳性率 100% 的检测器没有判别力，
        三条硬性违规的 WO-260708 和只差两句说明的 WO-260712 压成同一个标签，
        值班的人分不出哪张要紧。
        """
        import sqlite3
        conn = sqlite3.connect("data/海上风电维检.db")
        ids = [row[0] for row in conn.execute(
            "SELECT work_order_id FROM maintenance_records WHERE status='COMPLETED'")]
        hard = [i for i in ids if close_compliance(i)["verdict_kind"] == "硬性违规"]
        assert len(ids) == 11
        assert hard == ["WO-260708"]

    def test_unclosed_order_is_not_judged_for_closure(self):
        """WO-260703 状态 OPEN、备注为空。第 6.1~6.4 条是关闭**前**的要求，
        此时判「七项全缺、不合规」字面没错，却让人以为这张单关错了。"""
        r = close_compliance("WO-260703")
        assert r["verdict_kind"] == "未关闭"
        assert "尚未关闭" in r["verdict"]

    def test_param_item_not_triggered_by_bare_recovery(self):
        """WO-260702 写的是「等待电网电压恢复」，不是参数恢复。

        「恢复」原先在 ④ 的词表里，会把没写参数的单子判成写了——
        与 ① 相反方向的同一种错。
        """
        r = close_compliance("WO-260702")
        assert any(x.startswith("④") for x in r["facts"]["未在备注中体现"])

    def test_matched_terms_returned_for_audit(self):
        """判「已记录」必须说清楚是哪个词命中的，否则同样不可复核。"""
        r = close_compliance("WO-260709")
        row = [c for c in r["facts"]["逐项核对"] if c["项"].startswith("③")][0]
        assert row["状态"] == "已记录"
        assert "更换" in row["命中词"]


class TestSubjectResolution:
    """判定对象的两种握法之间要能互换，别逼模型自己推断。

    工具描述承诺了「给工单编号即可」，而 priority_required 的签名里没有
    work_order_id——这个参数会被过滤掉，只给工单号拿不到判定。模型于是自己
    补了个故障码：两次实测问「T05 的 WO-260705 优先级对不对」，分别编了
    24001 和 24003。24001 那次尤其危险，T05 确实有 24001，规则照常返回了一个
    有效结论（应为 NORMAL），只是回答的不是被问的问题。
    """

    @pytest.mark.parametrize("rule", ["repeat_fault", "priority_required",
                                      "work_order_assessment", "remote_reset_ban"])
    def test_work_order_id_alone_is_enough(self, rule):
        r = check_rule(rule, work_order_id="WO-260705")
        assert r["ok"] is True
        assert r["判定对象来源"] == "由工单 WO-260705 定位到 T05 / 24005"
        assert "未指定" not in r["verdict"]        # 不该退化成通则

    def test_priority_resolved_from_order_is_emergency(self):
        """换算出来的判定要和直接传参一致，不能只是"没报错"。"""
        by_order = check_rule("priority_required", work_order_id="WO-260705")
        by_pair = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        assert by_order["required_priority"] == by_pair["required_priority"] == "EMERGENCY"

    def test_conflicting_fault_code_is_called_out(self):
        """传了工单号又自己带一个对不上的故障码——正是编造的形态。

        不能悄悄挑一个用：挑工单的会掩盖模型在瞎猜，挑传入的会答错问题。
        """
        r = check_rule("priority_required", work_order_id="WO-260705",
                       turbine_id="T05", fault_code="24003")
        assert r["ok"] is False
        assert "24005" in r["error"] and "24003" in r["error"]

    @pytest.mark.parametrize("rule", ["close_compliance", "replace_precondition"])
    def test_turbine_and_code_resolve_to_order(self, rule):
        r = check_rule(rule, turbine_id="T08", fault_code="24010")
        assert r["判定对象来源"] == "由 T08 / 24010 定位到工单 WO-260707"

    def test_unknown_work_order_does_not_fall_back_to_general(self):
        r = check_rule("priority_required", work_order_id="WO-999999")
        assert "不存在" in r["verdict"]
        assert "未指定" not in r["verdict"]

    def test_pair_without_work_order_is_explicit(self):
        """T07/24012 有告警但没工单，换算不出来要说清楚，不能静默走通则。"""
        r = check_rule("close_compliance", turbine_id="T07", fault_code="24012")
        assert "不存在" in r["verdict"]

    def test_general_question_still_gets_general_clauses(self):
        """不传任何 id 的泛问不受影响，仍返回通则条款。"""
        r = check_rule("close_compliance")
        assert r["is_general"] is True
        assert "6.2" in r["clauses"]


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
        # 第 8 项是拿 part_available 判的，那个字段的语义由第 7.1 条定义
        assert "7.1" in r["clauses"]

    def test_switch_item_quotes_the_manual(self):
        """第 5.1 条第 7 项写的是「已按**故障手册**断开相关开关」，条款直接指向手册。

        不读手册，这一项就只剩一句"无法确认"：参考来源里没有手册，
        现场核实清单也说不出要去断哪几个开关。
        """
        r = check_rule("replace_precondition", work_order_id="WO-260707")
        item = next(c for c in r["facts"]["逐项核对"] if c["项"].startswith("7."))
        assert item["结论"] == "现有资料无法确认"      # 是否已断开仍要现场确认
        for switch in ("Q6", "Q13", "Q7", "Q8", "Q41"):
            assert switch in item["依据"]
        assert "fault_manual" in r["sources"]

    def test_switches_taken_only_from_the_disconnect_line(self):
        """手册别处也有 Q 编号（恢复步骤），多列一个开关在电力作业里不是小事。"""
        from tools.rules import _manual_switches
        assert _manual_switches("按步骤断开 Q6、Q13；\n更换后恢复 Q99。") == ["Q6", "Q13"]


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

    def test_manual_section_read_by_rule_is_retrievable(self):
        """chip 上说"用了故障处理手册"，就得点得开是哪一节。"""
        r = check_rule("replace_precondition", work_order_id="WO-260707")
        docs = {s["section_id"] for s in cited_sections(r)}
        assert "24010_SC_主变流器RAM自检失败" in docs

    def test_cited_sections_return_original_text(self):
        """chip 上的条款号要能点开原文，两者必须同源。"""
        r = check_rule("priority_required", turbine_id="T05", fault_code="24005")
        sections = cited_sections(r)
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


class TestGeneralQuestionMode:
    """泛问不绑定具体对象时，规则引擎给通则而不是缺席。

    起因是两次实测漏依据：
      「仓库里有备件是不是就可以直接开工了」—— check_rule 报「缺少 work_order_id」
        直接缺席，判定退回文档检索，而第 7.3 条在 BM25 里排第 12，没进上下文。
      「断电重启之后报警没了，这单子能结吗」—— 同理漏掉第 6.2 条。
    缺的是判定对象，不是条款；规则引擎明明知道自己管哪几条。
    """

    def test_missing_id_returns_general_clauses_not_error(self):
        result = check_rule("replace_precondition")
        assert result["ok"] is True
        assert result["is_general"] is True
        assert "7.3" in result["clauses"], "备件可用不等于允许作业，泛问时必须给出"
        assert "work_order_id" in result["missing_args"]

    def test_close_compliance_general_covers_restart_only(self):
        result = check_rule("close_compliance")
        assert "6.2" in result["clauses"], "「重新上电后故障消失」不得单独作为关闭依据"

    def test_general_answer_carries_clause_texts(self):
        """原文要一起给，否则模型还得再 get_doc_section 取一遍。"""
        result = check_rule("replace_precondition")
        assert "第 7.3 条" in result.get("clause_texts", {})
        assert [s["section_id"] for s in cited_sections(result)]

    def test_general_answer_states_it_judged_nothing(self):
        """最危险的误用是把通则当成对某张工单的判定，措辞必须堵死。"""
        result = check_rule("close_compliance")
        assert "未针对任何具体工单或风机作出判定" in result["verdict"]
        assert result["facts"] == {}

    def test_specific_id_still_runs_real_judgement(self):
        """泛问模式不能把正常判定带跑偏。"""
        result = check_rule("close_compliance", work_order_id="WO-260701")
        assert result.get("is_general") is None
        assert result["facts"]["工单"]["work_order_id"] == "WO-260701"

    def test_unknown_rule_still_errors(self):
        assert check_rule("no_such_rule")["ok"] is False


class TestRuleQueryAudit:
    """规则引擎自己发的 SQL 必须留痕。

    起因：一次链路的答案里写「现有工单 WO-260703 为 NORMAL」，而 span 明细里
    一条查工单表的 SQL 都没有 —— 工单号来自 priority_required 内部查询，不经过
    模型的工具调用。确定性的那条路反而比模型那条路更不透明，而它的结论权重更高。
    """

    def test_queries_recorded_with_row_counts(self):
        result = check_rule("priority_required", turbine_id="T03", fault_code="24002")
        queries = result.get("_audit_queries")
        assert queries, "规则内部查询未留痕"
        assert any("maintenance_records" in q["sql"] for q in queries), "工单号的出处没记下来"
        assert all(isinstance(q["row_count"], int) for q in queries)

    def test_nested_rule_queries_attributed_to_parent(self):
        """priority_required 内部会调 repeat_fault，子规则的查询也要算在父结果上。"""
        result = check_rule("remote_reset_ban", turbine_id="T03", fault_code="24002")
        assert len(result.get("_audit_queries") or []) >= 1

    def test_audit_keys_do_not_reach_the_model(self):
        """留痕和喂模型是两件事：SQL 进 span，不进上下文。"""
        from agent.loop import _for_model
        result = check_rule("priority_required", turbine_id="T03", fault_code="24002")
        assert "_audit_queries" in result
        assert not any(k.startswith("_audit_") for k in _for_model(result))
        assert "verdict" in _for_model(result), "剥离不能误伤正常字段"


class TestRemoteResetChecklist:
    """第 4.1 条五款确定性摊开，不靠模型记得写。"""

    def test_all_five_clauses_rendered(self):
        from agent.checklist import build
        result = check_rule("remote_reset_ban", turbine_id="T03", fault_code="24002")
        cards = build({"rules": [{"rule": "remote_reset_ban", "result": result}]})
        assert cards, "未生成清单卡"
        items = cards[0]["items"]
        assert len(items) == 5, "第 4.1 条是五款，少一款都不行"
        assert all(i["label"].startswith("4.1(") for i in items)

    def test_ban_condition_met_is_blocked_not_ok(self):
        """语义与其他清单相反：这里「成立」是禁止情形成立，是坏消息。"""
        from agent.checklist import build
        result = check_rule("remote_reset_ban", turbine_id="T03", fault_code="24002")
        items = build({"rules": [{"rule": "remote_reset_ban", "result": result}]})[0]["items"]
        by_verdict = {i["verdict"]: i["state"] for i in items}
        assert by_verdict.get("成立") == "blocked"
        assert by_verdict.get("不成立") == "ok"
        assert by_verdict.get("资料无法确认") == "pending"

    def test_no_alarm_record_yields_no_card(self):
        """没有告警记录时规则不出 facts，清单也不该凭空造。"""
        from agent.checklist import build
        result = check_rule("remote_reset_ban", turbine_id="T99", fault_code="24002")
        assert build({"rules": [{"rule": "remote_reset_ban", "result": result}]}) == []
