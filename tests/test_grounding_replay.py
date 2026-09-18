# -*- coding: utf-8 -*-
import agent.grounding as grounding
import agent.replay as replay

EVIDENCE = {
    "tables": [{"sql": "SELECT ...", "rows": [{"occurred_at": "2026-07-18 08:15:00"}]}],
    "rules": [{"rule": "repeat_fault", "result": {"facts": {"次数": 4}, "clauses": ["3.1"]}}],
    "docs": [{"doc": "规程.md", "section_id": "3.1", "text": "连续 24 小时内发生 3 次及以上"}],
}


class TestGrounding:
    def test_supported_numbers_not_flagged(self):
        r = grounding.check("窗口内发生 4 次，见第 3.1 条。", EVIDENCE)
        assert r["flagged"] == []

    def test_unsupported_number_flagged(self):
        r = grounding.check("窗口内发生 4 次，观察时间 999 分钟。", EVIDENCE)
        assert [f["value"] for f in r["flagged"]] == ["999"]

    def test_unsupported_clause_flagged(self):
        r = grounding.check("依据第 9.9 条。", EVIDENCE)
        assert r["flagged"][0]["type"] == "clause"

    def test_whitelisted_constants_not_flagged(self):
        """20V、24 小时、120 分钟是规程常量，不该被当成缺证据。"""
        r = grounding.check("母线电压约 20 V，连续 24 小时，观察 120 分钟。", EVIDENCE)
        assert r["flagged"] == []

    def test_shadow_mode_does_not_modify_answer(self, monkeypatch):
        """会误伤的判定层必须先影子跑：只标注，不改回答。"""
        monkeypatch.setattr(grounding, "MODE", "shadow")
        result = {"answer": "有 999 分钟", "evidence": EVIDENCE, "unverifiable": []}
        out = grounding.apply(result)
        assert out["unverifiable"] == []
        assert out["meta"]["grounding"]["flagged"]

    def test_enforce_mode_appends_to_unverifiable(self, monkeypatch):
        monkeypatch.setattr(grounding, "MODE", "enforce")
        result = {"answer": "有 999 分钟", "evidence": EVIDENCE, "unverifiable": []}
        out = grounding.apply(result)
        assert out["unverifiable"] and "999" in out["unverifiable"][0]


class TestSupportSurface:
    """比对面 = 证据 + 轨迹摘要 + 问句。

    首版只拿 evidence 比，34 条对抗用例标记 8 条、几乎全是误报，来源就这两类。
    压不下误报率，enforce 档就永远不能开——那这层判定等于装饰。
    """

    def test_numbers_from_the_question_are_not_the_models_invention(self):
        """问「IEC 61400-3 的力矩标准」，61400 是用户写的，不是模型编的。"""
        answer = "按 IEC 61400-3 无法回答，随题资料未收录该标准。"
        assert grounding.check(answer, EVIDENCE)["flagged"], "不带问句时会误报"
        r = grounding.check(answer, EVIDENCE, question="按 IEC 61400-3 的预紧力矩是多少？")
        assert r["flagged"] == []

    def test_number_seen_only_in_the_trace_summary_counts_as_grounded(self):
        """工具执行结果只进了轨迹摘要时，据它作答不算编造。

        取行数而不是条款号：条款号如今由目录兜底（见
        test_catalog_backed_enumeration_is_grounded），验不出轨迹这一面。
        """
        answer = "该条件下命中 37 行。"
        assert grounding.check(answer, EVIDENCE)["flagged"], "不带轨迹时会误报"
        trace = [{"tool": "query_db", "summary": "命中 37 行"}]
        assert grounding.check(answer, EVIDENCE, trace=trace)["flagged"] == []

    def test_fabricated_number_still_flagged_with_full_surface(self):
        """放宽比对面不等于放水：问句和轨迹里都没有的数照样标出来。

        取 37 而不是 16：后者恰好等于证据里 24 − 8，会被既有的差额派生豁免掉，
        那验的就不是本条改动了。
        """
        r = grounding.check("工单共 37 张。", EVIDENCE,
                            question="全场有多少张工单？",
                            trace=[{"tool": "query_db", "summary": "命中 4 行"}])
        assert [f["value"] for f in r["flagged"]] == ["37"]

    def test_question_numbers_do_not_feed_the_derived_pass(self):
        """派生只在真证据里算：否则问句带两个数就能洗白任意第三个数。"""
        r = grounding.check("缺口 900 分钟。", EVIDENCE, question="1000 和 100 差多少？")
        assert [f["value"] for f in r["flagged"]] == ["900"]
        assert r["derived"] == []

    def test_catalog_backed_enumeration_is_grounded(self):
        """目录是注进提示词的确定性事实，照它作答不该被判"没有支撑"。

        不收进比对面的话，「手册只收录这 9 个故障码」这种**正确**回答会一次挨 9 刀。
        """
        answer = "手册只收录 24001、24002、24005、24006、24010、24011、24012、24013、24014。"
        assert grounding.check(answer, {})["flagged"] == []

    def test_apply_passes_question_and_trace_through(self, monkeypatch):
        monkeypatch.setattr(grounding, "MODE", "shadow")
        out = grounding.apply({"answer": "按 IEC 61400-3 无法回答。", "evidence": EVIDENCE,
                               "question": "IEC 61400-3 怎么规定的？", "trace": []})
        assert out["meta"]["grounding"]["flagged"] == []


class TestReplay:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("REPLAY_MODE", raising=False)
        assert replay.enabled() is False

    def test_matches_stored_case(self):
        hit = replay.find("请查询 T06，风机型号是什么？最新一条告警记录对应的运行状态是什么？")
        assert hit is not None
        assert hit["meta"]["replay"]["matched_case"] == "Q1"

    def test_unrelated_question_returns_none(self):
        """相似度不够就返回 None，不能硬凑一条看起来像的答案。"""
        assert replay.find("今天广州天气怎么样") is None

    def test_replay_result_is_labeled(self):
        hit = replay.find("T04 在 2026-07-10 至 2026-07-15 期间有哪些告警记录？")
        assert hit["meta"]["replay"]["similarity"] > 0


class TestReplayIdentifierGate:
    def test_out_of_range_turbine_rejected(self):
        """库外风机必须拒绝。

        只靠字符重合度会出事：「T15 的型号是什么？」与「T01 的风机型号是什么？」
        重合度高达 0.86，会拿 T01 的答案去答 T15。标识符硬门负责挡住它。
        注意不要用 T20 做这个断言——探针用例 P1 问的就是 T20，它在回放素材里，
        理应命中。
        """
        assert replay.find("T15 的型号是什么？") is None

    def test_unknown_fault_code_rejected(self):
        assert replay.find("T03 的 29999 是否属于重复故障？") is None

    def test_unknown_work_order_rejected(self):
        assert replay.find("WO-999999 的关闭是否合规？") is None

    def test_same_entity_near_miss_is_surfaced_not_hidden(self):
        """同一实体上的近似命中是回放固有的：问 T20 的型号，可能命中「T20 的最新告警」。

        标识符硬门只能挡跨实体串答，挡不住同实体换问法。缓解方式是把匹配到的用例
        和它的原问题一并返回，界面据此显示横幅，由使用者自行判断——而不是假装精确命中。
        """
        hit = replay.find("T20 的型号是什么？")
        if hit is not None:
            assert hit["meta"]["replay"]["matched_question"]
            assert hit["meta"]["replay"]["matched_case"]

    def test_paraphrase_with_same_identifier_still_matches(self):
        """改写、缩写、换措辞都应命中，只要标识符一致。"""
        hit = replay.find("T03 的 24002 是否属于重复故障？")
        assert hit and hit["meta"]["replay"]["matched_case"] == "Q5"

    def test_question_without_identifier_still_works(self):
        hit = replay.find("规程规定哪些情形禁止远程强制复位？")
        assert hit and hit["meta"]["replay"]["matched_case"] == "Q4"


class TestReplaySkipsBrokenArtifacts:
    def test_failed_run_not_used_as_replay_material(self, tmp_path, monkeypatch):
        """跑测遇网络超时会留下 {"error": ...} 产物；回放必须跳过它。

        照单全收的话，现场断网演示时会返回一个空答案——比报错更糟。
        """
        import json
        monkeypatch.setattr(replay, "OUT_DIR", tmp_path)
        (tmp_path / "BAD.json").write_text(json.dumps({
            "case": {"id": "BAD", "question": "T03 的 24002 是否属于重复故障？"},
            "result": {"error": "The read operation timed out"},
        }, ensure_ascii=False), encoding="utf-8")
        assert replay.find("T03 的 24002 是否属于重复故障？") is None
