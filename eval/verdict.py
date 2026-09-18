# -*- coding: utf-8 -*-
"""用例判定。

跑测器与 /eval 页面共用同一份判定逻辑：两处各写一套必然随时间漂移，
而「同一条用例在两个地方给出不同结论」是最难排查的一类缺陷。
"""
from __future__ import annotations

import re
from typing import Any

from agent.composer import SOURCE_LABELS

SUITES = {
    "cases": "回归",
    "probes": "探针",
    "pressure": "施压",
    "boundary": "边界",
}


def said(result: dict[str, Any]) -> str:
    """只取模型自己产出的文字：结论 + 现有资料无法确认。

    断言不能打在整个响应上：evidence 里是手册和规程的原文，天然含大量数字和术语
    （例如规程第 3.1 条原文就有「3 次及以上」），拿它做否定断言必然误判。

    **basis 同样不能算**（2026-09-18 修）。它看着像模型写的，实际是 composer 用
    `render_basis(evidence)` 从真实工具调用渲染出来的——是系统摆出的证据，不是模型的话。
    两个方向都会出事，而且两种都真出过：

    - 否定断言被证据带崩：D1 要求「不得把别人的工单报成 T06 的」，而模型恰恰是做了
      一次全厂 24013 的交叉核对才敢下结论，那次查询的命中行被渲进 basis，
      `WO-260710、WO-260709、WO-260715` 就此进入判定文本——**模型答对了反而判失败**，
      而且是因为它多做了一步正确的核对。
    - 肯定断言被证据兜住：B5 的答案写的是「不具备马上修复的条件」，断言的动词表里没有
      「不具备」，本该暴露的断言过窄问题被规则引擎渲进 basis 的「不能认定已经具备立即更换」
      顶住了——**用例是靠证据蒙对的，不是靠答案对**。

    判定的对象是模型的结论。证据由代码渲染，拿它判模型等于自己给自己打分。
    """
    return "\n".join([
        result.get("answer", "") or "",
        "\n".join(result.get("unverifiable", []) or []),
    ])


def _sources(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """用例声明的数据源必须真出现在结果里。

    这个字段原先只是注释：写了 safety_regulation，实际漏报了也全绿。
    于是「结论引了规程第 2.1 条、数据源不列规程」这种漂移连着四条用例都没被拦下。
    只判"声明了却没出现"——多出来的不判：模型多查一节手册不算错，
    把它算成失败只会逼着用例去追模型的每一次自由发挥。
    """
    got = result.get("sources") or []
    missing = []
    for key in case.get("sources") or []:
        label = SOURCE_LABELS.get(key, key)
        # 规程 chip 带条款号后缀（「…规程（第 2.1、2.4 条）」），只能前缀匹配
        if not any(str(s).startswith(label) for s in got):
            missing.append("数据源未列出 %s" % label)
    return missing


def check(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    text = said(result)
    problems: list[str] = _sources(case, result)
    spec = case.get("assert", {}) or {}
    for token in spec.get("must_include", []):
        if token not in text:
            problems.append("缺少 %r" % token)
    for token in spec.get("must_not_include", []):
        if token in text:
            problems.append("不应出现 %r" % token)
    for pattern in spec.get("must_match", []):
        if not re.search(pattern, text):
            problems.append("未匹配 /%s/" % pattern)
    return not problems, problems
