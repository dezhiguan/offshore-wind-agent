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
}


def said(result: dict[str, Any]) -> str:
    """只取模型自己产出的文字。

    断言不能打在整个响应上：evidence 里是手册和规程的原文，天然含大量数字和术语
    （例如规程第 3.1 条原文就有「3 次及以上」），拿它做否定断言必然误判。
    """
    return "\n".join([
        result.get("answer", "") or "",
        "\n".join(result.get("basis", []) or []),
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
