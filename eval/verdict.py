# -*- coding: utf-8 -*-
"""用例判定。

跑测器与 /eval 页面共用同一份判定逻辑：两处各写一套必然随时间漂移，
而「同一条用例在两个地方给出不同结论」是最难排查的一类缺陷。
"""
from __future__ import annotations

import re
from typing import Any

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


def check(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    text = said(result)
    problems: list[str] = []
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
