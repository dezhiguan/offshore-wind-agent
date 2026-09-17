# -*- coding: utf-8 -*-
"""接地校验：回答里出现的数字与条款号，是否真的在证据里出现过。

刻意默认跑 shadow（只标注、不改动回答）。会误伤正确结果的判定层，
必须先影子跑一段时间标定再切 enforce——本机凭空想不出真实的误判形状，
上来就拦会把正确答案也拦掉。

三档：off / shadow（默认）/ enforce
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

MODE = os.getenv("GROUNDING_MODE", "shadow").lower()

# 只检查「实质性数字」：≥2 位的数、带单位的量、条款号。
# 单个数字（「1. 」「第 2 点」）满屏都是，检查它们只会制造噪音。
_NUM = re.compile(r"(?<![\d.])(\d{2,}(?:\.\d+)?)(?![\d.])")
_CLAUSE = re.compile(r"第\s*(\d+\.\d+)\s*条")
# 这些是提示词和规程本身的固定常量，不算需要证据支撑的事实
_WHITELIST = {"20", "24", "120", "2026"}


def _evidence_blob(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, ensure_ascii=False, default=str)


def check(answer: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if MODE == "off" or not answer:
        return {"mode": "off", "flagged": []}

    blob = _evidence_blob(evidence)
    flagged = []

    for num in set(_NUM.findall(answer)) - _WHITELIST:
        if num not in blob:
            flagged.append({"type": "number", "value": num,
                            "note": "回答中出现的数值未在证据中找到"})
    for clause in set(_CLAUSE.findall(answer)):
        if clause not in blob:
            flagged.append({"type": "clause", "value": clause,
                            "note": "引用的条款未在本次取回的文档或规则结论中出现"})

    return {"mode": MODE, "flagged": flagged}


def apply(result: dict[str, Any]) -> dict[str, Any]:
    """把校验结果挂到 meta 上。shadow 档只记录，不改回答。"""
    report = check(result.get("answer", ""), result.get("evidence", {}))
    result.setdefault("meta", {})["grounding"] = report
    if report["mode"] == "enforce" and report["flagged"]:
        result["meta"]["grounding_enforced"] = True
        result["unverifiable"] = list(result.get("unverifiable", [])) + [
            "以下内容未能在本次取回的证据中找到支撑，请人工复核：%s"
            % "、".join(f["value"] for f in report["flagged"])
        ]
    return result
