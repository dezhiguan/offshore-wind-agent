# -*- coding: utf-8 -*-
"""接地校验：回答里出现的数字与条款号，是否真的在证据里出现过。

刻意默认跑 shadow（只标注、不改动回答）。会误伤正确结果的判定层，
必须先影子跑一段时间标定再切 enforce——本机凭空想不出真实的误判形状，
上来就拦会把正确答案也拦掉。

三档：off / shadow（默认）/ enforce

**证据面的口径**（2026-09-18 对抗性测试后放宽）：首版只拿 evidence 比对，34 条
对抗用例标记 9 条，逐条核对几乎全是误报，来源就两类——

  1. 问句里带进来的数字（问「IEC 61400-3 的力矩标准」，61400 被当成模型编的）；
  2. 只出现在检索摘要里的条款号（search_docs 命中了「第 5.1 条 作业前置条件」但
     没取全文，回答引用 5.1 属实，evidence 里却没有这一节）。

两类都不是"模型编的"，而是"证据面画小了"。现在比对面 = 证据 + 执行轨迹摘要 + 问句原文。
这是切 enforce 的前置条件：误报率压不下去，enforce 会把正确答案一起拦掉。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from tools.retriever import catalog

MODE = os.getenv("GROUNDING_MODE", "shadow").lower()

# 只检查「实质性数字」：≥2 位的数、带单位的量、条款号。
# 单个数字（「1. 」「第 2 点」）满屏都是，检查它们只会制造噪音。
_NUM = re.compile(r"(?<![\d.])(\d{2,}(?:\.\d+)?)(?![\d.])")
_CLAUSE = re.compile(r"第\s*(\d+\.\d+)\s*条")
# 这些是提示词和规程本身的固定常量，不算需要证据支撑的事实
_WHITELIST = {"20", "24", "120", "2026"}
# 派生数容忍：差额类数字（「缺口 105 分钟」= 门限 120 − 实际 15）在证据里
# 一定找不到字面，但它不是编造，是从两个证据数算出来的。实测 11 条链路里
# 这类假阳每次都出现，而"会误伤正确结果的判定层必须先标定"——不收干净它，
# 切到 enforce 就会给正确答案挂上"未找到支撑"。
#
# 只认**差**，不认和：本机对 11 条真实链路量过，两位/三位整数空间里
# 差值派生的误放率平均 2.1%、最高 6.8%；一旦把"和"也算进来会升到 13%，
# 那就等于给编造的数字开了一条洗白通道。差额也正是本场景真实出现的形状
# （门限与实际值的缺口、次数差）。
#
# 被豁免的数字不静默丢弃，单列 derived 报出来，事后可审计。
_DERIVE_MAX_TERMS = 40  # 证据数字过多时不做派生：组合爆炸且判据本身会失去意义


def _evidence_blob(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, ensure_ascii=False, default=str)


def _diffs(blob: str) -> dict[str, str]:
    """证据里任意两个数之差 → 它是怎么算出来的。用于放过差额类派生数。"""
    vals = []
    for token in set(_NUM.findall(blob)):
        try:
            vals.append((token, float(token)))
        except ValueError:
            continue
    if len(vals) > _DERIVE_MAX_TERMS:
        return {}
    out: dict[str, str] = {}
    for a_tok, a in vals:
        for b_tok, b in vals:
            gap = a - b
            if gap <= 0:
                continue
            # 回答里写的是整数就按整数比，避免 105 与 105.0 对不上
            key = "%d" % gap if gap == int(gap) else "%g" % gap
            out.setdefault(key, "%s − %s" % (a_tok, b_tok))
    return out


def _support_blob(evidence: dict[str, Any], question: str,
                  trace: list[dict[str, Any]] | None) -> str:
    """可作为支撑的全部文本 = 证据 + 执行轨迹摘要 + 问句原文 + 资料目录。

    轨迹摘要要算进来：search_docs 只返回摘要时，命中章节的标题（含条款号）只在
    摘要里，"没取回全文"不等于"没检索到这一节"。
    问句要算进来：用户自己写进问题的数字不是模型编的。
    目录要算进来：它是注进系统提示词的确定性事实，模型答"手册只收录这 9 个故障码"
    引的就是它——不收进来，每个照目录作答的正确回答都要挨一刀。
    """
    parts = [_evidence_blob(evidence), question or "", catalog()]
    for entry in trace or []:
        parts.append(json.dumps(entry, ensure_ascii=False, default=str))
    return "\n".join(parts)


def check(answer: str, evidence: dict[str, Any], question: str = "",
          trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if MODE == "off" or not answer:
        return {"mode": "off", "flagged": [], "derived": []}

    blob = _support_blob(evidence, question, trace)
    flagged, derived = [], []
    # 派生仍只在**真证据**里算：拿问句里的数字凑差额，等于给编造的数开一条洗白通道
    diffs = _diffs(_evidence_blob(evidence))

    for num in set(_NUM.findall(answer)) - _WHITELIST:
        if num in blob:
            continue
        src = diffs.get(num)
        if src:
            derived.append({"type": "number", "value": num, "from": src,
                            "note": "未在证据中直接出现，但等于证据中 %s 之差" % src})
            continue
        flagged.append({"type": "number", "value": num,
                        "note": "回答中出现的数值未在证据中找到"})
    for clause in set(_CLAUSE.findall(answer)):
        if clause not in blob:
            flagged.append({"type": "clause", "value": clause,
                            "note": "引用的条款未在本次取回的文档或规则结论中出现"})

    return {"mode": MODE, "flagged": flagged, "derived": derived}


def apply(result: dict[str, Any]) -> dict[str, Any]:
    """把校验结果挂到 meta 上。shadow 档只记录，不改回答。"""
    report = check(result.get("answer", ""), result.get("evidence", {}),
                   question=result.get("question", ""), trace=result.get("trace"))
    result.setdefault("meta", {})["grounding"] = report
    # 只有 flagged 参与拦截；derived 是"算出来的"，仅留痕供审计
    if report["mode"] == "enforce" and report["flagged"]:
        result["meta"]["grounding_enforced"] = True
        result["unverifiable"] = list(result.get("unverifiable", [])) + [
            "以下内容未能在本次取回的证据中找到支撑，请人工复核：%s"
            % "、".join(f["value"] for f in report["flagged"])
        ]
    return result
