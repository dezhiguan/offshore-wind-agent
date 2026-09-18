# -*- coding: utf-8 -*-
"""离线回放。

现场演示是唯一的交付场景，而演示环境不保证有外网。这里把 eval/run.py 跑出的
完整链路产物（SQL、命中章节、工具轨迹、答案）当作 fixture，断网时仍能走完整
界面流程。回放结果会带 replay 标记，界面上明示——这是演示预案，不是遮掩。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from agent import evalview

# 回放素材就是跑测产物，位置跟着 evalview 走（EVAL_OUT_DIR 可覆盖）。
# 这里原先自己写了一份 ROOT/"eval"/"out"——第三份定义。跑测产物落到别处时，
# 回放会静默地读旧目录：回放模式最怕的就是"匹配到的是上一轮的答案"而界面照常显示。
OUT_DIR = evalview.OUT_DIR
MIN_SIMILARITY = 0.6


def enabled() -> bool:
    return os.getenv("REPLAY_MODE", "false").lower() in {"1", "true", "yes"}


def _norm(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z]+|[一-龥]", text or ""))


# 风机编号 / 故障代码 / 工单编号——问句里出现的标识符必须在被匹配的用例里也出现。
# 只看字符重合度会出事：「T20 的型号是什么？」与 S1「T01 的风机型号是什么？」
# 重合度高达 0.86，回放会拿 T01 的答案去答 T20——而 T20 根本不在库里。
# 判别信息集中在这几个 token 上，必须单独设硬门，不能让它们淹没在通用字里。
_IDENT = re.compile(r"T\d{2}|WO-\w+|(?<!\d)\d{5}(?!\d)")


def _idents(text: str) -> set[str]:
    return set(_IDENT.findall(text or ""))


def _load() -> list[dict[str, Any]]:
    if not OUT_DIR.exists():
        return []
    items = []
    for path in sorted(OUT_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        # 跑测时若遇网络超时等异常，产物里存的是 {"error": ...}。
        # 这类残缺产物不能当回放素材——现场断网时会返回一个空答案。
        if not (blob.get("result") or {}).get("answer"):
            continue
        items.append(blob)
    return items


def find(question: str) -> dict[str, Any] | None:
    """按字符重合度取最接近的一条。相似度太低就返回 None，不硬凑。"""
    items = _load()
    if not items:
        return None
    q = _norm(question)
    q_ids = _idents(question)
    if not q:
        return None

    best, best_score = None, 0.0
    for item in items:
        stored_q = item.get("case", {}).get("question", "")
        stored = _norm(stored_q)
        if not stored:
            continue
        # 硬门：问句里的标识符必须被覆盖，否则直接淘汰
        if q_ids - _idents(stored_q):
            continue
        # 覆盖度（问句有多少被命中用例覆盖），比 Jaccard 更耐改写与缩写
        score = len(q & stored) / len(q)
        if score > best_score:
            best, best_score = item, score

    if best is None or best_score < MIN_SIMILARITY:
        return None

    result = dict(best["result"])
    result.setdefault("meta", {})
    result["meta"]["replay"] = {
        "matched_case": best["case"]["id"],
        "matched_question": best["case"]["question"],
        "similarity": round(best_score, 3),
    }
    return result
