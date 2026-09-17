# -*- coding: utf-8 -*-
"""模型计价。

单价必须来自官方价目并注明核对日期——写死一个记忆里的数字，
会让整张成本表在毫无察觉的情况下全错。这里的取值与 askdb 生产配置一致
（2026-09-15 核对）。换模型或官方调价时改这里，或用环境变量覆盖。

缓存命中的输入单独计价：qwen 系列缓存输入约为标准输入的 1/8，
而实测链路中缓存命中占比很高，按标准价算会明显高估。
"""
from __future__ import annotations

import os
from typing import Any

# 价目核对日期。改单价必须同时改它，否则界面上"按官方单价"会一直停在旧日期。
PRICES_CHECKED_AT = "2026-09-15"

# 元 / 1K token
DEFAULT_RATES = {
    "qwen3.8-flash": {"input": 0.0008, "output": 0.0027, "cached_input": 0.0001},
    "qwen3.8-max":   {"input": 0.0024, "output": 0.0096, "cached_input": 0.0003},
    "qwen-plus":     {"input": 0.0008, "output": 0.0020, "cached_input": 0.0001},
}
FALLBACK = {"input": 0.0008, "output": 0.0027, "cached_input": 0.0001}


# 缓存输入相对标准输入的折扣。qwen 系列官方明码 0.1 / 0.8 = 1/8。
# 只覆盖了输入价却没覆盖缓存价时，按这个比例推导 —— 保留另一个模型的**绝对**
# 缓存价，会算出一张两个模型混起来的账，而缓存命中常年占六成以上，误差全落在总账上。
CACHED_INPUT_RATIO = 1 / 8


def rates(model: str) -> dict[str, float]:
    base = dict(DEFAULT_RATES.get(model, FALLBACK))
    env_in = os.getenv("LLM_PRICE_INPUT_PER_1K")
    env_out = os.getenv("LLM_PRICE_OUTPUT_PER_1K")
    env_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1K")
    if env_in:
        base["input"] = float(env_in)
        base["cached_input"] = base["input"] * CACHED_INPUT_RATIO
    if env_out:
        base["output"] = float(env_out)
    if env_cached:
        base["cached_input"] = float(env_cached)
    return base


def basis(model: str) -> dict[str, Any]:
    """这次的成本是按什么价算的 —— 界面据此决定写"官方单价"还是"估算"。

    价目表外的模型会落到 FALLBACK（flash 的价），数字照样算得出来，
    但那是估算不是账单。不把这件事带到界面上，"按官方单价"就成了一句
    无论如何都会显示的话。
    """
    known = is_known(model)
    overridden = any(os.getenv(k) for k in
                     ("LLM_PRICE_INPUT_PER_1K", "LLM_PRICE_OUTPUT_PER_1K",
                      "LLM_PRICE_CACHED_INPUT_PER_1K"))
    r = rates(model)
    return {
        "model": model,
        "known": known and not overridden,
        "overridden": overridden,
        "checked_at": PRICES_CHECKED_AT if known and not overridden else None,
        "input_per_1k": r["input"],
        "output_per_1k": r["output"],
        "cached_input_per_1k": r["cached_input"],
    }


def cost(model: str, prompt_tokens: int, completion_tokens: int,
         cached_tokens: int = 0) -> float:
    r = rates(model)
    fresh = max(prompt_tokens - cached_tokens, 0)
    return (fresh / 1000 * r["input"]
            + cached_tokens / 1000 * r["cached_input"]
            + completion_tokens / 1000 * r["output"])


def is_known(model: str) -> bool:
    """单价是否来自已核对的价目表。未知模型的成本只是估算，界面应标注。"""
    return model in DEFAULT_RATES
