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

# 元 / 1K token
DEFAULT_RATES = {
    "qwen3.8-flash": {"input": 0.0008, "output": 0.0027, "cached_input": 0.0001},
    "qwen3.8-max":   {"input": 0.0024, "output": 0.0096, "cached_input": 0.0003},
    "qwen-plus":     {"input": 0.0008, "output": 0.0020, "cached_input": 0.0001},
}
FALLBACK = {"input": 0.0008, "output": 0.0027, "cached_input": 0.0001}


def rates(model: str) -> dict[str, float]:
    override = os.getenv("LLM_PRICE_INPUT_PER_1K"), os.getenv("LLM_PRICE_OUTPUT_PER_1K")
    base = dict(DEFAULT_RATES.get(model, FALLBACK))
    if override[0]:
        base["input"] = float(override[0])
    if override[1]:
        base["output"] = float(override[1])
    return base


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
