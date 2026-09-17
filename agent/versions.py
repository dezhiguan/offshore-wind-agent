# -*- coding: utf-8 -*-
"""Agent 版本台账与运行指纹。

「版本」在这里不等于 git 提交号。同一份代码，把 `LLM_MODEL` 换成另一个模型、
把 `GROUNDING_MODE` 从 shadow 拧到 enforce、把步数上限从 6 改成 3，跑出来就是
另一个 agent——回答会变、成本会变、回归成绩也会变。所以版本按**影响行为的要素**
定义，共七项：提示词、工具集、模型、步数上限、思考模式、接地校验档位、上下文预算阈值。
这七项算出一个指纹。

台账（LEDGER）是人工登记的声明，指纹是从**当前进程实际生效的值**现算的。
两者对不上就是漂移——改了提示词没登记、或者被环境变量换掉了模型——页面上直接标出来，
而不是靠记忆。台账本身不做持久化，也不从这里改运行时配置：
配置的唯一真相是环境变量与代码，后台只负责如实呈现与核对。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

# 当前登记在跑的版本。改动了八要素中的任何一项，就在 LEDGER 里加一条并改这里。
CURRENT = "V1.1"

# 指纹要素 → 页面上的中文标签
FIELDS = {
    "model": "模型",
    "max_steps": "步数上限",
    "thinking": "思考模式",
    "grounding": "接地校验",
    "prompt_sha": "提示词",
    "tools_sha": "工具集",
    "row_ratio": "数据行阈值",
    "doc_ratio": "文档字符阈值",
}

LEDGER: list[dict[str, Any]] = [
    {
        "version": "V1.1",
        "date": "2026-09-17",
        "summary": "失败链路可观测",
        "changes": [
            "模型调用失败改为留下 ERROR span，并把已发生的 span 随异常带出",
            "失败链路进留存，后台新增「模型调用成功率」，分母含失败",
            "步数用尽的收口调用补上 span：此前这次调用的 token 与成本漏计",
        ],
        # 本次只动可观测性，八要素一项没变，因此与 V1.0 同指纹
        "config": {
            "model": "qwen3.8-flash",
            "max_steps": 6,
            "thinking": False,
            "grounding": "shadow",
            "prompt_sha": "6125e49c",
            "tools_sha": "e4dda04d",
            "row_ratio": 0.5,
            "doc_ratio": 0.7,
        },
    },
    {
        "version": "V1.0",
        "date": "2026-09-16",
        "summary": "设计说明 V1.0 定版",
        "changes": [
            "四工具 Function Calling 循环，步数上限 6",
            "未取证拦截：一次工具都没调就想作答，打回；二次仍不调则拒答",
            "上下文预算护栏：数据行 50% / 文档字符 70%，运行时拦截而非事后统计",
            "接地校验默认 shadow：只标注不改写，避免误杀正确回答",
        ],
        "config": {
            "model": "qwen3.8-flash",
            "max_steps": 6,
            "thinking": False,
            "grounding": "shadow",
            "prompt_sha": "6125e49c",
            "tools_sha": "e4dda04d",
            "row_ratio": 0.5,
            "doc_ratio": 0.7,
        },
    },
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def fingerprint(config: dict[str, Any]) -> str:
    """八要素的指纹。要素值变一个字符，指纹就变。"""
    return _sha(json.dumps(config, ensure_ascii=False, sort_keys=True))[:6]


def runtime() -> dict[str, Any]:
    """当前进程**实际生效**的配置。

    刻意读模块属性而不是重新读一遍环境变量：模块常量是进程启动时定死的，
    那才是这一轮问答真正会用到的值。模型除外——它在每次调用时现读 env，
    这里跟着现读，口径才一致。
    """
    from agent import grounding, loop
    from agent.prompts import SYSTEM_PROMPT
    from agent.tools_spec import TOOLS
    from tools import budget

    return {
        "model": os.getenv("LLM_MODEL", "qwen3.8-flash"),
        "max_steps": loop.MAX_STEPS,
        "thinking": loop.ENABLE_THINKING,
        "grounding": grounding.MODE,
        "prompt_sha": _sha(SYSTEM_PROMPT),
        "tools_sha": _sha(json.dumps(TOOLS, ensure_ascii=False, sort_keys=True)),
        "row_ratio": budget.MAX_ROW_RATIO,
        "doc_ratio": budget.MAX_DOC_RATIO,
    }


def _entry(version: str) -> dict[str, Any] | None:
    return next((e for e in LEDGER if e["version"] == version), None)


def overview() -> dict[str, Any]:
    """台账 + 运行指纹 + 两者的差异。"""
    from agent.tools_spec import TOOL_REGISTRY

    live = runtime()
    declared = _entry(CURRENT)
    drift = []
    if declared:
        for key, label in FIELDS.items():
            want, got = declared["config"].get(key), live.get(key)
            if want != got:
                drift.append({"key": key, "label": label, "declared": want, "actual": got})

    return {
        "current": CURRENT,
        "fingerprint": fingerprint(live),
        "declared_fingerprint": fingerprint(declared["config"]) if declared else None,
        "config": live,
        "fields": FIELDS,
        "tools": list(TOOL_REGISTRY),
        "drift": drift,
        "ledger": [{
            **e,
            "running": e["version"] == CURRENT,
            "fingerprint": fingerprint(e["config"]),
        } for e in LEDGER],
    }
