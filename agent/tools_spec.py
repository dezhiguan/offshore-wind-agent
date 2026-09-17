# -*- coding: utf-8 -*-
"""工具声明与执行注册表。

工具描述同时写「用于」和「不要用于」——只写功能，模型选错率明显更高。

六条规则收在 check_rule 一个工具里用 enum 区分，而不是拆成六个工具：
工具数一过五六个，模型选错工具的概率上升明显；收成一个之后，模型的决策
从「在九个工具里挑」降级为「挑对一个工具 + 挑对一个枚举值」，稳得多，
而且以后加规则不用改工具清单。
"""
from __future__ import annotations

from typing import Any, Callable

from tools.db import query_db
from tools.retriever import get_doc_section, search_docs
from tools.rules import check_rule

# 六条规则收在一个工具里，靠 enum 区分。枚举值的说明直接进 description，
# 模型只需要「挑对一个工具 + 挑对一个枚举值」，比在九个工具里挑稳得多。
# 「需 A，或只给 B」不是废话：前一版只写「需 turbine_id + fault_code」，
# 而工具描述又说「给工单编号即可」，两处对不上。模型手里只有工单号时
# 拿不到判定，就自己补了个故障码——实测两次分别编了 24001 和 24003。
# 换算由 check_rule 统一做，这里把两种握法都写明，别再让它猜。
_RULE_HINTS = {
    "repeat_fault": "24 小时滑动窗口重复故障判定（第 3.1 条），需 turbine_id + fault_code，"
                    "或只给 work_order_id 由规则换算；可选窗口",
    "priority_required": "该故障应有的工单优先级及是否需要升级（第 2.1~2.4 条），"
                         "需 turbine_id + fault_code，或只给 work_order_id 由规则换算",
    "work_order_assessment": "工单安排五要素核对：优先级/状态/处理记录/观察时间/备件，"
                             "需 turbine_id + fault_code，或只给 work_order_id 由规则换算",
    "remote_reset_ban": "是否禁止远程强制复位，逐款核对第 4.1 条，需 turbine_id + fault_code，"
                        "或只给 work_order_id 由规则换算",
    "close_compliance": "已完成工单的关闭是否合规（第 6.1~6.5 条），需 work_order_id，"
                        "或只给 turbine_id + fault_code 由规则换算",
    "replace_precondition": "更换单板/模块的九项前置条件核对（第 5.1~5.2 条），需 work_order_id，"
                            "或只给 turbine_id + fault_code 由规则换算",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "对只读 SQLite 执行单条 SELECT 查询。"
                "用于：查风机型号、运行状态、告警发生时间与次数、告警严重程度与状态、"
                "工单编号、优先级、状态、处理备注、观察时长、备件及其可用性。"
                "不要用于：查故障原理、排查步骤、规程条款——那些在文档里，请用 search_docs。"
                "查询结果为空是正常结果，不是错误，说明数据库里确实没有这条记录。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "单条 SELECT 语句。时间按闭区间比较；关联工单与告警必须同时用 turbine_id 和 fault_code。",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "在故障处理手册和安全管理规程中检索相关章节，返回章节标题与摘要（不返回全文）。"
                "用于：按故障代码查处理办法、按关键词查规程要求。"
                "确认哪一节相关后，再用 get_doc_section 取回该节全文。"
                "检索走关键词匹配，请用资料里的术语而不是口语提问用词（例如用「告警」不是「报警」、"
                "「转矩」不是「扭矩」、「工单」不是「单子」）。"
                "结果里的 recall_check.status 若为 low_confidence，说明本次很可能没捞到该捞的那一节，"
                "不要直接拿这批结果作答：按 recall_check.unknown_terms 换成资料用词重试，"
                "或改用故障代码、条款号定位。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词。可直接用裸故障码如 24002，或条款号如 第 4.1 条。"},
                    "doc": {
                        "type": "string",
                        "enum": ["fault_manual", "safety_regulation"],
                        "description": "限定文档：fault_manual=故障处理手册，safety_regulation=检修作业与安全管理规程。不确定就不要传。",
                    },
                    "top_k": {"type": "integer", "description": "返回条数，默认 4。"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_doc_section",
            "description": (
                "按章节标识取回某一节的完整原文，用于引用。"
                "section_id：故障手册用裸故障码（如 24002）或完整标题；规程用条款号（如 4.1）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {
                        "type": "string",
                        "enum": ["fault_manual", "safety_regulation"],
                        "description": "文档标识。",
                    },
                    "section_id": {"type": "string", "description": "章节标识，如 24002 或 4.1。"},
                },
                "required": ["doc", "section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_rule",
            "description": (
                "按《检修作业与安全管理规程》做确定性判定，返回结论、支撑事实和引用条款。"
                "凡涉及次数统计、时长阈值、优先级应为什么、能否远程复位、关单是否合规、"
                "是否具备更换条件——一律调用本工具，不要自己心算，也不要自己比较阈值。"
                "本工具会自己查库取数：给风机编号 + 故障代码，或只给工单编号，都能判——"
                "两种握法之间由规则自己换算。**不要为了凑参数自己推断故障代码或风机编号**，"
                "手里有哪个就传哪个；推断出来的码即使在库里存在，判出的也是另一个问题的答案。"
                "返回结果已在 clause_texts 中内联了引用到的规程条款原文，"
                "可直接据此引用，**不需要**再调用 get_doc_section 取回这些条款。"
                "用户问的是通则、不针对某台风机或某张工单时（如「备件有货就能开工吗」"
                "「断电重启后告警没了能关单吗」），照样调用本工具但不传 id："
                "会返回该规则覆盖的全部条款与原文（结果带 is_general=true），"
                "比自己用 search_docs 找条款更全、也不会漏。"
                "反过来，问题指向具体对象时必须带上 id，通则不能替代对具体工单的判定。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule": {
                        "type": "string",
                        "enum": list(_RULE_HINTS),
                        "description": "；".join("%s=%s" % kv for kv in _RULE_HINTS.items()),
                    },
                    "turbine_id": {"type": "string", "description": "风机编号，如 T03。"},
                    "fault_code": {"type": "string", "description": "裸故障代码，如 24002。"},
                    "work_order_id": {"type": "string", "description": "工单编号，如 WO-260708。"},
                    "window_start": {"type": "string", "description": "可选。考察窗口起点，'YYYY-MM-DD HH:MM:SS'，闭区间。"},
                    "window_end": {"type": "string", "description": "可选。考察窗口终点，'YYYY-MM-DD HH:MM:SS'，闭区间。"},
                },
                "required": ["rule"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "query_db": query_db,
    "search_docs": search_docs,
    "get_doc_section": get_doc_section,
    "check_rule": check_rule,
}
