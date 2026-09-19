# -*- coding: utf-8 -*-
"""Function Calling 循环。

没有引入 Agent 框架：本题只有三到四个工具、最多六步、单轮问答、不需要 checkpoint，
框架带来的调度抽象和依赖成本高于收益，且会削弱执行轨迹的可控性——而轨迹展示是
题面的明确要求（4.4 节「简要查询过程」）。详见 设计说明。

一个针对本题的关键处理：题面红线要求「必须以 rag 或某种策略检索，不得把整个
数据库或文档放入上下文」。模型有时会觉得问题简单而一个工具都不调、直接凭空作答，
那就直接踩线了。

最初用 tool_choice="required" 强制首步调用工具，但 qwen3 系列在 thinking 模式下
不支持该参数（400 InvalidParameter）。改为代码侧强制：一次工具都没调用就想作答时，
打回要求先检索；仍不调用则直接拒答，而不是放行一个没有依据的回答。这样不依赖任何
厂商的参数支持，换模型也不会失效。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from openai import OpenAI

from agent.prompts import OUT_OF_SCOPE_MARK, SYSTEM_PROMPT
from agent.tools_spec import TOOLS, TOOL_REGISTRY
from agent.tracing import DEGRADED, ERROR, OK, TOOL_KIND, Trace
from tools.budget import ContextBudget, measure
from tools.pricing import cost as price_of
from tools.rules import cited_sections

# span 明细里「详情」展开的单侧上限。取回的文档节可达数千字，
# 50 条留存乘上去会把内存吃掉，超出就截断并标出来，不假装是全文。
DETAIL_CAP = 4000

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
MAX_UNGROUNDED_RETRIES = 1
# 提示词强制的第一个二级标题，用作"最终答案已开始"的分界
ANSWER_MARKER = "## 结论"
# 正文被厂商的输出长度上限砍断时，span 摘要上的前缀。
# 起因是 2026-09-18 探针用例 P8 的一次实测：模型把要点写在「## 结论」之前的铺垫里，
# 到结论段只写出半句「T03 心跳丢失确实已达到重复故障标准——2026-07-18」就断了。
# 解析照常成功（标题在）、format_parsed 为 true、「产出可用回答」记满分 —— 一段 22 字
# 的残句被当成完整答案报上去，只有断言碰巧打在被砍掉的那半边才暴露。停止原因是
# 厂商给的机器事实，不是措辞判断，不存在误伤，因此直接标注，不走影子档。
_TRUNCATED_MARK = "⚠ 输出被长度上限截断："
# qwen3 系列默认开启 thinking，单次调用可达 30 秒，现场演示不可接受。
# 关闭后靠工具与规则引擎保证正确性，而不是靠模型的内部推理。
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in {"1", "true", "yes"}
# 流式返回下连接保持时间更长；实测 60 秒会偶发读超时，演示中途失败比慢更糟
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

# 采样参数。此前一次都没传，走的是厂商默认（qwen 系默认带随机性），于是同一条用例
# 跑两次是两份不同的答案：2026-09-19 的四轮全量跑测 53 条里挂 0~4 条，每轮挂的还不是
# 同一批 —— 两轮分数严格说不可比，改动有没有效果也就无从判断。
# 判定链路要的是可复现，不是文采；温度归零，seed 可按需固定。
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
_SEED = os.getenv("LLM_SEED")
SAMPLING: dict[str, Any] = {"temperature": TEMPERATURE}
if _SEED:
    SAMPLING["seed"] = int(_SEED)

# 模型把函数调用模板的残片当正文吐出来的形状。2026-09-19 回归用例 Q6 实测：规则引擎
# 已经把 WO-260708 / 120 分钟 / 不符合关闭要求原样交回来了，成文那一次却只回了
# `<previous_tool_call>\n\n</previous_tool_call>` 共 11 个 token，链路照单全收当成最终答案。
# 判据是字面控制残片，不做语义判断，因此不存在误伤正常正文的可能。
_JUNK_MARKUP = re.compile(
    r"</?\s*(?:previous_)?(?:tool_call|tool_response|function_call|function_results)\s*[^>]*>",
    re.I)
# 去掉残片后仍够不上这个字数，就不是一份能发出去的正文。三段标题本身就有三十多字，
# 真实最短的一份正文是 127 字（探针 P6），与这条线之间留着足够余量。
MIN_ANSWER_CHARS = 40


# 问题里点名的库内对象：工单号、风机号、故障码。命中说明这件事**有**对应的表可查，
# 「资料里没有这个字段」得查过才说得出口 —— 与「问题不落在四类资料范围内」不是一回事。
# 不用 \b 划边界：中文在 Unicode 下也是 \w，「24011故障」里数字两侧都不成边界，
# 整条规则会在没有空格的中文问句上静默失效。改用只排除数字/字母的前后瞻。
_DB_ENTITY = re.compile(r"WO-\d+|(?<![A-Za-z])[Tt]\d{2}(?!\d)|(?<!\d)\d{5}(?!\d)")


def _names_db_entity(question: str) -> bool:
    return bool(_DB_ENTITY.search(question or ""))


def _unusable_answer(content: str) -> str | None:
    """正文不可用的理由；可用则返回 None。

    只认两种确定性形态：整段只剩控制残片（或干脆是空的），以及短到不可能是一份
    三段式答案。范围外声明那条合规出口天然简短，按标记放行。
    """
    text = _JUNK_MARKUP.sub("", content or "").strip()
    if not text:
        return "正文为空或只剩控制残片"
    if OUT_OF_SCOPE_MARK in content:
        return None
    if len(text) < MIN_ANSWER_CHARS:
        return "正文只有 %d 字，不足以构成一份答案" % len(text)
    return None


class LlmNotConfigured(RuntimeError):
    pass


class AgentRunFailed(RuntimeError):
    """运行期失败（模型超时、网关报错等），携带已经发生过的 span。

    不带出来的话，失败链路在后台就完全不可见——而「模型调用成功率」的分母
    只剩下成功的那些，指标会恒为 100%，等于装饰。
    """

    def __init__(self, cause: Exception, partial: dict[str, Any]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.partial = partial


@lru_cache(maxsize=4)
def _client_for(key: str, base_url: str, timeout: float) -> OpenAI:
    """按配置缓存客户端，**跨问答复用同一个连接池**。

    原先每轮问答新建一个 client，也就是每个问题都重做一次 TCP + TLS 握手。
    本机实测（2026-09-18，经 VPN 出网）到 DashScope 的 RTT 为 386 ms，TLS 要
    2~3 个往返，建连一项就吃掉约 0.8~1.2 秒 —— 而一次问答的全部工具执行加起来
    才 0.92 秒。链路里"Agent 决策"那一段 1108 ms，大半是握手，不是模型在想。

    缓存键带上配置：换了 base_url 或超时仍然拿到新客户端，改配置不必重启。
    httpx 的连接池本身线程安全，多个请求共用一个 client 是 SDK 的推荐用法。
    """
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout)


def _client() -> OpenAI:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise LlmNotConfigured(
            "未配置 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
        )
    return _client_for(
        key,
        os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        TIMEOUT,
    )


class _Msg:
    """把流式增量拼回成与非流式一致的消息对象，后续逻辑无需分叉。"""

    def __init__(self, content: str, tool_calls: list[Any], *, truncated: bool = False) -> None:
        self.content = content
        self.tool_calls = tool_calls or None
        # 正文被厂商的输出长度上限砍断 —— 这一段不是完整答案
        self.truncated = truncated


class _Call:
    def __init__(self, cid: str, name: str, arguments: str) -> None:
        self.id = cid
        self.function = type("F", (), {"name": name, "arguments": arguments})()


def _call(client: OpenAI, model: str, messages: list[dict[str, Any]], trace: Trace,
          *, stream_answer: bool = True, release_mark: str | None = None):
    """发起一次流式调用。

    `stream_answer=False` 时正文先扣住不发。用在还没取到任何证据的那几轮上：
    那时候写出来的东西随时可能被「未取证拦截」整段作废，先吐后撤等于让用户看着
    一段完整答案被换掉（见 run_agent_stream 里 streamed 的说明）。

    但「扣住」不能一扣到底——范围外声明是护栏的合规出口，那一轮同样没有工具调用，
    却是要放行的。所以给一个放行标记 `release_mark`：正文里一出现它，就说明这轮
    不会被作废，立刻把攒下的补发出去、后面照常逐字流。标记按提示词要求写在
    「结论」段首句，扣住的时间因此只有一句话，出口路径几乎不损失首字时间。

    实测（见 设计说明 第 3.14 节）：一次多步问答里 69% 的耗时花在最后一次「写答案」上——
    1402 个字符按约 94 字符/秒生成，就是 13 秒。决策调用反而很快（2~3 秒）。
    逐字流式吐出后，首字时间降到约 1 秒，用户不必对着转圈等十几秒。

    工具调用的增量按 index 累加，拼回后与非流式结果同构，因此循环其余部分不用分叉。
    """
    span = trace.start("MODEL", "Agent 决策", detail=model)
    usage = None
    joined = ""           # 本轮已收到的全部内容，滚动累加（逐块重拼是 O(n²)）
    finish: str | None = None   # 厂商给的停止原因，"length" 表示正文被砍在半句上
    partial: dict[int, dict[str, str]] = {}
    answer_at = -1        # 正文（「## 结论」）在已收到内容里的起点，-1 表示还没出现
    emitted = 0           # 正文里已经吐给界面的字符数，补发时从这里接着发
    streaming = stream_answer

    # 失败（超时、限流、网关 5xx）同样是一次模型调用，要留下 ERROR span：
    # 只记成功的话，后台的模型调用成功率就永远是 100%。
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": ENABLE_THINKING}, **SAMPLING,
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            # 厂商把「为什么停」放在最后一个分片上，错过就再也问不到了
            finish = getattr(chunk.choices[0], "finish_reason", None) or finish
            delta = chunk.choices[0].delta
            if delta.content:
                joined += delta.content
                # 决策轮也会顺带输出几句自然语言（「我需要先查询…」）。
                # 那些不是答案，若照发给界面，会出现"冒出几个字又被冲掉"的闪烁。
                # 以提示词强制的三段标题为界：见到「## 结论」才算进入最终答案。
                if answer_at < 0:
                    answer_at = joined.find(ANSWER_MARKER)
                if answer_at >= 0:
                    body = joined[answer_at:]
                    # 见到放行标记就解扣：这一轮已经确定不会被未取证拦截作废
                    if not streaming and release_mark and release_mark in joined:
                        streaming = True
                    if streaming and len(body) > emitted:
                        yield {"type": "token", "text": body[emitted:]}
                        emitted = len(body)
            for tc in (delta.tool_calls or []):
                slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
    except Exception as exc:
        # 失败的调用同样烧了 token：厂商对已产出的部分照常计费。流式在末尾才发
        # usage，异常打断多半拿不到 —— 那就把"这次的用量没测到"显式记下来，
        # 而不是记成 0 让 Token / 成本那两格看起来天然不含它。
        span.close(input_summary="上下文 %d 条消息" % len(messages),
                   output_summary="%s：%s" % (type(exc).__name__, exc),
                   input_detail=_request_detail(model, messages),
                   output_detail=_detail({"error": "%s：%s" % (type(exc).__name__, exc)}),
                   status=ERROR, **_usage_of(model, usage))
        raise

    calls = [_Call(v["id"], v["name"], v["arguments"]) for _, v in sorted(partial.items())]
    content = joined
    truncated = finish == "length"

    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    span.name = "Agent 决策" if calls else "答案成文"
    span.close(
        input_summary="prompt %d tok%s" % (pt, "（缓存命中 %d）" % cached if cached else ""),
        output_summary=(("发起 %d 个工具调用：%s" % (len(calls), "、".join(c.function.name for c in calls)))
                        if calls else (_TRUNCATED_MARK if truncated else "")
                             + content[:110] + ("…" if len(content) > 110 else "")),
        status=DEGRADED if truncated else OK,
        finish_reason=finish if truncated else None,
        input_detail=_request_detail(model, messages),
        output_detail=_detail({
            "content": content,
            "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments}
                           for c in calls],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached},
        }),
        prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached,
        cost_cny=round(price_of(model, pt, ct, cached), 6),
    )
    if calls and emitted:
        # 极少见：同一轮里既出现了标题又发起了工具调用。已渲染的不是最终答案，要撤掉。
        # 一个字都没吐过就没什么可撤的，别发空指令让前端白闪一下。
        yield {"type": "answer_reset"}
    elif not calls and answer_at < 0 and content.strip():
        # 模型没按三段格式输出时的兜底：一次性给出，不让界面空着。
        # 同样只在这一轮不会被作废时才发——扣住期间宁可晚一点，也不能吐了再撤。
        if streaming or (release_mark and release_mark in content):
            yield {"type": "token", "text": content}
    yield {"type": "_message", "message": _Msg(content, calls, truncated=truncated)}


def _usage_of(model: str, usage: Any) -> dict[str, Any]:
    """把一次调用的用量折成 span 上的计量字段。

    ``usage`` 为空表示这次没测到（流式被异常打断，usage 在末尾那一帧还没到）。
    此时打 ``usage_missing`` 标记：后台据此在 Token 那格标注"另有 N 次调用未计量"，
    而不是把缺失当成零。
    """
    if usage is None:
        return {"usage_missing": True}
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached,
            "cost_cny": round(price_of(model, pt, ct, cached), 6)}


def _for_model(result: dict[str, Any]) -> dict[str, Any]:
    """发给模型的那一份：剥掉 `_audit_` 开头的键。

    留痕和喂模型是两件事。规则引擎内部发的 SQL 要进 span 供审计复现，但发给模型
    毫无用处 —— 它已经拿到结论和 facts 了，多出来的 SQL 只是烧 token，还会把
    护栏的上下文水位抬高。

    这条边界原本靠"记得别回写 result"的口头纪律守（见 _collect_evidence 里那段
    注释），加一条就得记一次。改成按前缀剥离，新增审计字段时自动生效。

    批量判定的审计字段藏在 results 里的每一条下面，只剥顶层等于没剥 ——
    而批量正是"同样的内容乘以对象数"最容易失控的地方。
    """
    stripped = {k: v for k, v in result.items() if not k.startswith("_audit_")}
    if stripped.get("batch"):
        stripped["results"] = [
            dict(item, result=_for_model(item.get("result") or {}))
            for item in stripped.get("results") or []
        ]
    return stripped


def _detail(obj: Any) -> str:
    """span 详情：完整的输入/输出参数，供后台展开查看。"""
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return text if len(text) <= DETAIL_CAP else text[:DETAIL_CAP] + "\n…（超出 %d 字符已截断）" % DETAIL_CAP


def _request_detail(model: str, messages: list[dict[str, Any]]) -> str:
    """模型调用的输入侧。

    完整提示词与历史不留存——system 提示词每轮重复、工具返回在对应的 TOOL span 里
    已有原文，再存一份是把同样的内容乘以轮数。这里给出构成与本轮最后一条消息，
    并把这件事写在 note 里，不让人误以为看到的就是全部输入。
    """
    last = messages[-1] if messages else {}
    roles: dict[str, int] = {}
    for m in messages:
        roles[m.get("role", "?")] = roles.get(m.get("role", "?"), 0) + 1
    return _detail({
        "model": model,
        "enable_thinking": ENABLE_THINKING,
        "tools": [t["function"]["name"] for t in TOOLS],
        "messages": len(messages),
        "by_role": roles,
        "last_message": {"role": last.get("role"),
                         "content": str(last.get("content") or "")[:600]},
        "note": "完整提示词与历史未留存；工具返回的原文见对应 TOOL / RAG / RULE span",
    })


def _plain(msg) -> dict[str, Any]:
    """只回填 role / content / tool_calls。

    thinking 模式下响应还带 reasoning_content 等字段，原样回填有些网关会报错，
    而且推理过程没有必要占用后续每一轮的上下文。
    """
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [{
            "id": c.id,
            "type": "function",
            "function": {"name": c.function.name, "arguments": c.function.arguments},
        } for c in msg.tool_calls]
    return out


# 批量判定摘要里数哪个字段、叫什么。各规则的"成立"含义不同（禁止复位成立是坏消息、
# 关闭合规成立是好消息），所以按规则逐条写明，不用一个笼统的"命中"糊过去。
# priority_required 的结论是优先级档位不是布尔，故意不在表里，只报对象数。
_BATCH_FLAG = {
    "repeat_fault": ("is_repeat_fault", "构成重复故障"),
    "remote_reset_ban": ("reset_banned", "禁止远程复位"),
    "close_compliance": ("is_compliant", "关闭合规"),
    "replace_precondition": ("can_replace_now", "具备立即更换条件"),
    "work_order_assessment": ("has_work_order", "已有工单"),
}


def _summarize(name: str, result: dict[str, Any]) -> str:
    """给 UI 时间线用的一行摘要。不放全量结果，避免界面被刷屏。"""
    if not result.get("ok", True):
        # 护栏拦下的那一次，结果里带着预算报表 —— 它不是"失败"，是按红线不放行
        if result.get("budget"):
            return "护栏拦截：%s" % result.get("error", "")[:80]
        return "失败：%s" % result.get("error", "")[:80]
    if name == "query_db":
        n = result.get("row_count", 0)
        return "命中 %d 行" % n if n else "无匹配记录"
    if name == "search_docs":
        hits = result.get("hits", [])
        check = result.get("recall_check") or {}
        # 漏召回原来是静默的：显示「命中 4 节」，绿的，而该命中的那节不在里面。
        # 把判据摊开写进摘要，退化才看得见。
        warn = ""
        if check.get("status") == "low_confidence":
            warn = "（低置信：%s）" % "；".join(check.get("reasons") or [])
        if not hits:
            return "未命中%s" % warn
        inlined = "（已带回《%s》全文）" % result["inlined"]["title"][:24] if result.get("inlined") else ""
        return "命中 %d 节：%s%s%s" % (
            len(hits), "、".join(h["title"][:24] for h in hits[:3]), inlined, warn)
    if name == "get_doc_section":
        return "取回《%s》%s" % (result.get("doc", ""), result.get("title", ""))
    if name == "check_rule":
        # 批量判定没有单一 verdict：不特判的话时间线上就是一行空白，
        # 一次判了几个、判出几个成立全看不见。
        if result.get("batch"):
            items = result.get("results") or []
            flag = _BATCH_FLAG.get(result.get("rule") or "")
            if not flag:
                return "批量判定 %d 个对象" % len(items)
            key, label = flag
            # 只数 True。None 是「资料不足以判定」，混进计数就等于把未知算成否定。
            hit = sum(1 for i in items if (i.get("result") or {}).get(key) is True)
            return "批量判定 %d 个对象：%d 个%s" % (len(items), hit, label)
        return str(result.get("verdict", ""))[:80]
    return "完成"


def warm_up() -> None:
    """把懒加载的几样先建好，别让它们落在第一个问题头上。

    检索索引要给两份文档分词，而 jieba 的词典是首次分词时才加载的：本机实测这一下
    要 2.4 秒。原先它落在**进程起来后的第一个提问**上——那一题白等两秒多，看上去
    像模型慢，其实一个 token 都还没发出去。预算分母（语料总量）同理。

    服务端与命令行跑测器都要调：只在服务端预热的话，跑测产物里第一条用例会多背
    一个冷启动，而那批数字正是用来判断优化有没有效果的。
    """
    from tools.budget import corpus_totals
    from tools.retriever import get_index

    get_index().search("预热")
    corpus_totals()


def run_agent(question: str) -> dict[str, Any]:
    """跑一轮问答，返回最终结果。内部复用流式实现，避免两套逻辑漂移。"""
    result: dict[str, Any] = {}
    for event in run_agent_stream(question):
        if event["type"] == "done":
            result = event["result"]
    return result


def run_agent_stream(question: str) -> Iterator[dict[str, Any]]:
    """跑一轮问答，边跑边吐事件。

    单次调用延迟受网络与模型波动影响很大（实测同一用例可差数倍），压不下去。
    因此让等待过程可见：每完成一次工具调用就推一条事件，界面实时渲染轨迹。
    题面 4.4 要求「查询中的状态反馈」，一个转圈图标干等半分钟不算合格的反馈。

    证据不是让模型自己报的，而是从真实发生过的工具调用里汇总出来的——
    这样「依据展示」在结构上就不可能是编的。
    """
    client = _client()
    model = os.getenv("LLM_MODEL", "qwen3.8-flash")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []
    tr = Trace()
    evidence: dict[str, list[dict[str, Any]]] = {"tables": [], "docs": [], "rules": []}
    budget = ContextBudget()
    started = time.monotonic()
    draft = ""
    stop_reason = "completed"

    ungrounded_retries = 0
    # 范围外声明被要求先取证的次数。上限 1，见下面那段说明。
    scope_challenges = 0
    # 界面上此刻是否挂着一段还没定稿的正文。只要这段会被后面的分支作废
    # （未取证打回、拒答、步数用尽重写），就必须先发 answer_reset 把它撤掉，
    # 否则新正文会直接续写在旧正文后面——线上实测一次提问被写成了两份答案。
    streamed = False

    yield {"type": "thinking", "message": "正在判断需要查询哪些资料…"}

    # 失败也要能被复核：把已发生的 span 连同异常一起带出去，由调用方留存链路。
    try:
        for _step in range(MAX_STEPS):
            msg = None
            # 一条证据都还没取到时，这一轮写出来的正文随时可能被下面的未取证拦截
            # 整段作废，所以先扣住。取到证据之后拦截不会再触发，正文照常逐字流；
            # 范围外声明那条合规出口靠 release_mark 当场解扣，不必等整轮跑完。
            for event in _call(client, model, messages, tr, stream_answer=bool(trace),
                               release_mark=OUT_OF_SCOPE_MARK):
                if event["type"] == "_message":
                    msg = event["message"]
                    continue
                if event["type"] == "token":
                    streamed = True
                elif event["type"] == "answer_reset":
                    streamed = False
                yield event

            if not msg.tool_calls:
                content = msg.content or ""
                # 范围外声明是护栏的**合规出口**：问题压根不落在四类资料里
                # （问模型是谁、问天气、问外部标准），没有任何工具能给出依据，
                # 打回只会逼它去查点无关的东西凑数——实测它就是这么干的，
                # 而凑出来的那段自由发挥里混着没人核对的数字。
                # 判据是固定串而不是语义猜测：出口必须确定，否则等于没有红线。
                if not trace and OUT_OF_SCOPE_MARK in content:
                    # 出口有一个前置条件：问题点名了库内对象（工单号 / 风机号 / 故障码）时，
                    # 「资料里没有这个字段」与「问题不落在四类资料范围内」是两件事，
                    # 前者必须查过那张表才说得出口。2026-09-19 边界用例 B8 实测：问
                    # 「WO-260708 这张工单是谁关闭的」，工单就在库里、只是没有人员字段，
                    # 模型却走了范围外出口，0 步作答 —— 话说对了，证据一条没取。
                    #
                    # 只拦一次。再声明一次就放行：真·范围外的问题只要捎带一个风机号
                    # （「T08 那边今天天气如何」）就被逼着去查无关的东西凑数的话，
                    # 正是这个出口当初要消掉的毛病，不能为了堵一个洞把它重新挖开。
                    if _names_db_entity(question) and not scope_challenges:
                        scope_challenges += 1
                        tr.record("GUARD", "范围外声明待核实", status=DEGRADED,
                                  output_summary="问题点名了库内对象，已要求先取证再判断是否属范围外")
                        if streamed:
                            yield {"type": "answer_reset"}
                            streamed = False
                        yield {"type": "thinking",
                               "message": "问题里点到了库内对象，先核实一次再判断…"}
                        messages.append(_plain(msg))
                        messages.append({"role": "user", "content": _SCOPE_CHALLENGE})
                        continue
                    tr.record("GUARD", "范围外声明", status=DEGRADED,
                              output_summary="模型声明问题超出四类资料范围，按合规拒答放行，不再要求取证")
                    stop_reason = "out_of_scope"
                    draft = content
                    break
                # 一次工具都没调就想作答 —— 踩红线，打回
                if not trace and ungrounded_retries < MAX_UNGROUNDED_RETRIES:
                    ungrounded_retries += 1
                    tr.record("GUARD", "未取证拦截", status=DEGRADED,
                              output_summary="模型未调用任何工具即试图作答，已打回要求先检索")
                    if streamed:
                        yield {"type": "answer_reset"}
                        streamed = False
                    # 打回意味着要重跑一整轮模型调用，界面上会静默好几秒。
                    # 不说一声的话，用户看到的是计时器在跳而什么都没发生。
                    yield {"type": "thinking",
                           "message": "这一轮没有检索任何资料，已打回要求先取证…"}
                    messages.append(_plain(msg))
                    messages.append({
                        "role": "user",
                        "content": "不允许在未检索任何资料的情况下作答。请先调用 query_db "
                                   "或 search_docs 取得依据，再回答。"
                                   "若该问题确实超出随题四类资料的范围（系统实现、密钥配置、"
                                   "天气实况、外部标准等），不要为了取证去查无关资料，"
                                   "直接拒答并在「结论」段首句写明「%s」。" % OUT_OF_SCOPE_MARK,
                    })
                    continue
                if not trace:
                    tr.record("GUARD", "未取证拦截", status=ERROR,
                              output_summary="二次仍未检索，拒绝作答而非放行无依据回答")
                    if streamed:
                        yield {"type": "answer_reset"}
                        streamed = False
                    stop_reason = "refused_ungrounded"
                    draft = ("## 结论\n无法回答：本系统要求所有结论必须有随题资料支撑，"
                             "本次未能检索到任何可用依据。\n\n## 依据\n（无）\n\n"
                             "## 现有资料无法确认\n- 全部事项，需要重新提问或缩小问题范围")
                    break
                draft = msg.content or ""
                # 正文不可用就重写一次。此前这里是无条件收下：只要这一轮没发工具调用，
                # 模型吐出来的东西就是最终答案 —— Q6 那次的「答案」是 11 个 token 的
                # 控制残片，照样进了成文、进了指标、进了页面。
                # 重写不带工具：证据已经齐了（trace 非空），这一步要的只是把话写出来。
                # 只重写一次：还是残片就是这一轮它写不出来，再烧一次多半是同样的东西，
                # 而耗时是实打实翻倍的。
                reason = _unusable_answer(draft)
                if reason:
                    tr.record("GUARD", "成文不可用", status=DEGRADED,
                              output_summary="%s，已要求重写" % reason)
                    if streamed:
                        yield {"type": "answer_reset"}
                        streamed = False
                    yield {"type": "thinking", "message": "这一轮没有写出可用正文，正在重写…"}
                    messages.append(_plain(msg))
                    messages.append({"role": "user", "content": _REWRITE_INSTRUCTION})
                    draft, _ = _closing_call(client, model, messages, tr,
                                             "成文重写", with_tools=False)
                    reason = _unusable_answer(draft)
                    if reason is None:
                        yield {"type": "token", "text": draft}
                if reason:
                    # 重写仍不可用：如实说出来，不拿残片冒充答案。措辞与未取证拒答一致 ——
                    # 两者都是"这次没有可用产出"，口径上也一并从「有效回答」里剔除。
                    tr.record("GUARD", "成文不可用", status=ERROR,
                              output_summary="重写后仍不可用（%s），未产出可用正文" % reason)
                    stop_reason = "answer_unusable"
                    draft = ("## 结论\n本次未能生成可用正文：模型连续两次输出不可用，"
                             "已取得的证据见下方依据，请重跑本问题。\n\n## 依据\n（见证据面板）\n\n"
                             "## 现有资料无法确认\n- 本轮结论未能成文，需重跑")
                break

            messages.append(_plain(msg))
            for call in msg.tool_calls:
                name = call.function.name
                guard: str | None = None      # 非空表示这次调用被护栏按红线拦下
                tool_span = tr.start(TOOL_KIND.get(name, "TOOL"), name)
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = {"ok": False, "error": "工具参数不是合法 JSON：%s" % exc}
                    args = {}
                else:
                    fn = TOOL_REGISTRY.get(name)
                    if fn is None:
                        result = {"ok": False, "error": "未知工具 %s" % name}
                    else:
                        try:
                            result = fn(**args)
                        except TypeError as exc:
                            result = {"ok": False, "error": "工具参数不匹配：%s" % exc}
                        else:
                            # 题面红线：不得把整个数据库或文档放入上下文。
                            # 这里做运行时拦截，而不是事后统计。
                            draw = measure(name, result)
                            refusal = budget.would_exceed(draw)
                            if refusal:
                                # 拦截**只记在这一次工具调用上**，不另开一条 GUARD span：
                                # 同一件事记两条，后台会把它读成"一次工具失败 + 一次护栏降级"，
                                # 异常清单里也会出现两行同样的文字。
                                guard = "上下文预算"
                                # 拒绝不计入用量，水位会停在阈值以下 ——
                                # 把"这次想压到哪"单独记下来，红线才画得出撞线
                                budget.note_refusal(draw, refusal)
                                result = {"ok": False, "error": refusal,
                                          "budget": budget.report()}
                            else:
                                budget.charge(draw)

                entry = {
                    "step": len(trace) + 1,
                    "tool": name,
                    "input": args,
                    "ok": result.get("ok", True),
                    "refused": bool(guard),
                    "summary": _summarize(name, result),
                }
                trace.append(entry)
                # 被护栏拒绝不是"工具失败"：工具本身跑通了，是系统按红线不让结果进上下文。
                # 记成 ERROR 会让后台的「工具调用成功率」被自家护栏拉低 —— 而同一个后台
                # 在「未取证拒答率」旁边写着"护栏生效，不是故障"。两处口径必须一致。
                tool_span.close(
                    input_summary=json.dumps(args, ensure_ascii=False)[:160],
                    output_summary=entry["summary"],
                    input_detail=_detail(args),
                    output_detail=_detail(result),
                    status=DEGRADED if guard else (OK if entry["ok"] else ERROR),
                    guard=guard,
                    budget=budget.report() if guard else None,
                )
                _collect_evidence(evidence, name, args, result)
                yield {"type": "step", "step": entry}

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(_for_model(result), ensure_ascii=False, default=str),
                })
        else:
            stop_reason = "max_steps"
            # 收口会重新生成一份正文，界面上半截的那份不能留着让它续写
            if streamed:
                yield {"type": "answer_reset"}
                streamed = False
            draft = _force_answer(client, model, messages, tr)
    except Exception as exc:
        raise AgentRunFailed(exc, {
            "question": question,
            "answer": "",
            "basis": [],
            "unverifiable": [],
            "sources": [],
            "evidence": evidence,
            "trace": trace,
            "spans": tr.as_list(),
            "trace_summary": tr.summary(),
            "meta": {
                "stop_reason": "llm_error",
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "steps": len(trace),
                "format_parsed": False,
                "budget": budget.report(),
                "model": model,
            },
        }) from exc

    # 正文写到一半被厂商的长度上限砍断 —— 取最后一次 MODEL span 的停止原因：
    # 无论正文出自流式的「答案成文」还是步数用尽后的收口调用，它都是那一次。
    # 拒答那条路的正文是代码写死的，模型那次写了什么都与最终答案无关，不参与判定。
    model_spans = [s for s in tr.as_list() if s["kind"] == "MODEL"]
    answer_truncated = (stop_reason != "refused_ungrounded"
                        and bool(model_spans)
                        and model_spans[-1].get("finish_reason") == "length")

    yield {"type": "composing", "message": "正在整理结论与依据…"}
    yield {
        "type": "done",
        "result": {
            "draft": draft,
            "answer_truncated": answer_truncated,
            "trace": trace,
            "evidence": evidence,
            "stop_reason": stop_reason,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "budget": budget.report(),
            "spans": tr.as_list(),
            "trace_summary": tr.summary(),
            "model": model,
        },
    }


_SCOPE_CHALLENGE = (
    "问题里点名的工单号 / 风机号 / 故障码就在随题数据库里，这件事有表可查。"
    "「资料里没有这个字段」与「问题超出四类资料范围」是两件事：前者必须先查过对应的表"
    "才能下结论，不能凭表结构直接断言。请先调用 query_db 取回该对象所在的记录再回答；"
    "取回之后，如果问到的要点确实不在四类资料里，照原格式说明即可。"
)

_REWRITE_INSTRUCTION = (
    "上一条回复不是一份可用的正文。请基于上面已经取得的证据，直接按三段格式重写答案："
    "「## 结论」「## 依据」「## 现有资料无法确认」。不要再调用工具，不要输出任何标签，"
    "不要复述这条指令。"
)


def _force_answer(client: OpenAI, model: str, messages: list[dict[str, Any]], trace: Trace) -> str:
    """步数用尽时，要求模型基于已有证据收口，而不是无声截断。

    **这一次也要带上 tools**（2026-09-18 改）。先前为了防止模型又去调工具而把它拿掉，
    结果是前缀变了、缓存从工具定义那里整段作废：实测 prompt 6005 token 只命中 1024，
    而带着工具的同类调用命中 3072 —— 多算约 2000 个 token 的 prefill，且这些 token
    按未缓存价计费，单次成本贵 43%。不让它调工具靠的是上面那句用户消息，不是靠
    把工具藏起来。

    万一它还是不听、回了个工具调用而正文为空，再补一次不带工具的调用兜底——
    那一次单独开 span，不并进上一次：两次调用各自烧了 token，合成一条记录会让
    「模型调用次数」少算一次，成本也跟着漏。
    """
    messages.append({
        "role": "user",
        "content": "已达到查询步数上限。请基于上面已经取得的证据直接作答，不要再调用工具。"
                   "证据不足分两种，不要混为一谈："
                   "①查过、资料里确实没有记录的（如现场安全条件、责任调查结论），"
                   "放进「现有资料无法确认」；"
                   "②本轮没来得及查的（如某张表、某个字段还没读过），"
                   "写成「本轮未完成查询：…（需补查）」，"
                   "**不得**说成资料里没有或无法确认——那是在把自己没查说成资料没有。",
    })
    content, called_tools = _closing_call(client, model, messages, trace,
                                          "步数用尽收口", with_tools=True)
    # 判据原先是「正文为空且又发了工具调用」。空只是不可用的一种形态：只剩控制残片
    # 同样发不出去，而它 .strip() 非空，照原判据会被直接放行（见 _unusable_answer）。
    if _unusable_answer(content) is None:
        return content
    # 撤下工具再问一次，这次它没有别的选择
    retry, _ = _closing_call(client, model, messages, trace,
                             "步数用尽收口（撤下工具重试）", with_tools=False)
    return retry or content


def _closing_call(client: OpenAI, model: str, messages: list[dict[str, Any]], trace: Trace,
                  name: str, *, with_tools: bool) -> tuple[str, bool]:
    """收口用的单次非流式调用。返回（正文，这次有没有发起工具调用）。

    同样开 span：它是货真价实的一次模型调用，不记的话步数用尽的链路在后台会少算
    一次调用，token 与成本也跟着漏计。
    """
    span = trace.start("MODEL", name, detail=model)
    extra: dict[str, Any] = {"tools": TOOLS} if with_tools else {}
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, extra_body={"enable_thinking": ENABLE_THINKING},
            **SAMPLING, **extra,
        )
    except Exception as exc:
        # 非流式调用异常时连 usage 对象都没有，同样标成"未计量"
        span.close(input_summary="上下文 %d 条消息" % len(messages),
                   output_summary="%s：%s" % (type(exc).__name__, exc),
                   input_detail=_request_detail(model, messages),
                   output_detail=_detail({"error": "%s：%s" % (type(exc).__name__, exc)}),
                   status=ERROR, usage_missing=True)
        raise
    message = resp.choices[0].message
    content = message.content or ""
    called_tools = bool(getattr(message, "tool_calls", None))
    truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    # 收口轮又发工具调用是降级：这一次没产出可用正文，得让它在异常清单里看得见
    empty_with_tools = called_tools and not content.strip()
    span.close(
        input_summary="prompt %d tok%s" % (pt, "（缓存命中 %d）" % cached if cached else ""),
        output_summary=("收口轮仍发起工具调用且正文为空，已撤下工具重试"
                        if empty_with_tools
                        else (_TRUNCATED_MARK if truncated else "")
                             + content[:110] + ("…" if len(content) > 110 else "")),
        input_detail=_request_detail(model, messages),
        output_detail=_detail({"content": content, "tool_calls": called_tools,
                               "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                         "cached_tokens": cached}}),
        status=DEGRADED if (empty_with_tools or truncated) else OK,
        finish_reason="length" if truncated else None,
        prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached,
        cost_cny=round(price_of(model, pt, ct, cached), 6),
    )
    return content, called_tools


# 收口时长度直接决定耗时，见 _call 的说明


def _add_doc(evidence, section: dict[str, Any]) -> None:
    """同一节可能既被模型取过、又被规则判定带出来，只留一份。"""
    key = (section.get("doc"), section.get("section_id"))
    if any((d.get("doc"), d.get("section_id")) == key for d in evidence["docs"]):
        return
    evidence["docs"].append(section)


def _collect_evidence(evidence, name, args, result) -> None:
    if not result.get("ok", True):
        return
    if name == "search_docs":
        # 带回全文的那一节要进证据：模型不再需要 get_doc_section，
        # 而依据、数据源与证据面板都是从 evidence 汇总出来的 ——
        # 不收在这里，界面上就会出现"引用了手册、数据源里却没有手册"。
        for hit in result.get("hits", []):
            if hit.get("text"):
                _add_doc(evidence, {"doc": hit.get("doc"), "section_id": hit.get("section_id"),
                                    "title": hit.get("title"), "path": hit.get("path"),
                                    "text": hit.get("text")})
    elif name == "query_db":
        evidence["tables"].append({
            "sql": result.get("sql"),
            "columns": result.get("columns", []),
            "rows": result.get("rows", []),
            "row_count": result.get("row_count", 0),
        })
    elif name == "get_doc_section":
        _add_doc(evidence, {
            "doc": result.get("doc"),
            "section_id": result.get("section_id"),
            "title": result.get("title"),
            "path": result.get("path"),
            "text": result.get("text"),
        })
    elif name == "check_rule":
        # 批量判定在证据里摊平成逐条：依据渲染、现场核实清单和数据源汇总都按
        # 「一条 rules 记录 = 一个判定对象」写的，收成一条会让面板上只剩一个结论。
        if result.get("batch"):
            for item in result.get("results", []):
                one = item.get("result") or {}
                if not one.get("ok", True):
                    continue
                evidence["rules"].append({"rule": args.get("rule"),
                                          "subject": item.get("subject"), "result": one})
                for section in cited_sections(one):
                    _add_doc(evidence, section)
            return
        evidence["rules"].append({"rule": args.get("rule"), "result": result})
        # 判定引用到的规程条款，原文一并留存：依据面板里点得开、数据源里数得到。
        # 只写 evidence，**不回写 result** —— 调用方随后会把 result 序列化进模型上下文，
        # 在这里塞原文等于把同一段文字再发一遍，护栏水位也会跟着虚高。
        for section in cited_sections(result):
            _add_doc(evidence, section)
