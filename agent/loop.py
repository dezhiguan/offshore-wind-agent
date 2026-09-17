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
import time
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT
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
# qwen3 系列默认开启 thinking，单次调用可达 30 秒，现场演示不可接受。
# 关闭后靠工具与规则引擎保证正确性，而不是靠模型的内部推理。
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in {"1", "true", "yes"}
# 流式返回下连接保持时间更长；实测 60 秒会偶发读超时，演示中途失败比慢更糟
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


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


def _client() -> OpenAI:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise LlmNotConfigured(
            "未配置 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
        )
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=TIMEOUT,
    )


class _Msg:
    """把流式增量拼回成与非流式一致的消息对象，后续逻辑无需分叉。"""

    def __init__(self, content: str, tool_calls: list[Any]) -> None:
        self.content = content
        self.tool_calls = tool_calls or None


class _Call:
    def __init__(self, cid: str, name: str, arguments: str) -> None:
        self.id = cid
        self.function = type("F", (), {"name": name, "arguments": arguments})()


def _call(client: OpenAI, model: str, messages: list[dict[str, Any]], trace: Trace):
    """发起一次流式调用。

    实测（见 设计说明 第 3.14 节）：一次多步问答里 69% 的耗时花在最后一次「写答案」上——
    1402 个字符按约 94 字符/秒生成，就是 13 秒。决策调用反而很快（2~3 秒）。
    逐字流式吐出后，首字时间降到约 1 秒，用户不必对着转圈等十几秒。

    工具调用的增量按 index 累加，拼回后与非流式结果同构，因此循环其余部分不用分叉。
    """
    span = trace.start("MODEL", "Agent 决策", detail=model)
    usage = None
    content_parts: list[str] = []
    partial: dict[int, dict[str, str]] = {}
    answering = False

    # 失败（超时、限流、网关 5xx）同样是一次模型调用，要留下 ERROR span：
    # 只记成功的话，后台的模型调用成功率就永远是 100%。
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": ENABLE_THINKING},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                # 决策轮也会顺带输出几句自然语言（「我需要先查询…」）。
                # 那些不是答案，若照发给界面，会出现"冒出几个字又被冲掉"的闪烁。
                # 以提示词强制的三段标题为界：见到「## 结论」才算进入最终答案。
                if answering:
                    yield {"type": "token", "text": delta.content}
                else:
                    joined = "".join(content_parts)
                    at = joined.find(ANSWER_MARKER)
                    if at >= 0:
                        answering = True
                        yield {"type": "token", "text": joined[at:]}
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
    content = "".join(content_parts)

    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    span.name = "Agent 决策" if calls else "答案成文"
    span.close(
        input_summary="prompt %d tok%s" % (pt, "（缓存命中 %d）" % cached if cached else ""),
        output_summary=(("发起 %d 个工具调用：%s" % (len(calls), "、".join(c.function.name for c in calls)))
                        if calls else (content[:110] + ("…" if len(content) > 110 else ""))),
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
    if calls and answering:
        # 极少见：同一轮里既出现了标题又发起了工具调用。已渲染的不是最终答案，要撤掉。
        yield {"type": "answer_reset"}
    elif not calls and not answering and content.strip():
        # 模型没按三段格式输出时的兜底：一次性给出，不让界面空着
        yield {"type": "token", "text": content}
    yield {"type": "_message", "message": _Msg(content, calls)}


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
    """
    return {k: v for k, v in result.items() if not k.startswith("_audit_")}


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
        return "命中 %d 节：%s%s" % (
            len(hits), "、".join(h["title"][:24] for h in hits[:3]), warn)
    if name == "get_doc_section":
        return "取回《%s》%s" % (result.get("doc", ""), result.get("title", ""))
    if name == "check_rule":
        return str(result.get("verdict", ""))[:80]
    return "完成"


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

    yield {"type": "thinking", "message": "正在判断需要查询哪些资料…"}

    # 失败也要能被复核：把已发生的 span 连同异常一起带出去，由调用方留存链路。
    try:
        for _step in range(MAX_STEPS):
            msg = None
            for event in _call(client, model, messages, tr):
                if event["type"] == "_message":
                    msg = event["message"]
                else:
                    yield event

            if not msg.tool_calls:
                # 一次工具都没调就想作答 —— 踩红线，打回
                if not trace and ungrounded_retries < MAX_UNGROUNDED_RETRIES:
                    ungrounded_retries += 1
                    tr.record("GUARD", "未取证拦截", status=DEGRADED,
                              output_summary="模型未调用任何工具即试图作答，已打回要求先检索")
                    messages.append(_plain(msg))
                    messages.append({
                        "role": "user",
                        "content": "不允许在未检索任何资料的情况下作答。请先调用 query_db "
                                   "或 search_docs 取得依据，再回答。",
                    })
                    continue
                if not trace:
                    tr.record("GUARD", "未取证拦截", status=ERROR,
                              output_summary="二次仍未检索，拒绝作答而非放行无依据回答")
                    stop_reason = "refused_ungrounded"
                    draft = ("## 结论\n无法回答：本系统要求所有结论必须有随题资料支撑，"
                             "本次未能检索到任何可用依据。\n\n## 依据\n（无）\n\n"
                             "## 现有资料无法确认\n- 全部事项，需要重新提问或缩小问题范围")
                    break
                draft = msg.content or ""
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

    yield {"type": "composing", "message": "正在整理结论与依据…"}
    yield {
        "type": "done",
        "result": {
            "draft": draft,
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


def _force_answer(client: OpenAI, model: str, messages: list[dict[str, Any]], trace: Trace) -> str:
    """步数用尽时，要求模型基于已有证据收口，而不是无声截断。

    这一次同样开 span：它是货真价实的一次模型调用，不记的话步数用尽的链路
    在后台会少算一次调用，token 与成本也跟着漏计。
    """
    messages.append({
        "role": "user",
        "content": "已达到查询步数上限。请基于上面已经取得的证据直接作答；"
                   "证据不足的部分放进「现有资料无法确认」，不要再调用工具。",
    })
    span = trace.start("MODEL", "步数用尽收口", detail=model)
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, extra_body={"enable_thinking": ENABLE_THINKING},
        )
    except Exception as exc:
        # 非流式调用异常时连 usage 对象都没有，同样标成"未计量"
        span.close(input_summary="上下文 %d 条消息" % len(messages),
                   output_summary="%s：%s" % (type(exc).__name__, exc),
                   input_detail=_request_detail(model, messages),
                   output_detail=_detail({"error": "%s：%s" % (type(exc).__name__, exc)}),
                   status=ERROR, usage_missing=True)
        raise
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    span.close(
        input_summary="prompt %d tok%s" % (pt, "（缓存命中 %d）" % cached if cached else ""),
        output_summary=content[:110] + ("…" if len(content) > 110 else ""),
        input_detail=_request_detail(model, messages),
        output_detail=_detail({"content": content,
                               "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                         "cached_tokens": cached}}),
        prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached,
        cost_cny=round(price_of(model, pt, ct, cached), 6),
    )
    return content


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
    if name == "query_db":
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
        evidence["rules"].append({"rule": args.get("rule"), "result": result})
        # 判定引用到的规程条款，原文一并留存：依据面板里点得开、数据源里数得到。
        # 只写 evidence，**不回写 result** —— 调用方随后会把 result 序列化进模型上下文，
        # 在这里塞原文等于把同一段文字再发一遍，护栏水位也会跟着虚高。
        for section in cited_sections(result):
            _add_doc(evidence, section)
