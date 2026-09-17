# -*- coding: utf-8 -*-
"""后台管理页面：Agent 质量中心与链路追踪，合并为一页，左侧切换。

质量中心里再分两档，两者口径不同，不能并成一张表看：
  · 离线评测 —— 固定用例 + 断言，有标准答案，判得出对错，数据来自 eval/run.py 的落盘产物；
  · 线上质量 —— 真实用户在问答页问出来的那些，没有标准答案，只能看系统留下的痕迹
    （收没收口、拒没拒答、标没标出无法确认、上下文压到哪、花了多少）。

按页面与问答界面分开——链路明细、质量指标是运维视角，混在问答页里会淹没
结论与依据，而后者才是用户要看的。

代码上单独成文件（职责分离），运行上是同一个进程、同一个端口：
链路留存为内存态，拆进程后后台就看不到问答侧记录的链路了。
路由以 APIRouter 形式挂进 app.py。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import evalview, grounding, replay, tracestore
from agent.composer import compose, stamp_served
from agent.loop import AgentRunFailed, run_agent

STATIC_DIR = Path(__file__).resolve().parent / "static"

admin_router = APIRouter(tags=["后台管理"])

# 页面本身不缓存。整个前端就是这一个文件（没有带 hash 的打包产物），FileResponse
# 默认又不发 Cache-Control，浏览器于是按启发式自己决定缓存多久 —— 改完前端普通
# 刷新可能还是旧的，而后端已经是新的：新页面去读旧接口没有的字段，看起来像功能
# 坏了，实际排查的是一个不存在的 bug。数据接口那边每个都已经各自带了 no-store。
_NO_STORE = {"Cache-Control": "no-store"}


class RunRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, max_length=40)


@admin_router.get("/admin")
def admin_page() -> FileResponse:
    """后台单页：左侧切换「Agent 质量中心」与「链路追踪」。"""
    return FileResponse(STATIC_DIR / "admin.html", headers=_NO_STORE)


@admin_router.get("/api/eval")
def eval_results(response: Response) -> dict:
    """离线评测只读接口。数据来自 eval/run.py 的落盘产物，不做任何持久化。"""
    response.headers["Cache-Control"] = "no-store"
    return evalview.load()


@admin_router.get("/api/quality/online")
def online_quality(response: Response) -> dict:
    """线上质量：只取 source=online 的链路，即真实用户在问答页留下的那些。

    与 /api/traces 的差别不只是过滤条件——后者是给链路追踪页看明细的，
    这里是**指标口径**：后台点「运行全部」跑出来的回归链路一条都不进来，
    否则挑好的用例会把线上成功率抬上去，指标就不再指示线上的真实状况。
    """
    response.headers["Cache-Control"] = "no-store"
    # 离线回放模式下问答走的是 eval 产物，不进链路留存 —— 这一档会停在旧数据上，
    # 不说出来就成了"看着一切正常"。回放是演示预案，但指标页必须自己承认。
    return {"stats": tracestore.stats(source="online"),
            "replay_mode": replay.enabled(),
            "items": tracestore.listing(source="online")}


@admin_router.post("/api/eval/run")
def eval_run(req: RunRequest) -> StreamingResponse:
    """现场直接跑用例：逐条执行并把结果流式推回，同时落盘更新产物。

    与命令行 `python eval/run.py` 走同一套判定（eval/verdict.py），不另写一份。
    """
    def events():
        def sse(payload: dict) -> str:
            return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False, default=str)

        try:
            cases = evalview.cases_by_id(req.ids)
        except Exception as exc:
            yield sse({"type": "error", "error": "读取用例失败：%s" % exc})
            return
        if not cases:
            yield sse({"type": "error", "error": "没有匹配的用例。"})
            return

        yield sse({"type": "begin", "total": len(cases)})
        for i, case in enumerate(cases, 1):
            yield sse({"type": "running", "id": case["id"], "index": i,
                       "question": case["question"]})
            try:
                t0 = time.monotonic()
                result = stamp_served(
                    grounding.apply(compose(case["question"], run_agent(case["question"]))), t0)
                # 标成 eval：链路追踪页照样看得到，但线上质量的分母里不能有它。
                # 编号随产物一起落盘 —— 离线评测那张表点一行要跳到链路追踪，
                # 有它才能定位到**同一次运行**留下的那条活链路。
                trace_id = tracestore.record(case["question"], result, source="eval")
                ok, problems = evalview.judge(case, result)
                evalview.persist(case, result, trace_id=trace_id)
            except AgentRunFailed as exc:
                # 跑挂的用例同样留一条链路，成功率与用例结果才对得上
                tracestore.record(case["question"], exc.partial, source="eval", error=str(exc.cause))
                yield sse({"type": "case", "id": case["id"], "ok": False,
                           "problems": ["模型调用失败：%s" % exc.cause], "elapsed_ms": None})
                continue
            except Exception as exc:
                yield sse({"type": "case", "id": case["id"], "ok": False,
                           "problems": ["异常：%s" % exc], "elapsed_ms": None})
                continue
            ts = result.get("trace_summary") or {}
            yield sse({"type": "case", "id": case["id"], "ok": ok, "problems": problems,
                       "elapsed_ms": result.get("meta", {}).get("elapsed_ms"),
                       "steps": result.get("meta", {}).get("steps"),
                       "tokens": (ts.get("prompt_tokens", 0) or 0) + (ts.get("completion_tokens", 0) or 0),
                       "cost_cny": ts.get("cost_cny")})
        # 跑完记一条指标快照：页面上的「较上次」靠它，与命令行 eval/run.py 走同一个函数
        snap = evalview.snapshot(ran=len(cases))
        yield sse({"type": "done", "at": snap["at"]})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_LIST_ONLY = ("spans", "answer", "unverifiable")


def _as_list_row(record: dict) -> dict:
    """与 tracestore.listing() 同一套剥字段口径，产物搭出来的那批也要照办。"""
    return dict({k: v for k, v in record.items() if k not in _LIST_ONLY},
                unverifiable_count=len(record.get("unverifiable") or []),
                answered=bool((record.get("answer") or "").strip())
                         and record.get("stop_reason") != "refused_ungrounded")


def _merged_traces() -> tuple[list[dict], int]:
    """留存的链路 + 产物搭出来的回归链路，按时间倒序合成一个列表。

    离线评测的每条用例都必须能在这一页找得到，否则那张表上的跳转在最常见的
    情形下就是坏的：留存是进程内的，命令行跑出的产物没有链路，后台跑的也会被
    挤出或随重启消失。同一次运行两边都有时以留存那条为准（它更全，含
    served_ms、格式解析、计价口径），靠产物里记下的 trace_id 去重。
    """
    live = tracestore.listing()
    live_ids = {t["id"] for t in live}
    extra = [_as_list_row(r) for rid, r in evalview.trace_records().items()
             if r.get("trace_id_in_store") not in live_ids]
    merged = sorted(live + extra, key=lambda t: t.get("at_ts") or 0, reverse=True)
    return merged, len(extra)


@admin_router.get("/api/traces")
def traces(response: Response) -> dict:
    """链路留存列表与聚合统计。进程内保留最近 N 条，不落库。

    这里**不分来源**：线上问答与回归跑测的链路都要能逐段复核，
    列表上按来源打标即可。做口径区分的是 /api/quality/online。

    列表里另掺了离线评测的产物（见 _merged_traces），但**统计口径不掺**：
    stats 仍然只算留存，否则"留存 N / 50"这个容量表述立刻失真。
    """
    response.headers["Cache-Control"] = "no-store"
    items, from_artifact = _merged_traces()
    return {"stats": tracestore.stats(), "items": items, "from_artifact": from_artifact}


@admin_router.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict:
    """id 是整数就查留存，带 eval: 前缀就拿产物现搭一条。"""
    if trace_id.startswith(evalview.TRACE_PREFIX):
        item = evalview.trace_records().get(trace_id)
        return item or {"error": "没有找到用例 %s 的评测产物。" %
                                 trace_id[len(evalview.TRACE_PREFIX):]}
    if not trace_id.lstrip("-").isdigit():
        return {"error": "链路编号 %r 格式不正确。" % trace_id}
    item = tracestore.get(int(trace_id))
    return item or {"error": "该链路不存在或已被新记录挤出（仅保留最近 %d 条）" % tracestore.MAX_TRACES}
