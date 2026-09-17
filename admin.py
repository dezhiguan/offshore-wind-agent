# -*- coding: utf-8 -*-
"""后台管理页面：Agent 质量中心、链路追踪与版本管理，合并为一页，左侧切换。

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
from pathlib import Path

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import evalview, grounding, tracestore, versions
from agent.composer import compose
from agent.loop import AgentRunFailed, run_agent

STATIC_DIR = Path(__file__).resolve().parent / "static"

admin_router = APIRouter(tags=["后台管理"])


class RunRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, max_length=40)


@admin_router.get("/admin")
def admin_page() -> FileResponse:
    """后台单页：左侧切换「Agent 质量中心」「链路追踪」「版本管理」。"""
    return FileResponse(STATIC_DIR / "admin.html")


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
    return {"stats": tracestore.stats(source="online"),
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
                result = grounding.apply(compose(case["question"], run_agent(case["question"])))
                # 标成 eval：链路追踪页照样看得到，但线上质量的分母里不能有它
                tracestore.record(case["question"], result, source="eval")
                ok, problems = evalview.judge(case, result)
                evalview.persist(case, result)
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
        yield sse({"type": "done"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@admin_router.get("/api/versions")
def version_overview(response: Response) -> dict:
    """版本台账与运行指纹。台账是声明，指纹从当前进程实际生效的配置现算。"""
    response.headers["Cache-Control"] = "no-store"
    return versions.overview()


@admin_router.get("/api/traces")
def traces(response: Response) -> dict:
    """链路留存列表与聚合统计。进程内保留最近 N 条，不落库。

    这里**不分来源**：线上问答与回归跑测的链路都要能逐段复核，
    列表上按来源打标即可。做口径区分的是 /api/quality/online。
    """
    response.headers["Cache-Control"] = "no-store"
    return {"stats": tracestore.stats(), "items": tracestore.listing()}


@admin_router.get("/api/traces/{trace_id}")
def trace_detail(trace_id: int) -> dict:
    item = tracestore.get(trace_id)
    return item or {"error": "该链路不存在或已被新记录挤出（仅保留最近 %d 条）" % tracestore.MAX_TRACES}
