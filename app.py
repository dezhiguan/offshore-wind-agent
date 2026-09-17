# -*- coding: utf-8 -*-
"""FastAPI 入口。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 必须在导入 agent.loop 之前，否则读不到 .env 里的模型配置

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from admin import admin_router  # noqa: E402
from agent import grounding, replay, tracestore  # noqa: E402
from agent.composer import compose  # noqa: E402
from agent.loop import AgentRunFailed, LlmNotConfigured, run_agent, run_agent_stream  # noqa: E402
from tools.db import DB_PATH, query_db  # noqa: E402
from tools.retriever import get_index  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="海上风电机组维检 Agent", version="1.0.0")

# 后台管理页面（Agent 质量中心、链路追踪）单独成文件，运行上同进程同端口
app.include_router(admin_router)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    idx = get_index()
    probe = query_db("SELECT COUNT(*) AS n FROM alarm_records")
    return {
        "ok": probe["ok"],
        "db": {"path": DB_PATH.name, "alarm_rows": probe["rows"][0]["n"] if probe["ok"] else None},
        "docs": {
            "chunks": len(idx.chunks),
            "fault_codes": len(idx.code_map),
        },
        "mode": {
            "replay": replay.enabled(),
            "grounding": grounding.MODE,
            "model": os.getenv("LLM_MODEL", "qwen3.8-flash"),
            "thinking": os.getenv("ENABLE_THINKING", "false"),
        },
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if replay.enabled():
        cached = replay.find(req.question)
        if cached:
            return {"ok": True, **cached}
        return {"ok": False, "kind": "replay_miss",
                "error": "离线回放模式下没有与该问题匹配的记录。"
                         "可改用示例问题，或关闭 REPLAY_MODE 走实时链路。"}
    try:
        run = run_agent(req.question)
    except LlmNotConfigured as exc:
        return {"ok": False, "error": str(exc), "kind": "not_configured"}
    except AgentRunFailed as exc:
        # 模型超时 / 网络异常，界面要给明确提示而不是白屏；
        # 同时把这条失败链路留存下来，否则后台的成功率里没有分母
        tracestore.record(req.question, exc.partial, source="online", error=str(exc.cause))
        return {"ok": False, "error": "模型调用失败：%s" % exc.cause, "kind": "llm_error"}
    except Exception as exc:
        return {"ok": False, "error": "模型调用失败：%s" % exc, "kind": "llm_error"}
    result = grounding.apply(compose(req.question, run))
    tracestore.record(req.question, result, source="online")
    return {"ok": True, **result}


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE 流式问答：每完成一次工具调用即推送一条事件，界面实时渲染轨迹。

    非流式的 /ask 保留，供回归跑测与离线回放使用。
    """
    def events():
        def sse(payload: dict) -> str:
            return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False, default=str)

        if replay.enabled():
            cached = replay.find(req.question)
            if cached:
                # 回放也逐步推，让现场演示的节奏与实时链路一致
                for entry in cached.get("trace", []):
                    yield sse({"type": "step", "step": entry})
                yield sse({"type": "done", "result": {"ok": True, **cached}})
            else:
                yield sse({"type": "error", "kind": "replay_miss",
                           "error": "离线回放模式下没有与该问题匹配的记录。"
                                    "可改用示例问题，或关闭 REPLAY_MODE 走实时链路。"})
            return

        try:
            for event in run_agent_stream(req.question):
                if event["type"] == "done":
                    result = grounding.apply(compose(req.question, event["result"]))
                    tracestore.record(req.question, result, source="online")
                    yield sse({"type": "done", "result": {"ok": True, **result}})
                else:
                    yield sse(event)
        except LlmNotConfigured as exc:
            yield sse({"type": "error", "kind": "not_configured", "error": str(exc)})
        except AgentRunFailed as exc:
            tracestore.record(req.question, exc.partial, source="online", error=str(exc.cause))
            yield sse({"type": "error", "kind": "llm_error",
                       "error": "模型调用失败：%s" % exc.cause})
        except Exception as exc:
            yield sse({"type": "error", "kind": "llm_error", "error": "模型调用失败：%s" % exc})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _port_free(port: int) -> bool:
    """先探端口再启动。

    直接交给 uvicorn 的话，报错行会夹在启动日志中间，而成功提示反而打在最后，
    看上去像起来了其实没有。
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


if __name__ == "__main__":
    # 直接运行本文件即可启动（IDE 里点 Run 也走这里）。
    # 命令行等价写法：uvicorn app:app --port 8000
    import sys

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    if not _port_free(port):
        print("端口 %d 已被占用，服务未启动。" % port)
        print("  查看占用：lsof -nP -iTCP:%d -sTCP:LISTEN" % port)
        print("  换个端口：PORT=%d python app.py" % (port + 1))
        sys.exit(1)

    base = "http://127.0.0.1:%d" % port
    # flush=True：stdout 被重定向（IDE 控制台、日志文件）时 Python 会缓冲，
    # 不强制刷新的话这段地址要等到进程退出才出现，等于没打印。
    print("\n  海上风电机组维检 Agent\n"
          "  问答      →  %s\n"
          "  后台管理  →  %s/admin    Agent 质量中心与链路追踪（左侧切换）\n" % (base, base),
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
