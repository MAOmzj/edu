"""小学教育知识问答系统 Web 服务。

给初学者的阅读提示：这个文件只负责“接收 HTTP 请求和返回 HTTP 响应”，
真正的回答逻辑在 :class:`agent.education_agent.EducationAgent` 中。

一次网页提问的大致路线是：

    浏览器 POST /api/ask
        -> FastAPI 把 JSON 校验成 AskRequest
        -> EducationRuntime 提供共享的 Agent 和 SQLite 连接
        -> EducationAgent.answer() 执行多 Agent 工作流
        -> AskResponse 把最终答案转换成 JSON 返回浏览器

把 Web 层和 Agent 层分开后，将来换前端或增加命令行入口时，不需要重写问答逻辑。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.education_agent import EducationAgent
from memory.runtime import EducationRuntime
from rag.vector_store import EducationVectorStore
from utils.config_handler import app_conf
from utils.logger_handler import logger
from utils.path_tool import PROJECT_ROOT, get_abs_path


WEB_ROOT = Path(PROJECT_ROOT) / "web"
SUPPORTED_SUBJECTS = tuple(app_conf["supported_subjects"])
IDENTIFIER_PATTERN = r"^[A-Za-z0-9_-]{8,80}$"
FINAL_ANSWER_CHUNK_SIZE = 8
FINAL_ANSWER_CHUNK_DELAY_SECONDS = 0.02
# 测试可以临时放入一个假 Agent，正常运行时它始终为 None。
# 这样 Web API 测试不需要真的调用 DeepSeek，线上代码仍使用 runtime.agent。
_agent_override: EducationAgent | None = None


def get_memory_database_path() -> Path:
    """读取环境变量或应用配置，返回长期记忆数据库的绝对路径。"""
    configured = os.getenv(
        "EDUCATION_MEMORY_DB_PATH",
        app_conf["memory"]["database_path"],
    )
    return Path(get_abs_path(configured))


@asynccontextmanager
async def lifespan(application: FastAPI):
    """在 Web 服务启动时打开记忆运行时，并在服务关闭时释放资源。"""
    # lifespan 可以理解为 Web 服务的“开机/关机钩子”：yield 之前是开机，
    # yield 之后是关机。运行时只创建一次，不能在每个请求里反复开关数据库。
    runtime = EducationRuntime(get_memory_database_path()).open()
    application.state.education_runtime = runtime
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(
    title=app_conf["name"],
    description="面向小学一至六年级的教育知识问答接口",
    version="1.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


class AskRequest(BaseModel):
    """前端必须提交的字段；Pydantic 会在进入 ask() 前自动完成基础校验。"""

    question: str = Field(min_length=1, max_length=500)
    subject: str = "综合"
    grade: int | None = Field(default=None, ge=1, le=6)
    student_id: str = Field(min_length=8, max_length=80, pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(min_length=8, max_length=80, pattern=IDENTIFIER_PATTERN)


class AskResponse(BaseModel):
    """问答接口固定返回的 JSON 结构，避免前后端随意猜字段名。"""

    answer: str
    subject: str
    grade: int | None
    student_id: str
    thread_id: str


def get_runtime(request: Request) -> EducationRuntime:
    """从 FastAPI 应用状态中取得已初始化的教育记忆运行时。"""
    runtime = getattr(request.app.state, "education_runtime", None)
    if runtime is None:
        raise RuntimeError("记忆运行时尚未初始化")
    return runtime


def split_final_answer(answer: str, chunk_size: int = FINAL_ANSWER_CHUNK_SIZE):
    """把已经审核完成的最终答案切成小段，供前端逐段展示。"""

    if chunk_size < 1:
        raise ValueError("流式分段大小必须大于 0")
    for start in range(0, len(answer), chunk_size):
        yield answer[start : start + chunk_size]


async def stream_final_answer(answer: str) -> AsyncIterator[str]:
    """只发送最终答案正文，不发送草稿、审核意见或内部工作流状态。"""

    for chunk in split_final_answer(answer):
        payload = json.dumps({"text": chunk}, ensure_ascii=False)
        yield f"event: answer_delta\ndata: {payload}\n\n"
        # 很短的让步让浏览器有机会逐段绘制，也避免一次把所有片段合并显示。
        await asyncio.sleep(FINAL_ANSWER_CHUNK_DELAY_SECONDS)
    yield 'event: done\ndata: {"completed": true}\n\n'


async def run_education_answer(payload: AskRequest, request: Request) -> str:
    """复用同一套校验和异常处理，生成已经审核、保存完成的最终答案。"""

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    if payload.subject not in SUPPORTED_SUBJECTS:
        raise HTTPException(status_code=422, detail="不支持这个学科")

    try:
        runtime = get_runtime(request)
        education_agent = _agent_override or runtime.agent
        # Agent、SQLite 和模型 SDK 都是普通同步代码。run_in_threadpool 会把它们
        # 放到工作线程执行，避免一次较慢的模型请求卡住 FastAPI 的异步事件循环。
        return await run_in_threadpool(
            education_agent.answer,
            question,
            payload.subject,
            payload.grade,
            student_id=payload.student_id,
            thread_id=payload.thread_id,
        )
    except EnvironmentError as exc:
        # 例如没有配置模型 Key，这属于服务暂不可用，所以返回 503。
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # 例如学科或年级不合法，这属于调用方参数错误，所以返回 422。
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("网页问答失败")
        raise HTTPException(
            status_code=500,
            detail="回答暂时失败，请稍后重试或检查服务日志。",
        ) from exc


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """返回小学知识问答系统的前端首页文件。"""
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    """返回服务、知识库索引和 SQLite 记忆模块的健康状态。"""
    runtime = get_runtime(request)
    return {
        "status": "ok",
        "name": app_conf["name"],
        "subjects": SUPPORTED_SUBJECTS,
        "index": EducationVectorStore().status(),
        "memory": {
            "backend": "sqlite",
            **runtime.memory_store.stats(),
        },
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """校验学生问题，在线程池中调用 Agent，并返回结构化问答结果。"""
    answer = await run_education_answer(payload, request)

    return AskResponse(
        answer=answer,
        subject=payload.subject,
        grade=payload.grade,
        student_id=payload.student_id,
        thread_id=payload.thread_id,
    )


@app.post("/api/ask/stream", response_class=StreamingResponse)
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """工作流全部结束后，仅将已经审核的最终答案按 SSE 小段发送给前端。"""

    # 必须先等待 answer() 完整结束。此时审核、必要重写、长期记忆保存都已完成，
    # 所以任何学科 Agent 草稿和审核意见都不会提前泄露给学生。
    answer = await run_education_answer(payload, request)
    return StreamingResponse(
        stream_final_answer(answer),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store",
            # 告诉常见反向代理不要缓冲，否则浏览器可能最后一次性看到全文。
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history/{thread_id}")
def history(
    thread_id: str,
    request: Request,
    student_id: str = Query(
        min_length=8,
        max_length=80,
        pattern=IDENTIFIER_PATTERN,
    ),
) -> dict[str, object]:
    """读取指定匿名学生在某个会话中的完整消息历史。"""
    try:
        messages = get_runtime(request).memory_store.get_history(
            student_id=student_id,
            thread_id=thread_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"thread_id": thread_id, "messages": messages}


@app.get("/api/conversations")
def conversations(
    request: Request,
    student_id: str = Query(
        min_length=8,
        max_length=80,
        pattern=IDENTIFIER_PATTERN,
    ),
) -> dict[str, object]:
    """列出指定匿名学生最近使用过的会话。"""
    items = get_runtime(request).memory_store.list_conversations(
        student_id=student_id
    )
    return {"conversations": items}


@app.get("/api/memories")
def memories(
    request: Request,
    student_id: str = Query(
        min_length=8,
        max_length=80,
        pattern=IDENTIFIER_PATTERN,
    ),
) -> dict[str, object]:
    """列出指定匿名学生已经形成的长期学习记忆。"""
    items = get_runtime(request).memory_store.list_memories(student_id=student_id)
    return {"memories": items}


@app.delete("/api/conversations/{thread_id}")
def delete_conversation(
    thread_id: str,
    request: Request,
    student_id: str = Query(
        min_length=8,
        max_length=80,
        pattern=IDENTIFIER_PATTERN,
    ),
) -> dict[str, object]:
    """删除指定会话及其 Checkpoint 和会话关联记忆。"""
    try:
        deleted = get_runtime(request).delete_conversation(
            student_id=student_id,
            thread_id=thread_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"deleted": deleted}


@app.delete("/api/students/{student_id}/memory")
def delete_student_memory(student_id: str, request: Request) -> dict[str, object]:
    """校验学生编号并清除该学生的全部会话、短期状态和长期记忆。"""
    if not re.fullmatch(IDENTIFIER_PATTERN, student_id):
        raise HTTPException(status_code=422, detail="匿名学生 ID 格式不正确")
    deleted_threads = get_runtime(request).delete_student_data(
        student_id=student_id
    )
    return {"deleted": True, "deleted_threads": deleted_threads}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8766, reload=False)
