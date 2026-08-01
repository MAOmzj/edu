"""小学教育知识问答系统 Web 服务。"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
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
_agent_override: EducationAgent | None = None


def get_memory_database_path() -> Path:
    configured = os.getenv(
        "EDUCATION_MEMORY_DB_PATH",
        app_conf["memory"]["database_path"],
    )
    return Path(get_abs_path(configured))


@asynccontextmanager
async def lifespan(application: FastAPI):
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
    question: str = Field(min_length=1, max_length=500)
    subject: str = "综合"
    grade: int | None = Field(default=None, ge=1, le=6)
    student_id: str = Field(min_length=8, max_length=80, pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(min_length=8, max_length=80, pattern=IDENTIFIER_PATTERN)


class AskResponse(BaseModel):
    answer: str
    subject: str
    grade: int | None
    student_id: str
    thread_id: str


def get_runtime(request: Request) -> EducationRuntime:
    runtime = getattr(request.app.state, "education_runtime", None)
    if runtime is None:
        raise RuntimeError("记忆运行时尚未初始化")
    return runtime


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict[str, object]:
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
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    if payload.subject not in SUPPORTED_SUBJECTS:
        raise HTTPException(status_code=422, detail="不支持这个学科")

    runtime = get_runtime(request)
    education_agent = _agent_override or runtime.agent
    try:
        answer = await run_in_threadpool(
            education_agent.answer,
            question,
            payload.subject,
            payload.grade,
            student_id=payload.student_id,
            thread_id=payload.thread_id,
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("网页问答失败")
        raise HTTPException(
            status_code=500,
            detail="回答暂时失败，请稍后重试或检查服务日志。",
        ) from exc

    return AskResponse(
        answer=answer,
        subject=payload.subject,
        grade=payload.grade,
        student_id=payload.student_id,
        thread_id=payload.thread_id,
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
    if not re.fullmatch(IDENTIFIER_PATTERN, student_id):
        raise HTTPException(status_code=422, detail="匿名学生 ID 格式不正确")
    deleted_threads = get_runtime(request).delete_student_data(
        student_id=student_id
    )
    return {"deleted": True, "deleted_threads": deleted_threads}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8766, reload=False)
