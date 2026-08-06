"""共享 SQLite 运行时：管理 Checkpointer、业务存储及资源释放。

同一个 SQLite 文件中有两类数据，但用途不同：

- SqliteSaver（Checkpointer）：保存 LangGraph 的完整 State，属于短期会话记忆；
- EducationMemoryStore：保存聊天记录和提炼后的学习偏好/主题，属于业务与长期记忆。

EducationRuntime 把连接生命周期集中管理，Web 服务启动时打开、关闭时释放。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import TracebackType

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.education_agent import EducationAgent
from memory.store import EducationMemoryStore
from utils.config_handler import app_conf
from utils.path_tool import get_abs_path


class EducationRuntime:
    """统一管理共享的 SQLite 连接、Checkpointer、长期存储和 Agent。"""

    def __init__(self, database_path: str | Path | None = None):
        """解析数据库路径并准备运行时属性，但暂不建立数据库连接。"""
        configured_path = database_path or app_conf["memory"]["database_path"]
        self.database_path = Path(get_abs_path(configured_path))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection | None = None
        self.checkpointer: SqliteSaver | None = None
        self.memory_store: EducationMemoryStore | None = None
        self._agent: EducationAgent | None = None
        self._agent_lock = threading.Lock()

    def open(self) -> "EducationRuntime":
        """打开 SQLite，初始化短期 Checkpointer 和长期记忆业务存储。"""
        if self.connection is not None:
            # open() 可以被多处放心调用；已经打开时直接复用，不重复创建连接。
            return self
        self.connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )
        # WAL 允许“读”和“写”更好地并发；busy_timeout 让短暂写锁等待一会儿，
        # 而不是立即报 database is locked。
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()
        self.memory_store = EducationMemoryStore(
            self.database_path,
            max_memories=int(app_conf["memory"]["max_long_term_memories"]),
        )
        return self

    @property
    def agent(self) -> EducationAgent:
        """按需创建并复用唯一的 EducationAgent 实例。"""
        if self.connection is None:
            self.open()
        if self._agent is None:
            # Web 请求可能并发到达。双重检查 + 锁保证昂贵的 Agent 只创建一次。
            with self._agent_lock:
                if self._agent is None:
                    self._agent = EducationAgent(
                        checkpointer=self.checkpointer,
                        memory_store=self.memory_store,
                    )
        return self._agent

    def delete_conversation(self, *, student_id: str, thread_id: str) -> bool:
        """同时删除指定会话的业务数据、关联长期记忆和短期状态。"""
        if self.connection is None:
            self.open()
        deleted = self.memory_store.delete_conversation(
            student_id=student_id,
            thread_id=thread_id,
        )
        self.checkpointer.delete_thread(
            EducationAgent.checkpoint_thread_id(student_id, thread_id)
        )
        return deleted

    def delete_student_data(self, *, student_id: str) -> int:
        """清除指定学生的所有业务记忆及其每个线程的 Checkpoint。"""
        if self.connection is None:
            self.open()
        thread_ids = self.memory_store.delete_student_data(student_id=student_id)
        for thread_id in thread_ids:
            self.checkpointer.delete_thread(
                EducationAgent.checkpoint_thread_id(student_id, thread_id)
            )
        return len(thread_ids)

    def close(self) -> None:
        """释放 Agent、存储引用和底层 SQLite 连接。"""
        self._agent = None
        self.checkpointer = None
        self.memory_store = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "EducationRuntime":
        """进入上下文管理器时打开运行时并返回自身。"""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文管理器时关闭运行时资源。"""
        del exc_type, exc_value, traceback
        self.close()
