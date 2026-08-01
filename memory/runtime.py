"""共享 SQLite 运行时：管理 Checkpointer、业务存储及资源释放。"""

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
    def __init__(self, database_path: str | Path | None = None):
        configured_path = database_path or app_conf["memory"]["database_path"]
        self.database_path = Path(get_abs_path(configured_path))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection | None = None
        self.checkpointer: SqliteSaver | None = None
        self.memory_store: EducationMemoryStore | None = None
        self._agent: EducationAgent | None = None
        self._agent_lock = threading.Lock()

    def open(self) -> "EducationRuntime":
        if self.connection is not None:
            return self
        self.connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )
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
        if self.connection is None:
            self.open()
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    self._agent = EducationAgent(
                        checkpointer=self.checkpointer,
                        memory_store=self.memory_store,
                    )
        return self._agent

    def delete_conversation(self, *, student_id: str, thread_id: str) -> bool:
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
        if self.connection is None:
            self.open()
        thread_ids = self.memory_store.delete_student_data(student_id=student_id)
        for thread_id in thread_ids:
            self.checkpointer.delete_thread(
                EducationAgent.checkpoint_thread_id(student_id, thread_id)
            )
        return len(thread_ids)

    def close(self) -> None:
        self._agent = None
        self.checkpointer = None
        self.memory_store = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "EducationRuntime":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
