# 文件用途：统一管理 SQLite 连接、LangGraph Checkpointer、长期记忆存储和 Agent 生命周期。
#
# 调用关系：（谁调用我，上游）
#   - app.py:68             创建：EducationRuntime(...).open()（Web 服务启动时）
#   - app.py:105            get_runtime()：每个请求取同一个运行时单例
#   - app.py:143/186/235/254/270/286/300  Web 接口经 runtime 访问 Agent 和存储
#   - main.py:140           with EducationRuntime() as runtime:（CLI 对话）
#   - main.py:143           runtime.agent.answer(...)
#   - tests                 测试
#
# 调用关系：（我调用谁，下游）
#   - agent/education_agent.py  EducationAgent（唯一创建者，见第 74 行）
#   - memory/store.py           EducationMemoryStore（第 59 行创建）
#   - langgraph 的 SqliteSaver   短期 Checkpointer（第 57 行，第三方库）
#   - utils/config_handler.py   app_conf（数据库路径、记忆上限）
#   - utils/path_tool.py         get_abs_path（解析相对路径）
#
# 修改易踩坑：同一连接要允许跨线程并设置 WAL；删除会话时业务历史与对应 Checkpoint 必须一起清理。
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


# 上游调用者：app.py:68（Web）和 main.py:140（CLI）创建本运行时。
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
        # 调用下游：langgraph 的 SqliteSaver（第三方 Checkpointer，存短期会话 State）。
        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()
        # 调用下游：memory/store.py 的 EducationMemoryStore（业务历史+长期记忆）。
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
                    # 调用下游：agent/education_agent.py 的 EducationAgent（唯一创建者）。
                    self._agent = EducationAgent(
                        checkpointer=self.checkpointer,
                        memory_store=self.memory_store,
                    )
        return self._agent

    def delete_conversation(self, *, student_id: str, thread_id: str) -> bool:
        """同时删除指定会话的业务数据、关联长期记忆和短期状态。"""
        if self.connection is None:
            self.open()
        # 调用下游：store.py 的 delete_conversation() 删除业务数据+关联长期记忆。
        deleted = self.memory_store.delete_conversation(
            student_id=student_id,
            thread_id=thread_id,
        )
        # 调用下游：SqliteSaver.delete_thread() 清掉对应的短期 Checkpoint。
        self.checkpointer.delete_thread(
            EducationAgent.checkpoint_thread_id(student_id, thread_id)
        )
        return deleted

    def delete_student_data(self, *, student_id: str) -> int:
        """清除指定学生的所有业务记忆及其每个线程的 Checkpoint。"""
        if self.connection is None:
            self.open()
        # 调用下游：store.py 的 delete_student_data() 清除业务记忆并返回线程列表。
        thread_ids = self.memory_store.delete_student_data(student_id=student_id)
        for thread_id in thread_ids:
            # 调用下游：SqliteSaver.delete_thread() 逐个清掉短期 Checkpoint。
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

    # 被 with 语句使用（main.py:140、app.py:68）：进入时打开运行时。
    def __enter__(self) -> "EducationRuntime":
        """进入上下文管理器时打开运行时并返回自身。"""
        return self.open()

    # 被 with 语句使用（main.py:140、app.py:73）：退出时关闭资源。
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文管理器时关闭运行时资源。"""
        del exc_type, exc_value, traceback
        self.close()
