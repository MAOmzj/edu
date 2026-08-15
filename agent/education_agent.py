# 文件用途：提供 EducationAgent.answer() 稳定入口，内部运行“学科回答—审核—记忆”三子 Agent 工作流。
# 调用关系：memory/runtime.py 创建本类，app.py 与 main.py 调用；本文件调用 multi_agent 工具构造器、workflow、memory/store 和 model/factory。
# 修改易踩坑：answer() 是对外唯一契约；Checkpoint 线程编号必须拼接 student_id 与 thread_id，不能只用其中一个。
"""小学生教育知识问答 Agent 的对外入口。

外部代码只需要认识 EducationAgent.answer()。这个“门面类”会在内部完成：

    参数校验 -> 登记会话 -> 准备 State -> 运行主协调图 -> 取出最终答案

它本身不编写答案；学科、审核和记忆三个子 Agent 的先后顺序由 workflow.py 决定。

上游调用者：memory/runtime.py（唯一创建者）、app.py（Web）、main.py（CLI）、测试。
下游调用者：三个 build_* 子 Agent 工具、workflow.py 主协调图、memory/store.py、
model/factory.py、utils/config_handler.py（详见各方法行内注释）。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agent.multi_agent import (
    EducationMultiAgentWorkflow,
    build_memory_agent_tool,
    build_review_agent_tool,
    build_subject_agent_tool,
)
from agent.multi_agent.state import message_content_to_text
from memory.store import EducationMemoryStore
from model.factory import create_chat_model
from utils.config_handler import app_conf


class EducationAgent:
    """对外提供稳定 answer 接口，内部运行三个子 Agent Tool。"""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        memory_store: EducationMemoryStore | None = None,
        *,
        subject_tool: BaseTool | None = None,
        review_tool: BaseTool | None = None,
        memory_tool: BaseTool | None = None,
    ):
        """创建三个子 Agent Tool，并把它们交给主协调工作流。"""

        self.memory_store = memory_store

        # 正常运行时三个 Tool 都需要同一个聊天模型。测试可以直接传入假 Tool，
        # 这样不需要 API Key，也不会访问在线模型。
        # 调用下游：model/factory.py 的 create_chat_model()（底层是通义 qwen3-max）。
        if subject_tool is None or review_tool is None or memory_tool is None:
            model = model or create_chat_model()

        # 这里体现了“把子 Agent 当作 Tool”：build_* 先创建一个独立子 Agent，
        # 再包成具有固定入参/出参的 Tool，主协调器只调用 Tool，不接触内部消息。
        # 调用下游：agent/multi_agent/subject_agent.py、review_agent.py、memory_agent.py。
        self.subject_tool = subject_tool or build_subject_agent_tool(model)
        self.review_tool = review_tool or build_review_agent_tool(model)
        self.memory_tool = memory_tool or build_memory_agent_tool(model)

        # Checkpointer 保存 LangGraph State（短期、多轮），memory_store 保存可查询的
        # 业务历史和跨会话长期记忆。两者职责不同，但正式运行时都落到 SQLite。
        # 调用下游：agent/multi_agent/workflow.py 编译出主协调图（后续 answer() 调它）。
        self.workflow = EducationMultiAgentWorkflow(
            subject_tool=self.subject_tool,
            review_tool=self.review_tool,
            memory_tool=self.memory_tool,
            memory_store=memory_store,
            checkpointer=checkpointer or InMemorySaver(),
            retrieval_limit=int(app_conf["memory"]["retrieval_limit"]),
            recent_message_limit=int(
                app_conf["context"]["recent_message_limit"]
            ),
            max_conversation_chars=int(
                app_conf["context"]["max_conversation_chars"]
            ),
            max_long_term_chars=int(
                app_conf["context"]["max_long_term_chars"]
            ),
            max_review_feedback_chars=int(
                app_conf["context"]["max_review_feedback_chars"]
            ),
            summary_model=model,
            summary_enabled=bool(app_conf["context"]["summary_enabled"]),
            summary_trigger_messages=int(
                app_conf["context"]["summary_trigger_messages"]
            ),
            summary_keep_recent_messages=int(
                app_conf["context"]["summary_keep_recent_messages"]
            ),
            summary_trigger_chars=int(
                app_conf["context"]["summary_trigger_chars"]
            ),
            max_summary_chars=int(
                app_conf["context"]["max_summary_chars"]
            ),
            max_review_retries=int(
                app_conf["agent"].get("max_review_retries", 1)
            ),
        )

        # 保留 agent 属性，方便旧代码或调试代码继续调用底层 invoke/get_state。
        self.agent = self.workflow.graph
        self.context_manager = self.workflow.context_manager

    @staticmethod
    def checkpoint_thread_id(student_id: str, thread_id: str) -> str:
        """组合学生和会话编号，避免不同学生误用同一个 Checkpoint。"""

        return f"{student_id}:{thread_id}"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """兼容旧调用，把模型内容统一转换成普通文本。"""

        return message_content_to_text(content)

    def answer(
        self,
        question: str,
        subject: str = "综合",
        grade: int | None = None,
        *,
        student_id: str,
        thread_id: str,
    ) -> str:
        """校验问题，执行完整多 Agent 工作流并返回最终审核答案。"""

        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        if grade is not None and grade not in range(1, 7):
            raise ValueError("年级必须是 1 到 6")
        if not student_id or not thread_id:
            raise ValueError("student_id 和 thread_id 不能为空")

        supported = set(app_conf["supported_subjects"])
        normalized_subject = subject if subject in supported else "综合"

        # 先登记会话。即使模型中途失败，删除会话接口仍能找到并清理 Checkpoint。
        # 调用下游：memory/store.py 的 EducationMemoryStore.ensure_conversation()。
        if self.memory_store is not None:
            self.memory_store.ensure_conversation(
                student_id=student_id,
                thread_id=thread_id,
                question=question,
                subject=normalized_subject,
                grade=grade,
            )

        # turn_id 由服务器生成，同一轮发生节点重试时仍保持不变，防止重复入库。
        turn_id = f"turn-{uuid4().hex}"
        # 调用下游：agent/multi_agent/workflow.py 的主协调图（invoke 跑完整张图）。
        # 传入的字典就是本轮初始 State。图中每个节点只返回自己修改的字段，
        # LangGraph 会把这些局部结果逐步合并到同一份 State 中。
        result = self.workflow.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "student_id": student_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "question": question,
                "subject": normalized_subject,
                "grade": grade,
            },
            config={
                "recursion_limit": int(app_conf["agent"]["recursion_limit"]),
                "configurable": {
                    # Checkpointer 以这个 thread_id 区分短期会话。这里同时拼入
                    # student_id，防止两个学生碰巧使用相同会话编号而串话。
                    "thread_id": self.checkpoint_thread_id(student_id, thread_id)
                },
            },
        )

        answer = str(result.get("final_answer", "")).strip()
        if not answer:
            raise RuntimeError("多 Agent 工作流没有返回最终答案")
        return answer
