"""小学生教育知识问答 Agent。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agent.context_manager import EducationContextManager
from agent.prompt_builder import EducationPromptBuilder
from agent.tools.education_tools import calculate_expression, search_knowledge
from agent.tools.middleware import log_before_model, monitor_tool
from memory.store import EducationMemoryStore
from model.factory import create_chat_model
from utils.config_handler import app_conf
from utils.prompt_loader import load_system_prompt


class EducationAgent:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        memory_store: EducationMemoryStore | None = None,
    ):
        self.memory_store = memory_store
        self.context_manager = EducationContextManager(
            memory_store,
            limit=int(app_conf["memory"]["retrieval_limit"]),
        )
        self.prompt_builder = EducationPromptBuilder()
        self.agent = create_agent(
            model=model or create_chat_model(),
            system_prompt=load_system_prompt(),
            tools=[search_knowledge, calculate_expression],
            middleware=[monitor_tool, log_before_model],
            checkpointer=checkpointer or InMemorySaver(),
        )

    @staticmethod
    def checkpoint_thread_id(student_id: str, thread_id: str) -> str:
        """将匿名学生和会话组合，避免不同学生误用同一 checkpoint。"""
        return f"{student_id}:{thread_id}"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts).strip()
        return str(content).strip()

    def answer(
        self,
        question: str,
        subject: str = "综合",
        grade: int | None = None,
        *,
        student_id: str,
        thread_id: str,
    ) -> str:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        if grade is not None and grade not in range(1, 7):
            raise ValueError("年级必须是 1 到 6")
        if not student_id or not thread_id:
            raise ValueError("student_id 和 thread_id 不能为空")

        supported = set(app_conf["supported_subjects"])
        normalized_subject = subject if subject in supported else "综合"
        long_term_context = self.context_manager.build_long_term_context(
            student_id=student_id,
            question=question,
            subject=normalized_subject,
        )
        user_message = self.prompt_builder.build_user_message(
            question=question,
            subject=normalized_subject,
            grade=grade,
            long_term_context=long_term_context,
        )
        if self.memory_store is not None:
            self.memory_store.ensure_conversation(
                student_id=student_id,
                thread_id=thread_id,
                question=question,
                subject=normalized_subject,
                grade=grade,
            )
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "recursion_limit": int(app_conf["agent"]["recursion_limit"]),
                "configurable": {
                    "thread_id": self.checkpoint_thread_id(student_id, thread_id)
                },
            },
        )
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = self._content_to_text(message.content)
                if answer:
                    if self.memory_store is not None:
                        self.memory_store.record_turn(
                            student_id=student_id,
                            thread_id=thread_id,
                            question=question,
                            answer=answer,
                            subject=normalized_subject,
                            grade=grade,
                        )
                    return answer
        raise RuntimeError("模型没有返回最终答案")
