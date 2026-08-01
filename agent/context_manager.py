"""为每次模型调用选择相关的长期学习记忆。"""

from __future__ import annotations

from memory.store import EducationMemoryStore


class EducationContextManager:
    def __init__(self, memory_store: EducationMemoryStore | None, limit: int = 5):
        self.memory_store = memory_store
        self.limit = limit

    def build_long_term_context(
        self,
        *,
        student_id: str,
        question: str,
        subject: str,
    ) -> str:
        if self.memory_store is None:
            return "无"
        memories = self.memory_store.retrieve(
            student_id=student_id,
            question=question,
            subject=subject,
            limit=self.limit,
        )
        if not memories:
            return "无"
        return "\n".join(
            f"- [{memory.memory_type}/{memory.subject}] {memory.content}"
            for memory in memories
        )
