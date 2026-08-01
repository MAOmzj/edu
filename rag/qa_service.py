"""面向小学生的知识检索与直接 RAG 问答服务。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import create_chat_model
from rag.vector_store import EducationVectorStore
from utils.prompt_loader import load_rag_answer_prompt


class EducationQAService:
    def __init__(self, vector_store: EducationVectorStore | None = None):
        self._vector_store = vector_store
        self._chain = None

    @property
    def vector_store(self) -> EducationVectorStore:
        if self._vector_store is None:
            self._vector_store = EducationVectorStore()
        return self._vector_store

    @staticmethod
    def format_documents(documents: list[Document]) -> str:
        if not documents:
            return "没有检索到相关资料。"
        sections = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "未知来源")
            subject = document.metadata.get("subject", "综合")
            sections.append(
                f"【资料{index}｜{subject}｜{source}】\n{document.page_content.strip()}"
            )
        return "\n\n".join(sections)

    def search_context(self, question: str, subject: str = "综合") -> str:
        documents = self.vector_store.search(question, subject=subject)
        return self.format_documents(documents)

    def _get_chain(self):
        if self._chain is None:
            prompt = PromptTemplate.from_template(load_rag_answer_prompt())
            self._chain = prompt | create_chat_model() | StrOutputParser()
        return self._chain

    def answer(
        self,
        question: str,
        subject: str = "综合",
        grade: int | None = None,
    ) -> str:
        context = self.search_context(question, subject)
        grade_text = f"小学{grade}年级" if grade else "小学生（年级未指定）"
        return self._get_chain().invoke(
            {
                "question": question,
                "subject": subject,
                "grade": grade_text,
                "context": context,
            }
        )
