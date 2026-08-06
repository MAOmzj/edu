"""面向小学生的知识检索与直接 RAG 问答服务。

当前多 Agent 主流程主要调用 search_context()，把它作为 search_knowledge Tool。
answer() 保留了一条较简单的“检索 -> Prompt -> 模型”直接 RAG 路径，可用于独立调用。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import create_chat_model
from rag.vector_store import EducationVectorStore
from utils.prompt_loader import load_rag_answer_prompt


class EducationQAService:
    """把底层向量库包装成“取资料”和“直接生成回答”两个易用接口。"""

    def __init__(self, vector_store: EducationVectorStore | None = None):
        """接收可选向量库，并为向量库和 RAG 链保留延迟初始化状态。"""
        self._vector_store = vector_store
        self._chain = None

    @property
    def vector_store(self) -> EducationVectorStore:
        """按需创建并复用教育知识向量库实例。"""
        if self._vector_store is None:
            self._vector_store = EducationVectorStore()
        return self._vector_store

    @staticmethod
    def format_documents(documents: list[Document]) -> str:
        """把检索文档整理成包含序号、学科和来源的提示词上下文。"""
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
        """按问题和学科检索知识库，并返回已经格式化的参考资料。"""
        documents = self.vector_store.search(question, subject=subject)
        return self.format_documents(documents)

    def _get_chain(self):
        """按需创建并缓存由提示词、聊天模型和文本解析器组成的 RAG 链。"""
        if self._chain is None:
            prompt = PromptTemplate.from_template(load_rag_answer_prompt())
            # LCEL 的 | 表示流水线：字典先填入 Prompt，再交给模型，最后只取文本。
            self._chain = prompt | create_chat_model() | StrOutputParser()
        return self._chain

    def answer(
        self,
        question: str,
        subject: str = "综合",
        grade: int | None = None,
    ) -> str:
        """检索相关资料并调用 RAG 链，生成适合指定年级的回答。"""
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
