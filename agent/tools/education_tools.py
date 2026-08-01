"""教育 Agent 可调用的工具。"""

from langchain_core.tools import tool

from rag.qa_service import EducationQAService
from utils.config_handler import app_conf
from utils.math_tool import safe_calculate


qa_service = EducationQAService()


def normalize_subject(subject: str) -> str:
    value = (subject or "综合").strip()
    supported = set(app_conf["supported_subjects"])
    return value if value in supported else "综合"


@tool
def search_knowledge(question: str, subject: str = "综合") -> str:
    """从小学语文、数学、英语、科学、安全与品德知识库检索资料。

    Args:
        question: 学生提出的完整问题或适合检索的关键词。
        subject: 语文、数学、英语、科学、安全与品德或综合。
    """
    return qa_service.search_context(question, normalize_subject(subject))


@tool
def calculate_expression(expression: str) -> str:
    """准确计算基础算术表达式，适用于加减乘除、括号、余数和乘方。

    Args:
        expression: 仅包含数字和数学运算符的算式，例如 (25+15)*3。
    """
    result = safe_calculate(expression)
    return f"计算结果：{result}"
