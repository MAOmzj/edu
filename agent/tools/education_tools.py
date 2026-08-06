"""教育 Agent 可调用的工具。

@tool 会读取函数名、参数类型和文档字符串，生成模型能理解的工具说明。
模型只是在对话中“申请调用”；LangChain 真正执行下面的 Python 函数并返回结果。
"""

from langchain_core.tools import tool

from rag.qa_service import EducationQAService
from utils.config_handler import app_conf
from utils.math_tool import safe_calculate


# 模块加载时只创建轻量外壳，内部向量库会在第一次 search_knowledge 时延迟创建。
qa_service = EducationQAService()


def normalize_subject(subject: str) -> str:
    """规范化学科名称，不受支持或为空时统一回退为“综合”。"""
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
