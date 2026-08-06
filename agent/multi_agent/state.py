"""多 Agent 共用的数据结构。

可以把 State 理解为三个子 Agent 之间传递的“工作单”：学科 Agent 写草稿，
审核 Agent 写审核结果，记忆 Agent 写需要长期保存的学习信息。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


MemoryType = Literal[
    "profile",
    "preference",
    "learning_topic",
    "learning_difficulty",
]
ModelType = TypeVar("ModelType", bound=BaseModel)


class ReviewResult(BaseModel):
    """审核 Agent 必须返回的固定格式，避免用字符串猜测是否通过。"""

    approved: bool = Field(description="答案是否可以直接交给学生")
    score: int = Field(ge=0, le=100, description="答案质量分，范围 0 到 100")
    feedback: str = Field(description="给学科 Agent 的明确修改意见")
    corrected_answer: str | None = Field(
        default=None,
        description="多次审核仍不通过时可使用的安全修正版",
    )


class MemoryCandidate(BaseModel):
    """记忆 Agent 提取的一条长期学习记忆候选。"""

    memory_type: MemoryType = Field(description="长期记忆类型")
    subject: str = Field(description="记忆所属学科")
    content: str = Field(min_length=1, max_length=180, description="简短记忆内容")
    confidence: float = Field(ge=0, le=1, description="提取结果的可信程度")


class MemoryExtraction(BaseModel):
    """记忆 Agent 的完整输出；每轮最多保存五条候选记忆。"""

    memories: list[MemoryCandidate] = Field(default_factory=list, max_length=5)

#定义state
class EducationWorkflowState(TypedDict, total=False):
    """主协调工作流使用的共享 State。

    TypedDict 只是在告诉编辑器“字典里有哪些键”，运行时仍然是普通 dict。
    ``total=False`` 表示初始 State 不必一次拥有所有字段，后续节点可以逐步补齐。
    """

    # Annotated[..., add_messages] 指定了 messages 的“合并规则”：节点返回新消息时
    # 是追加而不是覆盖。Checkpointer 保存合并后的 State，所以能恢复多轮对话。
    messages: Annotated[list[AnyMessage], add_messages]

    # 本轮请求的基本信息。
    student_id: str
    thread_id: str
    turn_id: str
    question: str
    subject: str
    grade: int | None

    # 上下文管理器为学科 Agent 准备的信息。
    conversation_summary: str
    long_term_context: str
    conversation_context: str
    skill_context: str

    # 学科 Agent 和审核 Agent 依次写入的中间结果。
    draft_answer: str
    approved: bool
    review_score: int
    review_feedback: str
    corrected_answer: str
    review_attempts: int

    # 最终回答和记忆 Agent 提取的候选记忆。
    final_answer: str
    memory_candidates: list[dict[str, Any]]


def message_content_to_text(content: Any) -> str:
    """把模型的字符串或文本内容块统一转换成普通字符串。"""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def final_ai_text(result: dict[str, Any]) -> str:
    """从 Agent 运行结果中找到最后一条不是工具调用的 AI 正文。"""

    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = message_content_to_text(message.content)
            if text:
                return text
    raise RuntimeError("子 Agent 没有返回最终文本")


def parse_json_model(text: str, model_type: type[ModelType]) -> ModelType:
    """从纯 JSON 或 Markdown JSON 代码块中解析并校验 Pydantic 对象。
        
    """
    #把模型吐出来的文本 一步步收拾成干净的JSON？  
    cleaned = text.strip()
    #把第一行 ```json 整行删掉
    if cleaned.startswith("```"):
        # 兼容模型偶尔返回的 ```json ... ``` 包装。
        first_newline = cleaned.find("\n")
        if first_newline >= 0:
            cleaned = cleaned[first_newline + 1 :]  #只保留 ```json后
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # 如果 JSON 前后带有一句说明，只截取最外层花括号之间的内容。
    start = cleaned.find("{")
    end = cleaned.rfind("}")  #.rfind("}")	r = reverse，从右边开始找，返回最后一个 } 的下标
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return model_type.model_validate_json(cleaned)
