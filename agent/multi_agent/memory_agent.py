# 文件用途：从学生原话提取结构化长期学习记忆候选，并包装成子 Agent Tool。
# 调用关系：education_agent.py 创建本 Tool，workflow.py 调用它；它调用聊天模型但不直接写数据库。
# 修改易踩坑：不能根据助手答案猜测掌握程度，输出必须通过 MemoryExtraction 且不得保存身份隐私。
"""学习记忆子 Agent，并将记忆提取能力包装成 Tool。

它只“提出候选记忆”，没有直接写数据库的权限。候选结果之后还会经过
memory/store.py 的类型白名单、可信度、脱敏和长度校验，合格后才真正持久化。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from agent.multi_agent.state import MemoryExtraction, final_ai_text, parse_json_model


MEMORY_SYSTEM_PROMPT = """
你是“小学生学习记忆提取员”，只分析学生本轮亲自表达的信息。

可以提取：
- profile：学生明确给出的年级等学习背景；
- preference：学生明确喜欢的讲解方式；
- learning_topic：本轮正在学习的知识主题；
- learning_difficulty：学生明确说不会、易错或分不清的知识点。

规则：
1. 不根据助手答案猜测学生已经掌握或没有掌握某个知识点。
2. 不保存姓名、学校、住址、电话、邮箱、证件号等身份信息。
3. 每条内容必须简短、客观，最多返回五条；没有值得长期保存的信息就返回空列表。
4. 只返回规定的结构化结果，不展示内部推理过程。

最终回复只能是 JSON，不要使用 Markdown 代码块。例如：
{"memories": [{"memory_type": "preference", "subject": "综合", "content": "学生偏好分步骤讲解。", "confidence": 0.9}]}
没有记忆时返回：{"memories": []}
""".strip()


def create_memory_agent(model: BaseChatModel):
    """创建只负责提取结构化长期记忆候选的学习记忆 Agent。"""

    return create_agent(
        model=model,
        system_prompt=MEMORY_SYSTEM_PROMPT,
        tools=[],
        name="learning_memory_agent",
    )


def build_memory_agent_tool(
    model: BaseChatModel | None = None,
    *,
    agent: Any | None = None,
) -> BaseTool:
    """把学习记忆 Agent 包装成返回记忆候选 JSON 的 LangChain Tool。"""

    if agent is None:
        if model is None:
            raise ValueError("创建记忆 Agent Tool 时必须提供 model 或 agent")
        agent = create_memory_agent(model)

    @tool("learning_memory_agent")
    def learning_memory_agent_tool(
        question: str,
        subject: str,
        grade: int | None,
    ) -> str:
        """从学生原话中提取可跨会话使用的长期学习记忆候选。"""

        grade_text = f"小学{grade}年级" if grade else "未指定年级"
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"学生年级：{grade_text}\n"
                            f"问题学科：{subject}\n"
                            f"学生本轮原话：{question}"
                        ),
                    }
                ]
            }
        )

        extraction = result.get("structured_response")
        if extraction is None:
            extraction = parse_json_model(final_ai_text(result), MemoryExtraction)
        elif not isinstance(extraction, MemoryExtraction):
            extraction = MemoryExtraction.model_validate(extraction)
        return extraction.model_dump_json()

    return learning_memory_agent_tool
