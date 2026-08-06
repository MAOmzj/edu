"""学科问答子 Agent，并将它包装成主工作流可以调用的 Tool。

这里有两层容易混淆：

1. create_subject_agent() 创建真正会和模型多轮交互、会调用知识库/计算器的 Agent；
2. build_subject_agent_tool() 再把这个 Agent 包成一个普通 Tool，交给主协调图调用。

因此“子 Agent 是 Tool”并不是模型本身变成函数，而是用 Tool 固定它的输入和输出边界。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from agent.context_manager import EducationContextManager
from agent.multi_agent.state import final_ai_text
from agent.tools.education_tools import calculate_expression, search_knowledge
from agent.tools.middleware import log_before_model, monitor_tool
from agent.tools.web_search import is_web_search_enabled, search_web


def create_subject_agent(model: BaseChatModel):
    """创建真正负责查知识、做计算和生成草稿答案的学科 Agent。"""

    # 子 Agent 不设置 Checkpointer。多轮历史由最外层主工作流统一保存，
    # 这样不会让同一段对话在多个 Agent 中重复持久化。
    tools = [search_knowledge, calculate_expression]
    # 联网工具可以通过配置整体关闭。即使开启但没有 Tavily Key，工具内部也会
    # 自动返回本地知识，不会让学科 Agent 因第三方服务失败而停止回答。
    if is_web_search_enabled():
        tools.append(search_web)
    return create_agent(
        model=model,
        system_prompt=EducationContextManager.build_subject_system_prompt(),
        tools=tools,
        middleware=[monitor_tool, log_before_model],
        name="subject_qa_agent",
    )


def build_subject_agent_tool(
    model: BaseChatModel | None = None,
    *,
    agent: Any | None = None,
) -> BaseTool:
    """把学科 Agent 的 invoke 调用封装成一个标准 LangChain Tool。"""

    if agent is None:
        if model is None:
            raise ValueError("创建学科 Agent Tool 时必须提供 model 或 agent")
        agent = create_subject_agent(model)

    @tool("subject_qa_agent")
    def subject_qa_agent_tool(
        context_message: str,
    ) -> str:
        """执行统一 Context Manager 已经组装好的学科问答消息。"""

        if not context_message.strip():
            raise ValueError("统一 Context Manager 生成的消息不能为空")

        # 子 Agent 自己可能经历“模型 -> 调工具 -> 模型”的小循环；invoke 返回时，
        # 工具调用已经结束。这里只取最后一条 AI 正文作为主工作流的答案草稿。
        result = agent.invoke(
            {"messages": [{"role": "user", "content": context_message}]}
        )
        return final_ai_text(result)

    return subject_qa_agent_tool
