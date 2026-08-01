"""Agent 中间件：记录运行状态，但不记录学生隐私或答案正文。"""

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentState,
    Runtime,
    before_model,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    logger.info("调用教育工具：%s", tool_name)
    try:
        result = handler(request)
        logger.info("教育工具调用成功：%s", tool_name)
        return result
    except Exception:
        logger.exception("教育工具调用失败：%s", tool_name)
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    del runtime
    logger.debug("准备调用模型，当前消息数：%d", len(state["messages"]))
