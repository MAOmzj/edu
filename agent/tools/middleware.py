# 文件用途：在模型和工具执行前后记录最少量运行指标，便于定位 Agent 故障。
# 调用关系：subject/review Agent 注册本中间件；LangChain 在每次模型或工具调用时触发它。
# 修改易踩坑：日志中不能写学生问题正文、模型答案、密钥或完整工具参数。
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
    """包裹每次工具调用，记录开始、成功或异常状态后返回调用结果。"""
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
    """在调用模型前记录当前消息数量，方便观察 Agent 运行状态。"""
    del runtime
    logger.debug("准备调用模型，当前消息数：%d", len(state["messages"]))
