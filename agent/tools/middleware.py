# 文件用途：执行每题各工具独立限额，并记录最少量模型和工具运行指标。
# 调用关系：subject/review Agent 注册本中间件；LangChain 在每次模型或工具调用时触发它。
# 修改易踩坑：日志中不能写学生问题正文、模型答案、密钥或完整工具参数。
"""Agent 中间件：记录运行状态，但不记录学生隐私或答案正文。"""

from collections.abc import Callable, Sequence

from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    Runtime,
    before_model,
    wrap_tool_call,
    wrap_model_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.tools.call_limits import (
    build_tool_limit_message,
    get_tool_call_budget,
    reserve_tool_call,
)
from utils.config_handler import get_tool_call_limits
from utils.logger_handler import logger


def build_tool_call_limit_prompt(tool_names: Sequence[str]) -> str:
    """告诉模型共享限额与超限后的处理方式，真正拦截由中间件执行。"""

    limits = get_tool_call_limits()
    entries = "、".join(
        f"{name} 最多 {limits[name]} 次" for name in tool_names if name in limits
    )
    return (
        "\n\n工具调用限额：按本轮学生问题计数，初稿、完整审核与重写共享同一工具的额度。"
        f"{entries}。各工具独立计数，没有工具调用总额限制。"
        "工具返回已达上限时，不要再次申请该工具；根据已有可靠结果完成任务，"
        "资料或计算不足时明确说明，不编造工具结果，也不能跳过工作流审核。"
    )


@wrap_model_call
def limit_available_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """只移除本轮已耗尽的工具，其他工具继续可用，不增加总额约束。"""

    budget = get_tool_call_budget()
    if budget is None:
        return handler(request)
    counts = budget.counts()
    available = []
    for candidate in request.tools:
        name = (
            candidate.get("name", candidate.get("function", {}).get("name"))
            if isinstance(candidate, dict)
            else candidate.name
        )
        limit = budget.limit_for(name)
        if limit is None or counts.get(name, 0) < limit:
            available.append(candidate)
    return handler(request.override(tools=available))


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """先预约该工具独立额度，超限不执行；失败调用也计入本轮额度。"""
    tool_name = request.tool_call["name"]
    if not reserve_tool_call(tool_name):
        budget = get_tool_call_budget()
        assert budget is not None
        limit = budget.limit_for(tool_name)
        assert limit is not None
        logger.warning("教育工具调用达到本轮上限：%s（%s 次）", tool_name, limit)
        return ToolMessage(
            content=build_tool_limit_message(tool_name, limit),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error",
        )
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
