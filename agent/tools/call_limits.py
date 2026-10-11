# 文件用途：为每道问题维护各工具独立的调用额度，达到上限后阻止继续执行。
# 调用关系：工作流入口创建预算作用域，工具中间件在执行前原子预约一次调用。
# 修改易踩坑：预算必须按请求隔离；失败不返还额度；不得把各工具额度合并为总额。
"""通过请求上下文共享工具计数，兼容同一请求中的并发工具调用。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock


class ToolCallBudget:
    """一轮问答的独立工具额度；初稿、重写和审核复用同一个实例。"""

    def __init__(self, limits: dict[str, int]) -> None:
        if not isinstance(limits, dict):
            raise ValueError("工具调用限额必须是工具名称到正整数的字典")
        for tool_name, limit in limits.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("工具调用限额的工具名称必须是非空字符串")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"工具 {tool_name} 的调用限额必须是正整数")
        self._limits = dict(limits)
        self._counts = {tool_name: 0 for tool_name in limits}
        self._lock = Lock()

    def try_acquire(self, tool_name: str) -> bool:
        """原子预约额度；未配置的工具不受限，已预约的失败调用也计数。"""
        limit = self._limits.get(tool_name)
        if limit is None:
            return True
        with self._lock:
            count = self._counts[tool_name]
            if count >= limit:
                return False
            self._counts[tool_name] = count + 1
            return True

    def limit_for(self, tool_name: str) -> int | None:
        """返回本轮该工具的限额，未配置时返回 None。"""
        return self._limits.get(tool_name)

    def counts(self) -> dict[str, int]:
        """返回计数快照，调用者无法通过修改结果改变本轮预算。"""
        with self._lock:
            return dict(self._counts)


_current_tool_call_budget: ContextVar[ToolCallBudget | None] = ContextVar(
    "education_tool_call_budget", default=None
)


@contextmanager
def tool_call_budget_scope(limits: dict[str, int]) -> Iterator[ToolCallBudget]:
    """建立新问题的预算作用域，并在正常结束或异常后恢复之前的上下文。"""
    budget = ToolCallBudget(limits)
    token = _current_tool_call_budget.set(budget)
    try:
        yield budget
    finally:
        _current_tool_call_budget.reset(token)


def get_tool_call_budget() -> ToolCallBudget | None:
    """读取当前请求预算；独立调用子 Agent 时可能没有工作流作用域。"""
    return _current_tool_call_budget.get()


def reserve_tool_call(tool_name: str) -> bool:
    """在执行工具前预约一次调用；无作用域时保留独立调用的兼容行为。"""
    budget = get_tool_call_budget()
    return budget is None or budget.try_acquire(tool_name)


def build_tool_limit_message(tool_name: str, limit: int) -> str:
    """告知模型调用被拦截，要求利用现有证据继续完成既定审核流程。"""
    return (
        f"工具 {tool_name} 在本轮问题中最多调用 {limit} 次，现已达到上限，"
        "本次调用未执行。请勿再次调用该工具；请根据已有结果继续处理，"
        "证据不足时明确说明，不要编造工具结果或跳过审核。"
    )
