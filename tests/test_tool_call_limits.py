# 文件用途：离线验证每题每个工具六次上限及真实 Agent 并行调用的请求隔离。
# 调用关系：unittest 调用本文件；测试使用计数器、中间件和无网络假模型执行真实 LangGraph。
# 修改易踩坑：不得使用在线模型或真实学生数据；初稿、审核和重写必须共享额度且没有总额。
"""独立工具额度、并发拦截和工作流作用域的行为测试。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import json
import unittest
from unittest.mock import Mock, patch

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from agent.education_agent import EducationAgent
from agent.multi_agent.workflow import EducationMultiAgentWorkflow
from agent.tools.call_limits import (
    ToolCallBudget,
    get_tool_call_budget,
    reserve_tool_call,
    tool_call_budget_scope,
)
from agent.tools.middleware import limit_available_tools, monitor_tool
from utils.config_handler import (
    app_conf,
    get_tool_call_limits,
    validate_tool_call_limits,
)


LIMITS = {
    "search_knowledge": 6,
    "calculate_expression": 6,
    "search_web": 6,
}


class BurstChatModel(BaseChatModel):
    """一次产生指定数量的并行工具请求，收到结果后直接结束，不访问网络。"""

    calls_per_tool: dict[str, int]

    @property
    def _llm_type(self) -> str:
        return "offline-tool-limit-burst"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(message, ToolMessage) for message in messages):
            answer = AIMessage(content="已基于可用工具结果完成回答。")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": {"value": index},
                        "id": f"{name}_{index}",
                        "type": "tool_call",
                    }
                    for name, count in self.calls_per_tool.items()
                    for index in range(count)
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


def make_offline_agent(calls_per_tool, executions, execution_lock, extra_middleware=()):
    """用真实工具节点记录执行，不伪造 LangGraph 的线程上下文传播。"""

    tools = []
    for name in calls_per_tool:
        def execute(value: int, tool_name=name) -> str:
            with execution_lock:
                executions.append((tool_name, value, get_tool_call_budget()))
            return f"结果 {value}"

        tools.append(StructuredTool.from_function(
            func=execute,
            name=name,
            description="用于离线计数验证，返回输入数字。",
        ))
    return create_agent(
        model=BurstChatModel(calls_per_tool=calls_per_tool),
        tools=tools,
        middleware=[limit_available_tools, *extra_middleware, monitor_tool],
    )


class ToolCallBudgetTests(unittest.TestCase):
    def test_each_tool_has_six_calls_with_no_total_limit(self):
        budget = ToolCallBudget(LIMITS)
        for name in LIMITS:
            self.assertEqual([budget.try_acquire(name) for _ in range(6)], [True] * 6)
        self.assertEqual(budget.counts(), LIMITS)
        self.assertEqual(sum(budget.counts().values()), 18)
        for name in LIMITS:
            self.assertFalse(budget.try_acquire(name))
        self.assertEqual(budget.counts(), LIMITS)

    def test_sub_agent_tools_are_not_counted(self):
        budget = ToolCallBudget(LIMITS)
        for _ in range(20):
            self.assertTrue(budget.try_acquire("subject_qa_agent"))
        self.assertEqual(budget.counts(), dict.fromkeys(LIMITS, 0))

    def test_parallel_reservations_never_exceed_six(self):
        budget = ToolCallBudget(LIMITS)
        with ThreadPoolExecutor(max_workers=12) as executor:
            granted = list(executor.map(
                lambda _: budget.try_acquire("search_knowledge"), range(100)
            ))
        self.assertEqual(sum(granted), 6)
        self.assertEqual(budget.counts()["search_knowledge"], 6)

    def test_nested_scope_restores_parent_even_after_exception(self):
        self.assertIsNone(get_tool_call_budget())
        with tool_call_budget_scope(LIMITS) as outer:
            self.assertTrue(reserve_tool_call("search_knowledge"))
            with self.assertRaisesRegex(RuntimeError, "失败"):
                with tool_call_budget_scope(LIMITS) as inner:
                    self.assertIsNot(inner, outer)
                    self.assertTrue(reserve_tool_call("search_knowledge"))
                    raise RuntimeError("失败")
            self.assertIs(get_tool_call_budget(), outer)
            self.assertEqual(outer.counts()["search_knowledge"], 1)
        self.assertIsNone(get_tool_call_budget())

    def test_simultaneous_questions_have_independent_counts(self):
        barrier = Barrier(2)

        def run_question(_):
            with tool_call_budget_scope(LIMITS) as budget:
                barrier.wait(timeout=10)
                granted = [reserve_tool_call("search_web") for _ in range(7)]
                return budget, granted, budget.counts()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(run_question, range(2)))
        self.assertIsNot(first[0], second[0])
        for result in (first, second):
            self.assertEqual(result[1], [True] * 6 + [False])
            self.assertEqual(result[2]["search_web"], 6)
        self.assertIsNone(get_tool_call_budget())


class ToolCallMiddlewareTests(unittest.TestCase):
    @staticmethod
    def request(name="search_knowledge", call_id="offline_call"):
        return ToolCallRequest(
            tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
            tool=None,
            state={},
            runtime=None,
        )

    def test_seventh_call_is_not_executed_and_other_tool_remains_available(self):
        request = self.request()
        success = ToolMessage(content="成功", tool_call_id=request.tool_call["id"])
        handler = Mock(return_value=success)
        with tool_call_budget_scope(LIMITS) as budget:
            for _ in range(6):
                self.assertIs(monitor_tool.wrap_tool_call(request, handler), success)
            rejected = monitor_tool.wrap_tool_call(self.request(call_id="seventh"), handler)
            self.assertEqual(handler.call_count, 6)
            self.assertEqual(rejected.status, "error")
            self.assertEqual(rejected.tool_call_id, "seventh")
            self.assertIn("6", rejected.content)
            self.assertIn("未执行", rejected.content)
            monitor_tool.wrap_tool_call(self.request("calculate_expression"), handler)
            self.assertEqual(handler.call_count, 7)
            self.assertEqual(budget.counts()["calculate_expression"], 1)

    def test_failed_execution_consumes_one_call(self):
        failing = Mock(side_effect=RuntimeError("离线失败"))
        succeeding = Mock(return_value=ToolMessage(content="成功", tool_call_id="offline_call"))
        with tool_call_budget_scope(LIMITS) as budget:
            with patch("agent.tools.middleware.logger"):
                with self.assertRaisesRegex(RuntimeError, "离线失败"):
                    monitor_tool.wrap_tool_call(self.request(), failing)
                for _ in range(5):
                    monitor_tool.wrap_tool_call(self.request(), succeeding)
                rejected = monitor_tool.wrap_tool_call(self.request(), succeeding)
            self.assertEqual(budget.counts()["search_knowledge"], 6)
            self.assertEqual(succeeding.call_count, 5)
            self.assertEqual(rejected.status, "error")

    def test_standalone_tool_without_request_scope_stays_compatible(self):
        self.assertIsNone(get_tool_call_budget())
        handler = Mock(return_value=ToolMessage(content="成功", tool_call_id="offline_call"))
        for _ in range(7):
            monitor_tool.wrap_tool_call(self.request(), handler)
        self.assertEqual(handler.call_count, 7)

    def test_model_guard_removes_only_exhausted_tools_and_preserves_request(self):
        def echo(value: int) -> str:
            return str(value)

        tools = [
            StructuredTool.from_function(func=echo, name="search_knowledge", description="离线工具。"),
            {"name": "calculate_expression", "description": "离线计算。"},
            {"type": "function", "function": {"name": "search_web", "description": "离线联网。"}},
            {"name": "answer_review_agent", "description": "不限额的协调工具。"},
        ]
        request = ModelRequest(
            model=BurstChatModel(calls_per_tool={}),
            messages=[HumanMessage(content="离线问题")],
            tools=tools,
        )
        handler = Mock(return_value=object())
        with tool_call_budget_scope(LIMITS):
            for _ in range(6):
                self.assertTrue(reserve_tool_call("search_knowledge"))
                self.assertTrue(reserve_tool_call("search_web"))
            self.assertTrue(reserve_tool_call("calculate_expression"))
            response = limit_available_tools.wrap_model_call(request, handler)
        self.assertIs(response, handler.return_value)
        forwarded = handler.call_args.args[0]
        self.assertEqual(forwarded.tools, [tools[1], tools[3]])
        self.assertEqual(request.tools, tools)
        self.assertEqual(len(request.tools), 4)

    def test_model_guard_without_scope_does_not_change_tool_list(self):
        request = ModelRequest(
            model=BurstChatModel(calls_per_tool={}),
            messages=[HumanMessage(content="离线问题")],
            tools=[{"name": "search_knowledge"}],
        )
        handler = Mock(return_value=object())
        limit_available_tools.wrap_model_call(request, handler)
        self.assertIs(handler.call_args.args[0], request)


class ToolCallAgentIntegrationTests(unittest.TestCase):
    def test_model_and_tool_guards_allow_answer_after_six_calls(self):
        advertised_tools = []

        @wrap_model_call
        def observe_tools(request, handler):
            advertised_tools.append([tool.name for tool in request.tools])
            return handler(request)

        executions = []
        agent = make_offline_agent(
            {"search_knowledge": 6, "calculate_expression": 0},
            executions,
            Lock(),
            extra_middleware=[observe_tools],
        )
        with tool_call_budget_scope(LIMITS) as budget:
            result = agent.invoke({"messages": [{"role": "user", "content": "离线问题"}]})
            self.assertEqual(budget.counts()["search_knowledge"], 6)
            self.assertEqual(budget.counts()["calculate_expression"], 0)
        self.assertEqual(advertised_tools, [
            ["search_knowledge", "calculate_expression"],
            ["calculate_expression"],
        ])
        self.assertEqual(len(executions), 6)
        self.assertEqual(result["messages"][-1].content, "已基于可用工具结果完成回答。")

    def test_real_workflow_shares_budget_across_initial_review_and_rewrite(self):
        executions = []
        lock = Lock()
        generation = make_offline_agent({"search_knowledge": 4}, executions, lock)
        review = make_offline_agent({"search_knowledge": 4}, executions, lock)
        stage_budgets = []

        def ask(context_message: str) -> str:
            stage_budgets.append(("generation", get_tool_call_budget()))
            result = generation.invoke({"messages": [{"role": "user", "content": context_message}]})
            return result["messages"][-1].content

        def full_review(question: str, draft_answer: str, subject: str, grade: int | None, review_attempt: int) -> str:
            stage_budgets.append(("review", get_tool_call_budget()))
            review.invoke({"messages": [{"role": "user", "content": question}]})
            return json.dumps({"approved": False, "score": 50, "feedback": "补充解释", "corrected_answer": None})

        def safety_review(question: str, draft_answer: str, subject: str, grade: int | None) -> str:
            stage_budgets.append(("safety", get_tool_call_budget()))
            return json.dumps({"safety_passed": True, "risk_categories": [], "feedback": "安全通过"})

        def extract_memory(question: str, subject: str, grade: int | None) -> str:
            return '{"memories": []}'

        def as_tool(function, name):
            return StructuredTool.from_function(func=function, name=name, description="离线工作流验证。")

        workflow = EducationMultiAgentWorkflow(
            subject_tool=as_tool(ask, "subject_qa_agent"),
            review_tool=as_tool(full_review, "answer_review_agent"),
            safety_review_tool=as_tool(safety_review, "answer_safety_review_agent"),
            memory_tool=as_tool(extract_memory, "learning_memory_agent"),
            memory_store=None,
            summary_enabled=False,
        )
        for turn in range(2):
            result = workflow.invoke(
                {
                    "messages": [HumanMessage(content=f"离线问题 {turn}")],
                    "question": f"离线问题 {turn}",
                    "subject": "数学",
                    "grade": 3,
                    "student_id": "offline_student",
                    "thread_id": "offline_thread",
                    "turn_id": f"offline_turn_{turn}",
                },
                {"configurable": {"thread_id": "offline_thread"}, "recursion_limit": 12},
            )
            self.assertTrue(result["approved"])
            self.assertEqual(result["review_mode"], "safety_only")
            self.assertIsNone(get_tool_call_budget())
        self.assertEqual(len(executions), 12)
        self.assertEqual([stage for stage, _ in stage_budgets], ["generation", "review", "generation", "safety"] * 2)
        first_budget = stage_budgets[0][1]
        second_budget = stage_budgets[4][1]
        self.assertIsNot(first_budget, second_budget)
        self.assertTrue(all(budget is first_budget for _, budget in stage_budgets[:4]))
        self.assertTrue(all(budget is second_budget for _, budget in stage_budgets[4:]))
        self.assertEqual(first_budget.counts()["search_knowledge"], 6)
        self.assertEqual(second_budget.counts()["search_knowledge"], 6)

    def test_real_parallel_tool_node_allows_eighteen_calls_and_blocks_three(self):
        executions = []
        agent = make_offline_agent(dict.fromkeys(LIMITS, 7), executions, Lock())
        with tool_call_budget_scope(LIMITS) as budget:
            result = agent.invoke({"messages": [{"role": "user", "content": "离线问题"}]})
            self.assertEqual(budget.counts(), LIMITS)
        self.assertEqual(len(executions), 18)
        self.assertTrue(all(entry[2] is budget for entry in executions))
        tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(tool_messages), 21)
        self.assertEqual(sum(message.status == "error" for message in tool_messages), 3)
        self.assertIsInstance(result["messages"][-1], AIMessage)

    def test_separate_agent_invocations_share_budget_until_next_question(self):
        executions = []
        lock = Lock()
        generation = make_offline_agent({"search_knowledge": 4}, executions, lock)
        review = make_offline_agent({"search_knowledge": 4}, executions, lock)
        payload = {"messages": [{"role": "user", "content": "离线问题"}]}
        with tool_call_budget_scope(LIMITS) as first:
            generation.invoke(payload)
            review.invoke(payload)
            self.assertEqual(first.counts()["search_knowledge"], 6)
        self.assertEqual(len(executions), 6)
        with tool_call_budget_scope(LIMITS) as second:
            generation.invoke(payload)
            self.assertEqual(second.counts()["search_knowledge"], 4)
        self.assertEqual(len(executions), 10)
        self.assertTrue(all(entry[2] is first for entry in executions[:6]))
        self.assertTrue(all(entry[2] is second for entry in executions[6:]))

    def test_simultaneous_real_agents_do_not_share_question_budget(self):
        executions = []
        agent = make_offline_agent({"search_web": 7}, executions, Lock())
        barrier = Barrier(2)

        def run_question(index):
            with tool_call_budget_scope(LIMITS) as budget:
                barrier.wait(timeout=10)
                agent.invoke({"messages": [{"role": "user", "content": f"离线问题 {index}"}]})
                return budget, budget.counts()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run_question, range(2)))
        self.assertEqual(len(executions), 12)
        self.assertIsNot(results[0][0], results[1][0])
        for budget, counts in results:
            self.assertEqual(counts["search_web"], 6)
            self.assertEqual(sum(entry[2] is budget for entry in executions), 6)


class ToolCallConfigAndWorkflowTests(unittest.TestCase):
    def test_legacy_agent_invoke_keeps_scope_kwargs_and_read_api_compatible(self):
        # 注入四个占位工具，构造真实门面；只有替换后的 graph 会执行。
        facade = EducationAgent(
            subject_tool=Mock(),
            review_tool=Mock(),
            safety_review_tool=Mock(),
            memory_tool=Mock(),
        )
        seen_budgets = []

        def execute(state, config=None, **kwargs):
            budget = get_tool_call_budget()
            seen_budgets.append(budget)
            self.assertIsNotNone(budget)
            self.assertEqual([reserve_tool_call("search_web") for _ in range(7)], [True] * 6 + [False])
            return {"final_answer": "完成"}

        facade.workflow.graph = Mock()
        facade.workflow.graph.invoke.side_effect = execute
        self.assertEqual(facade.agent.invoke({}, output_keys=["final_answer"]), {"final_answer": "完成"})
        facade.workflow.graph.invoke.assert_called_once_with({}, config=None, output_keys=["final_answer"])
        self.assertIsNone(get_tool_call_budget())
        facade.agent.invoke({}, {"configurable": {"thread_id": "same_thread"}})
        self.assertIsNot(seen_budgets[0], seen_budgets[1])
        read_config = {"configurable": {"thread_id": "same_thread"}}
        self.assertIs(facade.agent.get_state(read_config), facade.workflow.graph.get_state.return_value)
        facade.workflow.graph.get_state.assert_called_once_with(read_config)
        self.assertIs(facade.agent.get_graph(), facade.workflow.graph.get_graph.return_value)
        self.assertIsNone(get_tool_call_budget())

    def test_legacy_config_defaults_to_six_for_each_tool(self):
        with patch.dict(app_conf, {"agent": {"recursion_limit": 12}}):
            self.assertEqual(get_tool_call_limits(), LIMITS)

    def test_config_rejects_total_unknown_missing_or_invalid_limit(self):
        invalid = [None, [], {}, {**LIMITS, "total": 18}, {**LIMITS, "other_tool": 6}]
        invalid.append({"search_knowledge": 6, "search_web": 6})
        invalid.extend({**LIMITS, "search_web": value} for value in (True, 0, -1, 6.0, "6"))
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_tool_call_limits(config)

    def test_workflow_owns_new_scope_for_every_invocation(self):
        workflow = EducationMultiAgentWorkflow.__new__(EducationMultiAgentWorkflow)
        seen = []

        def execute(state, config):
            budget = get_tool_call_budget()
            seen.append(budget)
            self.assertIsNotNone(budget)
            self.assertEqual(budget.counts(), dict.fromkeys(LIMITS, 0))
            self.assertEqual([reserve_tool_call("calculate_expression") for _ in range(7)], [True] * 6 + [False])
            return {"final_answer": "完成"}

        workflow.graph = Mock()
        workflow.graph.invoke.side_effect = execute
        with patch.dict(app_conf, {"agent": {"tool_call_limits": LIMITS}}):
            for _ in range(2):
                result = workflow.invoke({}, {"configurable": {"thread_id": "same_thread"}})
                self.assertEqual(result["final_answer"], "完成")
                self.assertIsNone(get_tool_call_budget())
        self.assertIsNot(seen[0], seen[1])

    def test_workflow_exception_releases_request_scope(self):
        workflow = EducationMultiAgentWorkflow.__new__(EducationMultiAgentWorkflow)
        workflow.graph = Mock()
        workflow.graph.invoke.side_effect = RuntimeError("工作流失败")
        with self.assertRaisesRegex(RuntimeError, "工作流失败"):
            workflow.invoke({}, {})
        self.assertIsNone(get_tool_call_budget())


if __name__ == "__main__":
    unittest.main()
