# 文件用途：协调首次完整审核、一次重写、第二次独立安全审核和最终问答保存。
#
# 调用关系：（谁调用我，上游）
#   - agent/education_agent.py:63  创建：EducationMultiAgentWorkflow(...)（传三个子 Agent Tool）
#   - agent/education_agent.py:153 调用：self.workflow.invoke(state, config)
#   - tests/test_multi_agent.py    测试：直接构造 workflow 或通过 EducationAgent 调用
#
# 调用关系：（我调用谁，下游）
#   - agent/context_manager.py     EducationContextManager：压缩/收集/拼消息（见各节点行内注释）
#   - agent/multi_agent/subject_agent.py  subject_tool：学科 Agent 写草稿（inject）
#   - agent/multi_agent/review_agent.py   review_tool：审核 Agent 核查草稿
#   - agent/multi_agent/memory_agent.py   memory_tool：记忆 Agent 提候选
#   - memory/store.py             EducationMemoryStore.record_turn()：落库
#   - agent/multi_agent/state.py  EducationWorkflowState/ReviewResult/MemoryExtraction
#   - utils/logger_handler.py     logger：异常日志
#   - langgraph 框架               StateGraph/add_node/add_edge/compile（图本身）
#
# 修改易踩坑：RemoveMessage 只能删除已有 ID，审核重试必须有上限，只有最终答案可以追加到主 messages。
"""主协调工作流：按固定顺序调用三个“子 Agent Tool”。

给初学者的流程图：

    START
      -> prepare_context（读取短期对话和长期记忆）
      -> ask_subject_agent（生成答案草稿）
      -> review_answer（审核草稿）
           | 通过 ---------------------> save_learning_memory -> END
           | 未通过且还能重试 -> ask_subject_agent
           | 重写后 -> review_safety（仅安全）
                          | 安全通过 -> save_learning_memory -> END
                          | 不安全或审核异常 -> use_safe_fallback -> save_learning_memory -> END

每个节点收到完整 State，但只返回自己要更新的几个字段；LangGraph 负责合并。
子 Agent 内部的工具消息不会写入这张主图，只保存学生消息和最终答案。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.context_manager import EducationContextBundle, EducationContextManager
from agent.multi_agent.state import (
    EducationWorkflowState,
    MemoryExtraction,
    ReviewResult,
    SafetyReviewResult,
)
from memory.store import EducationMemoryStore
from agent.tools.call_limits import tool_call_budget_scope
from utils.config_handler import get_tool_call_limits
from utils.logger_handler import logger

SAFETY_BLOCKED_ANSWER = "这份回答可能包含不安全内容，暂时不能展示。请向老师或家长寻求帮助。"
SAFETY_REVIEW_UNAVAILABLE_ANSWER = "安全检查暂时无法完成，请稍后重试。"
FULL_REVIEW_FALLBACK_ANSWER = "这份回答暂未通过审核，请补充问题条件或向老师求助。"
#langgraph定义 state 节点 边，state在state.py EducationWorkflowState完成

class EducationMultiAgentWorkflow:
    """协调学科、首次完整审核、二次安全审核和学习记忆四个 Tool。"""

    def __init__(
        self,
        *,
        subject_tool: BaseTool,
        review_tool: BaseTool,
        safety_review_tool: BaseTool,
        memory_tool: BaseTool,
        memory_store: EducationMemoryStore | None,
        checkpointer: BaseCheckpointSaver | None = None,
        retrieval_limit: int = 5,
        recent_message_limit: int = 6,
        max_conversation_chars: int = 4000,
        max_long_term_chars: int = 1800,
        max_review_feedback_chars: int = 1200,
        summary_model: BaseChatModel | None = None,
        summary_enabled: bool = True,
        summary_trigger_messages: int = 10,
        summary_keep_recent_messages: int = 6,
        summary_trigger_chars: int = 4000,
        max_summary_chars: int = 1400,
        max_review_retries: int = 1,
    ):
        """保存四个 Tool 和记忆组件，编译带短期状态的固定流程。"""

        self.subject_tool = subject_tool
        self.review_tool = review_tool
        self.safety_review_tool = safety_review_tool
        self.memory_tool = memory_tool
        self.memory_store = memory_store
        # 本流程最多重写一次；第二次只检查安全，失败后不再进入生成循环。
        self.max_review_retries = min(1, max(0, max_review_retries))
        # 调用下游：agent/context_manager.py 的 EducationContextManager（上下文大脑）。
        self.context_manager = EducationContextManager(
            memory_store,
            limit=retrieval_limit,
            recent_message_limit=recent_message_limit,
            max_conversation_chars=max_conversation_chars,
            max_long_term_chars=max_long_term_chars,
            max_review_feedback_chars=max_review_feedback_chars,
            summary_model=summary_model,
            summary_enabled=summary_enabled,
            summary_trigger_messages=summary_trigger_messages,
            summary_keep_recent_messages=summary_keep_recent_messages,
            summary_trigger_chars=summary_trigger_chars,
            max_summary_chars=max_summary_chars,
        )
        self.graph = self._build_graph(checkpointer or InMemorySaver())

    def _build_graph(self, checkpointer: BaseCheckpointSaver):
        """连接各处理步骤，并把 Checkpointer 只挂在最外层工作流上。"""

        # StateGraph 可以理解为“带共享字典的流程图”。这里声明共享字典的结构。
        builder = StateGraph(EducationWorkflowState)  #建立state 定义state  就是字典

        # 每个节点只做一件事，代码小白可以沿着边从上往下阅读。
        builder.add_node("prepare_context", self._prepare_context)
        builder.add_node("ask_subject_agent", self._ask_subject_agent)
        builder.add_node("review_answer", self._review_answer)
        builder.add_node("review_safety", self._review_safety)
        builder.add_node("use_safe_fallback", self._use_safe_fallback)
        builder.add_node("save_learning_memory", self._save_learning_memory)

        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "ask_subject_agent")
        builder.add_conditional_edges(
            "ask_subject_agent", self._route_after_subject,
            {"full": "review_answer", "safety": "review_safety"},
        )
        # 普通边永远走向同一节点；conditional_edges 会读取审核结果选择下一条边。
        builder.add_conditional_edges(
            "review_answer",
            self._route_after_review,
            {
                "retry": "ask_subject_agent",
                "approved": "save_learning_memory",
                "fallback": "use_safe_fallback",
            },
        )
        builder.add_edge("use_safe_fallback", "save_learning_memory")
        builder.add_conditional_edges(
            "review_safety", self._route_after_safety,
            {"approved": "save_learning_memory", "fallback": "use_safe_fallback"},
        )
        builder.add_edge("save_learning_memory", END)
        # compile 后才得到可 invoke 的应用。Checkpointer 会在每个节点后保存 State，
        # 同一 configurable.thread_id 的下一轮调用就能恢复之前的 messages。
        return builder.compile(checkpointer=checkpointer)

    def _prepare_context(self, state: EducationWorkflowState) -> dict[str, Any]:
        """让统一 Context Manager 准备本轮所需的全部学科问答上下文。"""

        # 调用下游：context_manager.compact_conversation() 压缩长会话（判断要不要摘要）。
        compaction = self.context_manager.compact_conversation(
            messages=list(state.get("messages", [])),
            previous_summary=state.get("conversation_summary", ""),
        )
        # 调用下游：context_manager.build_context_bundle() 收集摘要/最近聊天/长期记忆/Skill。
        context_bundle = self.context_manager.build_context_bundle(
            student_id=state["student_id"],
            question=state["question"],
            subject=state["subject"],
            grade=state.get("grade"),
            messages=compaction.retained_messages,
            conversation_summary=compaction.summary,
        )

        # 返回的只是“局部更新”，不是要重新构造一份完整 State。
        return {
            "conversation_summary": context_bundle.conversation_summary,
            "long_term_context": context_bundle.long_term_context,
            "conversation_context": context_bundle.conversation_context,
            "skill_context": context_bundle.skill_context,
            "draft_answer": "",
            "approved": False,
            "review_score": None,
            "review_feedback": "",
            "corrected_answer": "",
            "review_attempts": 0,
            "review_mode": "full",
            "safety_passed": None,
            "safety_review_error": False,
            "final_answer": "",
            "memory_candidates": [],
            # add_messages reducer 识别 RemoveMessage 后，会从最新 Checkpoint State
            # 删除已经写入摘要的旧消息，只保留最近原文。没有压缩时不更新 messages。
            **(
                {
                    "messages": [
                        RemoveMessage(id=message_id)
                        for message_id in compaction.removed_message_ids
                    ]
                }
                if compaction.compacted
                else {}
            ),
        }

    def _ask_subject_agent(self, state: EducationWorkflowState) -> dict[str, str]:
        """调用学科子 Agent Tool；重试时把审核意见一并交给它修改。"""

        # 工作流只传结构化字段，不再自己拼 Prompt。上下文顺序、裁剪规则和
        # 默认文字都由 EducationContextManager 统一决定。
        context_bundle = EducationContextBundle(
            question=state["question"],
            subject=state["subject"],
            grade=state.get("grade"),
            conversation_summary=state.get("conversation_summary", "无"),
            long_term_context=state.get("long_term_context", "无"),
            conversation_context=state.get("conversation_context", "无"),
            skill_context=state.get(
                "skill_context", "本轮没有额外教育 Skill。"
            ),
        )
        # 调用下游：context_manager.build_subject_user_message() 拼完整消息（当前问题在最后）。
        context_message = self.context_manager.build_subject_user_message(
            context_bundle,
            review_feedback=state.get("review_feedback", ""),
        )
        # 调用下游：subject_agent.py 的学科子 Agent Tool（写草稿，内部自己调知识库/计算器）。
        draft = self.subject_tool.invoke(
            {"context_message": context_message}
        )
        draft_text = str(draft).strip()
        if not draft_text:
            raise RuntimeError("学科问答 Agent 返回了空答案")
        return {"draft_answer": draft_text}

    def _review_answer(self, state: EducationWorkflowState) -> dict[str, Any]:
        """调用审核子 Agent Tool，并把固定 JSON 结果写回共享 State。"""

        attempt = int(state.get("review_attempts", 0)) + 1
        # 调用下游：review_agent.py 的审核子 Agent Tool（独立核查草稿，返回固定JSON）。
        raw_result = self.review_tool.invoke(
            {
                "question": state["question"],
                "draft_answer": state["draft_answer"],
                "subject": state["subject"],
                "grade": state.get("grade"),
                "review_attempt": attempt,
            }
        )
        review = (
            ReviewResult.model_validate_json(raw_result)
            if isinstance(raw_result, str)
            else ReviewResult.model_validate(raw_result)
        )
        return {
            "approved": review.approved,
            "review_score": review.score,
            "review_feedback": review.feedback,
            "corrected_answer": review.corrected_answer or "",
            "review_attempts": attempt,
            "review_mode": "full",
        }

    @staticmethod
    def _route_after_subject(state: EducationWorkflowState) -> Literal["full", "safety"]:
        """本轮首次草稿完整审核；重写稿只交给独立安全审核。"""
        return "safety" if int(state.get("review_attempts", 0)) >= 1 else "full"

    def _review_safety(self, state: EducationWorkflowState) -> dict[str, Any]:
        """第二次只读取安全结论，质量分或额外 approved 字段不能影响路由。"""
        result: dict[str, Any] = {
            "approved": False,
            "review_mode": "safety_only",
            "review_score": None,
            "review_attempts": int(state.get("review_attempts", 0)) + 1,
            "corrected_answer": "",
            "safety_passed": None,
            "safety_review_error": False,
        }
        try:
            raw_result = self.safety_review_tool.invoke(
                {
                    "question": state["question"],
                    "draft_answer": state["draft_answer"],
                    "subject": state["subject"],
                    "grade": state.get("grade"),
                }
            )
            review = (
                SafetyReviewResult.model_validate_json(raw_result)
                if isinstance(raw_result, str)
                else SafetyReviewResult.model_validate(raw_result)
            )
            result.update(
                approved=review.safety_passed,
                safety_passed=review.safety_passed,
                review_feedback=review.feedback,
            )
        except Exception as exc:
            # 异常不等于安全通过；不打印原始结果或异常正文，避免泄露待审内容。
            logger.warning("第二次安全审核失败（%s），返回固定提示", type(exc).__name__)
            result.update(
                safety_review_error=True,
                review_feedback="安全检查暂时无法完成",
            )
        return result

    @staticmethod
    def _route_after_safety(state: EducationWorkflowState) -> Literal["approved", "fallback"]:
        """安全通过必放行，不再评估质量，也不再重写。"""
        if state.get("safety_passed") is True and not state.get("safety_review_error", False):
            return "approved"
        return "fallback"

    def _route_after_review(
        self,
        state: EducationWorkflowState,
    ) -> Literal["retry", "approved", "fallback"]:
        """审核通过就保存；未通过且有次数就重写，否则进入安全兜底。"""

        if state.get("approved", False):
            return "approved"
        # 例如 max_review_retries=1：第一次审核不通过后允许重写一次；第二次审核
        # 仍不通过就进入 fallback，不会无限循环消耗模型调用。
        if int(state.get("review_attempts", 0)) <= self.max_review_retries:
            return "retry"
        return "fallback"

    @staticmethod
    def _use_safe_fallback(state: EducationWorkflowState) -> dict[str, str]:
        """阻断或审核异常只返回固定提示，不展示未经安全检查的建议稿。"""
        if state.get("safety_review_error", False):
            fallback = SAFETY_REVIEW_UNAVAILABLE_ANSWER
        elif state.get("review_mode") == "safety_only":
            fallback = SAFETY_BLOCKED_ANSWER
        else:
            fallback = FULL_REVIEW_FALLBACK_ANSWER
        return {"final_answer": fallback}

    def _save_learning_memory(
        self,
        state: EducationWorkflowState,
    ) -> dict[str, Any]:
        """调用记忆子 Agent Tool，并把最终问答与候选记忆一起持久化。"""

        final_answer = state.get("final_answer", "").strip()
        if not final_answer:
            final_answer = state["draft_answer"].strip()

        # 提取失败或成功无候选时都只保存代码确定的年级和本轮主题。
        # 不启用旧关键词规则，避免把题干引用或临时请求误记为长期偏好/困难。
        candidates: list[dict[str, Any]] = []
        try:
            # 调用下游：memory_agent.py 的记忆子 Agent Tool（提长期记忆候选JSON）。
            raw_result = self.memory_tool.invoke(
                {
                    "question": state["question"],
                    "subject": state["subject"],
                    "grade": state.get("grade"),
                }
            )
            extraction = (
                MemoryExtraction.model_validate_json(raw_result)
                if isinstance(raw_result, str)
                else MemoryExtraction.model_validate(raw_result)
            )
            candidates = [
                item.model_dump() for item in extraction.memories
            ]
        except Exception as exc:
            # 记忆提取是增强功能。即使它暂时失败，也不能吞掉已经审核通过的答案。
            logger.warning(
                "学习记忆 Agent 执行失败（%s），仅保存年级与本轮提问主题",
                type(exc).__name__,
            )

        # 调用下游：memory/store.py 的 record_turn()（落库：会话+消息+长期记忆）。
        if self.memory_store is not None:
            self.memory_store.record_turn(
                student_id=state["student_id"],
                thread_id=state["thread_id"],
                turn_id=state["turn_id"],
                question=state["question"],
                answer=final_answer,
                subject=state["subject"],
                grade=state.get("grade"),
                memory_candidates=candidates,
            )

        # 只把最终答案加入主会话；三个子 Agent 的内部消息不会污染学生历史。
        return {
            "final_answer": final_answer,
            "memory_candidates": candidates,
            "messages": [AIMessage(content=final_answer)],
        }

    def invoke(
        self,
        state: EducationWorkflowState,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EducationWorkflowState:
        """使用给定初始 State 和线程配置执行一次完整多 Agent 问答。"""

        # 调用下游：LangGraph 框架的 CompiledStateGraph.invoke()（跑完整张图）。
        # 同题所有子 Agent 使用一个并发安全计数器；计数不写入 Checkpoint。
        # 作用域结束即释放，新问题（即使同一线程）重新按各工具独立计数。
        with tool_call_budget_scope(get_tool_call_limits()):
            return self.graph.invoke(state, config=config, **kwargs)
