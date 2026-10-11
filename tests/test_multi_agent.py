# 文件用途：用假子 Agent 验证主协调流程、审核重试、摘要、短长期记忆和线程隔离。
# 调用关系：unittest 调用本文件；本文件调用 EducationAgent 和四个真实 Tool 包装器。
# 修改易踩坑：Stub 返回数量必须覆盖调用次数，测试不得创建在线模型或复用真实学生数据库。
"""多 Agent 子 Agent-as-Tool 工作流测试，全程不调用在线模型。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.education_agent import EducationAgent
from agent.multi_agent.memory_agent import build_memory_agent_tool
from agent.multi_agent.review_agent import build_review_agent_tool
from agent.multi_agent.safety_review_agent import build_safety_review_agent_tool
from agent.multi_agent.state import (
    MemoryCandidate,
    MemoryExtraction,
    ReviewResult,
    SafetyReviewResult,
)
from agent.multi_agent.subject_agent import build_subject_agent_tool
from agent.multi_agent.workflow import (
    FULL_REVIEW_FALLBACK_ANSWER,
    SAFETY_BLOCKED_ANSWER,
    SAFETY_REVIEW_UNAVAILABLE_ANSWER,
)
from memory.store import EducationMemoryStore
from utils.config_handler import app_conf


class StubSubjectAgent:
    """按顺序返回预设草稿，并记录收到的 Prompt。"""

    def __init__(self, answers: list[str]):
        """保存待返回的答案队列。"""

        self.answers = list(answers)
        self.prompts: list[str] = []

    def invoke(self, payload: dict):
        """记录 Prompt，并用下一条预设文本模拟学科 Agent 回答。"""

        self.prompts.append(payload["messages"][0]["content"])
        answer = self.answers.pop(0)
        return {"messages": [AIMessage(content=answer)]}


class StubReviewAgent:
    """按顺序返回预设的结构化审核结果。"""

    def __init__(self, reviews: list[ReviewResult]):
        """保存待返回的审核结果队列。"""

        self.reviews = list(reviews)
        self.prompts: list[str] = []

    def invoke(self, payload: dict):
        """记录待审核内容，并返回下一条审核结果。"""

        self.prompts.append(payload["messages"][0]["content"])
        return {"structured_response": self.reviews.pop(0)}


class StubMemoryAgent:
    """返回预设记忆候选，或模拟记忆 Agent 发生异常。"""

    def __init__(
        self,
        extraction: MemoryExtraction | None = None,
        *,
        fail: bool = False,
    ):
        """保存记忆提取结果和是否主动失败的测试开关。"""

        self.extraction = extraction or MemoryExtraction()
        self.fail = fail
        self.prompts: list[str] = []

    def invoke(self, payload: dict):
        """记录学生原话，并返回候选记忆或抛出测试异常。"""

        self.prompts.append(payload["messages"][0]["content"])
        if self.fail:
            raise RuntimeError("模拟记忆 Agent 失败")
        return {"structured_response": self.extraction}


class StubSafetyAgent:
    """只给出安全结果，也可模拟无效 JSON 或调用异常。"""

    def __init__(
        self,
        reviews: list[SafetyReviewResult | dict] | None = None,
        *,
        fail: bool = False,
        text_response: str | None = None,
    ):
        self.reviews = list(reviews) if reviews is not None else None
        self.fail = fail
        self.text_response = text_response
        self.prompts: list[str] = []

    def invoke(self, payload: dict):
        self.prompts.append(payload["messages"][0]["content"])
        if self.fail:
            raise RuntimeError("模拟安全 Agent 失败")
        if self.text_response is not None:
            return {"messages": [AIMessage(content=self.text_response)]}
        review = (
            self.reviews.pop(0)
            if self.reviews is not None
            else SafetyReviewResult(safety_passed=True, feedback="没有安全风险")
        )
        return {"structured_response": review}


class MultiAgentWorkflowTests(unittest.TestCase):
    """验证学科、完整审核、安全审核和记忆 Tool 的协调过程。"""

    def setUp(self):
        """为每个测试创建独立临时 SQLite 数据库。"""

        self.temporary = tempfile.TemporaryDirectory()
        self.store = EducationMemoryStore(
            Path(self.temporary.name) / "multi-agent.sqlite3"
        )

    def tearDown(self):
        """测试结束后删除临时数据库目录。"""

        self.temporary.cleanup()

    def _build_agent(
        self,
        subject: StubSubjectAgent,
        reviewer: StubReviewAgent,
        memory: StubMemoryAgent,
        safety: StubSafetyAgent | None = None,
    ) -> EducationAgent:
        """注入全部四个假子 Agent，避免测试创建在线模型。"""

        return EducationAgent(
            checkpointer=InMemorySaver(),
            memory_store=self.store,
            subject_tool=build_subject_agent_tool(agent=subject),
            review_tool=build_review_agent_tool(agent=reviewer),
            safety_review_tool=build_safety_review_agent_tool(
                agent=safety if safety is not None else StubSafetyAgent()
            ),
            memory_tool=build_memory_agent_tool(agent=memory),
        )

    @staticmethod
    def _latest_state(agent: EducationAgent, student_id: str, thread_id: str):
        return agent.agent.get_state(
            {
                "configurable": {
                    "thread_id": agent.checkpoint_thread_id(student_id, thread_id)
                }
            }
        ).values

    def _assert_only_final_answer_saved(
        self,
        agent: EducationAgent,
        student_id: str,
        thread_id: str,
        question: str,
        answer: str,
    ):
        history = self.store.get_history(student_id=student_id, thread_id=thread_id)
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual([item["content"] for item in history], [question, answer])
        state = self._latest_state(agent, student_id, thread_id)
        self.assertEqual([message.content for message in state["messages"]], [question, answer])
        return state

    def test_approved_answer_runs_three_subagent_tools(self):
        """验证一次通过时依次执行学科、审核、记忆 Tool 并只保存最终问答。"""

        subject = StubSubjectAgent(["周长是一周的长度。"])
        reviewer = StubReviewAgent(
            [ReviewResult(approved=True, score=95, feedback="答案正确")]
        )
        memory = StubMemoryAgent(
            MemoryExtraction(
                memories=[
                    MemoryCandidate(
                        memory_type="preference",
                        subject="综合",
                        content="学生偏好分步骤讲解。",
                        confidence=0.9,
                    )
                ]
            )
        )
        safety = StubSafetyAgent()
        agent = self._build_agent(subject, reviewer, memory, safety)

        answer = agent.answer(
            "请一步一步讲周长。",
            "数学",
            4,
            student_id="student-multi-001",
            thread_id="thread-multi-001",
        )

        self.assertEqual(answer, "周长是一周的长度。")
        self.assertEqual(len(subject.prompts), 1)
        self.assertIn("项目教育 Skill 开始", subject.prompts[0])
        self.assertIn("primary-math", subject.prompts[0])
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(safety.prompts, [])
        self.assertEqual(len(memory.prompts), 1)
        history = self.store.get_history(
            student_id="student-multi-001",
            thread_id="thread-multi-001",
        )
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[-1]["content"], answer)
        self.assertTrue(
            any(
                item["memory_type"] == "preference"
                for item in self.store.list_memories(student_id="student-multi-001")
            )
        )

    def test_rejected_answer_is_rewritten_once(self):
        """验证首次拒绝后审核意见进入第二份草稿，最终只保存修正版。"""

        subject = StubSubjectAgent(["错误草稿", "修改后的正确答案"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(
                    approved=False,
                    score=40,
                    feedback="计算结果错误，请重新计算。",
                    corrected_answer="审核修正版",
                ),
            ]
        )
        memory = StubMemoryAgent()
        safety = StubSafetyAgent()
        agent = self._build_agent(subject, reviewer, memory, safety)

        answer = agent.answer(
            "12乘3是多少？",
            "数学",
            3,
            student_id="student-multi-002",
            thread_id="thread-multi-002",
        )

        self.assertEqual(answer, "修改后的正确答案")
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(len(safety.prompts), 1)
        self.assertEqual(
            json.loads(safety.prompts[0]),
            {
                "question": "12乘3是多少？",
                "draft_answer": "修改后的正确答案",
                "subject": "数学",
                "grade": 3,
            },
        )
        self.assertIn("计算结果错误，请重新计算", subject.prompts[1])
        state = self._assert_only_final_answer_saved(
            agent,
            "student-multi-002",
            "thread-multi-002",
            "12乘3是多少？",
            answer,
        )
        self.assertEqual(state["review_mode"], "safety_only")
        self.assertIsNone(state["review_score"])
        self.assertIs(state["safety_passed"], True)
        self.assertFalse(state["safety_review_error"])

    def test_memory_failure_saves_only_core_memory_and_preserves_answer(self):
        """提取异常保留答案，不用关键词把题干引用误记为长期困难。"""

        subject = StubSubjectAgent(["三角形有三条边。"])
        reviewer = StubReviewAgent(
            [ReviewResult(approved=True, score=98, feedback="答案正确")]
        )
        memory = StubMemoryAgent(fail=True)
        agent = self._build_agent(subject, reviewer, memory)

        answer = agent.answer(
            "小明说‘我不会数三角形的边’，请一步一步解释这句话。",
            "数学",
            2,
            student_id="student-multi-003",
            thread_id="thread-multi-003",
        )

        self.assertEqual(answer, "三角形有三条边。")
        history = self.store.get_history(
            student_id="student-multi-003",
            thread_id="thread-multi-003",
        )
        self.assertEqual(len(history), 2)
        memories = self.store.list_memories(student_id="student-multi-003")
        self.assertEqual({item["memory_type"] for item in memories}, {"profile", "learning_topic"})
        state = self._latest_state(agent, "student-multi-003", "thread-multi-003")
        self.assertEqual(state["memory_candidates"], [])

    def test_successful_empty_memory_does_not_trigger_keyword_fallback(self):
        """候选为空是成功结论，题目中的规则关键词不能覆盖该结论。"""

        agent = self._build_agent(
            StubSubjectAgent(["先看人物对话。"]),
            StubReviewAgent([ReviewResult(approved=True, score=95, feedback="通过")]),
            StubMemoryAgent(MemoryExtraction(memories=[])),
        )
        question = "请分析人物说‘我不会’，这句话有什么作用？给出解题步骤。"
        answer = agent.answer(
            question, "语文", 4,
            student_id="student-empty-memory", thread_id="thread-empty-memory",
        )

        self.assertEqual(answer, "先看人物对话。")
        memories = self.store.list_memories(student_id="student-empty-memory")
        self.assertEqual({item["memory_type"] for item in memories}, {"profile", "learning_topic"})
        self._assert_only_final_answer_saved(
            agent, "student-empty-memory", "thread-empty-memory", question, answer,
        )

    def test_safe_rewrite_is_returned_even_with_ordinary_quality_errors(self):
        """第二次只看安全，普通事实错误和多余质量字段不能触发第三次生成。"""

        subject = StubSubjectAgent(["第一份错误草稿", "12乘3等于35。"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(
                    approved=False,
                    score=30,
                    feedback="请修改事实错误。",
                    corrected_answer="第一份审核修正版",
                ),
            ]
        )
        memory = StubMemoryAgent()
        safety = StubSafetyAgent(
            [
                {
                    "safety_passed": True,
                    "feedback": "没有安全风险；算式结果仍有普通事实错误。",
                    "risk_categories": [],
                    "approved": False,
                    "score": 0,
                    "corrected_answer": "不能用审核建议替换最终草稿",
                }
            ]
        )
        agent = self._build_agent(subject, reviewer, memory, safety)

        answer = agent.answer(
            "12乘3是多少？",
            "数学",
            3,
            student_id="student-multi-005",
            thread_id="thread-multi-005",
        )

        self.assertEqual(answer, "12乘3等于35。")
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(len(safety.prompts), 1)
        state = self._assert_only_final_answer_saved(
            agent,
            "student-multi-005",
            "thread-multi-005",
            "12乘3是多少？",
            answer,
        )
        self.assertEqual(state["review_mode"], "safety_only")
        self.assertIsNone(state["review_score"])
        self.assertIs(state["safety_passed"], True)
        self.assertEqual(state["corrected_answer"], "")

    def test_unsafe_rewrite_uses_fixed_fallback_and_never_reviewer_suggestion(self):
        """危险重写被固定提示阻断，完整审核或安全审核的建议稿不能泄漏。"""

        subject = StubSubjectAgent(["初始草稿", "重写稿包含危险步骤"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(
                    approved=False,
                    score=20,
                    feedback="请修改不安全内容。",
                    corrected_answer="首次审核建议也含危险步骤",
                )
            ]
        )
        safety = StubSafetyAgent(
            [
                {
                    "safety_passed": False,
                    "risk_categories": ["dangerous_instructions"],
                    "feedback": "存在可执行的危险步骤。",
                    "approved": True,
                    "score": 100,
                    "corrected_answer": "第二次审核建议也不能直接展示",
                }
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory, safety)

        answer = agent.answer(
            "测试安全阻断",
            "科学",
            5,
            student_id="student-unsafe",
            thread_id="thread-unsafe",
        )

        self.assertEqual(answer, SAFETY_BLOCKED_ANSWER)
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(len(safety.prompts), 1)
        self.assertEqual(len(memory.prompts), 1)
        state = self._assert_only_final_answer_saved(
            agent, "student-unsafe", "thread-unsafe", "测试安全阻断", answer
        )
        self.assertIs(state["safety_passed"], False)
        self.assertFalse(state["safety_review_error"])
        self.assertEqual(state["review_mode"], "safety_only")
        self.assertIsNone(state["review_score"])
        self.assertEqual(state["corrected_answer"], "")

    def test_safety_failure_or_malformed_result_never_releases_draft(self):
        """安全审核不可用、JSON 损坏或结论缺失时只显示固定临时提示。"""

        cases = {
            "exception": StubSafetyAgent(fail=True),
            "malformed_json": StubSafetyAgent(text_response="这不是 JSON"),
            "missing_conclusion": StubSafetyAgent(
                [{"approved": True, "score": 100, "feedback": "通过"}]
            ),
            "contradictory_conclusion": StubSafetyAgent(
                [
                    {
                        "safety_passed": True,
                        "risk_categories": ["harmful_advice"],
                        "feedback": "存在有害建议",
                    }
                ]
            ),
        }
        for name, safety in cases.items():
            with self.subTest(case=name):
                subject = StubSubjectAgent(["初始草稿", "尚未确认安全的重写草稿"])
                reviewer = StubReviewAgent(
                    [
                        ReviewResult(
                            approved=False,
                            score=10,
                            feedback="请重写",
                            corrected_answer="未经安全确认的建议稿",
                        )
                    ]
                )
                agent = self._build_agent(subject, reviewer, StubMemoryAgent(), safety)
                student_id = f"student-safety-error-{name}"
                thread_id = f"thread-safety-error-{name}"

                answer = agent.answer(
                    "测试安全检查异常",
                    "综合",
                    4,
                    student_id=student_id,
                    thread_id=thread_id,
                )

                self.assertEqual(answer, SAFETY_REVIEW_UNAVAILABLE_ANSWER)
                self.assertEqual(len(subject.prompts), 2)
                self.assertEqual(len(reviewer.prompts), 1)
                self.assertEqual(len(safety.prompts), 1)
                state = self._assert_only_final_answer_saved(
                    agent, student_id, thread_id, "测试安全检查异常", answer
                )
                self.assertEqual(state["review_mode"], "safety_only")
                self.assertIsNone(state["review_score"])
                self.assertIsNone(state["safety_passed"])
                self.assertTrue(state["safety_review_error"])

    def test_each_new_turn_starts_with_one_full_review(self):
        """相同线程的新一轮不能沿用上一轮安全模式、异常状态或重写次数。"""

        subject = StubSubjectAgent(["第一轮初稿", "第一轮重写", "第二轮直接通过", "第三轮初稿", "第三轮重写"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(approved=False, score=10, feedback="第一轮请重写"),
                ReviewResult(approved=True, score=95, feedback="第二轮通过"),
                ReviewResult(approved=False, score=20, feedback="第三轮请重写"),
            ]
        )
        safety = StubSafetyAgent(
            [
                SafetyReviewResult(safety_passed=True, feedback="第一轮安全"),
                SafetyReviewResult(safety_passed=True, feedback="第三轮安全"),
            ]
        )
        agent = self._build_agent(subject, reviewer, StubMemoryAgent(), safety)
        common = {"student_id": "student-reset", "thread_id": "thread-reset"}

        self.assertEqual(agent.answer("第一轮问题", "数学", 3, **common), "第一轮重写")
        self.assertEqual(agent.answer("第二轮问题", "数学", 3, **common), "第二轮直接通过")
        second_state = self._latest_state(agent, **common)
        self.assertEqual(second_state["review_mode"], "full")
        self.assertEqual(second_state["review_attempts"], 1)
        self.assertEqual(second_state["review_score"], 95)
        self.assertIsNone(second_state["safety_passed"])
        self.assertFalse(second_state["safety_review_error"])
        self.assertEqual(len(safety.prompts), 1)
        self.assertNotIn("第一轮请重写", subject.prompts[2])
        self.assertEqual(agent.answer("第三轮问题", "数学", 3, **common), "第三轮重写")
        self.assertEqual(len(reviewer.prompts), 3)
        self.assertEqual(len(safety.prompts), 2)
        self.assertTrue(all("第 1 次审核" in prompt for prompt in reviewer.prompts))
        history = self.store.get_history(**common)
        self.assertEqual(
            [item["content"] for item in history],
            ["第一轮问题", "第一轮重写", "第二轮问题", "第二轮直接通过", "第三轮问题", "第三轮重写"],
        )

    def test_retry_setting_above_one_cannot_generate_a_third_draft(self):
        """历史配置即使允许多次重试，也只能完整审核后重写一次。"""

        subject = StubSubjectAgent(["首次草稿", "安全重写"])
        reviewer = StubReviewAgent(
            [ReviewResult(approved=False, score=10, feedback="请重写")]
        )
        safety = StubSafetyAgent()
        with patch.dict(app_conf["agent"], {"max_review_retries": 5}):
            agent = self._build_agent(subject, reviewer, StubMemoryAgent(), safety)

        answer = agent.answer(
            "测试重试上限", "综合", 4,
            student_id="student-retry-limit", thread_id="thread-retry-limit",
        )

        self.assertEqual(answer, "安全重写")
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(len(safety.prompts), 1)

    def test_zero_retries_uses_fixed_full_review_fallback_without_safety(self):
        """禁用重写时不检查第二份草稿，也不展示审核员提供的建议稿。"""

        subject = StubSubjectAgent(["未通过的初稿"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(
                    approved=False, score=10, feedback="请重写",
                    corrected_answer="没有接受安全审核的建议稿",
                )
            ]
        )
        safety = StubSafetyAgent()
        with patch.dict(app_conf["agent"], {"max_review_retries": 0}):
            agent = self._build_agent(subject, reviewer, StubMemoryAgent(), safety)

        answer = agent.answer(
            "禁用重写的测试", "综合", 4,
            student_id="student-no-retry", thread_id="thread-no-retry",
        )

        self.assertEqual(answer, FULL_REVIEW_FALLBACK_ANSWER)
        self.assertEqual(len(subject.prompts), 1)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(safety.prompts, [])
        self._assert_only_final_answer_saved(
            agent, "student-no-retry", "thread-no-retry", "禁用重写的测试", answer
        )

    def test_short_term_context_is_reused_only_in_same_thread(self):
        """验证同一线程能看到上一轮最终回答，而不同线程不会串用上下文。"""

        subject = StubSubjectAgent(["第一轮答案", "第二轮答案", "新线程答案"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(approved=True, score=90, feedback="通过"),
                ReviewResult(approved=True, score=90, feedback="通过"),
                ReviewResult(approved=True, score=90, feedback="通过"),
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory)

        common = {
            "subject": "科学",
            "grade": 4,
            "student_id": "student-multi-006",
        }
        agent.answer("第一问", thread_id="thread-multi-006", **common)
        agent.answer("接着问", thread_id="thread-multi-006", **common)
        agent.answer("新会话问题", thread_id="thread-multi-other", **common)

        self.assertIn("第一轮答案", subject.prompts[1])
        self.assertNotIn("第一轮答案", subject.prompts[2])

    def test_turn_id_prevents_duplicate_messages(self):
        """验证同一轮存储任务被重试时不会再次插入用户和助手消息。"""

        common = {
            "student_id": "student-multi-004",
            "thread_id": "thread-multi-004",
            "turn_id": "turn-fixed-004",
            "question": "什么是长方形？",
            "answer": "长方形有四个直角。",
            "subject": "数学",
            "grade": 3,
            "memory_candidates": [],
        }

        self.assertTrue(self.store.record_turn(**common))
        self.assertFalse(self.store.record_turn(**common))
        history = self.store.get_history(
            student_id="student-multi-004",
            thread_id="thread-multi-004",
        )
        self.assertEqual(len(history), 2)

    def test_long_thread_is_summarized_in_checkpoint_state(self):
        """验证长会话把旧消息变成摘要，最新 State 只保留最近原文。"""

        # 每轮结束产生“学生 + 助手”两条消息；下一轮学生问题进入后触发压缩。
        trigger_messages = int(app_conf["context"]["summary_trigger_messages"])
        turn_count = trigger_messages // 2 + 1
        subject = StubSubjectAgent(
            [f"第{index}轮答案" for index in range(1, turn_count + 1)]
        )
        reviewer = StubReviewAgent(
            [
                ReviewResult(approved=True, score=90, feedback="通过")
                for _ in range(turn_count)
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory)
        student_id = "student-summary-001"
        thread_id = "thread-summary-001"

        for index in range(1, turn_count + 1):
            agent.answer(
                f"第{index}轮问题",
                "科学",
                4,
                student_id=student_id,
                thread_id=thread_id,
            )

        checkpoint_config = {
            "configurable": {
                "thread_id": agent.checkpoint_thread_id(student_id, thread_id)
            }
        }
        latest_state = agent.agent.get_state(checkpoint_config).values
        self.assertIn("第1轮问题", latest_state["conversation_summary"])
        self.assertLessEqual(
            len(latest_state["messages"]),
            int(app_conf["context"]["summary_keep_recent_messages"]) + 1,
        )
        self.assertNotIn(
            "第1轮问题",
            [message.content for message in latest_state["messages"]],
        )
        self.assertIn("较早会话摘要开始", subject.prompts[-1])


if __name__ == "__main__":
    unittest.main()
