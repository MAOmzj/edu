"""多 Agent 子 Agent-as-Tool 工作流测试，全程不调用在线模型。"""

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.education_agent import EducationAgent
from agent.multi_agent.memory_agent import build_memory_agent_tool
from agent.multi_agent.review_agent import build_review_agent_tool
from agent.multi_agent.state import (
    MemoryCandidate,
    MemoryExtraction,
    ReviewResult,
)
from agent.multi_agent.subject_agent import build_subject_agent_tool
from memory.store import EducationMemoryStore


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


class MultiAgentWorkflowTests(unittest.TestCase):
    """验证学科、审核和记忆三个子 Agent Tool 的完整协调过程。"""

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
    ) -> EducationAgent:
        """把三个假子 Agent 包成真实 Tool，再构造不联网的 EducationAgent。"""

        return EducationAgent(
            checkpointer=InMemorySaver(),
            memory_store=self.store,
            subject_tool=build_subject_agent_tool(agent=subject),
            review_tool=build_review_agent_tool(agent=reviewer),
            memory_tool=build_memory_agent_tool(agent=memory),
        )

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
        agent = self._build_agent(subject, reviewer, memory)

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
                ReviewResult(approved=True, score=96, feedback="修改正确"),
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory)

        answer = agent.answer(
            "12乘3是多少？",
            "数学",
            3,
            student_id="student-multi-002",
            thread_id="thread-multi-002",
        )

        self.assertEqual(answer, "修改后的正确答案")
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 2)
        self.assertIn("计算结果错误，请重新计算", subject.prompts[1])
        history = self.store.get_history(
            student_id="student-multi-002",
            thread_id="thread-multi-002",
        )
        self.assertEqual(len(history), 2)
        self.assertNotIn("错误草稿", str(history))

    def test_memory_failure_falls_back_to_rule_memory(self):
        """验证记忆 Agent 异常不会吞掉通过审核的答案。"""

        subject = StubSubjectAgent(["三角形有三条边。"])
        reviewer = StubReviewAgent(
            [ReviewResult(approved=True, score=98, feedback="答案正确")]
        )
        memory = StubMemoryAgent(fail=True)
        agent = self._build_agent(subject, reviewer, memory)

        answer = agent.answer(
            "三角形有几条边？",
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
        self.assertTrue(any(item["memory_type"] == "learning_topic" for item in memories))

    def test_two_rejections_use_reviewer_safe_answer(self):
        """验证连续两次拒绝后停止循环，并返回审核 Agent 的安全修正版。"""

        subject = StubSubjectAgent(["第一份错误草稿", "第二份仍有错误的草稿"])
        reviewer = StubReviewAgent(
            [
                ReviewResult(
                    approved=False,
                    score=30,
                    feedback="请修改事实错误。",
                    corrected_answer="第一份审核修正版",
                ),
                ReviewResult(
                    approved=False,
                    score=50,
                    feedback="仍然不够准确。",
                    corrected_answer="这是审核员给出的安全正确答案。",
                ),
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory)

        answer = agent.answer(
            "测试审核兜底",
            "综合",
            5,
            student_id="student-multi-005",
            thread_id="thread-multi-005",
        )

        self.assertEqual(answer, "这是审核员给出的安全正确答案。")
        self.assertEqual(len(subject.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 2)
        history = self.store.get_history(
            student_id="student-multi-005",
            thread_id="thread-multi-005",
        )
        self.assertEqual(history[-1]["content"], answer)

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

        subject = StubSubjectAgent([f"第{index}轮答案" for index in range(1, 7)])
        reviewer = StubReviewAgent(
            [
                ReviewResult(approved=True, score=90, feedback="通过")
                for _ in range(6)
            ]
        )
        memory = StubMemoryAgent()
        agent = self._build_agent(subject, reviewer, memory)
        student_id = "student-summary-001"
        thread_id = "thread-summary-001"

        for index in range(1, 7):
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
        self.assertLessEqual(len(latest_state["messages"]), 8)
        self.assertNotIn(
            "第1轮问题",
            [message.content for message in latest_state["messages"]],
        )
        self.assertIn("较早会话摘要开始", subject.prompts[-1])


if __name__ == "__main__":
    unittest.main()
