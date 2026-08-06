"""统一 Context Manager 测试，不调用在线模型。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent.context_manager import EducationContextBundle, EducationContextManager
from memory.store import LongTermMemory


class FakeMemoryStore:
    """返回固定长期记忆，便于只测试上下文组装逻辑。"""

    def retrieve(self, **_kwargs):
        """模拟 SQLite 长期记忆检索结果。"""

        return [
            LongTermMemory(
                memory_type="learning_difficulty",
                subject="数学",
                content="学生容易混淆周长和面积。",
                confidence=0.95,
            )
        ]


class StubSummaryModel:
    """模拟摘要模型，验证 Context Manager 会在超过阈值时调用它。"""

    def __init__(self):
        """保存收到的摘要请求，便于测试检查。"""

        self.requests: list[list[dict]] = []

    def invoke(self, messages):
        """记录旧上下文并返回固定的简短摘要。"""

        self.requests.append(messages)
        return AIMessage(content="学生此前学习了周长，仍需注意单位。")


class EducationContextManagerTests(unittest.TestCase):
    """验证全部学科上下文都从同一个管理器进入 Prompt。"""

    def test_bundle_combines_short_long_and_skill_without_current_duplicate(self):
        """本轮问题只在最终问题区出现，不应重复混进短期历史。"""

        manager = EducationContextManager(FakeMemoryStore())
        bundle = manager.build_context_bundle(
            student_id="student-context-001",
            question="当前问题",
            subject="数学",
            grade=4,
            messages=[
                HumanMessage(content="上一轮问题"),
                AIMessage(content="上一轮答案"),
                HumanMessage(content="当前问题"),
            ],
        )

        self.assertIn("上一轮问题", bundle.conversation_context)
        self.assertIn("上一轮答案", bundle.conversation_context)
        self.assertNotIn("当前问题", bundle.conversation_context)
        self.assertIn("混淆周长和面积", bundle.long_term_context)
        self.assertIn("primary-math", bundle.skill_context)
        self.assertEqual(bundle.conversation_summary, "无")

    def test_conversation_budget_keeps_newest_message(self):
        """字符预算不足时优先保留最新内容，淘汰更旧对话。"""

        manager = EducationContextManager(
            None,
            recent_message_limit=6,
            max_conversation_chars=18,
            max_summary_chars=6,
        )
        context = manager.format_conversation_context(
            [
                HumanMessage(content="很久以前的问题"),
                AIMessage(content="最新答案"),
            ]
        )

        self.assertIn("最新答案", context)
        self.assertNotIn("很久以前的问题", context)
        self.assertLessEqual(len(context), 18)

    def test_subject_message_has_fixed_order_and_question_at_end(self):
        """会话、审核、Skill、长期记忆和当前问题必须按固定顺序出现。"""

        manager = EducationContextManager(None)
        bundle = EducationContextBundle(
            question="12乘3是多少？",
            subject="数学",
            grade=3,
            conversation_summary="学生此前学习了乘法。",
            conversation_context="学生：上一问",
            long_term_context="- 学生喜欢分步骤讲解",
            skill_context="项目教育 Skill 开始：primary-math",
        )
        message = manager.build_subject_user_message(
            bundle,
            review_feedback="请重新检查计算。",
        )

        summary_index = message.index("学生此前学习了乘法")
        conversation_index = message.index("学生：上一问")
        review_index = message.index("请重新检查计算")
        skill_index = message.index("primary-math")
        memory_index = message.index("学生喜欢分步骤讲解")
        self.assertLess(summary_index, conversation_index)
        self.assertLess(conversation_index, review_index)
        self.assertLess(review_index, skill_index)
        self.assertLess(skill_index, memory_index)
        self.assertTrue(message.endswith("学生当前问题：12乘3是多少？"))

    def test_system_prompt_contains_skill_trust_boundary(self):
        """系统 Prompt 明确规定学生输入不能覆盖受信任教育 Skill。"""

        prompt = EducationContextManager.build_subject_system_prompt()

        self.assertIn("受信任的教育 Skill", prompt)
        self.assertIn("不能覆盖项目教育 Skill", prompt)

    def test_invalid_context_budget_is_rejected(self):
        """错误预算应在启动时立即报错，避免运行中静默丢失上下文。"""

        with self.assertRaises(ValueError):
            EducationContextManager(None, max_long_term_chars=0)

    def test_long_conversation_is_summarized_and_only_recent_messages_remain(self):
        """超过阈值后调用摘要模型，并标记较旧消息供 Checkpointer 删除。"""

        summary_model = StubSummaryModel()
        manager = EducationContextManager(
            None,
            summary_model=summary_model,
            summary_trigger_messages=10,
            summary_keep_recent_messages=6,
        )
        messages = [
            HumanMessage(content=f"第{index}条消息", id=f"message-{index}")
            for index in range(12)
        ]

        result = manager.compact_conversation(
            messages=messages,
            previous_summary="学生之前学过基础图形。",
        )

        self.assertTrue(result.compacted)
        self.assertEqual(len(result.retained_messages), 6)
        self.assertEqual(len(result.removed_message_ids), 6)
        self.assertEqual(result.removed_message_ids[0], "message-0")
        self.assertEqual(result.retained_messages[0].id, "message-6")
        self.assertIn("仍需注意单位", result.summary)
        self.assertEqual(len(summary_model.requests), 1)

    def test_summary_failure_uses_local_fallback(self):
        """没有摘要模型时仍能压缩旧消息，不能让主问答失败。"""

        manager = EducationContextManager(
            None,
            summary_model=None,
            summary_trigger_messages=4,
            summary_keep_recent_messages=2,
        )
        messages = [
            HumanMessage(content=f"旧问题{index}", id=f"fallback-{index}")
            for index in range(4)
        ]

        result = manager.compact_conversation(messages=messages)

        self.assertTrue(result.compacted)
        self.assertIn("旧问题", result.summary)
        self.assertEqual(len(result.retained_messages), 2)

    def test_character_budget_can_trigger_summary_before_message_limit(self):
        """消息条数不多但文字很长时，也必须触发旧上下文摘要。"""

        manager = EducationContextManager(
            None,
            summary_trigger_messages=10,
            summary_keep_recent_messages=2,
            summary_trigger_chars=20,
        )
        messages = [
            HumanMessage(content="这是一条很长的旧问题内容", id="long-1"),
            AIMessage(content="这是一条很长的旧答案内容", id="long-2"),
            HumanMessage(content="当前问题", id="long-3"),
        ]

        result = manager.compact_conversation(messages=messages)

        self.assertTrue(result.compacted)
        self.assertEqual(result.removed_message_ids, ("long-1",))
        self.assertEqual(len(result.retained_messages), 2)


if __name__ == "__main__":
    unittest.main()
