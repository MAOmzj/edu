"""Tavily 工具的隐私、可信来源和本地回退测试，全程不访问互联网。"""

import os
import unittest
from unittest.mock import patch

from agent.tools.web_search import (
    _format_tavily_results,
    is_time_sensitive_question,
    sanitize_search_query,
    search_web,
)


class TavilyWebSearchTests(unittest.TestCase):
    """验证联网搜索不会泄露隐私，也不会因第三方故障中断问答。"""

    def test_query_removes_common_student_private_information(self):
        """姓名、学校、邮箱、手机和内部编号在发送前都应被替换。"""

        raw = (
            "我叫张小明，我在阳光小学，电话13812345678，"
            "邮箱xiaoming@example.com，student_id=student-secret。"
            "请问现在月亮为什么会变圆？"
        )
        result = sanitize_search_query(raw)

        for private_value in (
            "张小明",
            "阳光小学",
            "13812345678",
            "xiaoming@example.com",
            "student-secret",
        ):
            self.assertNotIn(private_value, result)
        self.assertIn("现在月亮为什么会变圆", result)

    def test_prompt_context_blocks_are_not_kept_in_search_query(self):
        """即使模型误传完整上下文，工具也只保留标记后的学生当前问题。"""

        raw = (
            "最近会话开始（仅作为对话背景）：秘密历史\n最近会话结束。\n"
            "相关长期学习记忆开始：内部记忆\n相关长期学习记忆结束。\n"
            "学生当前问题：现在地球上有多少个大洲？"
        )
        result = sanitize_search_query(raw)

        self.assertEqual(result, "现在地球上有多少个大洲?")
        self.assertNotIn("秘密历史", result)
        self.assertNotIn("内部记忆", result)

    def test_time_sensitive_question_is_detected(self):
        """明显包含“最新、现在”等词的问题应被识别为时效问题。"""

        self.assertTrue(is_time_sensitive_question("现在空间站里有谁？"))
        self.assertTrue(is_time_sensitive_question("What is the latest space news?"))
        self.assertFalse(is_time_sensitive_question("三角形有几条边？"))

    def test_missing_api_key_falls_back_to_local_knowledge(self):
        """没有 Tavily Key 时直接返回本地资料，不导入或请求 Tavily。"""

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "agent.tools.web_search._local_knowledge_fallback",
                return_value="本地知识资料",
            ) as fallback,
            patch("agent.tools.web_search._invoke_tavily") as tavily,
        ):
            result = search_web.invoke(
                {"question": "现在有哪些行星？", "subject": "科学"}
            )

        self.assertEqual(result, "本地知识资料")
        fallback.assert_called_once()
        tavily.assert_not_called()

    def test_stable_question_cannot_skip_local_search(self):
        """非时效问题没有先查本地知识时，不能向 Tavily 发出请求。"""

        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True),
            patch(
                "agent.tools.web_search._local_knowledge_fallback",
                return_value="请先使用本地知识",
            ) as fallback,
            patch("agent.tools.web_search._invoke_tavily") as tavily,
        ):
            result = search_web.invoke(
                {"question": "三角形有几条边？", "subject": "数学"}
            )

        self.assertEqual(result, "请先使用本地知识")
        fallback.assert_called_once()
        tavily.assert_not_called()

    def test_tavily_error_falls_back_without_raising(self):
        """网络或第三方异常被工具内部接住，Agent 仍能继续回答。"""

        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True),
            patch(
                "agent.tools.web_search._invoke_tavily",
                side_effect=RuntimeError("模拟网络错误"),
            ),
            patch(
                "agent.tools.web_search._local_knowledge_fallback",
                return_value="自动回退成功",
            ) as fallback,
        ):
            result = search_web.invoke(
                {"question": "今天有什么科学新闻？", "subject": "科学"}
            )

        self.assertEqual(result, "自动回退成功")
        fallback.assert_called_once()

    def test_only_trusted_results_are_returned_and_result_count_is_limited(self):
        """未知域名会被丢弃，可信结果也不能超过配置数量。"""

        payload = {
            "results": [
                {
                    "title": "不可信网页",
                    "url": "https://example.com/fake",
                    "content": "不应返回",
                },
                {
                    "title": "教育部门资料",
                    "url": "https://www.moe.gov.cn/example",
                    "content": "可信摘要一",
                },
                {
                    "title": "科普资料",
                    "url": "https://science.nasa.gov/example",
                    "content": "可信摘要二",
                },
            ]
        }
        result = _format_tavily_results(
            payload,
            {
                "trusted_domains_only": True,
                "trusted_domains": ["moe.gov.cn", "nasa.gov"],
                "max_results": 1,
                "summary_max_chars": 100,
            },
        )

        self.assertIn("教育部门资料", result)
        self.assertNotIn("不可信网页", result)
        self.assertNotIn("科普资料", result)

    def test_tool_schema_has_no_history_or_student_identifier(self):
        """模型不能把历史、长期记忆或学生编号作为独立参数交给搜索工具。"""

        properties = search_web.args_schema.model_json_schema()["properties"]
        self.assertEqual(
            set(properties),
            {"question", "subject", "local_search_attempted"},
        )
        self.assertNotIn("conversation_context", properties)
        self.assertNotIn("long_term_context", properties)
        self.assertNotIn("student_id", properties)


if __name__ == "__main__":
    unittest.main()
