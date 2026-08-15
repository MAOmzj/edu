# 文件用途：验证首页、健康检查、普通问答、最终答案流和历史 API 的 HTTP 合同。
# 调用关系：unittest/TestClient 调用 app.py；本文件用假 Runtime/Agent 隔离在线模型。
# 修改易踩坑：导入 app 前要设置测试环境，流事件只能断言公开数据，不能泄露内部 Agent 状态。
import os
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app as web_app


STUDENT_ID = "student-test-001"
THREAD_ID = "thread-test-001"


class FakeEducationAgent:
    def answer(
        self,
        question,
        subject,
        grade,
        *,
        student_id,
        thread_id,
    ):
        """返回包含全部输入字段的固定文本，避免接口测试调用在线模型。"""
        return f"{subject}-{grade}-{student_id}-{thread_id}-{question}"


class WebApiTests(unittest.TestCase):
    def setUp(self):
        """为每个接口测试创建临时数据库、测试客户端和假 Agent。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_database_path = os.environ.get("EDUCATION_MEMORY_DB_PATH")
        os.environ["EDUCATION_MEMORY_DB_PATH"] = str(
            Path(self.temporary.name) / "web-memory.sqlite3"
        )
        self.original_agent = web_app._agent_override
        web_app._agent_override = FakeEducationAgent()
        self.client_context = TestClient(web_app.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        """关闭测试客户端并恢复 Agent、环境变量和临时目录。"""
        self.client_context.__exit__(None, None, None)
        web_app._agent_override = self.original_agent
        if self.previous_database_path is None:
            os.environ.pop("EDUCATION_MEMORY_DB_PATH", None)
        else:
            os.environ["EDUCATION_MEMORY_DB_PATH"] = self.previous_database_path
        self.temporary.cleanup()

    def test_home_page(self):
        """验证首页能够访问并显示小学知识助手标题。"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("小学知识小助手", response.text)

    def test_health(self):
        """验证健康接口报告正常状态和 SQLite 记忆后端。"""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["memory"]["backend"], "sqlite")

    def test_ask(self):
        """验证问答接口正确接收参数并返回 Agent 的结构化结果。"""
        response = self.client.post(
            "/api/ask",
            json={
                "question": "三角形内角和？",
                "subject": "数学",
                "grade": 5,
                "student_id": STUDENT_ID,
                "thread_id": THREAD_ID,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("数学-5-student-test-001", response.json()["answer"])
        self.assertEqual(response.json()["thread_id"], THREAD_ID)

    def test_ask_stream_only_contains_final_answer_events(self):
        """验证流式接口只分段发送最终答案，并用 done 事件明确结束。"""

        response = self.client.post(
            "/api/ask/stream",
            json={
                "question": "三角形内角和？",
                "subject": "数学",
                "grade": 5,
                "student_id": STUDENT_ID,
                "thread_id": THREAD_ID,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        answer_parts = []
        event_names = []
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            event_name = next(
                line.removeprefix("event: ")
                for line in lines
                if line.startswith("event: ")
            )
            event_names.append(event_name)
            if event_name == "answer_delta":
                data_line = next(
                    line.removeprefix("data: ")
                    for line in lines
                    if line.startswith("data: ")
                )
                answer_parts.append(json.loads(data_line)["text"])

        self.assertEqual(event_names[-1], "done")
        self.assertNotIn("status", event_names)
        self.assertEqual(
            "".join(answer_parts),
            f"数学-5-{STUDENT_ID}-{THREAD_ID}-三角形内角和？",
        )

    def test_rejects_unknown_subject(self):
        """验证问答接口拒绝不在配置白名单中的未知学科。"""
        response = self.client.post(
            "/api/ask",
            json={
                "question": "测试",
                "subject": "未知学科",
                "student_id": STUDENT_ID,
                "thread_id": THREAD_ID,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_empty_history(self):
        """验证尚未产生消息的新会话返回空历史记录。"""
        response = self.client.get(
            f"/api/history/{THREAD_ID}",
            params={"student_id": STUDENT_ID},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["messages"], [])


if __name__ == "__main__":
    unittest.main()
