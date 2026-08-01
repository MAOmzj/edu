import os
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
        return f"{subject}-{grade}-{student_id}-{thread_id}-{question}"


class WebApiTests(unittest.TestCase):
    def setUp(self):
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
        self.client_context.__exit__(None, None, None)
        web_app._agent_override = self.original_agent
        if self.previous_database_path is None:
            os.environ.pop("EDUCATION_MEMORY_DB_PATH", None)
        else:
            os.environ["EDUCATION_MEMORY_DB_PATH"] = self.previous_database_path
        self.temporary.cleanup()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("小学知识小助手", response.text)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["memory"]["backend"], "sqlite")

    def test_ask(self):
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

    def test_rejects_unknown_subject(self):
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
        response = self.client.get(
            f"/api/history/{THREAD_ID}",
            params={"student_id": STUDENT_ID},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["messages"], [])


if __name__ == "__main__":
    unittest.main()
