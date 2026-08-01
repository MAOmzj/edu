import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph

from agent.context_manager import EducationContextManager
from memory.store import EducationMemoryStore


class EducationMemoryStoreTests(unittest.TestCase):
    def test_history_long_term_retrieval_and_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "memory.sqlite3"
            store = EducationMemoryStore(database_path)
            student_id = "student-memory-001"

            store.record_turn(
                student_id=student_id,
                thread_id="thread-memory-001",
                question="我总是分不清面积和周长，请简单一点讲。",
                answer="面积表示大小，周长表示一周的长度。",
                subject="数学",
                grade=4,
            )
            store.record_turn(
                student_id=student_id,
                thread_id="thread-memory-002",
                question="请举例说明长方形周长。",
                answer="例如长5米、宽3米，周长是16米。",
                subject="数学",
                grade=4,
            )
            store.record_turn(
                student_id=student_id,
                thread_id="thread-memory-001",
                question="我的邮箱是 student@example.com，请不要保存。",
                answer="我不会在学习记录中保留你的邮箱。",
                subject="安全与品德",
                grade=4,
            )

            history = store.get_history(
                student_id=student_id,
                thread_id="thread-memory-001",
            )
            self.assertEqual(len(history), 4)
            self.assertIn("[邮箱已隐藏]", history[2]["content"])
            self.assertNotIn("student@example.com", history[2]["content"])

            memories = store.retrieve(
                student_id=student_id,
                question="周长怎么计算？",
                subject="数学",
            )
            memory_types = {memory.memory_type for memory in memories}
            self.assertIn("profile", memory_types)
            self.assertTrue(
                {"preference", "learning_difficulty"} & memory_types
            )

            context = EducationContextManager(store).build_long_term_context(
                student_id=student_id,
                question="周长怎么计算？",
                subject="数学",
            )
            self.assertIn("小学4年级", context)

            self.assertEqual(len(store.list_conversations(student_id=student_id)), 2)
            thread_ids = store.delete_student_data(student_id=student_id)
            self.assertEqual(set(thread_ids), {"thread-memory-001", "thread-memory-002"})
            self.assertEqual(store.list_memories(student_id=student_id), [])


class SqliteCheckpointerTests(unittest.TestCase):
    @staticmethod
    def _build_graph(checkpointer):
        def reply(state: MessagesState):
            return {"messages": [AIMessage(content=f"消息数：{len(state['messages'])}")]}

        builder = StateGraph(MessagesState)
        builder.add_node("reply", reply)
        builder.add_edge(START, "reply")
        return builder.compile(checkpointer=checkpointer)

    def test_state_survives_new_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "checkpoints.sqlite3"
            config = {"configurable": {"thread_id": "student:thread-001"}}

            connection = sqlite3.connect(database_path, check_same_thread=False)
            saver = SqliteSaver(connection)
            saver.setup()
            graph = self._build_graph(saver)
            graph.invoke({"messages": [{"role": "user", "content": "第一问"}]}, config)
            graph.invoke({"messages": [{"role": "user", "content": "第二问"}]}, config)
            self.assertEqual(len(graph.get_state(config).values["messages"]), 4)
            connection.close()

            reopened = sqlite3.connect(database_path, check_same_thread=False)
            saver = SqliteSaver(reopened)
            graph = self._build_graph(saver)
            self.assertEqual(len(graph.get_state(config).values["messages"]), 4)
            saver.delete_thread("student:thread-001")
            self.assertFalse(graph.get_state(config).values)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
