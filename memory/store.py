"""SQLite 业务存储：会话历史和跨会话长期学习记忆。"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class LongTermMemory:
    memory_type: str
    subject: str
    content: str
    confidence: float


class EducationMemoryStore:
    """保存可查询的会话记录，并提炼少量长期学习记忆。"""

    def __init__(self, database_path: str | Path, max_memories: int = 120):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_memories = max_memories
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS education_conversations (
                    thread_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    grade INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_student_updated
                ON education_conversations(student_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS education_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(thread_id)
                        REFERENCES education_conversations(thread_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_thread_id
                ON education_messages(thread_id, id);

                CREATE TABLE IF NOT EXISTS education_long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source_thread_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, memory_key)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_student_subject
                ON education_long_term_memories(
                    student_id,
                    subject,
                    updated_at DESC
                );
                """
            )

    @staticmethod
    def _short_text(value: str, limit: int = 180) -> str:
        normalized = " ".join(value.strip().split())
        return normalized if len(normalized) <= limit else normalized[:limit] + "…"

    @staticmethod
    def _redact_sensitive(value: str) -> str:
        """在写入学生数据库前隐藏常见直接身份信息。"""
        patterns = (
            (r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]"),
            (
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                "[邮箱已隐藏]",
            ),
            (r"(?<!\d)\d{17}[\dXx](?!\d)", "[证件号已隐藏]"),
        )
        redacted = value
        for pattern, replacement in patterns:
            redacted = re.sub(pattern, replacement, redacted)
        return redacted

    @staticmethod
    def _topic_key(subject: str, question: str) -> str:
        normalized = " ".join(question.lower().split())
        digest = hashlib.sha256(f"{subject}|{normalized}".encode("utf-8")).hexdigest()
        return f"topic:{digest[:24]}"

    @staticmethod
    def _difficulty_key(subject: str, question: str) -> str:
        digest = hashlib.sha256(
            f"difficulty|{subject}|{question}".encode("utf-8")
        ).hexdigest()
        return f"difficulty:{digest[:24]}"

    @staticmethod
    def _assert_thread_owner(
        connection: sqlite3.Connection,
        thread_id: str,
        student_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT student_id FROM education_conversations WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row and row["student_id"] != student_id:
            raise ValueError("该会话不属于当前匿名学生")

    @staticmethod
    def _upsert_memory(
        connection: sqlite3.Connection,
        *,
        student_id: str,
        memory_key: str,
        memory_type: str,
        subject: str,
        content: str,
        confidence: float,
        source_thread_id: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO education_long_term_memories (
                student_id, memory_key, memory_type, subject,
                content, confidence, source_thread_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id, memory_key) DO UPDATE SET
                memory_type = excluded.memory_type,
                subject = excluded.subject,
                content = excluded.content,
                confidence = excluded.confidence,
                source_thread_id = excluded.source_thread_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                student_id,
                memory_key,
                memory_type,
                subject,
                content,
                confidence,
                source_thread_id,
            ),
        )

    def _extract_memories(
        self,
        connection: sqlite3.Connection,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        subject: str,
        grade: int | None,
    ) -> None:
        if grade is not None:
            self._upsert_memory(
                connection,
                student_id=student_id,
                memory_key="profile:grade",
                memory_type="profile",
                subject="综合",
                content=f"学生当前按小学{grade}年级难度学习。",
                confidence=1.0,
                source_thread_id=thread_id,
            )

        short_question = self._short_text(question)
        self._upsert_memory(
            connection,
            student_id=student_id,
            memory_key=self._topic_key(subject, short_question),
            memory_type="learning_topic",
            subject=subject,
            content=f"学生曾询问：{short_question}",
            confidence=0.8,
            source_thread_id=thread_id,
        )

        preference_rules = {
            "preference:examples": (
                ("举例", "例子", "举个例"),
                "学生偏好通过具体例子理解知识。",
            ),
            "preference:steps": (
                ("步骤", "一步一步", "详细过程"),
                "学生偏好分步骤讲解。",
            ),
            "preference:simple": (
                ("简单一点", "简单讲", "通俗"),
                "学生偏好简单、通俗的解释。",
            ),
            "preference:visual": (
                ("画图", "图形", "示意图"),
                "学生偏好使用图形或示意方式理解知识。",
            ),
        }
        for memory_key, (keywords, content) in preference_rules.items():
            if any(keyword in question for keyword in keywords):
                self._upsert_memory(
                    connection,
                    student_id=student_id,
                    memory_key=memory_key,
                    memory_type="preference",
                    subject="综合",
                    content=content,
                    confidence=0.9,
                    source_thread_id=thread_id,
                )

        if re.search(r"不懂|不会|不明白|搞不清|容易错|总是错|总弄混|分不清", question):
            self._upsert_memory(
                connection,
                student_id=student_id,
                memory_key=self._difficulty_key(subject, short_question),
                memory_type="learning_difficulty",
                subject=subject,
                content=f"学生曾表示这个问题较难：{short_question}",
                confidence=0.75,
                source_thread_id=thread_id,
            )

        connection.execute(
            """
            DELETE FROM education_long_term_memories
            WHERE id IN (
                SELECT id
                FROM education_long_term_memories
                WHERE student_id = ? AND memory_type = 'learning_topic'
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (student_id, self.max_memories),
        )

    def record_turn(
        self,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        answer: str,
        subject: str,
        grade: int | None,
    ) -> None:
        with self._connect() as connection:
            self._assert_thread_owner(connection, thread_id, student_id)
            safe_question = self._redact_sensitive(question)
            safe_answer = self._redact_sensitive(answer)
            title = self._short_text(safe_question, 36)
            connection.execute(
                """
                INSERT INTO education_conversations (
                    thread_id, student_id, title, subject, grade
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    subject = excluded.subject,
                    grade = COALESCE(excluded.grade, education_conversations.grade),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, student_id, title, subject, grade),
            )
            connection.executemany(
                """
                INSERT INTO education_messages (
                    thread_id, student_id, role, content
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (thread_id, student_id, "user", safe_question),
                    (thread_id, student_id, "assistant", safe_answer),
                ),
            )
            self._extract_memories(
                connection,
                student_id=student_id,
                thread_id=thread_id,
                question=safe_question,
                subject=subject,
                grade=grade,
            )

    def ensure_conversation(
        self,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        subject: str,
        grade: int | None,
    ) -> None:
        """在模型执行前登记线程，确保失败的 Checkpoint 也可被删除。"""
        with self._connect() as connection:
            self._assert_thread_owner(connection, thread_id, student_id)
            connection.execute(
                """
                INSERT INTO education_conversations (
                    thread_id, student_id, title, subject, grade
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    subject = excluded.subject,
                    grade = COALESCE(excluded.grade, education_conversations.grade),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    thread_id,
                    student_id,
                    self._short_text(self._redact_sensitive(question), 36),
                    subject,
                    grade,
                ),
            )

    @staticmethod
    def _character_terms(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text.lower())
        if len(normalized) < 2:
            return {normalized} if normalized else set()
        return {normalized[index : index + 2] for index in range(len(normalized) - 1)}

    def retrieve(
        self,
        *,
        student_id: str,
        question: str,
        subject: str,
        limit: int = 5,
    ) -> list[LongTermMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_type, subject, content, confidence, updated_at
                FROM education_long_term_memories
                WHERE student_id = ?
                  AND (
                    subject IN (?, '综合')
                    OR memory_type IN ('profile', 'preference')
                  )
                ORDER BY updated_at DESC
                LIMIT 200
                """,
                (student_id, subject),
            ).fetchall()

        query_terms = self._character_terms(question)
        type_score = {
            "profile": 12.0,
            "preference": 10.0,
            "learning_difficulty": 5.0,
            "learning_topic": 1.0,
        }
        ranked: list[tuple[float, sqlite3.Row]] = []
        for recency, row in enumerate(rows):
            memory_terms = self._character_terms(row["content"])
            overlap = len(query_terms & memory_terms) / max(len(query_terms), 1)
            score = type_score.get(row["memory_type"], 0.0)
            score += 2.0 if row["subject"] == subject else 0.0
            score += overlap * 8.0
            score += max(0.0, 1.0 - recency * 0.01)
            ranked.append((score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            LongTermMemory(
                memory_type=row["memory_type"],
                subject=row["subject"],
                content=row["content"],
                confidence=float(row["confidence"]),
            )
            for _, row in ranked[:limit]
        ]

    def get_history(self, *, student_id: str, thread_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._assert_thread_owner(connection, thread_id, student_id)
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM education_messages
                WHERE thread_id = ? AND student_id = ?
                ORDER BY id
                """,
                (thread_id, student_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_conversations(self, *, student_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT thread_id, title, subject, grade, created_at, updated_at
                FROM education_conversations
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (student_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_memories(self, *, student_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_type, subject, content, confidence, updated_at
                FROM education_long_term_memories
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (student_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, *, student_id: str, thread_id: str) -> bool:
        with self._connect() as connection:
            self._assert_thread_owner(connection, thread_id, student_id)
            cursor = connection.execute(
                """
                DELETE FROM education_conversations
                WHERE thread_id = ? AND student_id = ?
                """,
                (thread_id, student_id),
            )
            connection.execute(
                """
                DELETE FROM education_long_term_memories
                WHERE student_id = ? AND source_thread_id = ?
                  AND memory_type IN ('learning_topic', 'learning_difficulty')
                """,
                (student_id, thread_id),
            )
            return cursor.rowcount > 0

    def delete_student_data(self, *, student_id: str) -> list[str]:
        with self._connect() as connection:
            thread_ids = [
                row["thread_id"]
                for row in connection.execute(
                    """
                    SELECT thread_id FROM education_conversations
                    WHERE student_id = ?
                    """,
                    (student_id,),
                ).fetchall()
            ]
            connection.execute(
                "DELETE FROM education_conversations WHERE student_id = ?",
                (student_id,),
            )
            connection.execute(
                "DELETE FROM education_long_term_memories WHERE student_id = ?",
                (student_id,),
            )
        return thread_ids

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            conversations = connection.execute(
                "SELECT COUNT(*) FROM education_conversations"
            ).fetchone()[0]
            messages = connection.execute(
                "SELECT COUNT(*) FROM education_messages"
            ).fetchone()[0]
            memories = connection.execute(
                "SELECT COUNT(*) FROM education_long_term_memories"
            ).fetchone()[0]
        return {
            "conversations": conversations,
            "messages": messages,
            "long_term_memories": memories,
        }
