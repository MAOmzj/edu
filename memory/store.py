"""SQLite 业务存储：会话历史和跨会话长期学习记忆。

这个文件管理四张业务表：

- education_conversations：一行代表一个会话；
- education_messages：完整保存每轮学生问题和最终答案；
- education_committed_turns：记录已经提交的 turn_id，用于防重复写入；
- education_long_term_memories：少量提炼后的年级、偏好、主题和学习难点。

“完整历史”和“长期记忆”不是二选一：历史用于查看原对话，长期记忆用于跨会话
快速取回少量相关信息。LangGraph Checkpoint 则由 memory/runtime.py 另外管理。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class LongTermMemory:
    """从数据库取出的一条只读长期记忆，供上下文管理器使用。"""

    memory_type: str
    subject: str
    content: str
    confidence: float


class EducationMemoryStore:
    """保存可查询的会话记录，并提炼少量长期学习记忆。"""

    def __init__(self, database_path: str | Path, max_memories: int = 120):
        """记录数据库位置和长期主题记忆上限，并初始化业务数据表。"""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_memories = max_memories
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """创建一次 SQLite 连接，并统一处理事务提交、回滚和连接关闭。"""
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        # with self._connect() 中的所有 SQL 属于同一个事务：全部成功才 commit；
        # 任意一步失败就 rollback，避免只保存了问题却没有保存答案的半成品数据。
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        """创建会话、消息和长期记忆表，以及查询所需的索引。"""
        with self._connect() as connection:
            connection.executescript(
                """
                -- IF NOT EXISTS 让初始化可以重复执行，不会清空已有数据。
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
                    -- 删除会话时，ON DELETE CASCADE 会自动删除它的所有消息。
                    FOREIGN KEY(thread_id)
                        REFERENCES education_conversations(thread_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_thread_id
                ON education_messages(thread_id, id);

                CREATE TABLE IF NOT EXISTS education_committed_turns (
                    turn_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(thread_id)
                        REFERENCES education_conversations(thread_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_committed_turns_thread_id
                ON education_committed_turns(thread_id, created_at);

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
        """清理多余空白并截断过长文本，便于生成标题或记忆内容。"""
        normalized = " ".join(value.strip().split())
        return normalized if len(normalized) <= limit else normalized[:limit] + "…"

    @staticmethod
    def _redact_sensitive(value: str) -> str:
        """在写入数据库前隐藏手机号、邮箱和身份证号等敏感信息。"""
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
        """根据学科和问题生成稳定的主题键，防止相同主题被重复保存。"""
        normalized = " ".join(question.lower().split())
        digest = hashlib.sha256(f"{subject}|{normalized}".encode("utf-8")).hexdigest()
        return f"topic:{digest[:24]}"

    @staticmethod
    def _difficulty_key(subject: str, question: str) -> str:
        """根据学科和问题生成稳定的学习难点键。"""
        normalized = " ".join(question.lower().split())
        digest = hashlib.sha256(
            f"difficulty|{subject}|{normalized}".encode("utf-8")
        ).hexdigest()
        return f"difficulty:{digest[:24]}"

    @staticmethod
    def _agent_memory_key(memory_type: str, subject: str, content: str) -> str:
        """由存储层生成候选记忆键，绝不采用模型自行提供的数据库键。"""

        normalized = " ".join(content.lower().split())
        digest = hashlib.sha256(
            f"agent|{memory_type}|{subject}|{normalized}".encode("utf-8")
        ).hexdigest()
        return f"{memory_type}:agent:{digest[:24]}"

    @staticmethod
    def _assert_thread_owner(
        connection: sqlite3.Connection,
        thread_id: str,
        student_id: str,
    ) -> None:
        """检查会话是否属于当前学生，避免不同学生互相访问会话数据。"""
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
        """按照学生和记忆键新增记忆；已存在时更新内容、来源和时间。"""
        # SQL 中的 ? 是参数占位符。数据与 SQL 分开传入，可以避免引号问题和
        # SQL 注入；不要用 f-string 把学生文本直接拼进 SQL。
        connection.execute(
            """
            INSERT INTO education_long_term_memories (
                student_id, memory_key, memory_type, subject,
                content, confidence, source_thread_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            -- 同一学生的同一 memory_key 已存在时更新，而不是新增重复记忆。
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

    def _extract_core_memories(
        self,
        connection: sqlite3.Connection,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        subject: str,
        grade: int | None,
    ) -> str:
        """始终用确定性代码保存年级和本轮主题，并返回截短后的问题。"""

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
        return short_question

    def _save_memory_candidates(
        self,
        connection: sqlite3.Connection,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        subject: str,
        candidates: Sequence[dict[str, Any]],
    ) -> None:
        """校验、脱敏并保存记忆 Agent 返回的候选，最多处理五条。"""

        allowed_types = {
            "profile",
            "preference",
            "learning_topic",
            "learning_difficulty",
        }
        safe_question = self._short_text(self._redact_sensitive(question))

        for candidate in candidates[:5]:
            memory_type = str(candidate.get("memory_type", ""))
            if memory_type not in allowed_types:
                continue

            # 候选学科只能是当前学科或“综合”，防止模型写入任意分类。
            candidate_subject = str(candidate.get("subject", "综合"))
            if candidate_subject not in {subject, "综合"}:
                candidate_subject = subject

            content = self._short_text(
                self._redact_sensitive(str(candidate.get("content", "")))
            )
            if not content:
                continue

            try:
                confidence = float(candidate.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            confidence = min(1.0, max(0.0, confidence))
            if confidence < 0.6:
                continue

            # 主题和难点以学生原问题生成内容，避免保存模型猜测出的个人信息。
            if memory_type == "learning_topic":
                memory_key = self._topic_key(subject, safe_question)
                candidate_subject = subject
                content = f"学生曾询问：{safe_question}"
            elif memory_type == "learning_difficulty":
                memory_key = self._difficulty_key(subject, safe_question)
                candidate_subject = subject
                content = f"学生曾表示这个问题较难：{safe_question}"
            elif memory_type == "profile":
                # 当前系统的学生画像只允许保存由接口确定的年级，忽略自由画像。
                continue
            else:
                memory_key = self._agent_memory_key(
                    memory_type,
                    candidate_subject,
                    content,
                )

            self._upsert_memory(
                connection,
                student_id=student_id,
                memory_key=memory_key,
                memory_type=memory_type,
                subject=candidate_subject,
                content=content,
                confidence=confidence,
                source_thread_id=thread_id,
            )

    def _prune_topic_memories(
        self,
        connection: sqlite3.Connection,
        *,
        student_id: str,
    ) -> None:
        """只保留一个学生最近的指定数量主题记忆，防止数据库无限增长。"""

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
        """从本轮问题中提取年级、学习主题、表达偏好和学习难点。"""
        short_question = self._extract_core_memories(
            connection,
            student_id=student_id,
            thread_id=thread_id,
            question=question,
            subject=subject,
            grade=grade,
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

        self._prune_topic_memories(
            connection,
            student_id=student_id,
        )

    def record_turn(
        self,
        *,
        student_id: str,
        thread_id: str,
        turn_id: str | None = None,
        question: str,
        answer: str,
        subject: str,
        grade: int | None,
        memory_candidates: Sequence[dict[str, Any]] | None = None,
    ) -> bool:
        """原子保存一轮问答和长期记忆；相同 turn_id 重试时不会重复写入。"""
        # 下面“会话、两条消息、长期记忆”的写入共享一个事务，保证原子性。
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

            if turn_id:
                # LangGraph 节点可能因重试再次执行。INSERT OR IGNORE 把 turn_id
                # 当作幂等键：同一轮第二次执行会直接返回 False，不重复保存消息。
                committed = connection.execute(
                    """
                    INSERT OR IGNORE INTO education_committed_turns (
                        turn_id, thread_id, student_id
                    ) VALUES (?, ?, ?)
                    """,
                    (turn_id, thread_id, student_id),
                )
                if committed.rowcount == 0:
                    existing = connection.execute(
                        """
                        SELECT thread_id, student_id
                        FROM education_committed_turns
                        WHERE turn_id = ?
                        """,
                        (turn_id,),
                    ).fetchone()
                    if (
                        existing["thread_id"] != thread_id
                        or existing["student_id"] != student_id
                    ):
                        raise ValueError("turn_id 已被其他会话使用")
                    return False

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
            if memory_candidates is None:
                # 兼容旧调用：没有记忆 Agent 候选时继续使用原有关键词规则。
                self._extract_memories(
                    connection,
                    student_id=student_id,
                    thread_id=thread_id,
                    question=safe_question,
                    subject=subject,
                    grade=grade,
                )
            else:
                # 多 Agent 路径：基础信息由代码确定，复杂偏好/难点来自候选。
                self._extract_core_memories(
                    connection,
                    student_id=student_id,
                    thread_id=thread_id,
                    question=safe_question,
                    subject=subject,
                    grade=grade,
                )
                self._save_memory_candidates(
                    connection,
                    student_id=student_id,
                    thread_id=thread_id,
                    question=safe_question,
                    subject=subject,
                    candidates=memory_candidates,
                )
                self._prune_topic_memories(
                    connection,
                    student_id=student_id,
                )
            return True

    def ensure_conversation(
        self,
        *,
        student_id: str,
        thread_id: str,
        question: str,
        subject: str,
        grade: int | None,
    ) -> None:
        """模型执行前登记会话，确保失败请求产生的 Checkpoint 也能被清理。"""
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
        """把文本拆成连续双字集合，供轻量级文本相关度计算使用。"""
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
        """按学生、学科、文本相关度、类型和时间选出最相关的长期记忆。"""
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

        # 长期记忆数量很少，这里无需再建一套向量库；用中文双字重合度、记忆
        # 类型、学科和新旧程度做轻量排序，选出最适合当前问题的几条。
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
        """按消息写入顺序读取指定学生在一个会话中的完整聊天记录。"""
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
        """按最近更新时间倒序列出一个学生的会话摘要。"""
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
        """按最近更新时间倒序列出一个学生已经形成的长期记忆。"""
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
        """删除一个会话，以及由该会话产生的主题和学习难点记忆。"""
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
        """删除一个学生的全部会话和长期记忆，并返回待清理的线程编号。"""
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
        """统计当前数据库中的会话、消息和长期记忆数量。"""
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
