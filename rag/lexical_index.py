# 文件用途：用 SQLite 持久化倒排索引，在全库或指定学科范围内计算准确的 BM25。
# 调用关系：vector_store.py 同步和查询本索引；分词统一复用 hybrid_search.py。
# 修改易踩坑：读取不能创建索引；文件替换必须原子完成；BM25 统计必须遵守学科范围。
"""独立于 Embedding 的、按知识集合隔离的持久化 BM25 倒排索引。"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.documents import Document

from rag.hybrid_search import tokenize_for_bm25


SCHEMA_VERSION = 2
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS lexical_documents (
        document_id INTEGER PRIMARY KEY,
        namespace TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        text TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        subject TEXT NOT NULL,
        source TEXT NOT NULL,
        length INTEGER NOT NULL CHECK (length >= 0),
        UNIQUE (namespace, chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lexical_postings (
        term TEXT NOT NULL,
        document_id INTEGER NOT NULL,
        tf INTEGER NOT NULL CHECK (tf > 0),
        PRIMARY KEY (term, document_id),
        FOREIGN KEY (document_id)
            REFERENCES lexical_documents(document_id) ON DELETE CASCADE
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS lexical_files (
        namespace TEXT NOT NULL,
        source TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        ids_json TEXT NOT NULL,
        PRIMARY KEY (namespace, source)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lexical_state (
        namespace TEXT PRIMARY KEY,
        manifest_digest TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_lexical_documents_subject
        ON lexical_documents(namespace, subject)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_lexical_documents_source
        ON lexical_documents(namespace, source)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_lexical_postings_document
        ON lexical_postings(document_id)
    """,
)
_TABLES = {
    "lexical_documents", "lexical_postings", "lexical_files", "lexical_state"
}
_TERM_BATCH_SIZE = 400
_SYNC_INSTRUCTION = "BM25 索引尚未准备好，请运行 python main.py --sync-index 重建索引。"


class BM25InvertedIndex:
    """只在显式同步时写入 SQLite，查询只读取相关词项及最终返回的正文。"""

    def __init__(
        self,
        database_path: str | Path,
        namespace: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 的 k1 必须是大于 0 的有限数值")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("BM25 的 b 必须是 0 到 1 之间的有限数值")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("BM25 namespace 必须是非空字符串")
        self.database_path = Path(database_path)
        self.namespace = namespace

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection | None]:
        """缺失文件直接返回空；mode=ro 保证读取不会创建数据库或修改数据。"""
        if not self.database_path.is_file():
            yield None
            return
        uri = self.database_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA query_only = ON")
            # 多个统计查询和正文读取共用同一个快照，避免同步中途改变 N 或 df。
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        """所有写入均使用短连接和立即事务，异常自动回滚整次操作。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                connection.execute(statement)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _has_schema(connection: sqlite3.Connection) -> bool:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", ("table",)
            )
        }
        return _TABLES.issubset(tables) and any(
            row[1] == "document_id"
            for row in connection.execute("PRAGMA table_info(lexical_documents)")
        )

    def _ready_digest(self, connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            "SELECT manifest_digest FROM lexical_state WHERE namespace = ?",
            (self.namespace,),
        ).fetchone()
        return row[0] if row and row[0] else None

    def _invalidate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM lexical_state WHERE namespace = ?", (self.namespace,)
        )

    def invalidate(self) -> None:
        """在跨向量与关键词索引的同步开始前撤销就绪标记，保留已有正文。"""
        with self._write_connection() as connection:
            self._invalidate(connection)

    def replace_source(
        self,
        source: str,
        sha256: str,
        documents: list[Document],
        ids: list[str],
    ) -> None:
        """一次事务替换一个文件；预处理或写入失败时原文件及词项均保留。"""
        if not isinstance(source, str) or not source:
            raise ValueError("source 必须是非空字符串")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError("sha256 必须是非空字符串")
        document_list = list(documents)
        id_list = list(ids)
        if len(document_list) != len(id_list):
            raise ValueError("documents 和 ids 的数量必须相同")
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in id_list):
            raise ValueError("chunk_id 必须是非空字符串")
        if len(set(id_list)) != len(id_list):
            raise ValueError("同一个文件中的 chunk_id 不得重复")

        document_rows: list[tuple[Any, ...]] = []
        posting_rows: list[tuple[Any, ...]] = []
        # 在开始写事务前完成分词与 JSON 验证，减少持锁时间，避免先删除旧文件。
        for chunk_id, document in zip(id_list, document_list, strict=True):
            metadata = dict(document.metadata)
            if metadata.get("chunk_id", chunk_id) != chunk_id:
                raise ValueError("文档元数据 chunk_id 与 ids 不一致")
            if metadata.get("source", source) != source:
                raise ValueError("文档元数据 source 与替换文件不一致")
            subject = metadata.get("subject", "")
            if not isinstance(subject, str):
                raise ValueError("文档元数据 subject 必须是字符串")
            metadata["chunk_id"] = chunk_id
            metadata.setdefault("source", source)
            metadata_json = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            tokens = tokenize_for_bm25(document.page_content)
            document_rows.append(
                (
                    self.namespace, chunk_id, document.page_content, metadata_json,
                    subject, source, len(tokens),
                )
            )
            posting_rows.extend(
                (term, chunk_id, frequency)
                for term, frequency in sorted(Counter(tokens).items())
            )

        with self._write_connection() as connection:
            self._invalidate(connection)
            connection.execute(
                "DELETE FROM lexical_documents WHERE namespace = ? AND source = ?",
                (self.namespace, source),
            )
            # 不使用 upsert：其他文件已经持有相同 ID 时应报错并回滚旧文件删除。
            connection.executemany(
                """
                INSERT INTO lexical_documents
                    (namespace, chunk_id, text, metadata_json, subject, source, length)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                document_rows,
            )
            document_ids = dict(connection.execute(
                "SELECT chunk_id, document_id FROM lexical_documents "
                "WHERE namespace = ? AND source = ?",
                (self.namespace, source),
            ))
            # 用整数引用文档，避免每条词项重复保存两个 64 字符 ID。
            # 按主键顺序批量插入，减少 SQLite B-tree 随机写入。
            compact_postings = sorted(
                (term, document_ids[chunk_id], frequency)
                for term, chunk_id, frequency in posting_rows
            )
            connection.executemany(
                """
                INSERT INTO lexical_postings (term, document_id, tf)
                VALUES (?, ?, ?)
                """,
                compact_postings,
            )
            connection.execute(
                """
                INSERT INTO lexical_files (namespace, source, sha256, ids_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, source) DO UPDATE SET
                    sha256 = excluded.sha256, ids_json = excluded.ids_json
                """,
                (self.namespace, source, sha256, json.dumps(id_list)),
            )

    def delete_source(self, source: str) -> None:
        """删除一个文件及其词项，同时使当前集合需要重新标记就绪。"""
        with self._write_connection() as connection:
            self._invalidate(connection)
            connection.execute(
                "DELETE FROM lexical_documents WHERE namespace = ? AND source = ?",
                (self.namespace, source),
            )
            connection.execute(
                "DELETE FROM lexical_files WHERE namespace = ? AND source = ?",
                (self.namespace, source),
            )

    def reset(self) -> None:
        """只清空本 namespace，不影响共享数据库中的其他知识集合。"""
        with self._write_connection() as connection:
            self._invalidate(connection)
            connection.execute(
                "DELETE FROM lexical_documents WHERE namespace = ?", (self.namespace,)
            )
            connection.execute(
                "DELETE FROM lexical_files WHERE namespace = ?", (self.namespace,)
            )

    def sources(self) -> dict[str, dict[str, Any]]:
        """返回同步清单；正文行缺失时返回实际 ID，促使上层修复损坏文件。"""
        with self._read_connection() as connection:
            if connection is None or not self._has_schema(connection):
                return {}
            actual_ids: dict[str, list[str]] = defaultdict(list)
            for source, chunk_id in connection.execute(
                """
                SELECT source, chunk_id FROM lexical_documents
                WHERE namespace = ? ORDER BY source, chunk_id
                """,
                (self.namespace,),
            ):
                actual_ids[source].append(chunk_id)
            files: dict[str, dict[str, Any]] = {}
            for source, sha256, ids_json in connection.execute(
                """
                SELECT source, sha256, ids_json FROM lexical_files
                WHERE namespace = ? ORDER BY source
                """,
                (self.namespace,),
            ):
                actual = actual_ids.pop(source, [])
                try:
                    saved_ids = json.loads(ids_json)
                except (TypeError, ValueError):
                    saved_ids = None
                consistent = (
                    isinstance(saved_ids, list)
                    and all(isinstance(item, str) for item in saved_ids)
                    and sorted(saved_ids) == actual
                )
                files[source] = {
                    "sha256": sha256,
                    "ids": saved_ids if consistent else actual,
                }
            # 孤立正文也要出现在清单里，否则上层无法清理或修复它们。
            for source, actual in actual_ids.items():
                files[source] = {"sha256": "", "ids": actual}
            return files

    def mark_ready(self, manifest_digest: str) -> None:
        """只有上层完整同步成功后，才记录与向量清单对应的就绪指纹。"""
        if not isinstance(manifest_digest, str) or not manifest_digest:
            raise ValueError("manifest_digest 必须是非空字符串")
        with self._write_connection() as connection:
            connection.execute(
                """
                INSERT INTO lexical_state (namespace, manifest_digest) VALUES (?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    manifest_digest = excluded.manifest_digest
                """,
                (self.namespace, manifest_digest),
            )

    def is_ready(self, manifest_digest: str) -> bool:
        """仅读取就绪指纹；缺失文件、表或不同 namespace 均视为未就绪。"""
        with self._read_connection() as connection:
            return bool(
                connection is not None
                and self._has_schema(connection)
                and self._ready_digest(connection) == manifest_digest
            )

    def status(self) -> dict[str, Any]:
        """只读取得轻量统计，不加载全文、不分词，也不创建缺失的数据库。"""
        result: dict[str, Any] = {
            "exists": self.database_path.is_file(),
            "ready": False,
            "indexed_chunks": 0,
            "indexed_files": 0,
            "postings": 0,
        }
        with self._read_connection() as connection:
            if connection is None or not self._has_schema(connection):
                return result
            result["ready"] = self._ready_digest(connection) is not None
            for key, statement in (
                ("indexed_chunks", "SELECT COUNT(*) FROM lexical_documents WHERE namespace = ?"),
                ("indexed_files", "SELECT COUNT(*) FROM lexical_files WHERE namespace = ?"),
                ("postings", "SELECT COUNT(*) FROM lexical_postings p "
                 "JOIN lexical_documents d ON d.document_id = p.document_id WHERE d.namespace = ?"),
            ):
                result[key] = connection.execute(statement, (self.namespace,)).fetchone()[0]
        return result

    @staticmethod
    def _batches(values: Sequence[str]) -> Iterator[Sequence[str]]:
        for start in range(0, len(values), _TERM_BATCH_SIZE):
            yield values[start : start + _TERM_BATCH_SIZE]

    def search(
        self,
        query: str,
        *,
        k: int,
        subject: str | None = None,
    ) -> list[Document]:
        """使用范围内完整语料的 N、df、avgdl，只给匹配词项的文档计算 BM25。"""
        try:
            with self._read_connection() as connection:
                if (
                    connection is None
                    or not self._has_schema(connection)
                    or self._ready_digest(connection) is None
                ):
                    raise EnvironmentError(_SYNC_INSTRUCTION)
                if k < 1:
                    return []
                query_terms = sorted(set(tokenize_for_bm25(query)))
                if not query_terms:
                    return []

                scope_sql = "" if subject is None else " AND subject = ?"
                scope_params = (self.namespace,) if subject is None else (self.namespace, subject)
                document_count, average_length = connection.execute(
                    "SELECT COUNT(*), AVG(length) FROM lexical_documents "
                    "WHERE namespace = ?" + scope_sql,
                    scope_params,
                ).fetchone()
                if document_count == 0:
                    return []
                safe_average_length = average_length or 1.0
                scores: dict[str, float] = defaultdict(float)
                subject_sql = "" if subject is None else " AND d.subject = ?"

                for terms in self._batches(query_terms):
                    placeholders = ",".join("?" for _ in terms)
                    # 插入 SQL 的只有固定语法与问号；namespace、学科、词项均参数化。
                    where_sql = (
                        f"WHERE d.namespace = ? AND p.term IN ({placeholders})"
                        + subject_sql
                    )
                    params = (self.namespace, *terms)
                    if subject is not None:
                        params += (subject,)
                    joins_sql = (
                        # 固定先查词项倒排表，再按整数主键查元数据，避免逐文档查词项。
                        " FROM lexical_postings p CROSS JOIN lexical_documents d "
                        "ON d.document_id = p.document_id "
                    )
                    frequencies = dict(connection.execute(
                        "SELECT p.term, COUNT(*)" + joins_sql + where_sql
                        + " GROUP BY p.term",
                        params,
                    ))
                    inverse_frequencies = {
                        term: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
                        for term, count in frequencies.items()
                    }
                    for term, chunk_id, term_frequency, length in connection.execute(
                        "SELECT p.term, d.chunk_id, p.tf, d.length" + joins_sql
                        + where_sql + " ORDER BY p.term, p.document_id",
                        params,
                    ):
                        denominator = term_frequency + self.k1 * (
                            1.0 - self.b + self.b * length / safe_average_length
                        )
                        scores[chunk_id] += inverse_frequencies[term] * (
                            term_frequency * (self.k1 + 1.0) / denominator
                        )

                ranked_ids = sorted(
                    (chunk_id for chunk_id, score in scores.items() if score > 0),
                    key=lambda chunk_id: (-scores[chunk_id], chunk_id),
                )[:k]
                # 排序完成后只加载最终返回的正文，长查询或大 k 也分批避开变量上限。
                documents: dict[str, Document] = {}
                for chunk_ids in self._batches(ranked_ids):
                    placeholders = ",".join("?" for _ in chunk_ids)
                    for chunk_id, text, metadata_json in connection.execute(
                        "SELECT chunk_id, text, metadata_json FROM lexical_documents "
                        f"WHERE namespace = ? AND chunk_id IN ({placeholders})",
                        (self.namespace, *chunk_ids),
                    ):
                        metadata = json.loads(metadata_json)
                        metadata["chunk_id"] = chunk_id
                        metadata["bm25_score"] = scores[chunk_id]
                        documents[chunk_id] = Document(page_content=text, metadata=metadata)
                return [documents[chunk_id] for chunk_id in ranked_ids]
        except sqlite3.Error as exc:
            raise EnvironmentError(_SYNC_INSTRUCTION) from exc
