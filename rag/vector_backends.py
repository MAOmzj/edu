# 文件用途：实现可切换的 SQLite 与 Chroma 向量后端，以及统一的增删查接口。
# 调用关系：vector_store.py 选择并调用本文件；本文件调用 Embedding 与 hybrid_search.py。
# 修改易踩坑：两个后端必须返回相同语义的结果；批量写入、向量维度和 Chroma 原生崩溃风险要重点测试。
"""可以互相切换的 SQLite 与 Chroma 向量存储后端。

上层 EducationVectorStore 只依赖 VectorBackend 规定的四个操作，所以配置从
SQLite 切到 Chroma 时，上层的文件切分、索引清单和 Agent Tool 都不用改。

默认 SQLite 的检索过程是：问题向量化 -> 读取候选向量 -> 计算余弦 -> 计算 BM25
-> 加权排序。它适合当前几百个知识片段；数据量非常大时应换专门向量数据库。
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.hybrid_search import (
    calculate_cosine_similarity,
    combine_hybrid_scores,
    rank_documents_by_hybrid_score,
)


class VectorBackend(Protocol):
    """规定教育知识库底层存储必须提供的统一操作。

    Protocol 类似一份“接口合同”，本身不创建对象；SQLite 和 Chroma 只要实现
    这些方法，上层就能用同一种写法调用它们。
    """

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> list[str]:
        """把文档及其向量保存到底层存储。"""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """删除给定分片编号对应的文档。"""
        ...

    def reset_collection(self) -> None:
        """清空当前知识集合，但保留数据库结构。"""
        ...

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """按照查询向量和可选元数据条件返回最相似文档。"""
        ...


class SQLiteVectorBackend:
    """使用标准库 SQLite 保存向量，并用 Python 计算余弦相似度。"""

    def __init__(
        self,
        *,
        database_path: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
        embedding_batch_size: int = 10,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """保存数据库、Embedding 批次和混合检索参数，并准备向量表。"""
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size 必须大于 0")
        # 提前调用一次空分数融合，让错误权重在启动时立即被发现。
        combine_hybrid_scores(
            [],
            [],
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        if bm25_k1 <= 0:
            raise ValueError("BM25 的 k1 必须大于 0")
        if not 0 <= bm25_b <= 1:
            raise ValueError("BM25 的 b 必须在 0 到 1 之间")
        self.database_path = Path(database_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self._write_lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """打开短连接，成功时提交、失败时回滚，并始终关闭文件句柄。"""
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _setup(self) -> None:
        """首次使用时创建向量表和常用查询索引。"""
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS education_vector_chunks (
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    file_hash TEXT NOT NULL DEFAULT '',
                    -- 向量有 1024 个浮点数。保存成二进制 BLOB 比 JSON 更省空间，
                    -- dimension 用于还原长度，norm 用于减少检索时的重复计算。
                    embedding BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_norm REAL NOT NULL,
                    PRIMARY KEY (collection_name, chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_education_vectors_subject
                    ON education_vector_chunks(collection_name, subject);

                CREATE INDEX IF NOT EXISTS idx_education_vectors_source
                    ON education_vector_chunks(collection_name, source);
                """
            )

    @staticmethod
    def _validate_embedding(values: Sequence[float]) -> list[float]:
        """把模型向量转成有限浮点数列表，并拒绝空向量。"""
        vector = [float(value) for value in values]
        if not vector:
            raise ValueError("Embedding 模型返回了空向量")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding 模型返回了 NaN 或无穷大")
        return vector

    @staticmethod
    def _encode_embedding(values: Sequence[float]) -> tuple[bytes, int, float]:
        """把浮点向量压缩成小端 float32 BLOB，同时计算向量长度。"""
        vector = SQLiteVectorBackend._validate_embedding(values)
        encoded = struct.pack(f"<{len(vector)}f", *vector)
        norm = math.sqrt(math.fsum(value * value for value in vector))
        return encoded, len(vector), norm

    @staticmethod
    def _decode_embedding(value: bytes, dimension: int) -> tuple[float, ...]:
        """把数据库中的 float32 BLOB 还原成可计算的浮点元组。"""
        expected_bytes = dimension * 4
        if dimension < 1 or len(value) != expected_bytes:
            raise RuntimeError("SQLite 中的向量维度或字节长度不正确")
        return struct.unpack(f"<{dimension}f", value)

    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """每次最多发送配置数量的文本，避免超过在线模型批次上限。"""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = list(texts[start : start + self.embedding_batch_size])
            raw_batch = self.embedding_model.embed_documents(batch)
            if len(raw_batch) != len(batch):
                raise RuntimeError("Embedding 返回数量与输入文本数量不一致")
            embeddings.extend(
                self._validate_embedding(vector) for vector in raw_batch
            )
        return embeddings

    @staticmethod
    def _metadata_matches(
        metadata: dict[str, Any],
        filter_value: dict[str, Any] | None,
    ) -> bool:
        """检查文档元数据是否满足所有等值过滤条件。"""
        if not filter_value:
            return True
        return all(metadata.get(key) == value for key, value in filter_value.items())

    @staticmethod
    def _cosine_similarity(
        query_vector: Sequence[float],
        query_norm: float,
        document_vector: Sequence[float],
        document_norm: float,
    ) -> float:
        """计算两个同维向量的余弦相似度，零向量统一返回 0。"""
        if query_norm == 0.0 or document_norm == 0.0:
            return 0.0
        dot_product = math.fsum(
            left * right
            for left, right in zip(query_vector, document_vector, strict=True)
        )
        return dot_product / (query_norm * document_norm)

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> list[str]:
        """先分批生成向量，再用一个 SQLite 事务批量新增或更新文档。"""
        document_list = list(documents)
        id_list = [str(chunk_id) for chunk_id in ids]
        if len(document_list) != len(id_list):
            raise ValueError("documents 和 ids 的数量必须相同")
        if not document_list:
            return []

        # 先在数据库事务外调用在线 Embedding。网络请求慢时不会长时间占住
        # SQLite 写锁；全部向量成功生成后，再开启一个短事务批量写入。
        texts = [document.page_content for document in document_list]
        embeddings = self._embed_documents(texts)
        rows: list[tuple[Any, ...]] = []
        for chunk_id, document, embedding in zip(
            id_list,
            document_list,
            embeddings,
            strict=True,
        ):
            metadata = dict(document.metadata)
            encoded, dimension, norm = self._encode_embedding(embedding)
            rows.append(
                (
                    self.collection_name,
                    chunk_id,
                    document.page_content,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    str(metadata.get("subject", "")),
                    str(metadata.get("source", "")),
                    str(metadata.get("file_hash", "")),
                    encoded,
                    dimension,
                    norm,
                )
            )

        with self._write_lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO education_vector_chunks (
                    collection_name,
                    chunk_id,
                    document,
                    metadata_json,
                    subject,
                    source,
                    file_hash,
                    embedding,
                    embedding_dimension,
                    embedding_norm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, chunk_id) DO UPDATE SET
                    document = excluded.document,
                    metadata_json = excluded.metadata_json,
                    subject = excluded.subject,
                    source = excluded.source,
                    file_hash = excluded.file_hash,
                    embedding = excluded.embedding,
                    embedding_dimension = excluded.embedding_dimension,
                    embedding_norm = excluded.embedding_norm
                """,
                rows,
            )
        return id_list

    def delete(self, ids: Sequence[str]) -> None:
        """按分片编号分批删除记录，空列表不会访问数据库。"""
        id_list = [str(chunk_id) for chunk_id in ids]
        if not id_list:
            return
        with self._write_lock, self._connect() as connection:
            for start in range(0, len(id_list), 500):
                batch = id_list[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"""
                    DELETE FROM education_vector_chunks
                    WHERE collection_name = ?
                      AND chunk_id IN ({placeholders})
                    """,
                    [self.collection_name, *batch],
                )

    def reset_collection(self) -> None:
        """在一个事务中删除当前集合的所有向量分片。"""
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM education_vector_chunks
                WHERE collection_name = ?
                """,
                (self.collection_name,),
            )

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """为候选文档计算余弦与 BM25 混合分数，并返回前 k 条。"""
        if k < 1:
            return []
        # 知识文本建库时用 embed_documents，提问时用 embed_query；两者必须来自
        # 同一个模型，否则向量不在同一空间，余弦分数没有意义。
        query_vector = self._validate_embedding(
            self.embedding_model.embed_query(query)
        )
        query_norm = math.sqrt(
            math.fsum(value * value for value in query_vector)
        )

        sql = """
            SELECT chunk_id, document, metadata_json,
                   embedding, embedding_dimension, embedding_norm
            FROM education_vector_chunks
            WHERE collection_name = ?
        """
        parameters: list[Any] = [self.collection_name]
        subject = filter.get("subject") if filter else None
        if subject is not None:
            sql += " AND subject = ?"
            parameters.append(str(subject))

        # 当前知识库很小，因此先按 collection/subject 从 SQLite 取候选，再由
        # Python 逐条计算相似度。这样实现简单、无额外服务，但不适合百万级向量。
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()

        candidate_ids: list[str] = []
        candidate_documents: list[Document] = []
        cosine_scores: list[float] = []
        for chunk_id, text, metadata_json, blob, dimension, stored_norm in rows:
            if dimension != len(query_vector):
                continue
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"向量分片 {chunk_id} 的元数据已损坏") from exc
            if not isinstance(metadata, dict):
                raise RuntimeError(f"向量分片 {chunk_id} 的元数据不是对象")
            if not self._metadata_matches(metadata, filter):
                continue
            document_vector = self._decode_embedding(blob, dimension)
            score = self._cosine_similarity(
                query_vector,
                query_norm,
                document_vector,
                float(stored_norm),
            )
            candidate_ids.append(str(chunk_id))
            candidate_documents.append(
                Document(page_content=text, metadata=metadata)
            )
            cosine_scores.append(score)

        return rank_documents_by_hybrid_score(
            query,
            candidate_ids,
            candidate_documents,
            cosine_scores,
            k=k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )


class ChromaVectorBackend:
    """保留原来的 Chroma 实现，供配置切换或其他运行环境使用。"""

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """创建 Chroma 集合，并保存与 SQLite 相同的混合检索参数。"""
        from langchain_chroma import Chroma

        combine_hybrid_scores(
            [],
            [],
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        if bm25_k1 <= 0:
            raise ValueError("BM25 的 k1 必须大于 0")
        if not 0 <= bm25_b <= 1:
            raise ValueError("BM25 的 b 必须在 0 到 1 之间")
        directory = Path(persist_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(directory),
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> list[str]:
        """把文档交给原有 LangChain Chroma 适配器写入。"""
        return self._store.add_documents(list(documents), ids=list(ids))

    def delete(self, ids: Sequence[str]) -> None:
        """通过原有 Chroma 接口删除给定分片。"""
        self._store.delete(ids=list(ids))

    def reset_collection(self) -> None:
        """通过原有 Chroma 接口清空并重建当前集合。"""
        self._store.reset_collection()

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """从 Chroma 读取向量，再执行与 SQLite 相同的混合排序。"""
        if k < 1:
            return []
        result = self._store.get(
            where=filter,
            include=["documents", "metadatas", "embeddings"],
        )
        identifiers = [str(value) for value in (result.get("ids") or [])]
        raw_documents = result.get("documents")
        raw_metadatas = result.get("metadatas")
        raw_embeddings = result.get("embeddings")
        if (
            raw_documents is None
            or raw_metadatas is None
            or raw_embeddings is None
        ):
            return []

        query_vector = SQLiteVectorBackend._validate_embedding(
            self.embedding_model.embed_query(query)
        )
        candidate_ids: list[str] = []
        candidate_documents: list[Document] = []
        cosine_scores: list[float] = []
        for chunk_id, text, metadata, embedding in zip(
            identifiers,
            raw_documents,
            raw_metadatas,
            raw_embeddings,
            strict=True,
        ):
            if text is None or embedding is None:
                continue
            document_vector = SQLiteVectorBackend._validate_embedding(embedding)
            if len(document_vector) != len(query_vector):
                continue
            candidate_ids.append(chunk_id)
            candidate_documents.append(
                Document(page_content=str(text), metadata=dict(metadata or {}))
            )
            cosine_scores.append(
                calculate_cosine_similarity(query_vector, document_vector)
            )

        return rank_documents_by_hybrid_score(
            query,
            candidate_ids,
            candidate_documents,
            cosine_scores,
            k=k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )
