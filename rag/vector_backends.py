# 文件用途：实现可切换的 SQLite、Chroma 与 Qdrant 向量后端，以及统一的增删查接口。
# 调用关系：vector_store.py 选择并调用本文件；本文件调用 Embedding 与 hybrid_search.py。
# 修改易踩坑：三个后端必须返回相同语义的结果；Qdrant score 已是相似度，不能再做 1-score。
"""可以互相切换的 SQLite、Chroma 与 Qdrant 向量存储后端。

上层 EducationVectorStore 只依赖 VectorBackend 规定的四个操作，所以配置从
SQLite、Chroma 与 Qdrant 之间切换时，上层的文件切分、索引清单和 Agent Tool 都不用改。

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
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.hybrid_search import (
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
    """通过独立 Chroma Server 原生召回候选，再执行轻量混合排序。"""

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_model: Embeddings,
        deployment_mode: str = "http",
        persist_directory: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        ssl: bool = False,
        tenant: str = "default_tenant",
        database: str = "default_database",
        auth_token: str = "",
        native_candidate_multiplier: int = 4,
        native_max_candidates: int = 100,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """连接 Chroma，并保存原生召回与候选集混合排序参数。"""
        import chromadb
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
        if native_candidate_multiplier < 1:
            raise ValueError("Chroma 原生候选倍数必须大于 0")
        if native_max_candidates < 1:
            raise ValueError("Chroma 原生候选上限必须大于 0")

        normalized_mode = str(deployment_mode).strip().lower()
        if normalized_mode not in {"http", "persistent"}:
            raise ValueError("Chroma deployment_mode 只能是 http 或 persistent")

        self.embedding_model = embedding_model
        self.deployment_mode = normalized_mode
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.native_candidate_multiplier = int(native_candidate_multiplier)
        self.native_max_candidates = int(native_max_candidates)

        if normalized_mode == "http":
            headers = (
                {"Authorization": f"Bearer {auth_token.strip()}"}
                if auth_token.strip()
                else None
            )
            client = chromadb.HttpClient(
                host=str(host).strip(),
                port=int(port),
                ssl=bool(ssl),
                headers=headers,
                tenant=str(tenant).strip(),
                database=str(database).strip(),
            )
        else:
            if persist_directory is None:
                raise ValueError("persistent 模式必须配置 Chroma 持久化目录")
            directory = Path(persist_directory)
            directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(directory),
                tenant=str(tenant).strip(),
                database=str(database).strip(),
            )

        # 明确使用 cosine 空间，使 Chroma 返回的 distance 可以稳定转换为
        # cosine_similarity = 1 - distance。已有其他度量空间的集合需要重建。
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
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
        """用 Chroma ANN 召回有限候选，再在候选集上融合 BM25 分数。"""
        if k < 1:
            return []

        # 原生向量查询先把集合缩小到有限候选，避免旧实现 get() 全量拉取正文、
        # 元数据和向量。BM25 在候选集内校正精确关键词排序，随后上层还可 Rerank。
        candidate_k = min(
            self.native_max_candidates,
            max(k, k * self.native_candidate_multiplier),
        )
        scored_documents = self._store.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filter,
        )
        if not scored_documents:
            return []

        candidate_ids: list[str] = []
        candidate_documents: list[Document] = []
        cosine_scores: list[float] = []
        for position, (document, raw_distance) in enumerate(scored_documents):
            distance = float(raw_distance)
            if not math.isfinite(distance):
                raise RuntimeError("Chroma 返回了 NaN 或无穷距离")
            metadata = dict(document.metadata)
            stable_id = (
                getattr(document, "id", None)
                or metadata.get("chunk_id")
                or (
                    f"{metadata.get('source', '')}:"
                    f"{metadata.get('file_hash', '')}:"
                    f"{metadata.get('chunk_index', position)}"
                )
            )
            candidate_ids.append(str(stable_id))
            candidate_documents.append(document)
            cosine_scores.append(max(-1.0, min(1.0, 1.0 - distance)))

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


class QdrantVectorBackend:
    """通过独立 Qdrant Server 原生召回候选，再执行轻量混合排序。"""

    _FLAT_METADATA_FIELDS = frozenset(
        {"subject", "source", "file_hash", "chunk_id"}
    )

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_model: Embeddings,
        url: str = "http://127.0.0.1:6333",
        api_key: str = "",
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        timeout: int = 10,
        vector_size: int = 1024,
        distance: str = "cosine",
        embedding_batch_size: int = 10,
        upsert_batch_size: int = 64,
        delete_batch_size: int = 256,
        shard_number: int = 1,
        replication_factor: int = 1,
        write_consistency_factor: int = 1,
        write_ordering: str = "medium",
        read_consistency: str = "majority",
        on_disk_payload: bool = True,
        on_disk_vectors: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        hnsw_full_scan_threshold: int = 10_000,
        search_hnsw_ef: int = 128,
        native_candidate_multiplier: int = 4,
        native_max_candidates: int = 100,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        schema_fingerprint: str = "",
        client: Any | None = None,
    ):
        """连接 Qdrant，并保存集合、写入、ANN 召回与混合排序参数。"""
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise EnvironmentError(
                "缺少 qdrant-client，请先执行 python -m pip install -r requirements.txt"
            ) from exc

        combine_hybrid_scores(
            [],
            [],
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        positive_values = {
            "Qdrant gRPC 端口": grpc_port,
            "Qdrant 超时": timeout,
            "Qdrant 向量维度": vector_size,
            "Qdrant Embedding 批次": embedding_batch_size,
            "Qdrant Upsert 批次": upsert_batch_size,
            "Qdrant Delete 批次": delete_batch_size,
            "Qdrant shard_number": shard_number,
            "Qdrant replication_factor": replication_factor,
            "Qdrant write_consistency_factor": write_consistency_factor,
            "Qdrant HNSW m": hnsw_m,
            "Qdrant HNSW ef_construct": hnsw_ef_construct,
            "Qdrant HNSW full_scan_threshold": hnsw_full_scan_threshold,
            "Qdrant 查询 hnsw_ef": search_hnsw_ef,
            "Qdrant 原生候选倍数": native_candidate_multiplier,
            "Qdrant 原生候选上限": native_max_candidates,
        }
        for label, value in positive_values.items():
            if int(value) < 1:
                raise ValueError(f"{label} 必须大于 0")
        if int(write_consistency_factor) > int(replication_factor):
            raise ValueError(
                "Qdrant write_consistency_factor 不能大于 replication_factor"
            )
        if str(distance).strip().lower() != "cosine":
            raise ValueError("Qdrant distance 当前只支持 cosine")
        normalized_write_ordering = str(write_ordering).strip().lower()
        if normalized_write_ordering not in {"weak", "medium", "strong"}:
            raise ValueError(
                "Qdrant write_ordering 只能是 weak、medium 或 strong"
            )
        normalized_read_consistency = str(read_consistency).strip().lower()
        if normalized_read_consistency not in {"majority", "quorum", "all"}:
            raise ValueError(
                "Qdrant read_consistency 只能是 majority、quorum 或 all"
            )
        if bm25_k1 <= 0:
            raise ValueError("BM25 的 k1 必须大于 0")
        if not 0 <= bm25_b <= 1:
            raise ValueError("BM25 的 b 必须在 0 到 1 之间")

        normalized_url = str(url).strip().rstrip("/")
        parsed_url = urlparse(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Qdrant URL 必须是有效的 HTTP(S) URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("Qdrant URL 不能包含凭据，请使用 QDRANT_API_KEY")
        try:
            http_port = parsed_url.port
        except ValueError as exc:
            raise ValueError("Qdrant URL 端口必须在 1 到 65535 之间") from exc
        if http_port is not None and not 1 <= http_port <= 65_535:
            raise ValueError("Qdrant URL 端口必须在 1 到 65535 之间")
        if int(grpc_port) > 65_535:
            raise ValueError("Qdrant gRPC 端口必须在 1 到 65535 之间")
        normalized_collection = str(collection_name).strip()
        if not normalized_collection:
            raise ValueError("Qdrant collection_name 不能为空")

        self.collection_name = normalized_collection
        self.embedding_model = embedding_model
        self.url = normalized_url
        self.grpc_port = int(grpc_port)
        self.prefer_grpc = bool(prefer_grpc)
        self.timeout = int(timeout)
        self.vector_size = int(vector_size)
        self.embedding_batch_size = int(embedding_batch_size)
        self.upsert_batch_size = int(upsert_batch_size)
        self.delete_batch_size = int(delete_batch_size)
        self.shard_number = int(shard_number)
        self.replication_factor = int(replication_factor)
        self.write_consistency_factor = int(write_consistency_factor)
        self.write_ordering = models.WriteOrdering(normalized_write_ordering)
        self.read_consistency = models.ReadConsistencyType(
            normalized_read_consistency
        )
        self.on_disk_payload = bool(on_disk_payload)
        self.on_disk_vectors = bool(on_disk_vectors)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construct = int(hnsw_ef_construct)
        self.hnsw_full_scan_threshold = int(hnsw_full_scan_threshold)
        self.search_hnsw_ef = int(search_hnsw_ef)
        self.native_candidate_multiplier = int(native_candidate_multiplier)
        self.native_max_candidates = int(native_max_candidates)
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.schema_fingerprint = str(schema_fingerprint).strip()
        self._models = models
        self._collection_ready = False
        self._client = (
            client
            if client is not None
            else QdrantClient(
                url=self.url,
                api_key=str(api_key).strip() or None,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout,
            )
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """把任意稳定分片 ID 映射为 Qdrant 接受的确定性 UUID。"""
        return str(uuid5(NAMESPACE_URL, f"zzdx-education-vector:{chunk_id}"))

    def _validate_embedding(self, values: Sequence[float]) -> list[float]:
        """校验向量是指定维度的有限浮点数列表。"""
        vector = [float(value) for value in values]
        if len(vector) != self.vector_size:
            raise ValueError(
                "Embedding 向量维度与 Qdrant 配置不一致："
                f"期望 {self.vector_size}，实际 {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding 模型返回了 NaN 或无穷大")
        return vector

    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """按供应商批次限制生成文档向量，并逐批校验数量与维度。"""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = list(texts[start : start + self.embedding_batch_size])
            raw_batch = self.embedding_model.embed_documents(batch)
            if len(raw_batch) != len(batch):
                raise RuntimeError("Embedding 返回数量与输入文本数量不一致")
            embeddings.extend(self._validate_embedding(vector) for vector in raw_batch)
        return embeddings

    def _validate_collection(self) -> Any:
        """拒绝复用维度、距离或向量命名方式不兼容的已有集合。"""
        info = self._client.get_collection(collection_name=self.collection_name)
        vectors = info.config.params.vectors
        if not hasattr(vectors, "size") or not hasattr(vectors, "distance"):
            raise RuntimeError(
                "Qdrant 集合使用了命名向量，当前项目只支持单个未命名向量；"
                "请执行 python main.py --rebuild-index"
            )
        if int(vectors.size) != self.vector_size:
            raise RuntimeError(
                "Qdrant 集合向量维度不匹配："
                f"期望 {self.vector_size}，实际 {vectors.size}；"
                "请执行 python main.py --rebuild-index"
            )
        if vectors.distance != self._models.Distance.COSINE:
            raise RuntimeError(
                "Qdrant 集合距离度量不是 Cosine；"
                "请执行 python main.py --rebuild-index"
            )
        if self.schema_fingerprint:
            collection_metadata = getattr(info.config, "metadata", None) or {}
            if (
                collection_metadata.get("managed_by")
                != "zzdx-education-qa"
                or collection_metadata.get("schema_fingerprint")
                != self.schema_fingerprint
            ):
                raise RuntimeError(
                    "Qdrant 集合不属于当前索引版本或索引参数已经变化；"
                    "请执行 python main.py --rebuild-index"
                )

        collection_params = info.config.params
        expected_params = {
            "shard_number": self.shard_number,
            "replication_factor": self.replication_factor,
            "write_consistency_factor": self.write_consistency_factor,
            "on_disk_payload": self.on_disk_payload,
        }
        for field, expected in expected_params.items():
            actual = getattr(collection_params, field, None)
            if actual is not None and actual != expected:
                raise RuntimeError(
                    f"Qdrant 集合参数 {field} 不匹配："
                    f"期望 {expected}，实际 {actual}；"
                    "请执行 python main.py --rebuild-index"
                )
        if vectors.on_disk is not None and vectors.on_disk != self.on_disk_vectors:
            raise RuntimeError(
                "Qdrant 集合 on_disk_vectors 参数不匹配；"
                "请执行 python main.py --rebuild-index"
            )
        hnsw_config = getattr(info.config, "hnsw_config", None)
        expected_hnsw = {
            "m": self.hnsw_m,
            "ef_construct": self.hnsw_ef_construct,
            "full_scan_threshold": self.hnsw_full_scan_threshold,
        }
        for field, expected in expected_hnsw.items():
            actual = getattr(hnsw_config, field, None)
            if actual is not None and int(actual) != expected:
                raise RuntimeError(
                    f"Qdrant HNSW 参数 {field} 不匹配："
                    f"期望 {expected}，实际 {actual}；"
                    "请执行 python main.py --rebuild-index"
                )

        subject_index = (getattr(info, "payload_schema", None) or {}).get(
            "subject"
        )
        if (
            subject_index is not None
            and getattr(subject_index, "data_type", None)
            != self._models.PayloadSchemaType.KEYWORD
        ):
            raise RuntimeError(
                "Qdrant subject payload 索引不是 KEYWORD；"
                "请执行 python main.py --rebuild-index"
            )
        return info

    def _ensure_collection(self) -> None:
        """幂等创建并验证集合，同时建立常用学科过滤索引。"""
        if self._collection_ready:
            return

        if not self._client.collection_exists(
            collection_name=self.collection_name
        ):
            try:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self._models.VectorParams(
                        size=self.vector_size,
                        distance=self._models.Distance.COSINE,
                        on_disk=self.on_disk_vectors,
                    ),
                    shard_number=self.shard_number,
                    replication_factor=self.replication_factor,
                    write_consistency_factor=self.write_consistency_factor,
                    on_disk_payload=self.on_disk_payload,
                    hnsw_config=self._models.HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct,
                        full_scan_threshold=self.hnsw_full_scan_threshold,
                    ),
                    metadata=(
                        {
                            "managed_by": "zzdx-education-qa",
                            "schema_fingerprint": self.schema_fingerprint,
                        }
                        if self.schema_fingerprint
                        else None
                    ),
                )
            except Exception:
                # 多副本应用首次同时启动时，另一个进程可能已经创建成功。
                if not self._client.collection_exists(
                    collection_name=self.collection_name
                ):
                    raise

        info = self._validate_collection()
        payload_schema = getattr(info, "payload_schema", None) or {}
        if "subject" not in payload_schema:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="subject",
                field_schema=self._models.PayloadSchemaType.KEYWORD,
                wait=True,
                ordering=self.write_ordering,
            )
        self._collection_ready = True

    def _build_filter(self, filter_value: dict[str, Any] | None) -> Any:
        """把上层等值过滤字典转换成 Qdrant 原生 Filter。"""
        if not filter_value:
            return None
        conditions = []
        for key, value in filter_value.items():
            if not isinstance(value, (str, int, bool)):
                raise ValueError(
                    f"Qdrant 等值过滤暂不支持 {key}={value!r}"
                )
            payload_key = (
                key if key in self._FLAT_METADATA_FIELDS else f"metadata.{key}"
            )
            conditions.append(
                self._models.FieldCondition(
                    key=payload_key,
                    match=self._models.MatchValue(value=value),
                )
            )
        return self._models.Filter(must=conditions)

    @staticmethod
    def _json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """把元数据规范化为 Qdrant payload 支持的 JSON 值。"""
        try:
            normalized = json.loads(
                json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("文档元数据必须可以序列化为 JSON") from exc
        if not isinstance(normalized, dict):
            raise ValueError("文档元数据必须是对象")
        return normalized

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> list[str]:
        """生成向量并分批 upsert；所有写入都等待服务端确认。"""
        document_list = list(documents)
        id_list = [str(chunk_id) for chunk_id in ids]
        if len(document_list) != len(id_list):
            raise ValueError("documents 和 ids 的数量必须相同")
        if not document_list:
            return []

        embeddings = self._embed_documents(
            [document.page_content for document in document_list]
        )
        return self.upsert_precomputed_documents(
            document_list,
            ids=id_list,
            embeddings=embeddings,
        )

    def upsert_precomputed_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """写入已经生成的向量，用于无损迁移且不会重复调用 Embedding。"""

        document_list = list(documents)
        id_list = [str(chunk_id) for chunk_id in ids]
        embedding_list = [
            self._validate_embedding(vector) for vector in embeddings
        ]
        if len(document_list) != len(id_list):
            raise ValueError("documents 和 ids 的数量必须相同")
        if len(embedding_list) != len(id_list):
            raise ValueError("embeddings 和 ids 的数量必须相同")
        if not document_list:
            return []

        self._ensure_collection()
        points = []
        for chunk_id, document, embedding in zip(
            id_list,
            document_list,
            embedding_list,
            strict=True,
        ):
            metadata = self._json_metadata(dict(document.metadata))
            metadata_chunk_id = metadata.get("chunk_id")
            if (
                metadata_chunk_id is not None
                and str(metadata_chunk_id) != chunk_id
            ):
                raise ValueError("文档 metadata.chunk_id 与写入 ID 不一致")
            metadata["chunk_id"] = chunk_id
            payload: dict[str, Any] = {
                "page_content": document.page_content,
                "metadata": metadata,
                "chunk_id": chunk_id,
            }
            for field in ("subject", "source", "file_hash"):
                value = metadata.get(field)
                if isinstance(value, (str, int, bool)):
                    payload[field] = value
            points.append(
                self._models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        # Qdrant upsert 使用确定性 UUID，失败后可以幂等重跑。不能把之前成功的
        # 批次直接删除：其中可能是对既有点的覆盖，删除并不等于回滚旧值。
        for start in range(0, len(points), self.upsert_batch_size):
            batch = points[start : start + self.upsert_batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
                ordering=self.write_ordering,
            )
        return id_list

    def _delete_point_ids(self, point_ids: Sequence[str]) -> None:
        """把 Qdrant UUID 分批删除，避免大文件产生超大请求。"""
        for start in range(0, len(point_ids), self.delete_batch_size):
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=self._models.PointIdsList(
                    points=list(point_ids[start : start + self.delete_batch_size])
                ),
                wait=True,
                ordering=self.write_ordering,
            )

    def delete(self, ids: Sequence[str]) -> None:
        """幂等删除原始分片 ID 对应的 Qdrant UUID 点。"""
        id_list = [str(chunk_id) for chunk_id in ids]
        if not id_list:
            return
        if not self._client.collection_exists(
            collection_name=self.collection_name
        ):
            self._collection_ready = False
            return
        self._delete_point_ids(
            [self._point_id(chunk_id) for chunk_id in id_list]
        )

    def count_documents(self) -> int:
        """精确返回集合点数；集合不存在时返回 0，用于与 manifest 对账。"""
        if not self._client.collection_exists(
            collection_name=self.collection_name
        ):
            self._collection_ready = False
            return 0
        self._ensure_collection()
        result = self._client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)

    def reset_collection(self) -> None:
        """幂等删除并按当前生产参数重建集合。"""
        if self._client.collection_exists(
            collection_name=self.collection_name
        ):
            self._client.delete_collection(collection_name=self.collection_name)
        self._collection_ready = False
        self._ensure_collection()

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """用 Qdrant ANN 召回有限候选，再在候选集上融合 BM25 分数。"""
        if k < 1:
            return []

        query_vector = self._validate_embedding(
            self.embedding_model.embed_query(query)
        )
        self._ensure_collection()
        candidate_k = min(
            self.native_max_candidates,
            max(k, k * self.native_candidate_multiplier),
        )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_filter(filter),
            search_params=self._models.SearchParams(
                hnsw_ef=self.search_hnsw_ef,
                exact=False,
            ),
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
            consistency=self.read_consistency,
        )

        candidate_ids: list[str] = []
        candidate_documents: list[Document] = []
        cosine_scores: list[float] = []
        for point in response.points:
            payload = point.payload or {}
            page_content = payload.get("page_content")
            metadata = payload.get("metadata") or {}
            if not isinstance(page_content, str) or not isinstance(metadata, dict):
                raise RuntimeError("Qdrant 候选点的正文或元数据格式错误")
            metadata = dict(metadata)
            stable_id = str(payload.get("chunk_id") or point.id)
            metadata.setdefault("chunk_id", stable_id)
            for field in ("subject", "source", "file_hash"):
                if field not in metadata and field in payload:
                    metadata[field] = payload[field]

            score = float(point.score)
            if not math.isfinite(score):
                raise RuntimeError("Qdrant 返回了 NaN 或无穷相似度")
            candidate_ids.append(stable_id)
            candidate_documents.append(
                Document(page_content=page_content, metadata=metadata)
            )
            # Qdrant Cosine 返回的已经是“越大越相似”的 score，不做 1-score。
            cosine_scores.append(max(-1.0, min(1.0, score)))

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
