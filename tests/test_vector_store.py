# 文件用途：验证 SQLite 向量持久化、批处理、增量同步和后端状态报告。
# 调用关系：unittest 调用本文件；本文件调用 rag/vector_store.py 与 vector_backends.py。
# 修改易踩坑：使用假 Embedding 保持离线；所有临时索引、清单和连接都要在测试后释放。
import json
import sqlite3
import unittest
import warnings
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models as qdrant_models

from rag.vector_backends import (
    ChromaVectorBackend,
    QdrantVectorBackend,
    SQLiteVectorBackend,
)
from rag.vector_store import EducationVectorStore


class KeywordEmbeddings(Embeddings):
    """用关键词生成固定三维向量，让离线测试不调用在线模型。"""

    def __init__(self):
        """创建批次记录列表，供测试确认自动拆批是否生效。"""
        self.batch_sizes: list[int] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        """把数学、科学和其他文本映射到不同方向的测试向量。"""
        if any(word in text for word in ("数学", "三角形", "长方形")):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ("科学", "植物", "太阳")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """记录本批文本数量，并为每段文本返回固定测试向量。"""
        self.batch_sizes.append(len(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """使用与文档相同的规则生成查询向量。"""
        return self._vector(text)


class FailingKeywordEmbeddings(KeywordEmbeddings):
    """允许测试主动触发文档向量化失败。"""

    def __init__(self):
        super().__init__()
        self.fail_documents = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_documents:
            raise RuntimeError("模拟在线 Embedding 中断")
        return super().embed_documents(texts)


class VectorStoreTests(unittest.TestCase):
    @patch("langchain_chroma.Chroma")
    @patch("chromadb.HttpClient")
    def test_chroma_uses_http_native_query_without_full_collection_get(
        self,
        mock_http_client,
        mock_chroma,
    ):
        """验证 Chroma 走独立 HTTP 服务，并只对原生 ANN 候选做混合排序。"""

        fake_store = MagicMock()
        fake_store.similarity_search_with_score.return_value = [
            (
                Document(
                    id="time",
                    page_content="时间知识：一年有十二个月。",
                    metadata={"subject": "数学"},
                ),
                0.0,
            ),
            (
                Document(
                    id="triangle",
                    page_content="图形知识：三角形有三条边。",
                    metadata={"subject": "数学"},
                ),
                0.2,
            ),
        ]
        mock_chroma.return_value = fake_store

        backend = ChromaVectorBackend(
            collection_name="native-query-test",
            embedding_model=KeywordEmbeddings(),
            deployment_mode="http",
            host="chroma.internal",
            port=8000,
            native_candidate_multiplier=4,
            native_max_candidates=50,
        )
        results = backend.similarity_search(
            "三角形有几条边？",
            k=2,
            filter={"subject": "数学"},
        )

        mock_http_client.assert_called_once_with(
            host="chroma.internal",
            port=8000,
            ssl=False,
            headers=None,
            tenant="default_tenant",
            database="default_database",
        )
        fake_store.similarity_search_with_score.assert_called_once_with(
            "三角形有几条边？",
            k=8,
            filter={"subject": "数学"},
        )
        fake_store.get.assert_not_called()
        self.assertIn("三角形有三条边", results[0].page_content)

    def test_qdrant_uses_native_query_and_keeps_cosine_score_direction(self):
        """验证 Qdrant 只查有限 ANN 候选，并直接使用越大越好的 cosine score。"""
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        fake_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=qdrant_models.VectorParams(
                        size=3,
                        distance=qdrant_models.Distance.COSINE,
                    )
                )
            ),
            payload_schema={
                "subject": qdrant_models.PayloadIndexInfo(
                    data_type=qdrant_models.PayloadSchemaType.KEYWORD,
                    points=0,
                )
            },
        )
        fake_client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="00000000-0000-0000-0000-000000000001",
                    score=0.9,
                    payload={
                        "page_content": "候选甲",
                        "metadata": {"subject": "数学"},
                        "chunk_id": "high-score",
                        "subject": "数学",
                    },
                ),
                SimpleNamespace(
                    id="00000000-0000-0000-0000-000000000002",
                    score=0.1,
                    payload={
                        "page_content": "候选乙",
                        "metadata": {"subject": "数学"},
                        "chunk_id": "low-score",
                        "subject": "数学",
                    },
                ),
            ]
        )
        backend = QdrantVectorBackend(
            collection_name="native-query-test",
            embedding_model=KeywordEmbeddings(),
            client=fake_client,
            vector_size=3,
            native_candidate_multiplier=4,
            native_max_candidates=50,
            vector_weight=1.0,
            bm25_weight=0.0,
        )

        results = backend.similarity_search(
            "没有匹配词的查询",
            k=2,
            filter={"subject": "数学"},
        )

        query_call = fake_client.query_points.call_args.kwargs
        self.assertEqual(query_call["collection_name"], "native-query-test")
        self.assertEqual(query_call["limit"], 8)
        self.assertTrue(query_call["with_payload"])
        self.assertFalse(query_call["with_vectors"])
        self.assertEqual(
            query_call["consistency"],
            qdrant_models.ReadConsistencyType.MAJORITY,
        )
        self.assertEqual(query_call["query_filter"].must[0].key, "subject")
        self.assertEqual(
            query_call["query_filter"].must[0].match.value,
            "数学",
        )
        fake_client.scroll.assert_not_called()
        fake_client.retrieve.assert_not_called()
        fake_client.get.assert_not_called()
        self.assertEqual(results[0].metadata["chunk_id"], "high-score")

    def test_qdrant_batches_upserts_and_maps_sha_ids_to_stable_uuids(self):
        """验证 Qdrant 写入分批、payload 和 SHA-256 ID 的 UUID 兼容层。"""
        fake_client = MagicMock()
        fake_client.collection_exists.side_effect = [False, True]
        fake_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=qdrant_models.VectorParams(
                        size=3,
                        distance=qdrant_models.Distance.COSINE,
                    )
                )
            ),
            payload_schema={},
        )
        embeddings = KeywordEmbeddings()
        backend = QdrantVectorBackend(
            collection_name="write-test",
            embedding_model=embeddings,
            client=fake_client,
            vector_size=3,
            embedding_batch_size=2,
            upsert_batch_size=2,
        )
        chunk_ids = ["a" * 64, "b" * 64, "c" * 64]
        documents = [
            Document(
                page_content=f"数学片段 {index}",
                metadata={
                    "subject": "数学",
                    "source": "数学.txt",
                    "file_hash": str(index),
                },
            )
            for index in range(3)
        ]

        returned_ids = backend.add_documents(documents, ids=chunk_ids)
        backend.delete(chunk_ids[:2])

        self.assertEqual(returned_ids, chunk_ids)
        self.assertEqual(embeddings.batch_sizes, [2, 1])
        create_call = fake_client.create_collection.call_args.kwargs
        self.assertEqual(create_call["vectors_config"].size, 3)
        self.assertEqual(
            create_call["vectors_config"].distance,
            qdrant_models.Distance.COSINE,
        )
        fake_client.create_payload_index.assert_called_once_with(
            collection_name="write-test",
            field_name="subject",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            wait=True,
            ordering=qdrant_models.WriteOrdering.MEDIUM,
        )
        self.assertEqual(fake_client.upsert.call_count, 2)
        written_points = [
            point
            for call in fake_client.upsert.call_args_list
            for point in call.kwargs["points"]
        ]
        self.assertEqual(len(written_points), 3)
        self.assertEqual(written_points[0].payload["chunk_id"], chunk_ids[0])
        self.assertEqual(
            written_points[0].payload["metadata"]["chunk_id"],
            chunk_ids[0],
        )
        self.assertEqual(
            str(UUID(str(written_points[0].id))),
            QdrantVectorBackend._point_id(chunk_ids[0]),
        )
        self.assertNotEqual(
            QdrantVectorBackend._point_id(chunk_ids[0]),
            QdrantVectorBackend._point_id(chunk_ids[1]),
        )
        delete_selector = fake_client.delete.call_args.kwargs["points_selector"]
        self.assertEqual(
            delete_selector.points,
            [QdrantVectorBackend._point_id(value) for value in chunk_ids[:2]],
        )
        self.assertTrue(fake_client.delete.call_args.kwargs["wait"])
        self.assertEqual(
            fake_client.delete.call_args.kwargs["ordering"],
            qdrant_models.WriteOrdering.MEDIUM,
        )

    def test_qdrant_partial_upsert_keeps_completed_points_for_idempotent_retry(self):
        """验证多批写入中断时不误删可能原本存在的已完成点。"""
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        fake_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=qdrant_models.VectorParams(
                        size=3,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                    shard_number=None,
                    replication_factor=None,
                    write_consistency_factor=None,
                    on_disk_payload=None,
                ),
                hnsw_config=SimpleNamespace(
                    m=None,
                    ef_construct=None,
                    full_scan_threshold=None,
                ),
                metadata=None,
            ),
            payload_schema={
                "subject": qdrant_models.PayloadIndexInfo(
                    data_type=qdrant_models.PayloadSchemaType.KEYWORD,
                    points=0,
                )
            },
        )
        fake_client.upsert.side_effect = [MagicMock(), RuntimeError("第二批失败")]
        backend = QdrantVectorBackend(
            collection_name="compensation-test",
            embedding_model=KeywordEmbeddings(),
            client=fake_client,
            vector_size=3,
            upsert_batch_size=1,
        )
        chunk_ids = ["a" * 64, "b" * 64]

        with self.assertRaisesRegex(RuntimeError, "第二批失败"):
            backend.add_documents(
                [
                    Document(page_content="数学片段一"),
                    Document(page_content="数学片段二"),
                ],
                ids=chunk_ids,
            )

        self.assertEqual(fake_client.upsert.call_count, 2)
        fake_client.delete.assert_not_called()

    def test_qdrant_rejects_embedding_dimension_before_writing(self):
        """验证模型向量维度错误时不会创建集合或写入部分数据。"""
        fake_client = MagicMock()
        backend = QdrantVectorBackend(
            collection_name="dimension-test",
            embedding_model=KeywordEmbeddings(),
            client=fake_client,
            vector_size=4,
        )

        with self.assertRaisesRegex(ValueError, "期望 4，实际 3"):
            backend.add_documents(
                [Document(page_content="数学片段")],
                ids=["a" * 64],
            )

        fake_client.collection_exists.assert_not_called()
        fake_client.create_collection.assert_not_called()
        fake_client.upsert.assert_not_called()

    def test_qdrant_rejects_incompatible_existing_collection(self):
        """验证已有集合维度不匹配时明确要求重建，不静默写入错误集合。"""
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        fake_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=qdrant_models.VectorParams(
                        size=4,
                        distance=qdrant_models.Distance.COSINE,
                    )
                )
            ),
            payload_schema={
                "subject": qdrant_models.PayloadIndexInfo(
                    data_type=qdrant_models.PayloadSchemaType.KEYWORD,
                    points=0,
                )
            },
        )
        backend = QdrantVectorBackend(
            collection_name="incompatible-test",
            embedding_model=KeywordEmbeddings(),
            client=fake_client,
            vector_size=3,
        )

        with self.assertRaisesRegex(RuntimeError, "维度不匹配"):
            backend.similarity_search("数学", k=2)

        fake_client.query_points.assert_not_called()

    def test_qdrant_client_in_memory_round_trip(self):
        """用真实 qdrant-client 本地引擎验证创建、写入、查询、删除和重置。"""
        client = QdrantClient(":memory:")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                backend = QdrantVectorBackend(
                    collection_name="round-trip-test",
                    embedding_model=KeywordEmbeddings(),
                    client=client,
                    vector_size=3,
                    on_disk_payload=False,
                    on_disk_vectors=False,
                    schema_fingerprint="round-trip-schema-v1",
                )
                chunk_ids = ["a" * 64, "b" * 64]
                backend.add_documents(
                    [
                        Document(
                            page_content="数学知识：三角形有三条边。",
                            metadata={"subject": "数学"},
                        ),
                        Document(
                            page_content="科学知识：植物生长需要太阳。",
                            metadata={"subject": "科学"},
                        ),
                    ],
                    ids=chunk_ids,
                )
                results = backend.similarity_search(
                    "三角形",
                    k=2,
                    filter={"subject": "数学"},
                )
                self.assertEqual(backend.count_documents(), 2)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].metadata["chunk_id"], chunk_ids[0])

                backend.delete([chunk_ids[0]])
                self.assertEqual(
                    backend.similarity_search(
                        "三角形",
                        k=2,
                        filter={"subject": "数学"},
                    ),
                    [],
                )
                backend.reset_collection()
                self.assertTrue(client.collection_exists("round-trip-test"))
                self.assertEqual(backend.count_documents(), 0)
        finally:
            client.close()

    @patch("rag.vector_store.QdrantVectorBackend")
    def test_store_factory_selects_qdrant_without_affecting_other_backends(
        self,
        mock_qdrant_backend,
    ):
        """验证上层能按配置创建 Qdrant，同时保留现有统一接口。"""
        store = EducationVectorStore(
            embedding_model=KeywordEmbeddings(),
            backend_name="qdrant",
        )

        self.assertIs(store._get_store(), mock_qdrant_backend.return_value)
        options = mock_qdrant_backend.call_args.kwargs
        self.assertEqual(options["url"], "http://127.0.0.1:6333")
        self.assertEqual(options["vector_size"], 1024)
        self.assertEqual(options["distance"], "cosine")
        self.assertEqual(options["native_max_candidates"], 100)
        self.assertEqual(options["delete_batch_size"], 256)
        self.assertEqual(options["write_ordering"], "medium")
        self.assertEqual(options["read_consistency"], "majority")
        self.assertEqual(len(options["schema_fingerprint"]), 64)

    def test_search_can_disable_automatic_sync_for_read_only_replicas(self):
        """验证生产服务副本可只查询，不与唯一索引任务争抢 manifest。"""
        store = EducationVectorStore(
            embedding_model=KeywordEmbeddings(),
            backend_name="sqlite",
        )
        store.sync_on_search = False
        fake_backend = MagicMock()
        fake_backend.similarity_search.return_value = []
        store._vector_store = fake_backend

        with patch.object(store, "sync") as mock_sync:
            self.assertEqual(store.search("三角形", subject="数学"), [])

        mock_sync.assert_not_called()
        fake_backend.similarity_search.assert_called_once()

    def test_qdrant_sync_reconciles_missing_manifest_and_collection_points(self):
        """验证本地 manifest 与远端点数任一丢失时都会受控全量重建。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            store = EducationVectorStore(
                embedding_model=KeywordEmbeddings(),
                backend_name="qdrant",
            )
            store.data_path = knowledge
            store.manifest_path = root / "manifest.json"
            fake_backend = MagicMock(spec=QdrantVectorBackend)
            store._vector_store = fake_backend

            # manifest 不存在时不信任远端同名集合，先清空应用专属集合。
            store.sync()
            fake_backend.reset_collection.assert_called_once()
            saved_manifest = json.loads(
                store.manifest_path.read_text(encoding="utf-8")
            )
            self.assertNotIn("_manifest_valid", saved_manifest)

            # manifest 声称有一个点、远端实际为 0，模拟 volume 丢失。
            saved_manifest["files"] = {
                "data/knowledge/数学旧文件.txt": {
                    "sha256": "old",
                    "ids": ["a" * 64],
                }
            }
            store._save_manifest(saved_manifest)
            fake_backend.reset_collection.reset_mock()
            fake_backend.count_documents.return_value = 0

            store.sync()

            fake_backend.count_documents.assert_called_once_with()
            fake_backend.reset_collection.assert_called_once_with()
            fake_backend.delete.assert_not_called()

    def test_update_keeps_old_chunks_when_embedding_fails(self):
        """验证新版本向量化失败时，旧文件切片仍保留在数据库中。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            source = knowledge / "数学测试.txt"
            old_text = "长方形面积等于长乘宽。"
            source.write_text(old_text, encoding="utf-8")

            embeddings = FailingKeywordEmbeddings()
            store = EducationVectorStore(
                embedding_model=embeddings,
                backend_name="sqlite",
            )
            store.data_path = knowledge
            store.sqlite_database_path = root / "vectors.sqlite3"
            store.persist_directory = store.sqlite_database_path
            store.manifest_path = root / "manifest.json"
            store.sync()

            source.write_text("更新后的数学内容。", encoding="utf-8")
            embeddings.fail_documents = True
            with self.assertRaisesRegex(RuntimeError, "模拟在线 Embedding 中断"):
                store.sync()

            with closing(sqlite3.connect(store.sqlite_database_path)) as connection:
                rows = connection.execute(
                    "SELECT document FROM education_vector_chunks"
                ).fetchall()
            self.assertEqual(rows, [(old_text,)])

    def test_add_update_search_and_remove_file_with_sqlite(self):
        """验证真实 SQLite 后端能够增量增删、更新并按学科检索。"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            source = knowledge / "数学测试.txt"
            source.write_text("长方形面积等于长乘宽。", encoding="utf-8")

            store = EducationVectorStore(
                embedding_model=KeywordEmbeddings(),
                backend_name="sqlite",
            )
            store.data_path = knowledge
            store.sqlite_database_path = root / "vectors.sqlite3"
            store.persist_directory = store.sqlite_database_path
            store.manifest_path = root / "manifest.json"

            first = store.sync()
            self.assertEqual(first.added_files, 1)
            self.assertGreater(first.indexed_chunks, 0)
            documents = store.search("长方形面积", subject="数学")
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["subject"], "数学")

            second = store.sync()
            self.assertEqual(second.unchanged_files, 1)

            source.write_text(
                "长方形面积等于长乘宽，周长等于长与宽的和乘二。",
                encoding="utf-8",
            )
            third = store.sync()
            self.assertEqual(third.updated_files, 1)

            source.unlink()
            fourth = store.sync()
            self.assertEqual(fourth.removed_files, 1)
            self.assertEqual(store.search("长方形", subject="数学"), [])

    def test_sqlite_vectors_survive_new_backend_instance(self):
        """验证关闭旧对象后，新 SQLite 后端仍能读取已经保存的向量。"""
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "vectors.sqlite3"
            embeddings = KeywordEmbeddings()
            first = SQLiteVectorBackend(
                database_path=database_path,
                collection_name="test",
                embedding_model=embeddings,
                embedding_batch_size=10,
            )
            first.add_documents(
                [
                    Document(
                        page_content="数学知识：三角形有三条边。",
                        metadata={"subject": "数学", "source": "数学.txt"},
                    ),
                    Document(
                        page_content="科学知识：植物生长需要阳光。",
                        metadata={"subject": "科学", "source": "科学.txt"},
                    ),
                ],
                ids=["math-1", "science-1"],
            )

            reopened = SQLiteVectorBackend(
                database_path=database_path,
                collection_name="test",
                embedding_model=KeywordEmbeddings(),
            )
            results = reopened.similarity_search(
                "三角形",
                k=2,
                filter={"subject": "数学"},
            )
            self.assertEqual(len(results), 1)
            self.assertIn("三条边", results[0].page_content)

    def test_embedding_requests_are_split_into_configured_batches(self):
        """验证后端会按指定批次处理向量，避免一次占用过多内存。"""
        with TemporaryDirectory() as directory:
            embeddings = KeywordEmbeddings()
            backend = SQLiteVectorBackend(
                database_path=Path(directory) / "vectors.sqlite3",
                collection_name="batch-test",
                embedding_model=embeddings,
                embedding_batch_size=10,
            )
            documents = [
                Document(page_content=f"数学片段 {index}")
                for index in range(23)
            ]
            backend.add_documents(
                documents,
                ids=[f"chunk-{index}" for index in range(23)],
            )

            self.assertEqual(embeddings.batch_sizes, [10, 10, 3])
            with closing(sqlite3.connect(backend.database_path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM education_vector_chunks"
                ).fetchone()[0]
            self.assertEqual(count, 23)

    def test_status_reports_selected_backend_without_loading_model(self):
        """验证状态接口显示所选后端，且不需要创建在线 Embedding 客户端。"""
        store = EducationVectorStore(backend_name="sqlite")
        status = store.status()
        self.assertEqual(status["backend"], "sqlite")
        self.assertEqual(
            status["retrieval_mode"],
            "bm25_cosine_hybrid_aliyun_rerank",
        )
        self.assertTrue(status["storage_path"].endswith("knowledge_vectors.sqlite3"))
        self.assertEqual(status["persist_directory"], status["storage_path"])

        chroma_status = EducationVectorStore(backend_name="chroma").status()
        self.assertEqual(chroma_status["backend"], "chroma")
        self.assertEqual(chroma_status["storage_path"], "http://127.0.0.1:8000")
        self.assertEqual(chroma_status["chroma_server"]["mode"], "http")
        self.assertEqual(
            chroma_status["chroma_server"]["native_max_candidates"],
            100,
        )

        qdrant_status = EducationVectorStore(backend_name="qdrant").status()
        self.assertEqual(qdrant_status["backend"], "qdrant")
        self.assertEqual(qdrant_status["storage_path"], "http://127.0.0.1:6333")
        self.assertIsNone(qdrant_status["chroma_server"])
        self.assertEqual(qdrant_status["qdrant_server"]["vector_size"], 1024)
        self.assertEqual(
            qdrant_status["qdrant_server"]["native_max_candidates"],
            100,
        )
        self.assertEqual(
            qdrant_status["qdrant_server"]["write_ordering"],
            "medium",
        )
        self.assertEqual(
            len(qdrant_status["qdrant_server"]["schema_fingerprint"]),
            64,
        )


if __name__ == "__main__":
    unittest.main()
