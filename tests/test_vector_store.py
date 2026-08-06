import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vector_backends import SQLiteVectorBackend
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


class VectorStoreTests(unittest.TestCase):
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
        self.assertEqual(status["retrieval_mode"], "bm25_cosine_hybrid")
        self.assertTrue(status["storage_path"].endswith("knowledge_vectors.sqlite3"))
        self.assertEqual(status["persist_directory"], status["storage_path"])

        chroma_status = EducationVectorStore(backend_name="chroma").status()
        self.assertEqual(chroma_status["backend"], "chroma")
        chroma_path = Path(chroma_status["storage_path"])
        self.assertEqual(chroma_path.parts[-2:], ("storage", "chroma"))


if __name__ == "__main__":
    unittest.main()
