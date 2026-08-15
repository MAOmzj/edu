# 文件用途：验证中文分词、BM25、余弦融合和 SQLite 后端最终排序。
# 调用关系：unittest 调用本文件；本文件调用 rag/hybrid_search.py 与向量后端。
# 修改易踩坑：断言应验证相对排序而非脆弱的小数细节，临时数据库必须彼此隔离。
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.hybrid_search import (
    calculate_bm25_scores,
    combine_hybrid_scores,
    tokenize_for_bm25,
)
from rag.vector_backends import SQLiteVectorBackend


class ConflictingEmbeddings(Embeddings):
    """故意让纯向量排序偏向错误文档，用于验证 BM25 能参与纠正。"""

    @staticmethod
    def _document_vector(text: str) -> list[float]:
        """让时间文档与查询完全同向，三角形文档只保持较高语义相似度。"""
        if "时间" in text:
            return [1.0, 0.0]
        if "三角形" in text:
            return [0.8, 0.6]
        return [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为每篇测试文档生成预先设计的冲突向量。"""
        return [self._document_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """返回会使纯余弦检索错误偏向时间文档的查询向量。"""
        del text
        return [1.0, 0.0]


class HybridSearchTests(unittest.TestCase):
    def test_chinese_tokenizer_keeps_characters_and_bigrams(self):
        """验证中文词语同时生成单字和相邻双字，英文则统一成小写。"""
        tokens = tokenize_for_bm25("三角形 Area")
        self.assertIn("三", tokens)
        self.assertIn("三角", tokens)
        self.assertIn("角形", tokens)
        self.assertIn("area", tokens)

    def test_bm25_prefers_document_with_exact_terms(self):
        """验证 BM25 把包含“三角形、三条边”的文档排在时间知识前面。"""
        scores = calculate_bm25_scores(
            "三角形有几条边？",
            [
                "时间知识：一年有十二个月。",
                "图形知识：三角形有三条边。",
            ],
        )
        self.assertGreater(scores[1], scores[0])

    def test_score_fusion_can_correct_a_close_vector_mismatch(self):
        """验证精确关键词可以纠正仍有一定语义相似度的错误向量第一名。"""
        scores = combine_hybrid_scores(
            cosine_scores=[1.0, 0.8],
            bm25_scores=[0.0, 5.0],
            vector_weight=0.65,
            bm25_weight=0.35,
        )
        self.assertGreater(scores[1].combined, scores[0].combined)

    def test_sqlite_backend_returns_hybrid_first_place(self):
        """验证 SQLite 最终检索结果确实使用 BM25 与余弦混合排序。"""
        with TemporaryDirectory() as directory:
            backend = SQLiteVectorBackend(
                database_path=Path(directory) / "vectors.sqlite3",
                collection_name="hybrid-test",
                embedding_model=ConflictingEmbeddings(),
                vector_weight=0.65,
                bm25_weight=0.35,
            )
            backend.add_documents(
                [
                    Document(
                        page_content="时间知识：一年有十二个月。",
                        metadata={"subject": "数学"},
                    ),
                    Document(
                        page_content="图形知识：三角形有三条边。",
                        metadata={"subject": "数学"},
                    ),
                ],
                ids=["time", "triangle"],
            )

            results = backend.similarity_search(
                "三角形有几条边？",
                k=2,
                filter={"subject": "数学"},
            )
            self.assertIn("三角形有三条边", results[0].page_content)


if __name__ == "__main__":
    unittest.main()
