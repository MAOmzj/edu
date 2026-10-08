# 文件用途：离线验证持久化 BM25 倒排索引、学科统计和增量写入安全性。
# 调用关系：unittest 调用本文件；本文件调用 rag.lexical_index 和 BM25 纯计算工具。
# 修改易踩坑：所有数据库必须放在临时目录，查询不能重新分词全部文档正文。
"""独立关键词召回的离线持久化和统计回归测试。"""

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document

from rag.hybrid_search import calculate_bm25_scores, tokenize_for_bm25
from rag.lexical_index import BM25InvertedIndex


def make_document(identifier, text, subject="数学", source="数学.txt"):
    return Document(
        page_content=text,
        metadata={"chunk_id": identifier, "subject": subject, "source": source},
    )


class BM25InvertedIndexTests(unittest.TestCase):
    def test_persistence_preserves_metadata_sources_and_ready_marker(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "keywords.sqlite3"
            index = BM25InvertedIndex(database, "collection-a")
            document = make_document("math-1", "三角形面积使用底乘高除以二。")
            document.metadata["page"] = 7
            index.replace_source("数学.txt", "hash-a", [document], ["math-1"])
            index.mark_ready("manifest-a")

            reopened = BM25InvertedIndex(database, "collection-a")
            self.assertEqual(
                reopened.sources(),
                {"数学.txt": {"sha256": "hash-a", "ids": ["math-1"]}},
            )
            self.assertTrue(reopened.is_ready("manifest-a"))
            self.assertFalse(reopened.is_ready("manifest-b"))
            results = reopened.search("三角形面积", k=4, subject="数学")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].page_content, document.page_content)
            self.assertEqual(results[0].metadata["page"], 7)
            self.assertEqual(results[0].metadata["chunk_id"], "math-1")
            self.assertGreater(results[0].metadata["bm25_score"], 0)
            self.assertEqual(reopened.status()["indexed_chunks"], 1)

    def test_namespaces_isolate_documents_ready_markers_and_reset(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "keywords.sqlite3"
            first = BM25InvertedIndex(database, "collection-a")
            second = BM25InvertedIndex(database, "collection-b")
            # 同一来源和分片 ID 在不同集合中应互不覆盖。
            first.replace_source("数学.txt", "a", [make_document("same", "alpha")], ["same"])
            second.replace_source("数学.txt", "b", [make_document("same", "beta")], ["same"])
            first.mark_ready("a")
            second.mark_ready("b")
            self.assertEqual(first.search("beta", k=3), [])
            self.assertEqual(second.search("alpha", k=3), [])
            first.reset()
            self.assertEqual(first.sources(), {})
            self.assertFalse(first.is_ready("a"))
            self.assertTrue(second.is_ready("b"))
            self.assertEqual(second.search("beta", k=3)[0].page_content, "beta")

    def test_replace_and_delete_remove_old_postings_and_invalidate_readiness(self):
        with TemporaryDirectory() as directory:
            index = BM25InvertedIndex(Path(directory) / "keywords.sqlite3", "test")
            index.replace_source("数学.txt", "old", [make_document("old", "triangle")], ["old"])
            index.mark_ready("old-manifest")
            index.replace_source("数学.txt", "new", [make_document("new", "rectangle")], ["new"])
            self.assertFalse(index.is_ready("old-manifest"))
            with self.assertRaises(EnvironmentError):
                index.search("rectangle", k=5)
            index.mark_ready("new-manifest")
            self.assertEqual(index.search("triangle", k=5), [])
            self.assertEqual(index.search("rectangle", k=5)[0].metadata["chunk_id"], "new")
            self.assertEqual(index.sources()["数学.txt"], {"sha256": "new", "ids": ["new"]})
            index.mark_ready("new-manifest")
            index.delete_source("数学.txt")
            self.assertEqual(index.sources(), {})
            self.assertFalse(index.is_ready("new-manifest"))
            index.mark_ready("empty-manifest")
            self.assertEqual(index.search("rectangle", k=5), [])
            self.assertEqual(index.status()["indexed_chunks"], 0)
            index.delete_source("already-absent.txt")

    def test_subject_statistics_match_bm25_over_the_entire_filtered_corpus(self):
        with TemporaryDirectory() as directory:
            index = BM25InvertedIndex(Path(directory) / "keywords.sqlite3", "test")
            math_documents = [
                make_document("m1", "ratio ratio ratio"),
                make_document("m2", "ratio fraction"),
                make_document("m3", "fraction " + "other " * 20),
                make_document("m4", "unrelated short"),
            ]
            other_documents = [
                make_document(f"s{number}", "ratio", "科学", "科学.txt")
                for number in range(25)
            ]
            index.replace_source("数学.txt", "math", math_documents, [d.metadata["chunk_id"] for d in math_documents])
            index.replace_source("科学.txt", "science", other_documents, [d.metadata["chunk_id"] for d in other_documents])
            index.mark_ready("manifest")
            query = "ratio fraction"
            scores = calculate_bm25_scores(query, [d.page_content for d in math_documents])
            expected = {
                document.metadata["chunk_id"]: score
                for document, score in zip(math_documents, scores, strict=True)
                if score > 0
            }
            results = index.search(query, k=100, subject="数学")
            self.assertEqual({d.metadata["chunk_id"] for d in results}, set(expected))
            for document in results:
                self.assertEqual(document.metadata["subject"], "数学")
                self.assertAlmostEqual(document.metadata["bm25_score"], expected[document.metadata["chunk_id"]], places=12)
            self.assertEqual(
                [d.metadata["chunk_id"] for d in results],
                sorted(expected, key=lambda identifier: (-expected[identifier], identifier)),
            )
            self.assertEqual(index.search(query, k=100, subject="语文"), [])

    def test_chinese_and_normalized_alphanumeric_terms_do_not_return_irrelevant_documents(self):
        with TemporaryDirectory() as directory:
            index = BM25InvertedIndex(Path(directory) / "keywords.sqlite3", "test")
            documents = [
                make_document("matched", "三角形面积计算示例：Ａ12。"),
                make_document("unrelated", "photosynthesis needs sunlight"),
            ]
            index.replace_source("数学.txt", "hash", documents, ["matched", "unrelated"])
            index.mark_ready("manifest")
            results = index.search("面积 a12", k=50)
            self.assertEqual([d.metadata["chunk_id"] for d in results], ["matched"])
            self.assertEqual(index.search("absentkeyword", k=50), [])
            self.assertEqual(index.search("！？", k=50), [])

    def test_queries_tokenize_only_the_query_and_use_persisted_postings(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "keywords.sqlite3"
            index = BM25InvertedIndex(database, "test")
            documents = [make_document("one", "面积计算依赖底与高。"), make_document("two", "面积公式用于三角形。")]
            index.replace_source("数学.txt", "hash", documents, ["one", "two"])
            index.mark_ready("manifest")
            reopened = BM25InvertedIndex(database, "test")
            query = "面积"

            def tokenize_query_only(text):
                if text != query:
                    raise AssertionError("查询阶段重新分词了文档正文")
                return tokenize_for_bm25(text)

            with patch("rag.lexical_index.tokenize_for_bm25", side_effect=tokenize_query_only) as tokenizer:
                results = reopened.search(query, k=2)
            self.assertEqual(len(results), 2)
            tokenizer.assert_called_once_with(query)

    def test_invalid_replacements_roll_back_source_content_and_ready_marker(self):
        with TemporaryDirectory() as directory:
            index = BM25InvertedIndex(Path(directory) / "keywords.sqlite3", "test")
            original = make_document("old", "originalkeyword")
            index.replace_source("数学.txt", "old-hash", [original], ["old"])
            index.mark_ready("original-manifest")
            invalid_metadata = make_document("new", "newkeyword")
            invalid_metadata.metadata["bad"] = object()
            bad_batches = [
                ([make_document("new", "newkeyword")], []),
                ([make_document("new", "newkeyword"), make_document("new", "otherkeyword")], ["new", "new"]),
                ([invalid_metadata], ["new"]),
            ]
            for documents, identifiers in bad_batches:
                with self.subTest(identifiers=identifiers):
                    with self.assertRaises((TypeError, ValueError)):
                        index.replace_source("数学.txt", "new-hash", documents, identifiers)
                    self.assertTrue(index.is_ready("original-manifest"))
                    self.assertEqual(index.sources()["数学.txt"], {"sha256": "old-hash", "ids": ["old"]})
                    self.assertEqual(index.search("originalkeyword", k=1)[0].metadata["chunk_id"], "old")
                    self.assertEqual(index.search("newkeyword", k=10), [])

    def test_reading_missing_database_does_not_create_files_or_directories(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "missing-parent" / "keywords.sqlite3"
            index = BM25InvertedIndex(database, "test")
            self.assertEqual(index.sources(), {})
            self.assertFalse(index.is_ready("manifest"))
            self.assertFalse(index.status()["exists"])
            with self.assertRaisesRegex(EnvironmentError, "sync-index|sync-keywords"):
                index.search("triangle", k=3)
            self.assertFalse(database.exists())
            self.assertFalse(database.parent.exists())

    def test_id_collision_rolls_back_all_changes_after_source_replacement_starts(self):
        with TemporaryDirectory() as directory:
            index = BM25InvertedIndex(Path(directory) / "keywords.sqlite3", "test")
            index.replace_source("数学.txt", "math", [make_document("old", "originalkeyword")], ["old"])
            index.replace_source("科学.txt", "science", [make_document("occupied", "sciencekeyword", "科学", "科学.txt")], ["occupied"])
            index.mark_ready("original-manifest")
            prior_sources = index.sources()
            # 第二个分片才发生约束冲突，必须回滚来源删除和第一个分片写入。
            replacements = [make_document("new", "newkeyword"), make_document("occupied", "newkeyword")]
            with self.assertRaises((ValueError, RuntimeError, sqlite3.IntegrityError)):
                index.replace_source("数学.txt", "replacement", replacements, ["new", "occupied"])
            self.assertEqual(index.sources(), prior_sources)
            self.assertTrue(index.is_ready("original-manifest"))
            self.assertEqual(index.search("originalkeyword", k=5)[0].metadata["chunk_id"], "old")
            self.assertEqual(index.search("newkeyword", k=5), [])


if __name__ == "__main__":
    unittest.main()
