# 文件用途：离线验证加权 RRF 双路召回以及知识清单与关键词索引的同步。
# 调用关系：unittest 调用本文件；本文件调用 hybrid_search 和 EducationVectorStore。
# 修改易踩坑：隔离向量库、清单和关键词路径；测试不能连接在线 Embedding 或 Rerank。
"""关键词独立召回和 RRF 融合的离线回归测试。"""

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.hybrid_search import reciprocal_rank_fusion
from rag.vector_store import EducationVectorStore


def candidate(identifier, text=None, subject="数学"):
    return Document(
        page_content=text or identifier,
        metadata={"chunk_id": identifier, "subject": subject},
    )


class CountingEmbeddings(Embeddings):
    """固定小向量使 SQLite 写入离线，同时记录是否重复调用 Embedding。"""

    def __init__(self):
        self.document_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def temporary_store(root, embeddings):
    store = EducationVectorStore(embedding_model=embeddings, backend_name="sqlite")
    store.data_path = root / "knowledge"
    store.data_path.mkdir(exist_ok=True)
    store.sqlite_database_path = root / "vectors.sqlite3"
    store.persist_directory = store.sqlite_database_path
    store.manifest_path = root / "manifest.json"
    store.sync_on_search = False
    store.rerank_enabled = False
    store.vector_candidate_k = 1
    store.bm25_candidate_k = 10
    store.vector_weight = 0.75
    store.bm25_weight = 0.25
    store.rrf_rank_constant = 60
    return store


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_vector_weight_dominates_equal_opposing_rankings(self):
        first, second = candidate("first"), candidate("second")
        results = reciprocal_rank_fusion([[first, second], [second, first]], weights=[0.75, 0.25], k=2)
        self.assertEqual([d.metadata["chunk_id"] for d in results], ["first", "second"])
        reversed_weights = reciprocal_rank_fusion([[first, second], [second, first]], weights=[0.25, 0.75], k=2)
        self.assertEqual(reversed_weights[0].metadata["chunk_id"], "second")

    def test_support_from_both_routes_can_outrank_a_single_route_first_place(self):
        dense_only, lexical_only, shared = candidate("dense"), candidate("lexical"), candidate("shared")
        results = reciprocal_rank_fusion([[dense_only, shared], [lexical_only, shared]], weights=[0.75, 0.25], k=3)
        self.assertEqual([d.metadata["chunk_id"] for d in results], ["shared", "dense", "lexical"])

    def test_union_preserves_lexical_only_documents_and_deduplicates_chunk_ids(self):
        dense = candidate("dense")
        shared_dense = candidate("shared", "dense copy")
        shared_lexical = candidate("shared", "lexical copy")
        lexical = candidate("lexical")
        results = reciprocal_rank_fusion([[dense, shared_dense], [shared_lexical, lexical]], weights=[0.75, 0.25], k=20)
        self.assertEqual({d.metadata["chunk_id"] for d in results}, {"dense", "shared", "lexical"})
        self.assertEqual(len(results), 3)

    def test_zero_weight_lexical_route_excludes_lexical_only_documents(self):
        dense = candidate("dense")
        shared = candidate("shared")
        lexical_only = candidate("lexical-only")
        results = reciprocal_rank_fusion(
            [[dense, shared], [lexical_only, shared]],
            weights=[1.0, 0.0],
            k=20,
        )
        self.assertEqual(
            [document.metadata["chunk_id"] for document in results],
            ["dense", "shared"],
        )

    def test_repeated_document_inside_one_route_does_not_cast_extra_votes(self):
        first, supported = candidate("first"), candidate("supported")
        results = reciprocal_rank_fusion([[first, first, first, supported], [supported]], weights=[0.5, 0.5], k=2)
        self.assertEqual([d.metadata["chunk_id"] for d in results], ["supported", "first"])

    def test_weight_validation_rejects_nonfinite_negative_zero_and_mismatched_values(self):
        rankings = [[candidate("one")], [candidate("two")]]
        invalid = [[0.75], [-1.0, 1.0], [0.0, 0.0], [math.nan, 0.25], [0.75, math.inf], [0.75, -math.inf]]
        for weights in invalid:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    reciprocal_rank_fusion(rankings, weights=weights)

    def test_rank_constant_validation_and_deterministic_ties(self):
        for constant in (-1, math.nan, math.inf):
            with self.subTest(rank_constant=constant):
                with self.assertRaises(ValueError):
                    reciprocal_rank_fusion([[candidate("one")]], weights=[1.0], rank_constant=constant)
        rankings = [[candidate("z")], [candidate("a")]]
        first = reciprocal_rank_fusion(rankings, weights=[0.5, 0.5], k=2)
        second = reciprocal_rank_fusion(rankings, weights=[0.5, 0.5], k=2)
        self.assertEqual([d.metadata["chunk_id"] for d in first], [d.metadata["chunk_id"] for d in second])


class IndependentRetrievalTests(unittest.TestCase):
    def test_keyword_backfill_rejects_changed_boundaries_even_when_chunk_ids_match(self):
        with TemporaryDirectory() as directory:
            embeddings = CountingEmbeddings()
            store = temporary_store(Path(directory), embeddings)
            source = store.data_path / "数学切分.txt"
            source.write_text("abcdefghijklmnopqrstuvwx", encoding="utf-8")
            store.splitter = RecursiveCharacterTextSplitter(
                chunk_size=12, chunk_overlap=0, separators=[""],
            )
            store.sync()
            manifest = store._load_manifest()
            item = manifest["files"]["data/knowledge/数学切分.txt"]
            original_chunks, original_ids = store._split_file(source, item["sha256"])
            original_calls = list(embeddings.document_calls)
            store.splitter = RecursiveCharacterTextSplitter(
                chunk_size=13, chunk_overlap=0, separators=[""],
            )
            changed_chunks, changed_ids = store._split_file(source, item["sha256"])
            self.assertEqual(original_ids, changed_ids)
            self.assertEqual(len(original_ids), 2)
            self.assertNotEqual(
                [document.page_content for document in original_chunks],
                [document.page_content for document in changed_chunks],
            )
            with patch.object(store, "_get_store", side_effect=AssertionError("切分不匹配不应加载向量后端")):
                with self.assertRaisesRegex(EnvironmentError, "sync-index|rebuild-index"):
                    store.sync_keyword_index()
            self.assertEqual(embeddings.document_calls, original_calls)
            self.assertFalse(
                store._get_lexical_index().is_ready(store._manifest_digest(manifest)),
            )

    def test_sqlite_sync_reconciles_manifest_loss_and_orphan_chunks(self):
        for failure in ("manifest-loss", "orphan-chunk"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                store = temporary_store(Path(directory), CountingEmbeddings())
                source = store.data_path / "数学旧资料.txt"
                source.write_text("stalekeyword retired concept", encoding="utf-8")
                store.sync()
                backend = store._get_store()
                if failure == "manifest-loss":
                    source.unlink()
                    source = store.data_path / "数学当前.txt"
                    store.manifest_path.unlink()
                else:
                    orphan = candidate("orphan-id", "stalekeyword orphan concept")
                    backend.add_documents([orphan], ids=["orphan-id"])
                    store._get_lexical_index().replace_source(
                        "data/knowledge/数学遗留.txt", "orphan-hash",
                        [orphan], ["orphan-id"],
                    )
                    self.assertEqual(backend.count_documents(), 2)
                source.write_text("currentkeyword active concept", encoding="utf-8")
                store.sync()
                self.assertEqual(backend.count_documents(), 1)
                self.assertEqual(
                    [document.page_content for document in backend.dense_search("anything", k=10)],
                    ["currentkeyword active concept"],
                )
                lexical = store._get_lexical_index()
                self.assertEqual(lexical.search("stalekeyword", k=10), [])
                self.assertEqual(
                    [document.page_content for document in lexical.search("currentkeyword", k=10)],
                    ["currentkeyword active concept"],
                )
                manifest = store._load_manifest()
                self.assertEqual(sum(len(item["ids"]) for item in manifest["files"].values()), 1)
                self.assertTrue(lexical.is_ready(store._manifest_digest(manifest)))

    def test_default_candidate_pool_keeps_exact_keyword_only_match_for_reranking(self):
        with TemporaryDirectory() as directory:
            embeddings = CountingEmbeddings()
            configured = EducationVectorStore(
                embedding_model=embeddings,
                backend_name="sqlite",
            )
            store = temporary_store(Path(directory), embeddings)
            for name in (
                "vector_candidate_k",
                "bm25_candidate_k",
                "rrf_rank_constant",
                "vector_weight",
                "bm25_weight",
                "rerank_candidate_k",
            ):
                setattr(store, name, getattr(configured, name))
            self.assertEqual(
                (
                    store.vector_candidate_k,
                    store.bm25_candidate_k,
                    store.rrf_rank_constant,
                    store.vector_weight,
                    store.bm25_weight,
                    store.rerank_candidate_k,
                ),
                (80, 80, 5, 0.75, 0.25, 20),
            )
            keyword_text = "lexicalneedle exact textbook match"
            (store.data_path / "数学公式.txt").write_text(keyword_text, encoding="utf-8")
            store.sync()
            store.rerank_enabled = True
            dense_documents = [
                candidate(f"dense-{index:03d}", f"semantic candidate {index}")
                for index in range(80)
            ]
            reranker = MagicMock()
            reranker.rerank.side_effect = lambda query, documents, top_n: documents[:top_n]
            with (
                patch.object(
                    store._get_store(),
                    "dense_search",
                    return_value=dense_documents,
                ) as dense_search,
                patch.object(store, "_get_reranker", return_value=reranker),
            ):
                results = store.search("lexicalneedle", subject="数学", k=5)
            self.assertEqual(len(results), 5)
            self.assertEqual(dense_search.call_args.kwargs["k"], 80)
            reranker.rerank.assert_called_once()
            fused_candidates = reranker.rerank.call_args.args[1]
            self.assertEqual(len(fused_candidates), 20)
            self.assertIn(
                keyword_text,
                [document.page_content for document in fused_candidates],
            )

    def test_missing_keyword_index_fails_read_only_search_with_sync_instructions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = CountingEmbeddings()
            initial = temporary_store(root, embeddings)
            (initial.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            initial.sync()
            initial.manifest_path.with_suffix(".bm25.sqlite3").unlink()
            store = temporary_store(root, embeddings)
            fake_backend = MagicMock()
            store._vector_store = fake_backend
            with patch.object(store, "sync") as synchronize:
                with self.assertRaisesRegex(EnvironmentError, "sync-keywords|sync-index"):
                    store.search("hypotenuse", subject="数学", k=2)
            synchronize.assert_not_called()
            fake_backend.dense_search.assert_not_called()
            self.assertTrue(store.manifest_path.exists())
            self.assertFalse(store.manifest_path.with_suffix(".bm25.sqlite3").exists())

    def test_stale_keyword_ready_digest_blocks_read_only_search_before_dense_query(self):
        with TemporaryDirectory() as directory:
            store = temporary_store(Path(directory), CountingEmbeddings())
            (store.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            store.sync()
            store._get_lexical_index().mark_ready("different-manifest")
            with patch.object(store._get_store(), "dense_search") as dense_search:
                with self.assertRaisesRegex(EnvironmentError, "sync-keywords|sync-index"):
                    store.search("hypotenuse", subject="数学", k=2)
            dense_search.assert_not_called()

    def test_keyword_backfill_from_existing_manifest_avoids_embeddings_and_backend_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = CountingEmbeddings()
            store = temporary_store(root, embeddings)
            (store.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            store.sync()
            original_calls = list(embeddings.document_calls)
            keyword_path = store.manifest_path.with_suffix(".bm25.sqlite3")
            keyword_path.unlink()
            reopened = temporary_store(root, embeddings)
            with patch.object(reopened, "_get_store", side_effect=AssertionError("关键词回填不应访问向量后端")):
                report = reopened.sync_keyword_index()
            self.assertEqual(embeddings.document_calls, original_calls)
            self.assertEqual(report.indexed_chunks, 0)
            self.assertGreater(report.lexical_indexed_chunks, 0)
            lexical_results = reopened._get_lexical_index().search("hypotenuse", k=3, subject="数学")
            self.assertEqual(len(lexical_results), 1)
            self.assertIn("theorem", lexical_results[0].page_content)

    def test_unchanged_sync_backfills_keyword_index_without_new_embeddings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = CountingEmbeddings()
            store = temporary_store(root, embeddings)
            (store.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            store.sync()
            original_calls = list(embeddings.document_calls)
            store.manifest_path.with_suffix(".bm25.sqlite3").unlink()
            reopened = temporary_store(root, embeddings)
            report = reopened.sync()
            self.assertEqual(report.unchanged_files, 1)
            self.assertEqual(embeddings.document_calls, original_calls)
            self.assertEqual(len(reopened._get_lexical_index().search("hypotenuse", k=3)), 1)

    def test_independent_lexical_route_recalls_document_missing_from_dense_candidates_with_same_subject_filter(self):
        with TemporaryDirectory() as directory:
            store = temporary_store(Path(directory), CountingEmbeddings())
            (store.data_path / "数学基础.txt").write_text("addition basics", encoding="utf-8")
            (store.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            (store.data_path / "科学资料.txt").write_text("hypotenuse science example", encoding="utf-8")
            store.sync()
            manifest = store._load_manifest()
            dense_identifier = manifest["files"]["data/knowledge/数学基础.txt"]["ids"][0]
            dense_document = candidate(dense_identifier, "addition basics")
            backend = store._get_store()
            lexical = store._get_lexical_index()
            with patch.object(store, "_get_lexical_index", return_value=lexical), patch.object(backend, "dense_search", return_value=[dense_document]) as dense_search, patch.object(backend, "similarity_search", side_effect=AssertionError("不能再次融合候选池分数")), patch.object(lexical, "search", wraps=lexical.search) as lexical_search:
                results = store.search("hypotenuse", subject="数学", k=2)
            self.assertEqual({d.page_content for d in results}, {"addition basics", "hypotenuse theorem"})
            self.assertTrue(all(d.metadata["subject"] == "数学" for d in results))
            self.assertEqual(dense_search.call_args.kwargs["filter"], {"subject": "数学"})
            self.assertEqual(lexical_search.call_args.kwargs["subject"], "数学")

    def test_sync_updates_and_removes_keyword_documents_with_vector_manifest(self):
        with TemporaryDirectory() as directory:
            store = temporary_store(Path(directory), CountingEmbeddings())
            source = store.data_path / "数学公式.txt"
            source.write_text("oldkeyword theorem", encoding="utf-8")
            store.sync()
            source.write_text("newkeyword equation", encoding="utf-8")
            updated = store.sync()
            lexical = store._get_lexical_index()
            self.assertEqual(updated.updated_files, 1)
            self.assertEqual(lexical.search("oldkeyword", k=10), [])
            self.assertEqual(lexical.search("newkeyword", k=10)[0].page_content, "newkeyword equation")
            source.unlink()
            removed = store.sync()
            self.assertEqual(removed.removed_files, 1)
            self.assertEqual(lexical.sources(), {})
            self.assertEqual(lexical.search("newkeyword", k=10), [])

    def test_rerank_failure_preserves_the_rrf_order(self):
        with TemporaryDirectory() as directory:
            store = temporary_store(Path(directory), CountingEmbeddings())
            (store.data_path / "数学基础.txt").write_text("addition basics", encoding="utf-8")
            (store.data_path / "数学公式.txt").write_text("hypotenuse theorem", encoding="utf-8")
            store.sync()
            manifest = store._load_manifest()
            identifier = manifest["files"]["data/knowledge/数学基础.txt"]["ids"][0]
            dense_document = candidate(identifier, "addition basics")
            store.rerank_enabled = True
            store.rerank_candidate_k = 2
            broken_reranker = MagicMock()
            broken_reranker.rerank.side_effect = RuntimeError("模拟 Rerank 服务中断")
            with patch.object(store._get_store(), "dense_search", return_value=[dense_document]), patch.object(store, "_get_reranker", return_value=broken_reranker):
                results = store.search("hypotenuse", subject="数学", k=1)
            self.assertEqual([d.metadata["chunk_id"] for d in results], [identifier])
            broken_reranker.rerank.assert_called_once()
            candidates = broken_reranker.rerank.call_args.args[1]
            self.assertEqual([d.page_content for d in candidates], ["addition basics", "hypotenuse theorem"])


if __name__ == "__main__":
    unittest.main()
