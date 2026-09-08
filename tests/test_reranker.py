# 文件用途：离线验证阿里云百炼 Rerank 客户端的响应解析和短候选直通逻辑。
# 调用关系：unittest 调用本文件；本文件不发起真实网络请求。
# 修改易踩坑：这里只测试解析与排序，不要依赖真实 DASHSCOPE_API_KEY。
"""阿里云 Rerank 客户端离线测试。"""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from rag.reranker import DashScopeReranker


def make_documents(texts):
    """把文本列表包装成 Document 列表。"""
    return [Document(page_content=text) for text in texts]


class DashScopeRerankerTests(unittest.TestCase):
    """验证 Rerank 响应解析、稳定排序和无需请求的短候选场景。"""

    def test_rerank_sorts_documents_by_relevance_score(self):
        """高分文档应排在前面，即使它的原召回顺序靠后。"""
        documents = make_documents(["低分文档", "高分文档", "中分文档"])
        fake_response = {
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            }
        }

        with patch.object(
            DashScopeReranker, "_post", return_value=fake_response
        ) as post:
            reranker = DashScopeReranker(
                model="gte-rerank",
                api_key="test-key",
                base_url="https://example.invalid/rerank",
            )
            ranked = reranker.rerank("问题", documents, top_n=2)

        self.assertEqual(post.call_count, 1)
        self.assertEqual(
            [document.page_content for document in ranked],
            ["高分文档", "中分文档"],
        )

    def test_rerank_does_not_call_api_when_candidates_fit_top_n(self):
        """候选数量不超过 top_n 时直接原序返回，不产生网络请求。"""
        documents = make_documents(["片段一", "片段二"])

        with patch.object(DashScopeReranker, "_post") as post:
            reranker = DashScopeReranker(api_key="test-key")
            ranked = reranker.rerank("问题", documents, top_n=5)

        self.assertEqual([document.page_content for document in ranked], ["片段一", "片段二"])
        post.assert_not_called()

    def test_missing_api_key_raises_without_network_request(self):
        """候选超过 top_n 且缺少 API Key 时应快速失败，由上层决定是否回退。"""
        documents = make_documents(["一", "二", "三"])

        with patch.object(DashScopeReranker, "_post") as post:
            reranker = DashScopeReranker(api_key="", base_url="https://example.invalid")
            with self.assertRaises(EnvironmentError):
                reranker.rerank("问题", documents, top_n=1)

        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
