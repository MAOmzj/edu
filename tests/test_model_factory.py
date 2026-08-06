"""DeepSeek 模型配置和本地 Embedding 的离线测试。"""

import math
import os
import unittest
from unittest.mock import patch

from model.factory import (
    create_embedding_model,
    require_dashscope_api_key,
    require_deepseek_api_key,
)
from utils.config_handler import model_conf


class ModelFactoryTests(unittest.TestCase):
    """确认 DeepSeek 聊天和两种 Embedding 配置都能正确创建。"""

    def test_local_embeddings_are_deterministic_and_normalized(self):
        """同一段文字每次生成相同单位向量，过程中不调用在线 API。"""

        with patch.dict(
            model_conf,
            {
                "embedding_model_name": "local-hashing-zh-v1",
                "local_embedding_dimension": 1024,
            },
        ):
            embeddings = create_embedding_model()
            first = embeddings.embed_query("三角形有三条边")
            second = embeddings.embed_query("三角形有三条边")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1024)
        norm = math.sqrt(sum(value * value for value in first))
        self.assertAlmostEqual(norm, 1.0)

    def test_documents_and_query_use_the_same_embedding_algorithm(self):
        """知识文本与查询必须使用同一算法，余弦相似度才有意义。"""

        with patch.dict(
            model_conf,
            {"embedding_model_name": "local-hashing-zh-v1"},
        ):
            embeddings = create_embedding_model()
            document_vector = embeddings.embed_documents(["植物生长需要阳光"])[0]
            query_vector = embeddings.embed_query("植物生长需要阳光")
        self.assertEqual(document_vector, query_vector)

    def test_text_embedding_v4_client_uses_dashscope_key(self):
        """在线向量模型只读取百炼 Key，创建客户端时不发起网络请求。"""

        with (
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}),
            patch.dict(
                model_conf,
                {"embedding_model_name": "text-embedding-v4"},
            ),
        ):
            embeddings = create_embedding_model()

        self.assertEqual(embeddings.model, "text-embedding-v4")

    def test_missing_dashscope_key_has_clear_embedding_message(self):
        """选择在线向量模型但没有 Key 时，明确指出缺少哪项配置。"""

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
            with self.assertRaisesRegex(EnvironmentError, "DASHSCOPE_API_KEY"):
                require_dashscope_api_key()

    def test_missing_deepseek_key_has_clear_message(self):
        """没有官方 DeepSeek Key 时直接提示配置方法，而不是模糊网络错误。"""

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(EnvironmentError, "DEEPSEEK_API_KEY"):
                require_deepseek_api_key()


if __name__ == "__main__":
    unittest.main()
