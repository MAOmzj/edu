# 文件用途：提供无需网络的确定性中文特征哈希 Embedding，作为在线向量模型的备用方案。
# 调用关系：model/factory.py 创建本对象，向量后端调用 embed_documents/embed_query。
# 修改易踩坑：文档和查询算法必须完全一致；改变维度或算法后必须重建全部知识索引。
"""无需账号和网络下载的本地中文特征哈希 Embedding。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings

from rag.hybrid_search import tokenize_for_bm25


class LocalHashEmbeddings(Embeddings):
    """把中英文词项稳定映射为本地向量，供余弦相似度检索使用。

    这个实现不是大模型语义 Embedding，不会理解所有同义词；它的优点是完全本地、
    不需要百炼账号、不下载模型，并且和现有 BM25 混合后适合当前小型教育知识库。
    """

    def __init__(self, dimension: int = 1024):
        """保存向量维度；维度过小会产生大量哈希碰撞，因此至少使用 128 维。"""

        if dimension < 128:
            raise ValueError("本地 Embedding 维度不能小于 128")
        self.dimension = int(dimension)

    def _embed(self, text: str) -> list[float]:
        """把文本词项哈希到固定维度，并返回已经归一化的浮点向量。"""

        tokens = tokenize_for_bm25(text)
        vector = [0.0] * self.dimension
        if not tokens:
            return vector

        # 同一个词出现越多权重越高，但使用对数抑制重复句子带来的过大影响。
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(float(count)))

        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """在本机为每段知识文本生成向量，不发送任何数据到第三方。"""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """使用和知识文本相同的算法生成查询向量。"""

        return self._embed(text)
