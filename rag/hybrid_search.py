# 文件用途：实现中文 BM25、余弦相似度归一化与两者的加权融合排序。
# 调用关系：vector_backends.py 调用本文件；本文件只做纯计算，不读文件或数据库。
# 修改易踩坑：权重、分词或归一化变化会改变全部排序结果，必须同时更新配置与离线测试。
"""BM25 与向量余弦相似度的轻量混合评分工具。

两种分数解决不同问题：

- 余弦相似度比较 Embedding，擅长找到“意思相近但说法不同”的内容；
- BM25 比较关键词，擅长精确匹配术语、数字和公式。

最后将两种分数归一化并加权，避免只靠一种方法漏掉正确资料。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document


# 连续英文/数字作为一个词；连续汉字随后会拆成单字和相邻双字。
TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[._'-][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


@dataclass(frozen=True)
class HybridScore:
    """保存一个文档的原始余弦、BM25 和最终融合分数。"""

    cosine: float
    bm25: float
    combined: float


def calculate_cosine_similarity(
    left_vector: Sequence[float],
    right_vector: Sequence[float],
) -> float:
    """计算两个同维向量的余弦相似度，任一零向量统一返回 0。"""
    if len(left_vector) != len(right_vector):
        raise ValueError("计算余弦相似度的两个向量维度必须相同")
    left_norm = math.sqrt(math.fsum(value * value for value in left_vector))
    right_norm = math.sqrt(math.fsum(value * value for value in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = math.fsum(
        left * right
        for left, right in zip(left_vector, right_vector, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def tokenize_for_bm25(text: str) -> list[str]:
    """把中英文文本转成 BM25 词项；中文同时保留单字和双字。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for matched in TOKEN_PATTERN.findall(normalized):
        if CJK_PATTERN.fullmatch(matched):
            characters = list(matched)
            tokens.extend(characters)
            tokens.extend(
                "".join(characters[index : index + 2])
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(matched)
    return tokens


def calculate_bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """按照经典 BM25 公式计算查询对每篇候选文档的关键词相关度。"""
    if k1 <= 0:
        raise ValueError("BM25 的 k1 必须大于 0")
    if not 0 <= b <= 1:
        raise ValueError("BM25 的 b 必须在 0 到 1 之间")
    if not documents:
        return []

    query_terms = set(tokenize_for_bm25(query))
    tokenized_documents = [tokenize_for_bm25(text) for text in documents]
    if not query_terms:
        return [0.0] * len(documents)

    document_count = len(tokenized_documents)
    average_length = (
        sum(len(tokens) for tokens in tokenized_documents) / document_count
    )
    # 空文档集合的平均长度会是 0；用 1 避免后续长度归一化除零。
    safe_average_length = average_length or 1.0
    # document_frequencies 统计“某个查询词出现在多少篇文档”，不是出现总次数。
    # 越少见的词区分度越高，后面的 IDF 会给它更大权重。
    document_frequencies = Counter(
        term
        for tokens in tokenized_documents
        for term in query_terms.intersection(tokens)
    )

    scores: list[float] = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        length_ratio = len(tokens) / safe_average_length
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            matching_documents = document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0
                + (document_count - matching_documents + 0.5)
                / (matching_documents + 0.5)
            )
            denominator = term_frequency + k1 * (1.0 - b + b * length_ratio)
            score += inverse_document_frequency * (
                term_frequency * (k1 + 1.0) / denominator
            )
        scores.append(score)
    return scores


def combine_hybrid_scores(
    cosine_scores: Sequence[float],
    bm25_scores: Sequence[float],
    *,
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> list[HybridScore]:
    """把余弦分数映射到 0—1、BM25 按最大值归一化，再加权融合。"""
    if len(cosine_scores) != len(bm25_scores):
        raise ValueError("余弦分数和 BM25 分数数量必须相同")
    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("混合检索权重不能是负数")
    total_weight = vector_weight + bm25_weight
    if total_weight == 0:
        raise ValueError("混合检索至少需要一个大于 0 的权重")

    normalized_vector_weight = vector_weight / total_weight
    normalized_bm25_weight = bm25_weight / total_weight
    maximum_bm25 = max(bm25_scores, default=0.0)

    results: list[HybridScore] = []
    for cosine, bm25 in zip(cosine_scores, bm25_scores, strict=True):
        # 余弦理论范围为 -1 到 1，映射后不会因为候选集合变化而改变尺度。
        bounded_cosine = max(-1.0, min(1.0, float(cosine)))
        normalized_cosine = (bounded_cosine + 1.0) / 2.0
        normalized_bm25 = bm25 / maximum_bm25 if maximum_bm25 > 0 else 0.0
        combined = (
            normalized_vector_weight * normalized_cosine
            + normalized_bm25_weight * normalized_bm25
        )
        results.append(
            HybridScore(
                cosine=float(cosine),
                bm25=float(bm25),
                combined=combined,
            )
        )
    return results


def rank_documents_by_hybrid_score(
    query: str,
    identifiers: Sequence[str],
    documents: Sequence[Document],
    cosine_scores: Sequence[float],
    *,
    k: int,
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[Document]:
    """统一计算 BM25 与余弦混合分数，并稳定返回排名最高的文档。"""
    if not (len(identifiers) == len(documents) == len(cosine_scores)):
        raise ValueError("分片编号、文档和余弦分数数量必须相同")
    if k < 1 or not documents:
        return []
    bm25_scores = calculate_bm25_scores(
        query,
        [document.page_content for document in documents],
        k1=bm25_k1,
        b=bm25_b,
    )
    hybrid_scores = combine_hybrid_scores(
        cosine_scores,
        bm25_scores,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    ranked_documents = list(
        zip(hybrid_scores, identifiers, documents, strict=True)
    )
    # 先按融合分数排序；完全相同时再依次比较 BM25、余弦和稳定 ID，确保同样
    # 的输入每次都得到同样顺序，便于测试和排查问题。
    ranked_documents.sort(
        key=lambda item: (
            -item[0].combined,
            -item[0].bm25,
            -item[0].cosine,
            item[1],
        )
    )
    return [document for _, _, document in ranked_documents[:k]]
