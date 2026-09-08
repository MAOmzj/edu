# 文件用途：调用阿里云百炼（DashScope）Rerank 服务，对混合检索候选做 Cross-Encoder 式精排。
# 调用关系：rag/vector_store.py 的 EducationVectorStore.search() 调用本文件；
#           本文件只调用阿里云百炼 HTTP 接口，不访问项目数据库。
# 修改易踩坑：Rerank 只负责精排，不能替代召回；候选数量过大会增加接口成本与延迟。
"""阿里云百炼 Rerank 精排客户端。

这是两阶段检索的第二阶段：

    第一阶段：向量 + BM25 混合召回 top-N 候选；
    第二阶段：把 query 与候选文档一起发给阿里云 Rerank 模型，
              按返回的相关度分数重新排序，最后取 top-K。

Rerank 属于 API 型 Cross-Encoder：它会把问题和每个候选拼在一起做交互式
相关度打分，因此比单纯的向量余弦相似度更准确。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from langchain_core.documents import Document

DEFAULT_RERANK_MODEL = "gte-rerank"
DEFAULT_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


class DashScopeReranker:
    """调用阿里云百炼文本重排序接口，返回按相关度重新排序后的文档。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        base_url: str | None = None,
    ) -> None:
        """保存模型、API Key、超时和接口地址，但不在初始化时发起网络请求。"""
        self.model = (
            str(model or "").strip()
            or os.getenv("DASHSCOPE_RERANK_MODEL", "").strip()
            or DEFAULT_RERANK_MODEL
        )
        # None 表示调用方未指定，允许读取环境变量；显式空字符串表示禁用 Key，
        # 便于离线测试和受控降级，不应被本机环境中的真实 Key 悄悄覆盖。
        self.api_key = (
            os.getenv("DASHSCOPE_API_KEY", "").strip()
            if api_key is None
            else str(api_key).strip()
        )
        self.timeout = float(timeout)
        self.base_url = (
            str(base_url or "").strip()
            or os.getenv("DASHSCOPE_RERANK_BASE_URL", "").strip()
            or DEFAULT_RERANK_ENDPOINT
        ).rstrip("/")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        *,
        top_n: int,
    ) -> list[Document]:
        """对候选文档调用阿里云 Rerank，并按相关度分数降序返回 top_n 条。

        Args:
            query: 本轮用户问题。
            documents: 第一阶段混合检索得到的候选文档，顺序为召回顺序。
            top_n: 最终要返回给 Agent 的文档数量。

        Returns:
            重排序后的前 top_n 个文档。
        """
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            raise ValueError("Rerank 查询文本不能为空")
        if top_n < 1:
            raise ValueError("Rerank top_n 必须大于 0")
        if not documents:
            return []
        if len(documents) <= top_n:
            return list(documents)

        if not self.api_key:
            raise EnvironmentError(
                "启用阿里云 Rerank 需要 DASHSCOPE_API_KEY，请在 .env 中填写百炼 Key。"
            )

        payload = {
            "model": self.model,
            "input": {
                "query": cleaned_query,
                "documents": [document.page_content for document in documents],
            },
            "parameters": {
                "top_n": min(top_n, len(documents)),
                # 不要求返回原文，节省响应体积；返回的 index 已足够映射回原文档。
                "return_documents": False,
            },
        }
        response = self._post(payload)
        return self._parse_response(response, documents, top_n)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送百炼 Rerank HTTP 请求，并把响应 JSON 解析为字典。"""
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"阿里云 Rerank 接口返回 HTTP {exc.code}：{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"阿里云 Rerank 接口连接失败：{exc.reason}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"阿里云 Rerank 返回了非 JSON 响应：{body[:200]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"阿里云 Rerank 响应格式异常：{body[:200]}")
        return data

    @staticmethod
    def _parse_response(
        response: dict[str, Any],
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """从百炼响应中读取 index 与 relevance_score，并稳定排序。"""
        output = response.get("output")
        if not isinstance(output, dict):
            raise RuntimeError(f"阿里云 Rerank 响应缺少 output 字段：{response}")
        results = output.get("results")
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"阿里云 Rerank 响应缺少有效 results：{response}")

        scored: list[tuple[float, int, Document]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(documents):
                continue
            try:
                score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            scored.append((score, index, documents[index]))

        if not scored:
            raise RuntimeError(f"阿里云 Rerank 未返回可用文档索引：{response}")

        # 分数降序；分数相同时保持原召回顺序，让结果可复现。
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, _, document in scored[:top_n]]


__all__ = ["DashScopeReranker", "DEFAULT_RERANK_ENDPOINT", "DEFAULT_RERANK_MODEL"]
