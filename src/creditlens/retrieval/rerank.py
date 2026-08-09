"""Rerank Provider（任务 16，文档 §8.11）。

- Reranker 只判断相关性，不判断权限；时点/ACL/质量必须在此之前硬过滤；
- 无 GPU/API 时使用 LexicalOverlapReranker 兜底（确定性、可测试），
  评测报告必须标注 reranker_version；
- CrossEncoderReranker 对接 OpenAI 兼容或 TEI /rerank 服务
  （Qwen3-Reranker/bge-reranker-v2-m3）。
"""

from typing import Protocol

from creditlens.retrieval.sparse import tokenize


class RerankProvider(Protocol):
    @property
    def version(self) -> str: ...

    async def score(self, query: str, documents: list[str]) -> list[float]: ...


class LexicalOverlapReranker:
    """确定性词面重叠打分：|Q ∩ D| / |Q|，同分保持原序（稳定排序）。"""

    version = "lexical-overlap-v1"

    async def score(self, query: str, documents: list[str]) -> list[float]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return [0.0] * len(documents)
        scores: list[float] = []
        for doc in documents:
            doc_terms = set(tokenize(doc))
            scores.append(len(query_terms & doc_terms) / len(query_terms))
        return scores


class HttpCrossEncoderReranker:
    """对接 TEI 风格 /rerank 接口的 Cross-Encoder（部署 bge-reranker-v2-m3 等）。"""

    def __init__(self, base_url: str, model: str, version: str, api_key: str = ""):
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60)
        self._model = model
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def score(self, query: str, documents: list[str]) -> list[float]:
        response = await self._client.post(
            "/rerank", json={"model": self._model, "query": query, "documents": documents}
        )
        response.raise_for_status()
        payload = response.json()
        scores = [0.0] * len(documents)
        for item in payload.get("results", payload.get("data", [])):
            scores[item["index"]] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    async def aclose(self) -> None:
        """关闭底层 HTTP 连接池。"""
        await self._client.aclose()


def build_reranker(settings) -> RerankProvider | None:
    if settings.rerank_provider == "disabled":
        return None
    if settings.rerank_provider == "lexical_fallback":
        return LexicalOverlapReranker()
    if settings.rerank_provider == "http":
        api_key = settings.rerank_api_key or settings.embedding_api_key  # 硅基流动共用 Key
        if not (settings.rerank_api_base and settings.rerank_model and api_key):
            raise ValueError("http rerank 需要 RERANK_API_BASE/MODEL 与可用 API Key")
        return HttpCrossEncoderReranker(
            base_url=settings.rerank_api_base,
            model=settings.rerank_model,
            version=f"{settings.rerank_model}@api",
            api_key=api_key,
        )
    raise NotImplementedError(f"rerank provider {settings.rerank_provider} 未配置")
