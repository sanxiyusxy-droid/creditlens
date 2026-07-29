"""Embedding Provider 实现。

- HashEmbedding：确定性词袋哈希向量，无 GPU/API 时的离线兜底，保证管线、
  评测与幂等契约可测试；语义质量有限，评测报告必须注明 embedding_version。
- OpenAICompatEmbedding：对接 OpenAI 兼容 /v1/embeddings（后续启用）。
"""

import hashlib
import math

import jieba


class HashEmbedding:
    """确定性哈希 Embedding：jieba 分词 -> 特征哈希 -> TF 权重 -> L2 归一化。"""

    def __init__(self, dim: int = 256, version: str = "hash-embed-v1"):
        self._dim = dim
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = [t for t in jieba.cut(text) if t.strip()]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAICompatEmbedding:
    """OpenAI 兼容 Embedding API（硅基流动 bge-m3 / Qwen3-Embedding 等）。"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, version: str,
                 batch_size: int = 16):
        import httpx

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        self._model = model
        self._dim = dim
        self._version = version
        self._batch_size = batch_size

    @property
    def version(self) -> str:
        return self._version

    @property
    def dim(self) -> int:
        return self._dim

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            "/embeddings", json={"model": self._model, "input": texts}
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


# 常见模型维度（未知模型在 build 时经一次探测调用确定）
KNOWN_EMBEDDING_DIMS = {
    "BAAI/bge-m3": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


def _probe_dim(base_url: str, api_key: str, model: str) -> int:
    """同步探测一次向量维度（仅未知模型；启动时执行）。"""
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": ["dim probe"]},
        timeout=30,
    )
    response.raise_for_status()
    return len(response.json()["data"][0]["embedding"])


def build_embedding_provider(settings) -> HashEmbedding | OpenAICompatEmbedding:
    from creditlens.common.config import Settings

    assert isinstance(settings, Settings)
    if settings.embedding_provider == "hash_fallback":
        return HashEmbedding(dim=settings.embedding_dim, version=settings.embedding_version)
    if settings.embedding_provider == "openai_compatible":
        if not (settings.embedding_api_base and settings.embedding_api_key and settings.embedding_model):
            raise ValueError("openai_compatible embedding 需要 EMBEDDING_API_BASE/KEY/MODEL")
        dim = KNOWN_EMBEDDING_DIMS.get(settings.embedding_model) or _probe_dim(
            settings.embedding_api_base, settings.embedding_api_key, settings.embedding_model
        )
        return OpenAICompatEmbedding(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=dim,
            version=settings.effective_embedding_version,
        )
    raise NotImplementedError(f"embedding provider {settings.embedding_provider} 未配置")
