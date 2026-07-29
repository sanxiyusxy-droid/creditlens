"""应用层 Port 定义（domain/application 不依赖具体基础设施）。"""

from typing import Protocol


class ObjectStorePort(Protocol):
    """对象存储端口：MinIO 或本地文件系统。

    路径由服务端构造，拒绝用户提供完整对象键（文档 §6.7）。
    """

    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> str:
        """写入对象，返回 object_uri（s3://bucket/key 或 file://...）。"""
        ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...


class EmbeddingProvider(Protocol):
    """Embedding 端口（文档 §3.2）：记录 provider/model/revision。"""

    @property
    def version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
