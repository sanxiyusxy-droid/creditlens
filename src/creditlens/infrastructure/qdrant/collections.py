"""Qdrant Collection/Alias 管理（任务 8，文档 §6.6）。

- 物理 Collection 带版本号（credit_chunks_v1），Alias 指向当前版本；
- Named Vectors：dense + sparse（BM25）；
- 高频过滤字段建立 Payload Index；
- Alias 只用于创建新 Snapshot 时解析当前索引版本，已启动 Run 绝不跟随 Alias。
"""

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from creditlens.common.config import Settings

CHUNKS_FAMILY = "CHUNKS"
SUMMARIES_FAMILY = "SUMMARIES"

# 高频过滤字段 Payload Index（文档 §6.6）
_KEYWORD_INDEXES = [
    "tenant_id",
    "point_type",
    "document_id",
    "document_version_id",
    "parse_run_id",
    "document_type",
    "product_codes",
    "entity_ids",
    "confidentiality",
    "acl_tags",
    "quality_status",
]
_BOOL_INDEXES = ["tombstoned"]
_DATETIME_INDEXES = ["source_available_at", "valid_from", "valid_to"]
# valid_from/valid_to 自 v0.6 起在向量层做范围过滤（D3 下推）；回表复核仍为第二道防线。


def build_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=settings.qdrant_url)


class CollectionManager:
    def __init__(self, client: QdrantClient, dense_dim: int):
        self._client = client
        self._dense_dim = dense_dim

    def ensure_collection(self, collection_name: str, alias_name: str | None = None) -> None:
        """创建物理 Collection（幂等），并可选把 Alias 指向它。

        Payload Index 每次都幂等确保（已有 Collection 也补建，如 D3 新增的
        valid_from/valid_to datetime index）。"""
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": qm.VectorParams(size=self._dense_dim, distance=qm.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
                },
            )
        self._create_payload_indexes(collection_name)
        if alias_name:
            self.switch_alias(alias_name, collection_name)

    def _create_payload_indexes(self, collection_name: str) -> None:
        for field in _KEYWORD_INDEXES:
            self._client.create_payload_index(
                collection_name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
            )
        for field in _BOOL_INDEXES:
            self._client.create_payload_index(
                collection_name, field_name=field, field_schema=qm.PayloadSchemaType.BOOL
            )
        for field in _DATETIME_INDEXES:
            self._client.create_payload_index(
                collection_name, field_name=field, field_schema=qm.PayloadSchemaType.DATETIME
            )

    def switch_alias(self, alias_name: str, collection_name: str) -> None:
        """原子切换 Alias -> 物理 Collection（蓝绿切换入口）。"""
        self._client.update_collection_aliases(
            change_aliases_operations=[
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(
                        collection_name=collection_name, alias_name=alias_name
                    )
                )
            ]
        )

    def resolve_alias(self, alias_name: str) -> str | None:
        """解析 Alias 当前指向的物理 Collection 名（创建 Snapshot 时使用）。"""
        aliases = self._client.get_aliases().aliases
        for alias in aliases:
            if alias.alias_name == alias_name:
                return alias.collection_name
        return None
