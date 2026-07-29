"""标识符生成。

业务主键使用随机 UUID；能够从内容确定的索引点使用 UUIDv5 派生，便于幂等（文档 §6.2/§6.6）。
"""

import uuid

# 项目固定命名空间：Qdrant Point ID 派生
QDRANT_POINT_NAMESPACE = uuid.UUID("6c2f1df2-3d55-4c85-9d0b-3a1a3d1c9f01")


def new_id() -> uuid.UUID:
    return uuid.uuid4()


def deterministic_point_id(
    section_id: uuid.UUID | str,
    text_hash: str,
    embedding_version: str,
) -> uuid.UUID:
    """Point ID = UUIDv5(namespace, section_id + text_hash + embedding_version)。

    Outbox Worker 至少一次消费时，Upsert 幂等的关键。
    """
    return uuid.uuid5(
        QDRANT_POINT_NAMESPACE,
        f"{section_id}:{text_hash}:{embedding_version}",
    )
