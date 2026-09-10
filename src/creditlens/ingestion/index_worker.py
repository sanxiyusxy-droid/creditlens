"""Index Worker（任务 9，文档 §6.8）。

消费 index_outbox：
1. 领取 PENDING 且 available_at 到期的任务（生产用 FOR UPDATE SKIP LOCKED；
   SQLite 测试环境退化为普通领取）；
2. 读取 Section，生成 Embedding（text_hash + embedding_version 相同可复用）；
3. 按确定性 Point ID Upsert Qdrant（至少一次消费也安全）；
4. 成功置 COMPLETED，失败置 FAILED 并记录 last_error、退避重试。
"""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import EmbeddingProvider
from creditlens.common.clock import utc_now
from creditlens.common.ids import deterministic_point_id
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    Document,
    DocumentSection,
    DocumentVersion,
    IndexOutbox,
)

MAX_ATTEMPTS = 5


@dataclass
class WorkerStats:
    processed: int = 0
    failed: int = 0


class IndexWorker:
    def __init__(self, qdrant: QdrantClient, embedder: EmbeddingProvider, sparse_encoder=None):
        self._qdrant = qdrant
        self._embedder = embedder
        if sparse_encoder is None:
            from creditlens.retrieval.sparse import Bm25SparseEncoder

            sparse_encoder = Bm25SparseEncoder()
        self._sparse = sparse_encoder

    async def process_batch(self, session: AsyncSession, batch_size: int = 32) -> WorkerStats:
        stats = WorkerStats()
        now = utc_now()
        dialect_name = session.bind.dialect.name if session.bind is not None else ""
        entries = (
            await session.scalars(self._claim_pending_entries_stmt(now, batch_size, dialect_name))
        ).all()
        for entry in entries:
            entry.status = "PROCESSING"
            entry.locked_at = now
            entry.attempts += 1
        await session.flush()

        for entry in entries:
            try:
                if entry.operation == "UPSERT" and entry.aggregate_type == "SECTION":
                    await self._upsert(session, entry)
                elif entry.operation == "UPSERT" and entry.aggregate_type == "SUMMARY":
                    await self._upsert_summary(session, entry)
                elif entry.operation == "TOMBSTONE":
                    self._tombstone(entry)
                entry.status = "COMPLETED"
                entry.completed_at = utc_now()
                stats.processed += 1
            except Exception as exc:
                entry.last_error = str(exc)[:2000]
                if entry.attempts >= MAX_ATTEMPTS:
                    entry.status = "FAILED"
                else:
                    entry.status = "PENDING"  # 退避重试
                    from datetime import timedelta

                    entry.available_at = utc_now() + timedelta(seconds=30 * entry.attempts)
                stats.failed += 1
        await session.flush()
        return stats

    @staticmethod
    def _claim_pending_entries_stmt(now, batch_size: int, dialect_name: str):
        """构造待领取任务查询。

        PostgreSQL 下用行锁加 ``SKIP LOCKED`` 让多个 Worker 不会阻塞在同一批
        Outbox 记录上；SQLite 不支持该语法，测试/本地模式保留普通查询。
        """
        statement = (
            select(IndexOutbox)
            .where(IndexOutbox.status == "PENDING", IndexOutbox.available_at <= now)
            .order_by(IndexOutbox.created_at)
            .limit(batch_size)
        )
        if dialect_name == "postgresql":
            return statement.with_for_update(skip_locked=True)
        return statement

    async def _upsert(self, session: AsyncSession, entry: IndexOutbox) -> None:
        section = await session.get(DocumentSection, entry.aggregate_id)
        if section is None:
            raise ValueError(f"section {entry.aggregate_id} 不存在")
        version = await session.get(DocumentVersion, section.document_version_id)
        document = await session.get(Document, version.document_id) if version else None
        if version is None or document is None:
            raise ValueError("document version 缺失")

        entity_ids, product_codes = await self._payload_scopes(session, version)

        # Embedding 输入拼接受控上下文（文档 §7.10）
        heading_path = " > ".join(section.heading_path or [])
        embed_input = (
            f"[文档类型] {document.document_type}\n[标题路径] {heading_path}\n[正文] {section.text}"
        )
        vector = (await self._embedder.embed_documents([embed_input]))[0]

        # Sparse/BM25（任务 14）：仅当 Outbox 声明了 sparse_encoder_version 时写入
        vectors: dict = {"dense": vector}
        if entry.sparse_encoder_version:
            if entry.sparse_encoder_version != self._sparse.version:
                raise ValueError(
                    f"sparse encoder 版本不匹配: {entry.sparse_encoder_version} != {self._sparse.version}"
                )
            indices, values = self._sparse.encode_document(section.text)
            if indices:
                vectors["sparse"] = qm.SparseVector(indices=indices, values=values)

        point_id = deterministic_point_id(section.id, section.text_hash, entry.embedding_version)
        payload = {
            "tenant_id": str(section.tenant_id),
            "point_type": "original_section",
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "parse_run_id": str(section.parse_run_id),
            "section_id": str(section.id),
            "parent_section_id": str(section.parent_section_id)
            if section.parent_section_id
            else None,
            "entity_ids": entity_ids,
            "document_type": document.document_type,
            "product_codes": product_codes,
            "valid_from": version.valid_from.isoformat() if version.valid_from else None,
            "valid_to": version.valid_to.isoformat() if version.valid_to else None,
            "source_available_at": version.source_available_at.isoformat(),
            "tombstoned": False,
            "confidentiality": document.confidentiality,
            "acl_tags": ["credit_analyst", "risk_reviewer"],  # MVP 静态派生；正式授权仍回 PG
            "quality_status": section.quality_status,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "heading_path": section.heading_path,
            "text_hash": section.text_hash,
            "embedding_version": entry.embedding_version,
        }
        self._qdrant.upsert(
            collection_name=entry.target_collection_name,
            points=[qm.PointStruct(id=str(point_id), vector=vectors, payload=payload)],
        )

    async def _upsert_summary(self, session: AsyncSession, entry: IndexOutbox) -> None:
        """摘要节点入索引（任务 17）：point_type=summary_node，永不作为正式证据。"""
        from creditlens.infrastructure.postgres.models import SummaryNode

        node = await session.get(SummaryNode, entry.aggregate_id)
        if node is None:
            raise ValueError(f"summary node {entry.aggregate_id} 不存在")
        if node.grounding_status != "VERIFIED":
            raise ValueError("未通过 grounding 校验的摘要不得进入摘要召回")
        version = await session.get(DocumentVersion, node.document_version_id)
        document = await session.get(Document, version.document_id) if version else None
        if version is None or document is None:
            raise ValueError("document version 缺失")

        vector = (await self._embedder.embed_documents([node.summary_text]))[0]
        point_id = deterministic_point_id(node.id, node.summary_hash, entry.embedding_version)
        payload = {
            "tenant_id": str(node.tenant_id),
            "point_type": "summary_node",
            "summary_node_id": str(node.id),
            "summary_level": node.summary_level,
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "parse_run_id": str(node.parse_run_id),
            "document_type": document.document_type,
            "valid_from": version.valid_from.isoformat() if version.valid_from else None,
            "valid_to": version.valid_to.isoformat() if version.valid_to else None,
            "source_available_at": version.source_available_at.isoformat(),
            "tombstoned": False,
            "confidentiality": document.confidentiality,
            "quality_status": "PASS",
            "text_hash": node.summary_hash,
            "embedding_version": entry.embedding_version,
        }
        self._qdrant.upsert(
            collection_name=entry.target_collection_name,
            points=[qm.PointStruct(id=str(point_id), vector={"dense": vector}, payload=payload)],
        )

    def _tombstone(self, entry: IndexOutbox) -> None:
        """撤销/删除：先置 tombstoned=true，物理删除由 GC 异步执行（文档 §6.6）。"""
        self._qdrant.set_payload(
            collection_name=entry.target_collection_name,
            payload={"tombstoned": True},
            points=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="section_id", match=qm.MatchValue(value=str(entry.aggregate_id))
                    )
                ]
            ),
        )

    @staticmethod
    async def _payload_scopes(
        session: AsyncSession, version: DocumentVersion
    ) -> tuple[list[str], list[str]]:
        """派生 Payload 中的 entity_ids 与 product_codes（召回前过滤用）。"""
        entity_ids: list[str] = []
        case_rows = (
            await session.execute(
                select(CaseDocument.case_id).where(CaseDocument.document_version_id == version.id)
            )
        ).all()
        if case_rows:
            from creditlens.infrastructure.postgres.models import CreditCase

            for (case_id,) in case_rows:
                case = await session.get(CreditCase, case_id)
                if case is not None:
                    entity_ids.append(str(case.borrower_entity_id))
        # MVP：单产品；政策文件默认适用 working_capital
        product_codes = ["working_capital"]
        return sorted(set(entity_ids)), product_codes


async def count_pending(session: AsyncSession) -> int:
    from sqlalchemy import func

    return (
        await session.scalar(
            select(func.count()).select_from(IndexOutbox).where(IndexOutbox.status == "PENDING")
        )
    ) or 0


async def reconcile_sections_vs_points(
    session: AsyncSession, qdrant: QdrantClient, collection_name: str, embedding_version: str
) -> list[uuid.UUID]:
    """Reconciler 最小版：检查已 COMPLETED 的 Section 是否真的存在向量，
    返回缺失的 section_id 列表（产生修复任务，不静默删除）。"""
    completed = (
        await session.scalars(
            select(IndexOutbox).where(
                IndexOutbox.status == "COMPLETED", IndexOutbox.aggregate_type == "SECTION"
            )
        )
    ).all()
    missing: list[uuid.UUID] = []
    for entry in completed:
        point_id = deterministic_point_id(entry.aggregate_id, entry.content_hash, embedding_version)
        found = qdrant.retrieve(collection_name, ids=[str(point_id)], with_payload=False)
        if not found:
            missing.append(entry.aggregate_id)
    return missing
