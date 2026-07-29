"""Dense-only Retriever（任务 10，文档 §8.6/§8.7 Route A/§8.8）。

流程：
1. 服务端追加硬过滤（tenant / tombstoned / quality / 时点 / 政策有效期）；
2. Qdrant dense 召回；
3. PostgreSQL 回表复核：Section 存在、ParseRun 属激活集、text_hash 一致、
   有效期与可获得时间、质量状态；失败候选带 rejection_reason，不进入结果。

硬过滤不能改为"在最终得分中降权"。
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import EmbeddingProvider
from creditlens.common.errors import AclDeniedError
from creditlens.infrastructure.postgres.models import (
    DocumentSection,
    DocumentVersion,
    ParseRun,
)
from creditlens.retrieval.contracts import (
    RetrievalResult,
    RetrievedCandidate,
    TrustedRequestContext,
)

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext


def build_hard_filter(
    trusted: TrustedRequestContext,
    snapshot: "SnapshotContext | None" = None,
) -> qm.Filter:
    """召回前硬过滤（文档 §8.6）。

    - 空 allowed_document_ids = 默认拒绝（授权集合为空不代表放行全部）；
    - 提供 Snapshot 时追加 parse_run_id IN snapshot.allowed_parse_run_ids，
      已启动 Run 只看冻结的输入世界。
    """
    if not trusted.allowed_document_ids:
        raise AclDeniedError(
            "空文档授权集合：TrustedRequestContext 必须由服务端从案件绑定派生",
            {"case_id": str(trusted.case_id)},
        )
    must: list[qm.Condition] = [
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(trusted.tenant_id))),
        qm.FieldCondition(key="tombstoned", match=qm.MatchValue(value=False)),
        qm.FieldCondition(
            key="document_id",
            match=qm.MatchAny(any=[str(d) for d in trusted.allowed_document_ids]),
        ),
    ]
    must_not: list[qm.Condition] = [
        qm.FieldCondition(key="quality_status", match=qm.MatchValue(value="BLOCKED")),
    ]
    cutoff = trusted.decision_cutoff_at.astimezone(UTC).isoformat()
    must.append(
        qm.FieldCondition(key="source_available_at", range=qm.DatetimeRange(lte=cutoff))
    )
    # D3（v0.6）：政策有效期下推向量层。as_of ∉ [valid_from, valid_to) 的政策类
    # 候选直接排除；null 值不匹配 range 条件，非政策文档不受影响。
    # 回表复核仍保留同一检查作为第二道防线（文档 §8.6/§8.8）。
    policy_types = ["REGULATION", "INTERNAL_POLICY"]
    as_of = trusted.as_of_date.isoformat()
    must_not.append(
        qm.Filter(
            must=[
                qm.FieldCondition(key="document_type", match=qm.MatchAny(any=policy_types)),
                qm.FieldCondition(key="valid_from", range=qm.DatetimeRange(gt=as_of)),
            ]
        )
    )
    must_not.append(
        qm.Filter(
            must=[
                qm.FieldCondition(key="document_type", match=qm.MatchAny(any=policy_types)),
                qm.FieldCondition(key="valid_to", range=qm.DatetimeRange(lte=as_of)),
            ]
        )
    )
    if snapshot is not None:
        must.append(
            qm.FieldCondition(
                key="parse_run_id",
                match=qm.MatchAny(any=[str(p) for p in snapshot.allowed_parse_run_ids]),
            )
        )
    return qm.Filter(must=must, must_not=must_not)


class DenseRetriever:
    def __init__(self, qdrant: QdrantClient, embedder: EmbeddingProvider):
        self._qdrant = qdrant
        self._embedder = embedder

    async def retrieve(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        query: str,
        collection_name: str,
        top_k: int = 30,
        snapshot: "SnapshotContext | None" = None,
    ) -> RetrievalResult:
        query_vector = await self._embedder.embed_query(query)
        hits = self._qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            query_filter=build_hard_filter(trusted, snapshot),
            limit=top_k,
            with_payload=True,
        ).points

        candidates: list[RetrievedCandidate] = []
        rejected: list[RetrievedCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            candidate = RetrievedCandidate(
                section_id=payload["section_id"],
                document_id=payload["document_id"],
                document_version_id=payload["document_version_id"],
                parse_run_id=payload["parse_run_id"],
                page_start=payload.get("page_start", 0),
                page_end=payload.get("page_end", 0),
                heading_path=payload.get("heading_path") or [],
                text="",
                text_hash=payload.get("text_hash", ""),
                channel="DENSE",
                rank=rank,
                raw_score=float(hit.score),
            )
            reason = await self._verify(session, trusted, candidate, payload, snapshot)
            if reason is None:
                candidates.append(candidate)
            else:
                candidate.rejection_reason = reason
                rejected.append(candidate)

        return RetrievalResult(
            query=query,
            candidates=candidates,
            rejected=rejected,
            channel_config={
                "channel": "DENSE",
                "top_k": top_k,
                "embedding_version": self._embedder.version,
                "collection": collection_name,
            },
        )

    async def _verify(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        candidate: RetrievedCandidate,
        payload: dict,
        snapshot: "SnapshotContext | None" = None,
    ) -> str | None:
        return await verify_candidate(session, trusted, candidate, payload, snapshot)


async def verify_candidate(
    session: AsyncSession,
    trusted: TrustedRequestContext,
    candidate: RetrievedCandidate,
    payload: dict,
    snapshot: "SnapshotContext | None" = None,
) -> str | None:
    """PostgreSQL 回表复核（文档 §8.8），各召回路线共用。
    返回 None 表示通过；否则返回拒绝原因。"""
    section = await session.get(DocumentSection, candidate.section_id)
    if section is None:
        return "STALE_INDEX"
    if section.text_hash != candidate.text_hash:
        return "STALE_INDEX"
    if str(section.tenant_id) != str(trusted.tenant_id):
        return "ACL_DENIED"
    if section.quality_status == "BLOCKED":
        return "QUALITY_BLOCKED"

    # ACL 第二道防线：候选必须属于授权文档集合
    if payload.get("document_id") and all(
        str(d) != str(payload["document_id"]) for d in trusted.allowed_document_ids
    ):
        return "ACL_DENIED"

    parse_run = await session.get(ParseRun, candidate.parse_run_id)
    if parse_run is None or parse_run.activation_status in {"REVOKED", "TOMBSTONED"}:
        return "PARSE_RUN_NOT_IN_SNAPSHOT"
    # Snapshot 冻结：只接受冻结集合内的 Parse Run（含 SUPERSEDED，
    # 支持历史 Run 继续访问旧解析批次）
    if snapshot is not None and candidate.parse_run_id not in set(
        snapshot.allowed_parse_run_ids
    ):
        return "PARSE_RUN_NOT_IN_SNAPSHOT"

    version = await session.get(DocumentVersion, candidate.document_version_id)
    if version is None:
        return "STALE_INDEX"
    if version.quality_status == "BLOCKED":
        return "QUALITY_BLOCKED"

    # 可获得时间（防止使用审查截止后才可获得的材料）
    available_at = version.source_available_at
    if available_at.tzinfo is None:
        available_at = available_at.replace(tzinfo=UTC)
    if available_at > trusted.decision_cutoff_at.astimezone(UTC):
        return "NOT_AVAILABLE_AT_CUTOFF"

    # 政策有效期：as_of_date ∈ [valid_from, valid_to)
    if payload.get("document_type") in {"REGULATION", "INTERNAL_POLICY"}:
        if version.valid_from is not None and trusted.as_of_date < version.valid_from:
            return "OUT_OF_EFFECTIVE_DATE"
        if version.valid_to is not None and trusted.as_of_date >= version.valid_to:
            return "OUT_OF_EFFECTIVE_DATE"

    # 通过：回填授权原文（向量 Payload 不存完整敏感原文）
    candidate.text = section.text
    return None


def ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
