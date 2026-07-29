"""Summary Navigation（任务 17 检索侧，文档 §8.7 Route C）。

流程：检索摘要 Collection（L0/L1）-> 选 Top 分支 -> 下钻其来源 Leaf Section
-> 用原始子问题对 Leaf 打分排序 -> 只返回 Leaf 作为候选。

摘要本身永不进入最终 Evidence；下钻产生的 Leaf 是"新候选"，
必须再次通过完整回表复核。
"""

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.infrastructure.postgres.models import (
    Document,
    DocumentSection,
    DocumentVersion,
)
from creditlens.ingestion.summaries import child_section_ids
from creditlens.retrieval.contracts import (
    RetrievalResult,
    RetrievedCandidate,
    TrustedRequestContext,
)
from creditlens.retrieval.dense import build_hard_filter, verify_candidate
from creditlens.retrieval.sparse import tokenize

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext


class SummaryNavigator:
    def __init__(self, qdrant: QdrantClient, embedder):
        self._qdrant = qdrant
        self._embedder = embedder

    async def retrieve(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        query: str,
        summary_collection: str,
        summary_top_k: int = 5,
        child_candidate_limit: int = 40,
        leaf_top_k: int = 8,
        snapshot: "SnapshotContext | None" = None,
    ) -> RetrievalResult:
        query_vector = await self._embedder.embed_query(query)
        hits = self._qdrant.query_points(
            collection_name=summary_collection,
            query=query_vector,
            using="dense",
            query_filter=build_hard_filter(trusted, snapshot),
            limit=summary_top_k,
            with_payload=True,
        ).points

        # 下钻：摘要 -> 来源 Leaf Section
        leaf_ids: list = []
        seen: set = set()
        for hit in hits:
            payload = hit.payload or {}
            if payload.get("point_type") != "summary_node":
                continue
            for section_id in await child_section_ids(session, payload["summary_node_id"]):
                if section_id not in seen:
                    seen.add(section_id)
                    leaf_ids.append(section_id)
                if len(leaf_ids) >= child_candidate_limit:
                    break

        # 用原始问题对 Leaf 重新打分（词面重叠，确定性）
        query_terms = set(tokenize(query))
        scored: list[tuple[float, DocumentSection]] = []
        for section_id in leaf_ids:
            section = await session.get(DocumentSection, section_id)
            if section is None or section.section_type not in {"ARTICLE", "PARAGRAPH"}:
                continue
            terms = set(tokenize(section.text))
            score = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
            scored.append((score, section))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        candidates: list[RetrievedCandidate] = []
        rejected: list[RetrievedCandidate] = []
        for rank, (score, section) in enumerate(scored[:leaf_top_k], start=1):
            version = await session.get(DocumentVersion, section.document_version_id)
            document = await session.get(Document, version.document_id) if version else None
            candidate = RetrievedCandidate(
                section_id=section.id,
                document_id=document.id if document else section.document_version_id,
                document_version_id=section.document_version_id,
                parse_run_id=section.parse_run_id,
                page_start=section.page_start,
                page_end=section.page_end,
                heading_path=section.heading_path or [],
                text="",
                text_hash=section.text_hash,
                channel="SUMMARY",
                rank=rank,
                raw_score=score,
            )
            payload = {
                "document_type": document.document_type if document else None,
                "document_id": str(document.id) if document else None,
            }
            reason = await verify_candidate(session, trusted, candidate, payload, snapshot)
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
                "channel": "SUMMARY",
                "summary_top_k": summary_top_k,
                "child_candidate_limit": child_candidate_limit,
                "leaf_top_k": leaf_top_k,
                "collection": summary_collection,
            },
        )
