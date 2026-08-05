"""多路混合检索（任务 14/15/16 编排）：Dense + Sparse/BM25 -> 回表复核 -> RRF -> 精排。"""

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import EmbeddingProvider
from creditlens.retrieval.contracts import (
    RetrievalResult,
    RetrievedCandidate,
    TrustedRequestContext,
)
from creditlens.retrieval.dense import build_hard_filter, verify_candidate
from creditlens.retrieval.fusion import FusedCandidate, rrf_fuse
from creditlens.retrieval.sparse import Bm25SparseEncoder

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext


def _hit_to_candidate(hit, rank: int, channel: str) -> RetrievedCandidate:
    payload = hit.payload or {}
    return RetrievedCandidate(
        section_id=payload["section_id"],
        document_id=payload["document_id"],
        document_version_id=payload["document_version_id"],
        parse_run_id=payload["parse_run_id"],
        page_start=payload.get("page_start", 0),
        page_end=payload.get("page_end", 0),
        heading_path=payload.get("heading_path") or [],
        text="",
        text_hash=payload.get("text_hash", ""),
        channel=channel,
        rank=rank,
        raw_score=float(hit.score),
    )


class SparseRetriever:
    """Route B：版本化 BM25 稀疏召回（文档 §8.7）。"""

    def __init__(self, qdrant: QdrantClient, encoder: Bm25SparseEncoder | None = None):
        self._qdrant = qdrant
        self._encoder = encoder or Bm25SparseEncoder()

    async def retrieve(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        query: str,
        collection_name: str,
        top_k: int = 30,
        exact_terms: list[str] | None = None,
        snapshot: "SnapshotContext | None" = None,
    ) -> RetrievalResult:
        indices, values = self._encoder.encode_query(query, extra_terms=exact_terms)
        if not indices:
            return RetrievalResult(query=query, candidates=[], rejected=[], channel_config={})
        hits = self._qdrant.query_points(
            collection_name=collection_name,
            query=qm.SparseVector(indices=indices, values=values),
            using="sparse",
            query_filter=build_hard_filter(trusted, snapshot),
            limit=top_k,
            with_payload=True,
        ).points

        candidates: list[RetrievedCandidate] = []
        rejected: list[RetrievedCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            candidate = _hit_to_candidate(hit, rank, "SPARSE")
            reason = await verify_candidate(
                session, trusted, candidate, hit.payload or {}, snapshot
            )
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
                "channel": "SPARSE",
                "top_k": top_k,
                "sparse_encoder_version": self._encoder.version,
                "collection": collection_name,
            },
        )


class HybridRetriever:
    """Dense + Sparse 并行召回 -> RRF 融合 ->（可选）Cross-Encoder 精排。"""

    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: EmbeddingProvider,
        sparse_encoder: Bm25SparseEncoder | None = None,
        reranker=None,
        rrf_k: int = 60,
        route_weights: dict[str, float] | None = None,
    ):
        from creditlens.retrieval.dense import DenseRetriever

        self._dense = DenseRetriever(qdrant, embedder)
        self._sparse = SparseRetriever(qdrant, sparse_encoder)
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._route_weights = route_weights

    async def retrieve(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        query: str,
        collection_name: str,
        top_k_per_route: int = 30,
        fused_limit: int = 80,
        final_limit: int = 12,
        exact_terms: list[str] | None = None,
        enable_sparse: bool = True,
        enable_rerank: bool = True,
        snapshot: "SnapshotContext | None" = None,
    ) -> RetrievalResult:
        ranked_lists: dict[str, list[RetrievedCandidate]] = {}
        rejected: list[RetrievedCandidate] = []

        dense = await self._dense.retrieve(
            session, trusted, query, collection_name, top_k=top_k_per_route, snapshot=snapshot
        )
        ranked_lists["DENSE"] = dense.candidates
        rejected.extend(dense.rejected)

        if enable_sparse:
            sparse = await self._sparse.retrieve(
                session,
                trusted,
                query,
                collection_name,
                top_k=top_k_per_route,
                exact_terms=exact_terms,
                snapshot=snapshot,
            )
            ranked_lists["SPARSE"] = sparse.candidates
            rejected.extend(sparse.rejected)

        fused = rrf_fuse(
            ranked_lists,
            rrf_k=self._rrf_k,
            route_weights=self._route_weights,
            limit=fused_limit,
            # 融合层不做来源上限：单文档语料会直接损失召回（q12 教训）。
            # "不让同一文档占满上下文"属于 Context Packing 层（文档 §8.13）。
            max_candidates_per_document=fused_limit,
        )

        rerank_applied = False
        rerank_degraded = False
        if enable_rerank and self._reranker is not None:
            try:
                fused = await self._rerank(query, fused)
                rerank_applied = True
            except Exception:
                # Reranker 不可用降级到 RRF 顺序，但必须记录，不假成功（文档 §8.11）
                rerank_degraded = True

        # 重排后按最终顺序重编 rank
        final: list[RetrievedCandidate] = []
        for rank, item in enumerate(fused[:final_limit], start=1):
            candidate = item.candidate.model_copy()
            candidate.rank = rank
            candidate.channel = "FUSED"
            final.append(candidate)

        return RetrievalResult(
            query=query,
            candidates=final,
            rejected=rejected,
            channel_config={
                "channel": "HYBRID",
                "routes": list(ranked_lists),
                "rrf_k": self._rrf_k,
                "rerank": rerank_applied,
                "rerank_degraded": rerank_degraded,
                "top_k_per_route": top_k_per_route,
                "collection": collection_name,
            },
        )

    async def _rerank(self, query: str, fused: list[FusedCandidate]) -> list[FusedCandidate]:
        scores = await self._reranker.score(
            query, [f.candidate.text or " ".join(f.candidate.heading_path) for f in fused]
        )
        # 平分确定性 tie-break（WP6）
        order = sorted(
            range(len(fused)),
            key=lambda i: (scores[i], fused[i].candidate.section_id),
            reverse=True,
        )
        return [fused[i] for i in order]
