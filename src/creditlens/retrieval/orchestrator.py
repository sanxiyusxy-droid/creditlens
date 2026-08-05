"""统一在线 Retrieval Orchestrator（v1.1，文档 §8.3-§8.13）。

完整管线：
    QuerySpec (build + validate)
    -> 子问题/变体路由 (query_variants 按 route 分发)
    -> Route A: Dense (per variant)
    -> Route B: Sparse/BM25 (per variant, exact_terms)
    -> Route C: Summary Navigation (L0/L1 递归下钻)
    -> Route D: Exact Match (条款号/数字 PG LIKE)
    -> PG 回表复核 (verify_candidate)
    -> RRF 融合
    -> Cross-Encoder 精排 (可选, 记录 rerank_degraded)
    -> Context Packing
    -> Trace 元数据

API、Policy Agent、Challenger 共用此编排器（不同 OrchestratorConfig）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import EmbeddingProvider
from creditlens.retrieval.context_packing import PackedContext, pack_context
from creditlens.retrieval.contracts import (
    RetrievedCandidate,
    TrustedRequestContext,
)
from creditlens.retrieval.dense import DenseRetriever, verify_candidate
from creditlens.retrieval.fusion import FusedCandidate, rrf_fuse
from creditlens.retrieval.hybrid import SparseRetriever
from creditlens.retrieval.query_spec import (
    build_query_spec,
    safe_fallback,
    validate_query_spec,
)
from creditlens.retrieval.rerank import RerankProvider
from creditlens.retrieval.summary_navigation import SummaryNavigator

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext


class OrchestratorConfig(BaseModel):
    """编排器运行配置（不同调用方使用不同配置）。"""

    top_k_per_route: int = 30
    fused_limit: int = 80
    final_limit: int = 12
    enable_sparse: bool = True
    enable_summary: bool = True
    enable_exact: bool = True
    enable_rerank: bool = True
    enable_packing: bool = True
    token_budget: int = 4096
    max_per_document_ratio: float = 0.6
    expand_adjacent: bool = True
    summary_top_k: int = 5
    summary_leaf_top_k: int = 10


class RouteTrace(BaseModel):
    """单路召回 Trace。"""

    route: str
    variant_id: str = ""
    candidates_count: int = 0
    rejected_count: int = 0
    rejection_reasons: dict[str, int] = {}


class OrchestratedResult(BaseModel):
    """编排器统一输出。"""

    query: str
    candidates: list[RetrievedCandidate]
    rejected: list[RetrievedCandidate]
    query_spec: dict = {}
    trace: dict = {}
    packing: dict | None = None
    channel_config: dict = {}


@dataclass
class RetrievalOrchestrator:
    """统一在线检索编排器。

    持有 Dense/Sparse/Summary 子检索器 + 可选 Reranker；
    通过 OrchestratorConfig 控制各路由开关与参数。
    """

    qdrant: Any
    embedder: EmbeddingProvider
    reranker: RerankProvider | None = None
    rrf_k: int = 60
    route_weights: dict[str, float] | None = None

    def __post_init__(self):
        self._dense = DenseRetriever(self.qdrant, self.embedder)
        self._sparse = SparseRetriever(self.qdrant)
        self._summary = SummaryNavigator(self.qdrant, self.embedder)

    async def retrieve(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        query: str,
        collection_name: str,
        config: OrchestratorConfig | None = None,
        snapshot: SnapshotContext | None = None,
        summaries_collection: str | None = None,
    ) -> OrchestratedResult:
        """执行完整编排管线。"""
        cfg = config or OrchestratorConfig()

        # 1. QuerySpec 构建 + 验证
        spec = build_query_spec(trusted, query)
        validation = validate_query_spec(trusted, spec)
        if not validation.ok:
            spec = safe_fallback(trusted, query)

        # 2. 按变体路由分发：每个 Query Variant 作为独立排名列表参与 RRF，
        # 不先顺序拼接（文档 §8.9）；通道名 "ROUTE:variant_id"。
        ranked_lists: dict[str, list[RetrievedCandidate]] = {}
        all_rejected: list[RetrievedCandidate] = []
        route_traces: list[RouteTrace] = []

        # Route A: Dense（每个 dense 变体独立检索、独立排名列表）
        dense_variants = [v for v in spec.query_variants if v.route == "dense"]
        dense_variant_texts = (
            [(v.variant_id, v.text) for v in dense_variants]
            if dense_variants
            else [("fallback", query)]
        )
        for variant_id, variant_text in dense_variant_texts:
            result = await self._dense.retrieve(
                session,
                trusted,
                variant_text,
                collection_name,
                top_k=cfg.top_k_per_route,
                snapshot=snapshot,
            )
            list_key = f"DENSE:{variant_id}"
            ranked_lists[list_key] = result.candidates
            all_rejected.extend(result.rejected)
            route_traces.append(
                _make_trace("DENSE", result.candidates, result.rejected, variant_id)
            )

        # Route B: Sparse/BM25（每个 sparse 变体独立排名列表 + exact_terms）
        if cfg.enable_sparse:
            sparse_variants = [v for v in spec.query_variants if v.route == "sparse"]
            sparse_variant_texts = (
                [(v.variant_id, v.text) for v in sparse_variants]
                if sparse_variants
                else [("fallback", query)]
            )
            for variant_id, variant_text in sparse_variant_texts:
                result = await self._sparse.retrieve(
                    session,
                    trusted,
                    variant_text,
                    collection_name,
                    top_k=cfg.top_k_per_route,
                    exact_terms=spec.exact_terms,
                    snapshot=snapshot,
                )
                list_key = f"SPARSE:{variant_id}"
                ranked_lists[list_key] = result.candidates
                all_rejected.extend(result.rejected)
                route_traces.append(
                    _make_trace("SPARSE", result.candidates, result.rejected, variant_id)
                )

        # Route C: Summary Navigation
        if cfg.enable_summary and summaries_collection:
            summary_result = await self._summary.retrieve(
                session,
                trusted,
                query,
                summaries_collection,
                summary_top_k=cfg.summary_top_k,
                leaf_top_k=cfg.summary_leaf_top_k,
                snapshot=snapshot,
            )
            ranked_lists["SUMMARY"] = summary_result.candidates
            all_rejected.extend(summary_result.rejected)
            route_traces.append(
                _make_trace("SUMMARY", summary_result.candidates, summary_result.rejected)
            )

        # Route D: Exact Match（条款号/数字 PG LIKE）
        if cfg.enable_exact and spec.exact_terms:
            exact_candidates, exact_rejected = await self._exact_match(
                session, trusted, spec.exact_terms, snapshot
            )
            if exact_candidates:
                ranked_lists["EXACT"] = exact_candidates
                all_rejected.extend(exact_rejected)
                route_traces.append(_make_trace("EXACT", exact_candidates, exact_rejected))

        # 3. RRF 融合
        fused = rrf_fuse(
            ranked_lists,
            rrf_k=self.rrf_k,
            route_weights=self.route_weights,
            limit=cfg.fused_limit,
            max_candidates_per_document=cfg.fused_limit,
        )

        # 4. Cross-Encoder 精排（可选）：请求精排但 Reranker 缺失/异常 -> 降级必须记录
        rerank_applied = False
        rerank_degraded = False
        if cfg.enable_rerank:
            if self.reranker is None:
                rerank_degraded = True
            else:
                try:
                    fused = await self._rerank(query, fused)
                    rerank_applied = True
                except Exception:
                    rerank_degraded = True

        # 5. 重编 rank + 截断
        final: list[RetrievedCandidate] = []
        for rank, item in enumerate(fused[: cfg.final_limit], start=1):
            candidate = item.candidate.model_copy()
            candidate.rank = rank
            candidate.channel = "FUSED"
            final.append(candidate)

        # 6. Context Packing：在完整融合候选池上执行（而非 final_limit 截断后），
        # 丢弃超预算候选后继续用后续候选补位（文档 §8.13）。
        packing_result: PackedContext | None = None
        if cfg.enable_packing and fused:
            packing_result = await pack_context(
                session,
                [f.candidate for f in fused],
                trusted=trusted,
                snapshot=snapshot,
                token_budget=cfg.token_budget,
                max_per_document_ratio=cfg.max_per_document_ratio,
                expand_adjacent=cfg.expand_adjacent,
            )

        # 7. 组装 Trace
        trace = {
            "routes": [t.model_dump() for t in route_traces],
            "rrf_k": self.rrf_k,
            "fusion": {
                "rrf_k": self.rrf_k,
                "input_lists": list(ranked_lists.keys()),
                "fused_count": len(fused),
            },
            "rerank_applied": rerank_applied,
            "rerank_degraded": rerank_degraded,
            "reranker_version": self.reranker.version if self.reranker else None,
            "fused_count": len(fused),
            "final_count": len(final),
            "query_spec_confidence": spec.rewrite_confidence,
            "query_variants_count": len(spec.query_variants),
        }

        channel_config = {
            "channel": "ORCHESTRATED",
            "routes": list(ranked_lists.keys()),
            "rrf_k": self.rrf_k,
            "rerank": rerank_applied,
            "rerank_degraded": rerank_degraded,
            "top_k_per_route": cfg.top_k_per_route,
            "collection": collection_name,
            "packing_tokens": packing_result.total_tokens_est if packing_result else None,
            "packing_dropped": len(packing_result.dropped) if packing_result else 0,
        }

        return OrchestratedResult(
            query=query,
            candidates=final,
            rejected=all_rejected,
            query_spec=spec.model_dump(mode="json"),
            trace=trace,
            packing=packing_result.model_dump(mode="json") if packing_result else None,
            channel_config=channel_config,
        )

    async def _rerank(self, query: str, fused: list[FusedCandidate]) -> list[FusedCandidate]:
        """Cross-Encoder 精排。"""
        scores = await self.reranker.score(
            query, [f.candidate.text or " ".join(f.candidate.heading_path) for f in fused]
        )
        # 平分确定性 tie-break（WP6：消除精排平分题的 ±1 排序抖动）
        order = sorted(
            range(len(fused)),
            key=lambda i: (scores[i], fused[i].candidate.section_id),
            reverse=True,
        )
        return [fused[i] for i in order]

    async def _exact_match(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        exact_terms: list[str],
        snapshot: SnapshotContext | None = None,
    ) -> tuple[list[RetrievedCandidate], list[RetrievedCandidate]]:
        """Route D：条款号/精确术语 PG LIKE 匹配（v1.1 P0 收口）。

        安全约束（与 Dense/Sparse 同等强度）：
        - SQL 层：租户 + 案件绑定（CaseDocument）+ Section 质量状态；
        - 验证通过前不加载 Section 原文（只选 ID/hash/定位信息）；
        - 回表复核用真实 document_type 执行政策有效期、时点、Snapshot、
          ParseRun 激活检查；rejected 候选不携带未授权原文（text 恒为 ""）。
        """
        from creditlens.infrastructure.postgres.models import (
            CaseDocument,
            Document,
            DocumentSection,
            DocumentVersion,
        )

        candidates: list[RetrievedCandidate] = []
        rejected: list[RetrievedCandidate] = []

        for term in exact_terms[:5]:  # 限制精确查询数
            rows = (
                await session.execute(
                    select(
                        DocumentSection,
                        DocumentVersion,
                        Document,
                    )
                    .join(
                        DocumentVersion,
                        DocumentVersion.id == DocumentSection.document_version_id,
                    )
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .join(
                        CaseDocument,
                        and_(
                            CaseDocument.document_version_id == DocumentVersion.id,
                            CaseDocument.case_id == trusted.case_id,
                        ),
                    )
                    .where(
                        DocumentSection.tenant_id == trusted.tenant_id,
                        DocumentSection.text.ilike(f"%{term}%"),
                        DocumentSection.section_type.in_(["ARTICLE", "PARAGRAPH"]),
                        DocumentSection.quality_status != "BLOCKED",
                        DocumentVersion.quality_status != "BLOCKED",
                    )
                    .limit(10)
                )
            ).all()
            for rank, (section, version, document) in enumerate(rows, start=1):
                # 验证前不携带原文：text=""，通过后由 verify_candidate 回填
                candidate = RetrievedCandidate(
                    section_id=section.id,
                    document_id=document.id,
                    document_version_id=version.id,
                    parse_run_id=section.parse_run_id,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    heading_path=section.heading_path or [],
                    text="",
                    text_hash=section.text_hash,
                    channel="EXACT",
                    rank=rank,
                    raw_score=1.0,
                )
                payload = {
                    "document_id": str(document.id),
                    "document_type": document.document_type,  # 真实类型 -> 政策有效期检查
                }
                reason = await verify_candidate(session, trusted, candidate, payload, snapshot)
                if reason is None:
                    candidates.append(candidate)
                else:
                    candidate.rejection_reason = reason
                    rejected.append(candidate)

        return _dedup_candidates(candidates), rejected


def _dedup_candidates(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    """按 section_id 去重，保留最优（最早出现的）rank。"""
    seen: set = set()
    result: list[RetrievedCandidate] = []
    for c in candidates:
        if c.section_id not in seen:
            seen.add(c.section_id)
            result.append(c)
    # 重编 rank
    for i, c in enumerate(result, start=1):
        c.rank = i
    return result


def _make_trace(
    route: str,
    candidates: list[RetrievedCandidate],
    rejected: list[RetrievedCandidate],
    variant_id: str = "",
) -> RouteTrace:
    """生成单路 Trace。"""
    reasons: dict[str, int] = {}
    for r in rejected:
        reason = r.rejection_reason or "UNKNOWN"
        reasons[reason] = reasons.get(reason, 0) + 1
    return RouteTrace(
        route=route,
        variant_id=variant_id,
        candidates_count=len(candidates),
        rejected_count=len(rejected),
        rejection_reasons=reasons,
    )


# --- 预置配置 ---

ONLINE_CONFIG = OrchestratorConfig(
    final_limit=8,
    enable_rerank=True,
    enable_packing=True,
)

AGENT_CONFIG = OrchestratorConfig(
    final_limit=8,
    enable_rerank=True,
    enable_packing=True,  # v1.1：Agent 消费 Packed Sections（而非仅候选元数据）
    enable_summary=True,
)

EVAL_CONFIG = OrchestratorConfig(
    final_limit=30,
    enable_rerank=False,
    enable_packing=False,
    enable_summary=False,
    enable_exact=False,
)
