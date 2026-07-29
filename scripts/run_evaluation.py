"""Recall@K 评测脚本（任务 12 + 消融对比）。

用法：
    uv run python scripts/run_evaluation.py [--dataset evaluation/datasets/smoke_v1.json]

一次运行三个配置（文档 §16.10 累积实验的关键点）：
- E0  Dense-only
- E23 Dense + Sparse/BM25 + RRF
- E7  E23 + Rerank（本地确定性兼容实现）
输出 Recall@5/10/20、MRR@10、AllRequiredEvidence@K，报告落盘 evaluation/reports/。
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from creditlens.application.snapshot_service import freeze_snapshot
from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context
from creditlens.common.clock import utc_now
from creditlens.common.config import get_settings
from creditlens.evaluation.gold_schema import GoldDataset
from creditlens.evaluation.recall import EvalReport, evaluate_question
from creditlens.infrastructure.llm.embedding import build_embedding_provider
from creditlens.infrastructure.objectstore import build_object_store
from creditlens.infrastructure.postgres.models import Base
from creditlens.infrastructure.postgres.session import (
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client
from creditlens.retrieval.contracts import RetrievalResult
from creditlens.retrieval.dense import DenseRetriever
from creditlens.retrieval.fusion import rrf_fuse
from creditlens.retrieval.hybrid import HybridRetriever
from creditlens.retrieval.query_spec import build_query_spec, safe_fallback, validate_query_spec
from creditlens.retrieval.rerank import LexicalOverlapReranker, build_reranker
from creditlens.retrieval.summary_navigation import SummaryNavigator
from seed_synthetic_data import CASE_ID, DEMO_USER_ID, TENANT_ID, seed_environment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def audit_final_candidates(session, trusted, candidates) -> int:
    """P0-5：独立回表审计最终 candidates（不信任检索链路自身的拒绝记录）。

    对每个最终候选独立验证：文档绑定于本案件、审查截止前可获得、
    政策类在有效期内、租户匹配。返回违规计数（目标恒为 0）。
    """
    from datetime import UTC

    from sqlalchemy import select as sa_select

    from creditlens.infrastructure.postgres.models import (
        CaseDocument,
        Document,
        DocumentSection,
        DocumentVersion,
    )

    violations = 0
    for candidate in candidates:
        section = await session.get(DocumentSection, candidate.section_id)
        version = await session.get(DocumentVersion, candidate.document_version_id)
        if section is None or version is None:
            violations += 1
            continue
        if str(section.tenant_id) != str(trusted.tenant_id):
            violations += 1
            continue
        bound = await session.scalar(
            sa_select(CaseDocument.case_id).where(
                CaseDocument.case_id == trusted.case_id,
                CaseDocument.document_version_id == version.id,
            )
        )
        if bound is None:
            violations += 1
            continue
        available = version.source_available_at
        if available.tzinfo is None:
            available = available.replace(tzinfo=UTC)
        if available > trusted.decision_cutoff_at.astimezone(UTC):
            violations += 1
            continue
        document = await session.get(Document, version.document_id)
        if document is not None and document.document_type in {"REGULATION", "INTERNAL_POLICY"}:
            if version.valid_from is not None and trusted.as_of_date < version.valid_from:
                violations += 1
                continue
            if version.valid_to is not None and trusted.as_of_date >= version.valid_to:
                violations += 1
    return violations


async def evaluate_channel(
    channel_name: str,
    retrieve_fn,
    dataset: GoldDataset,
    factory,
    settings,
    trusted,
    reranker_version: str = "-",
) -> tuple[EvalReport, int]:
    import time

    results = []
    leakage = 0
    independent_violations = 0
    unanswerable_skipped = 0
    latencies_ms: list[float] = []
    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        for question in dataset.questions:
            # 评测问题的时点必须与案件上下文一致；request_id 每题独立
            question_trusted = trusted.model_copy(
                update={
                    "request_id": uuid.uuid4(),
                    "as_of_date": question.as_of_date,
                    "decision_cutoff_at": question.decision_cutoff_at.astimezone(UTC),
                }
            )
            started = time.perf_counter()
            retrieval = await retrieve_fn(session, question_trusted, question.question)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            leakage += sum(
                1
                for r in retrieval.rejected
                if r.rejection_reason in {"NOT_AVAILABLE_AT_CUTOFF", "ACL_DENIED"}
            )
            # P0-5：独立回表审计最终 candidates（真正的 Leakage 证据）
            independent_violations += await audit_final_candidates(
                session, question_trusted, retrieval.candidates
            )
            if not question.answerable:
                # 拒答题不计入 Recall（无黄金证据）；正确拒答属答案层指标
                unanswerable_skipped += 1
                continue
            results.append(
                await evaluate_question(session, dataset, question, retrieval.candidates)
            )
    report = EvalReport(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        config={
            "channel": channel_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_version": settings.effective_embedding_version,
            "sparse_encoder_version": settings.sparse_encoder_version,
            "reranker_version": reranker_version,
            "dense_top_k": settings.dense_top_k,
            "rrf_k": settings.rrf_k,
            "collection": settings.chunks_collection_name,
            "parser": f"{settings.parser_name}:{settings.parser_version}",
        },
        question_results=results,
    )
    report.config["unanswerable_skipped"] = unanswerable_skipped
    report.config["independent_leakage_violations"] = independent_violations
    if latencies_ms:
        ordered = sorted(latencies_ms)
        report.config["latency_p50_ms"] = round(ordered[len(ordered) // 2], 1)
        report.config["latency_p95_ms"] = round(ordered[int(len(ordered) * 0.95) - 1], 1)
    return report, leakage


async def main(dataset_path: Path) -> None:
    settings = get_settings()
    dataset = GoldDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))

    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    store = build_object_store(settings)
    qdrant = build_qdrant_client(settings)

    await seed_environment(factory, store, qdrant, settings)

    # 服务端派生可信上下文并冻结 Snapshot（v0.2：检索只读冻结的输入世界）
    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        trusted = await build_trusted_context(session, TENANT_ID, CASE_ID)
        snapshot = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=settings.chunks_collection_name,
            summaries_collection=settings.summaries_collection_name,
            acl_hash=acl_scope_hash(trusted),
        )

    embedder = build_embedding_provider(settings)
    # E7 精排：优先使用配置的真实 Reranker；未配置时退回词面兜底并如实标注
    reranker = build_reranker(settings) or LexicalOverlapReranker()
    dense = DenseRetriever(qdrant, embedder)
    hybrid = HybridRetriever(qdrant, embedder, rrf_k=settings.rrf_k)
    hybrid_rerank = HybridRetriever(qdrant, embedder, reranker=reranker, rrf_k=settings.rrf_k)

    async def run_dense(session, trusted, query):
        return await dense.retrieve(
            session, trusted, query, snapshot.chunks_collection,
            top_k=settings.dense_top_k, snapshot=snapshot,
        )

    async def run_hybrid(session, trusted, query):
        return await hybrid.retrieve(
            session, trusted, query, snapshot.chunks_collection,
            top_k_per_route=settings.dense_top_k, final_limit=30,
            enable_rerank=False, snapshot=snapshot,
        )

    async def run_hybrid_rerank(session, trusted, query):
        return await hybrid_rerank.retrieve(
            session, trusted, query, snapshot.chunks_collection,
            top_k_per_route=settings.dense_top_k, final_limit=30,
            enable_rerank=True, snapshot=snapshot,
        )

    # E4：Dense + Summary 导航（下钻 Leaf 后 RRF 融合；摘要不作为证据）
    summary_nav = SummaryNavigator(qdrant, embedder)

    async def run_dense_summary(session, trusted, query):
        dense_result = await dense.retrieve(
            session, trusted, query, snapshot.chunks_collection,
            top_k=settings.dense_top_k, snapshot=snapshot,
        )
        summary_result = await summary_nav.retrieve(
            session, trusted, query, snapshot.summaries_collection,
            leaf_top_k=10, snapshot=snapshot,
        )
        fused = rrf_fuse(
            {"DENSE": dense_result.candidates, "SUMMARY": summary_result.candidates},
            rrf_k=settings.rrf_k, limit=80, max_candidates_per_document=80,
        )
        final = []
        for rank, item in enumerate(fused[:30], start=1):
            candidate = item.candidate.model_copy()
            candidate.rank = rank
            final.append(candidate)
        return RetrievalResult(
            query=query,
            candidates=final,
            rejected=dense_result.rejected + summary_result.rejected,
            channel_config={"channel": "E4_DENSE_SUMMARY"},
        )

    # E5：QuerySpec 规则化 Rewrite（术语归一 + 词法扩展进 Sparse/exact_terms），
    # 输出必须过 Rewrite Validator，失败退回 safe_fallback（文档 §8.4）
    async def run_hybrid_rewrite(session, trusted, query):
        spec = build_query_spec(trusted, query)
        if not validate_query_spec(trusted, spec).ok:
            spec = safe_fallback(trusted, query)
        return await hybrid.retrieve(
            session, trusted, spec.standalone_query, snapshot.chunks_collection,
            top_k_per_route=settings.dense_top_k, final_limit=30,
            enable_rerank=False, exact_terms=spec.exact_terms, snapshot=snapshot,
        )

    channels = [
        ("E0_dense_only", run_dense, "-"),
        ("E23_hybrid_rrf", run_hybrid, "-"),
        ("E4_dense_summary", run_dense_summary, "-"),
        ("E5_hybrid_rewrite", run_hybrid_rewrite, "-"),
        ("E7_hybrid_rerank", run_hybrid_rerank, reranker.version),
    ]

    all_summaries = {}
    per_channel = {}
    for name, fn, reranker_version in channels:
        report, leakage = await evaluate_channel(
            name, fn, dataset, factory, settings, trusted, reranker_version
        )
        summary = report.summary()
        summary["temporal_or_acl_leakage_into_candidates"] = leakage
        all_summaries[name] = summary
        per_channel[name] = {
            "config": report.config,
            "per_question": [
                {
                    "question_id": r.question_id,
                    "hit_rank": r.hit_rank,
                    "recall_at": r.recall_at,
                    "all_required_at": {str(k): v for k, v in r.all_required_at.items()},
                    "unmapped_keys": r.unmapped_keys,
                }
                for r in report.question_results
            ],
        }
        print(f"\n===== {name} =====")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    out_dir = PROJECT_ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{dataset.dataset_id}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"

    # P0-5：可复现 Manifest——dataset hash、迁移版本、Collection 点数、模型版本
    from creditlens.common.hashing import sha256_bytes

    manifest = {
        "dataset_file": str(dataset_path.name),
        "dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
        "embedding_version": settings.effective_embedding_version,
        "sparse_encoder_version": settings.sparse_encoder_version,
        "reranker_version": reranker.version,
        "collection": settings.chunks_collection_name,
        "collection_point_count": qdrant.count(settings.chunks_collection_name).count,
        "summaries_point_count": qdrant.count(settings.summaries_collection_name).count,
        "generated_at": utc_now().isoformat(),
    }
    try:
        from sqlalchemy import text as sa_text

        async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
            manifest["alembic_revision"] = await session.scalar(
                sa_text("SELECT version_num FROM alembic_version")
            )
    except Exception:
        manifest["alembic_revision"] = "n/a (create_all baseline)"

    out_path.write_text(
        json.dumps(
            {"manifest": manifest, "summaries": all_summaries, "channels": per_channel},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入: {out_path}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "smoke_v1.json",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dataset))
