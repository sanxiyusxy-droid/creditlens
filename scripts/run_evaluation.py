"""Recall@K 评测脚本（统一 Orchestrator 消融对比）。

用法：
    uv run python scripts/run_evaluation.py [--dataset evaluation/datasets/frozen_v2.json] [--split test]

WP4 口径：
- 按题目 case_key 建立案件映射（case001/002/003），逐案件冻结 TrustedContext/Snapshot；
- `--split dev|test`：dev 仅用于调参，简历指标只在冻结 test 上报；
- 输出整体 + per-case + 宏平均指标；
- Leakage/unmapped 违规时评测失败，不落盘成看似成功的报告。

消融配置（通过 OrchestratorConfig 控制）：
- E0  Dense-only（关闭 Sparse/Summary/Exact/Rerank）
- E23 Dense + Sparse/BM25 + RRF
- E4  Dense + Summary Navigation
- E5  Dense + Sparse + QuerySpec Rewrite
- E7  全链路（Dense + Sparse + Summary + Exact + Rerank）
输出 Recall@5/10/20、NDCG@K、Precision@K、MRR@10、Retrieved Evidence P/R，报告落盘 evaluation/reports/。
口径说明：无答案生成层，不宣称 Faithfulness / Citation Accuracy / Refusal Accuracy。
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
from creditlens.retrieval.orchestrator import (
    OrchestratorConfig,
    RetrievalOrchestrator,
)
from creditlens.retrieval.rerank import LexicalOverlapReranker, build_reranker
from seed_synthetic_data import (
    CASE_ID,
    CASE_ID_002,
    CASE_ID_003,
    DEMO_USER_ID,
    TENANT_ID,
    seed_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# WP4：case_key → case_id 映射（评测资产用稳定键，运行时映射到 Seed UUID）
CASE_KEY_MAP = {
    "golden_case_001": CASE_ID,
    "golden_case_002": CASE_ID_002,
    "golden_case_003": CASE_ID_003,
}


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
    case_contexts: dict,
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
            # WP4：按题目 case_key 取对应案件的 TrustedContext/Snapshot
            if question.case_key not in case_contexts:
                raise KeyError(f"案件 {question.case_key} 未在 Seed 映射中定义")
            base_trusted, case_snapshot = case_contexts[question.case_key]
            # 评测问题的时点必须与案件上下文一致；request_id 每题独立
            question_trusted = base_trusted.model_copy(
                update={
                    "request_id": uuid.uuid4(),
                    "as_of_date": question.as_of_date,
                    "decision_cutoff_at": question.decision_cutoff_at.astimezone(UTC),
                }
            )
            started = time.perf_counter()
            retrieval = await retrieve_fn(
                session, question_trusted, question.question, case_snapshot
            )
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


def summarize_per_case(dataset: GoldDataset, report: EvalReport) -> tuple[dict, dict]:
    """WP4：按 case_key 分组计算 per-case 指标与案件级宏平均。"""
    question_case = {q.question_id: q.case_key for q in dataset.questions}
    by_case: dict[str, list] = {}
    for result in report.question_results:
        by_case.setdefault(question_case.get(result.question_id, "unknown"), []).append(result)
    per_case: dict[str, dict] = {}
    for case_key in sorted(by_case):
        sub = EvalReport(
            dataset_id=report.dataset_id,
            dataset_version=report.dataset_version,
            config=report.config,
            question_results=by_case[case_key],
        )
        per_case[case_key] = sub.summary()
    # 宏平均：案件级指标再取均值（案件间样本量差异不传递到整体指标）
    macro: dict = {}
    if per_case:
        numeric_keys = {
            key
            for case_summary in per_case.values()
            for key, value in case_summary.items()
            if isinstance(value, float)
        }
        for key in sorted(numeric_keys):
            values = [s[key] for s in per_case.values() if key in s]
            macro[key] = sum(values) / len(values)
    return per_case, macro


async def main(dataset_path: Path, split: str) -> None:
    settings = get_settings()
    dataset = GoldDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))

    # WP4：--split dev|test，dev 仅用于调参，冻结 test 才报简历指标
    all_questions = dataset.questions
    dataset.questions = [q for q in all_questions if q.split == split]
    if not dataset.questions:
        raise SystemExit(f"数据集 {dataset.dataset_id} 中没有 split={split} 的题目")
    print(
        f"数据集 {dataset.dataset_id} split={split}: "
        f"{len(dataset.questions)}/{len(all_questions)} 题，"
        f"案件={sorted({q.case_key for q in dataset.questions})}"
    )

    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    store = build_object_store(settings)
    qdrant = build_qdrant_client(settings)

    await seed_environment(factory, store, qdrant, settings)

    # 服务端派生可信上下文并冻结 Snapshot（WP4：逐案件独立冻结）
    case_contexts: dict[str, tuple] = {}
    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        for case_key in sorted({q.case_key for q in dataset.questions}):
            case_id = CASE_KEY_MAP.get(case_key)
            if case_id is None:
                raise SystemExit(f"题目引用了未知案件 {case_key}")
            trusted = await build_trusted_context(session, TENANT_ID, case_id)
            snapshot = await freeze_snapshot(
                session,
                trusted,
                chunks_collection=settings.chunks_collection_name,
                summaries_collection=settings.summaries_collection_name,
                acl_hash=acl_scope_hash(trusted),
            )
            case_contexts[case_key] = (trusted, snapshot)

    embedder = build_embedding_provider(settings)
    reranker = build_reranker(settings) or LexicalOverlapReranker()

    # 统一使用 RetrievalOrchestrator，通过不同 OrchestratorConfig 实现消融
    orchestrator = RetrievalOrchestrator(
        qdrant=qdrant, embedder=embedder, reranker=reranker, rrf_k=settings.rrf_k
    )

    # 消融配置定义
    channels: list[tuple[str, OrchestratorConfig, str]] = [
        (
            "E0_dense_only",
            OrchestratorConfig(
                final_limit=30,
                enable_sparse=False,
                enable_summary=False,
                enable_exact=False,
                enable_rerank=False,
                enable_packing=False,
            ),
            "-",
        ),
        (
            "E23_hybrid_rrf",
            OrchestratorConfig(
                final_limit=30,
                enable_sparse=True,
                enable_summary=False,
                enable_exact=False,
                enable_rerank=False,
                enable_packing=False,
            ),
            "-",
        ),
        (
            "E4_dense_summary",
            OrchestratorConfig(
                final_limit=30,
                enable_sparse=False,
                enable_summary=True,
                enable_exact=False,
                enable_rerank=False,
                enable_packing=False,
            ),
            "-",
        ),
        (
            "E5_hybrid_rewrite",
            OrchestratorConfig(
                final_limit=30,
                enable_sparse=True,
                enable_summary=False,
                enable_exact=True,
                enable_rerank=False,
                enable_packing=False,
            ),
            "-",
        ),
        (
            "E7_full_orchestrator",
            OrchestratorConfig(
                final_limit=30,
                enable_sparse=True,
                enable_summary=True,
                enable_exact=True,
                enable_rerank=True,
                enable_packing=False,
            ),
            reranker.version,
        ),
    ]

    async def make_retrieve_fn(config: OrchestratorConfig):
        async def retrieve_fn(session, trusted_ctx, query, snapshot):
            # WP4：传入正确的 chunks/summaries Collection（来自案件 Snapshot）
            return await orchestrator.retrieve(
                session,
                trusted_ctx,
                query,
                snapshot.chunks_collection,
                config=config,
                snapshot=snapshot,
                summaries_collection=snapshot.summaries_collection,
            )

        return retrieve_fn

    all_summaries = {}
    per_channel = {}
    hard_failures: list[str] = []
    for name, config, reranker_version in channels:
        retrieve_fn = await make_retrieve_fn(config)
        report, leakage = await evaluate_channel(
            name, retrieve_fn, dataset, factory, settings, case_contexts, reranker_version
        )
        summary = report.summary()
        # 拒绝计数属预期防护行为（正确拦截），不是泄漏；真正泄漏看独立回表审计
        summary["temporal_or_acl_rejections"] = leakage
        # WP6：检索延迟 P50/P95（evaluate_channel 采集，提升到 summary 供对比）
        summary["latency_p50_ms"] = report.config.get("latency_p50_ms", 0.0)
        summary["latency_p95_ms"] = report.config.get("latency_p95_ms", 0.0)
        all_summaries[name] = summary
        per_case, macro = summarize_per_case(dataset, report)
        per_channel[name] = {
            "config": report.config,
            "per_case": per_case,
            "macro_average": macro,
            "per_question": [
                {
                    "question_id": r.question_id,
                    "hit_rank": r.hit_rank,
                    "recall_at": r.recall_at,
                    "ndcg_at": r.ndcg_at,
                    "precision_at": r.precision_at,
                    "all_required_at": {str(k): v for k, v in r.all_required_at.items()},
                    "unmapped_keys": r.unmapped_keys,
                }
                for r in report.question_results
            ],
        }
        print(f"\n===== {name} =====")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        for case_key, case_summary in per_case.items():
            print(
                f"  [{case_key}] recall@10={case_summary.get('recall@10'):.4f} "
                f"ndcg@10={case_summary.get('ndcg@10'):.4f}"
            )

        # WP4：评测失败不能落盘成看似成功的报告（Leakage/unmapped 必须为 0）
        if report.config.get("independent_leakage_violations"):
            hard_failures.append(
                f"{name}: 独立回表审计违规={report.config['independent_leakage_violations']}"
            )
        if summary.get("unmapped_questions"):
            hard_failures.append(f"{name}: 存在 unmapped 证据键={summary['unmapped_questions']}")

    if hard_failures:
        await engine.dispose()
        raise SystemExit(
            "评测失败，未写入报告:\n" + "\n".join(f"  - {msg}" for msg in hard_failures)
        )

    out_dir = PROJECT_ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"ablation_{dataset.dataset_id}_{split}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    # 可复现 Manifest
    from creditlens.common.hashing import sha256_bytes

    manifest = {
        "dataset_file": str(dataset_path.name),
        "dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
        "split": split,
        "question_count": len(dataset.questions),
        "case_keys": sorted({q.case_key for q in dataset.questions}),
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
        default=PROJECT_ROOT / "evaluation" / "datasets" / "frozen_v2.json",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="dev 仅用于调参；冻结 test 才上报简历指标",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.split))
