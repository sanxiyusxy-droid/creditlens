"""WP1/WP2 检索安全与深度 RAG 专项测试（SQLite + 内存 Qdrant）。

覆盖：
- Exact Route：旧政策有效期、未来材料 cutoff、跨案件隔离、非 Snapshot ParseRun；
- Exact Route：rejected 候选不携带未授权原文（text 恒为 ""）；
- Summary Navigation：L0 -> L1 -> Leaf 递归下钻，摘要本身不作为候选。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from creditlens.application.snapshot_service import SnapshotContext
from creditlens.infrastructure.llm.embedding import HashEmbedding
from creditlens.infrastructure.postgres.models import CreditCase, Entity
from creditlens.infrastructure.qdrant.collections import CollectionManager
from creditlens.retrieval.orchestrator import OrchestratorConfig, RetrievalOrchestrator
from creditlens.retrieval.summary_navigation import SummaryNavigator
from tests.e2e.test_ingest_retrieve_e2e import (
    TENANT_ID,
    _upload_cmd,
)

COLLECTION = "credit_chunks_v1"
SUMMARY_COLLECTION = "credit_summaries_v1"


@pytest.fixture
async def seeded_with_summaries(session, qdrant, object_store, policy_pdf_bytes):
    """入库同时构建 L0/L1 摘要 Collection。"""
    from creditlens.infrastructure.postgres.models import Tenant
    from creditlens.ingestion.index_worker import IndexWorker, count_pending
    from creditlens.ingestion.pipeline import IngestionPipeline, activate_parse_run_if_complete
    from creditlens.ingestion.upload_service import UploadService

    session.add(Tenant(id=TENANT_ID, name="测试租户"))
    borrower_id = uuid.uuid4()
    case_id = uuid.uuid4()
    session.add(
        Entity(
            id=borrower_id,
            tenant_id=TENANT_ID,
            entity_type="COMPANY",
            canonical_name="示例制造有限公司",
        )
    )
    session.add(
        CreditCase(
            id=case_id,
            tenant_id=TENANT_ID,
            case_number="tc-wp1",
            borrower_entity_id=borrower_id,
            product_code="working_capital",
            requested_amount=Decimal("5000000.00"),
            application_date=date(2026, 6, 30),
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
        )
    )
    await session.flush()

    embedder = HashEmbedding()
    manager = CollectionManager(qdrant, dense_dim=embedder.dim)
    manager.ensure_collection(COLLECTION)
    manager.ensure_collection(SUMMARY_COLLECTION)

    upload = UploadService(object_store, "creditlens-raw")
    result = await upload.upload(session, _upload_cmd(policy_pdf_bytes, case_id))
    pipeline = IngestionPipeline(
        object_store,
        target_collection_name=COLLECTION,
        embedding_version=embedder.version,
        summary_collection_name=SUMMARY_COLLECTION,
    )
    ingest = await pipeline.ingest(session, result.document_version_id)
    worker = IndexWorker(qdrant, embedder)
    while await count_pending(session) > 0:
        await worker.process_batch(session)
    await activate_parse_run_if_complete(session, ingest.parse_run_id)
    await session.commit()
    return {"case_id": case_id, "embedder": embedder, "parse_run_id": ingest.parse_run_id}


async def _trusted_from_db(session, case_id, **overrides):
    from creditlens.application.trusted_context import build_trusted_context

    trusted = await build_trusted_context(session, TENANT_ID, case_id)
    return trusted.model_copy(update=overrides) if overrides else trusted


def _orchestrator(qdrant, embedder):
    return RetrievalOrchestrator(qdrant=qdrant, embedder=embedder, reranker=None)


# ==================== WP1：Exact Route 安全收口 ====================


async def test_exact_route_returns_verified_text(session, qdrant, seeded_with_summaries):
    """正常路径：Exact 命中并通过回表复核后携带授权原文。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(session, seeded_with_summaries["case_id"])
    candidates, _rejected = await orch._exact_match(session, trusted, ["资产负债率"])
    assert candidates, "Exact Route 应命中政策条款"
    assert all(c.text for c in candidates), "通过复核后必须回填原文"
    assert all(c.channel == "EXACT" for c in candidates)


async def test_exact_route_blocks_expired_policy(session, qdrant, seeded_with_summaries):
    """旧政策：as_of_date 早于 valid_from -> OUT_OF_EFFECTIVE_DATE，且不带原文。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(
        session,
        seeded_with_summaries["case_id"],
        as_of_date=date(2025, 12, 31),  # 政策 2026-01-01 生效
        decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
    )
    candidates, rejected = await orch._exact_match(session, trusted, ["资产负债率"])
    assert candidates == []
    assert rejected, "过期政策必须进入 rejected 而非静默消失"
    assert all(r.rejection_reason == "OUT_OF_EFFECTIVE_DATE" for r in rejected)
    # 安全约束：rejected 不得携带未授权原文
    assert all(r.text == "" for r in rejected)


async def test_exact_route_blocks_future_material(session, qdrant, seeded_with_summaries):
    """未来材料：cutoff 早于可获得时间 -> NOT_AVAILABLE_AT_CUTOFF，且不带原文。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(
        session,
        seeded_with_summaries["case_id"],
        as_of_date=date(2025, 6, 30),
        decision_cutoff_at=datetime(2025, 6, 30, tzinfo=UTC),  # 早于材料可获得时间
    )
    candidates, rejected = await orch._exact_match(session, trusted, ["资产负债率"])
    assert candidates == []
    assert rejected
    assert all(
        r.rejection_reason in {"NOT_AVAILABLE_AT_CUTOFF", "OUT_OF_EFFECTIVE_DATE"} for r in rejected
    )
    assert all(r.text == "" for r in rejected)


async def test_exact_route_cross_case_isolated(session, qdrant, seeded_with_summaries):
    """跨案件隔离：未绑定该文档的另一案件检索不到任何条款。"""
    other_entity_id = uuid.uuid4()
    other_case_id = uuid.uuid4()
    session.add(
        Entity(
            id=other_entity_id,
            tenant_id=TENANT_ID,
            entity_type="COMPANY",
            canonical_name="另一公司",
        )
    )
    session.add(
        CreditCase(
            id=other_case_id,
            tenant_id=TENANT_ID,
            case_number="tc-other",
            borrower_entity_id=other_entity_id,
            product_code="working_capital",
            requested_amount=Decimal("1000000.00"),
            application_date=date(2026, 6, 30),
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
        )
    )
    await session.flush()

    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    # 手工构造 trusted（该案件无绑定文档，build_trusted_context 会因无材料失败）
    from creditlens.retrieval.contracts import TrustedRequestContext

    trusted = TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        case_id=other_case_id,
        borrower_entity_id=other_entity_id,
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
        allowed_document_ids=[],
    )
    candidates, rejected = await orch._exact_match(session, trusted, ["资产负债率"])
    assert candidates == []
    assert rejected == [], "跨案件条款根本不应出现在候选或拒绝列表中"


async def test_exact_route_snapshot_enforced(session, qdrant, seeded_with_summaries):
    """非 Snapshot ParseRun：冻结集合之外的解析批次一律拒绝。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(session, seeded_with_summaries["case_id"])
    empty_snapshot = SnapshotContext(
        snapshot_id=uuid.uuid4(),
        allowed_parse_run_ids=[],  # 故意不含任何 Parse Run
        chunks_collection=COLLECTION,
    )
    candidates, rejected = await orch._exact_match(
        session, trusted, ["资产负债率"], snapshot=empty_snapshot
    )
    assert candidates == []
    assert rejected
    assert all(r.rejection_reason == "PARSE_RUN_NOT_IN_SNAPSHOT" for r in rejected)
    assert all(r.text == "" for r in rejected)


# ==================== WP2：Summary L0 -> L1 -> Leaf ====================


async def test_summary_navigation_drills_l0_to_leaf(session, qdrant, seeded_with_summaries):
    """L0 命中必须递归下钻到 Leaf Section；摘要本身永不作为候选。"""
    from creditlens.infrastructure.postgres.models import DocumentSection, SummaryNode

    # 前置：摘要树包含 L0(DOCUMENT) 与 L1(CHAPTER)
    levels = set((await session.scalars(select(SummaryNode.summary_level))).all())
    assert {"DOCUMENT", "CHAPTER"} <= levels, "入库应生成 L0/L1 摘要"

    navigator = SummaryNavigator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(session, seeded_with_summaries["case_id"])
    result = await navigator.retrieve(
        session, trusted, "资产负债率 准入条件 风险预警", SUMMARY_COLLECTION, leaf_top_k=8
    )
    assert result.candidates, "Summary 导航应下钻出 Leaf 候选"
    # 候选必须是叶子 Section（ARTICLE/PARAGRAPH），不是摘要节点
    for candidate in result.candidates:
        section = await session.get(DocumentSection, candidate.section_id)
        assert section is not None
        assert section.section_type in {"ARTICLE", "PARAGRAPH"}
        assert candidate.text, "下钻 Leaf 通过复核后携带原文"
        assert candidate.channel == "SUMMARY"


async def test_orchestrator_summary_route_participates_rrf(session, qdrant, seeded_with_summaries):
    """编排器中 Summary Route 作为独立排名列表参与 RRF，Trace 可见。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(session, seeded_with_summaries["case_id"])
    config = OrchestratorConfig(enable_rerank=False, enable_packing=False)
    result = await orch.retrieve(
        session,
        trusted,
        "资产负债率上限",
        COLLECTION,
        config=config,
        summaries_collection=SUMMARY_COLLECTION,
    )
    assert "SUMMARY" in result.trace["fusion"]["input_lists"]
    summary_traces = [t for t in result.trace["routes"] if t["route"] == "SUMMARY"]
    assert summary_traces
    # reranker=None 且 enable_rerank=False 时不标记降级
    assert result.trace["rerank_degraded"] is False


async def test_orchestrator_rerank_degraded_recorded(session, qdrant, seeded_with_summaries):
    """reranker=None 但请求精排：必须记录 rerank_degraded=True（WP2）。"""
    orch = _orchestrator(qdrant, seeded_with_summaries["embedder"])
    trusted = await _trusted_from_db(session, seeded_with_summaries["case_id"])
    config = OrchestratorConfig(enable_rerank=True, enable_packing=False)
    result = await orch.retrieve(session, trusted, "资产负债率上限", COLLECTION, config=config)
    assert result.trace["rerank_degraded"] is True
    assert result.channel_config["rerank_degraded"] is True
