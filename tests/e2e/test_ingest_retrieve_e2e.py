"""端到端测试：上传 -> 入库 -> Outbox/Worker -> Dense 检索 -> Evidence Preview。

覆盖第 2 周 DoD：
- 重复入库不重复；
- 政策文档完整闭环；
- 任一候选可打开原始 PDF 页；
- 质量/时点失败不进入候选。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from creditlens.evidence.preview import EvidencePreviewService, candidate_to_evidence_ref
from creditlens.infrastructure.llm.embedding import HashEmbedding
from creditlens.infrastructure.postgres.models import (
    CreditCase,
    DocumentSection,
    Entity,
    IndexOutbox,
    Tenant,
)
from creditlens.infrastructure.qdrant.collections import CollectionManager
from creditlens.ingestion.index_worker import IndexWorker, count_pending
from creditlens.ingestion.pipeline import IngestionPipeline, activate_parse_run_if_complete
from creditlens.ingestion.upload_service import UploadCommand, UploadService
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.retrieval.dense import DenseRetriever

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
COLLECTION = "credit_chunks_v1"


def _upload_cmd(data: bytes, case_id=None) -> UploadCommand:
    return UploadCommand(
        tenant_id=TENANT_ID,
        case_id=case_id,
        logical_key="policy_manufacturing_wc",
        title="合成政策",
        document_type="INTERNAL_POLICY",
        document_role="BANK_POLICY",
        filename="policy.pdf",
        mime_type="application/pdf",
        data=data,
        version_label="2026",
        valid_from=date(2026, 1, 1),
        source_available_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
async def seeded(session, qdrant, object_store, policy_pdf_bytes):
    """完成 上传->入库->索引->激活，返回上下文。"""
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
            case_number="tc-001",
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
    CollectionManager(qdrant, dense_dim=embedder.dim).ensure_collection(COLLECTION)

    upload = UploadService(object_store, "creditlens-raw")
    result = await upload.upload(session, _upload_cmd(policy_pdf_bytes, case_id))

    pipeline = IngestionPipeline(
        object_store, target_collection_name=COLLECTION, embedding_version=embedder.version
    )
    ingest = await pipeline.ingest(session, result.document_version_id)

    worker = IndexWorker(qdrant, embedder)
    while await count_pending(session) > 0:
        await worker.process_batch(session)
    await activate_parse_run_if_complete(session, ingest.parse_run_id)
    await session.commit()
    return {
        "upload": result,
        "ingest": ingest,
        "embedder": embedder,
        "case_id": case_id,
    }


def _trusted(**overrides) -> TrustedRequestContext:
    defaults = dict(
        request_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
    )
    defaults.update(overrides)
    return TrustedRequestContext(**defaults)


async def _trusted_from_db(session, case_id, **overrides) -> TrustedRequestContext:
    """v0.2：可信上下文由服务端从案件绑定派生（含 allowed_document_ids）。"""
    from creditlens.application.trusted_context import build_trusted_context

    trusted = await build_trusted_context(session, TENANT_ID, case_id)
    return trusted.model_copy(update=overrides) if overrides else trusted


async def test_upload_dedup(session, object_store, policy_pdf_bytes, seeded):
    """同租户相同内容：复用原始对象，业务绑定单独保存。"""
    upload = UploadService(object_store, "creditlens-raw")
    second = await upload.upload(session, _upload_cmd(policy_pdf_bytes))
    assert second.deduplicated is True
    assert second.object_uri == seeded["upload"].object_uri
    assert second.document_version_id != seeded["upload"].document_version_id


async def test_reingest_is_idempotent(session, object_store, seeded):
    """相同 version + parser + config 不重复解析，不新增 Outbox。"""
    before = await session.scalar(select(func.count()).select_from(IndexOutbox))
    pipeline = IngestionPipeline(
        object_store, target_collection_name=COLLECTION, embedding_version="hash-embed-v1"
    )
    again = await pipeline.ingest(session, seeded["upload"].document_version_id)
    assert again.reused is True
    after = await session.scalar(select(func.count()).select_from(IndexOutbox))
    assert before == after


async def test_dense_retrieval_returns_verified_candidates(session, qdrant, seeded):
    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(session, seeded["case_id"])
    result = await retriever.retrieve(
        session, trusted, "资产负债率不得高于多少？", COLLECTION, top_k=10
    )
    assert result.candidates, "应有通过回表复核的候选"
    texts = " ".join(c.text for c in result.candidates[:5])
    assert "资产负债率" in texts
    for candidate in result.candidates:
        assert candidate.rejection_reason is None
        assert candidate.text  # 回表后携带授权原文


async def test_temporal_cutoff_blocks_future_material(session, qdrant, seeded):
    """decision_cutoff_at 早于材料可获得时间 => 候选为空（Temporal Leakage = 0）。"""
    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(
        session,
        seeded["case_id"],
        as_of_date=date(2025, 6, 30),
        decision_cutoff_at=datetime(2025, 6, 30, tzinfo=UTC),  # 政策 2026-01-01 才可获得
    )
    result = await retriever.retrieve(session, trusted, "资产负债率要求", COLLECTION, top_k=10)
    assert result.candidates == []


async def test_policy_validity_checked_on_verify(session, qdrant, seeded):
    """as_of_date 早于政策 valid_from：召回后回表复核必须拦截。"""
    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(
        session,
        seeded["case_id"],
        as_of_date=date(2025, 12, 31),
        decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
    )
    result = await retriever.retrieve(session, trusted, "准入条件", COLLECTION, top_k=10)
    assert result.candidates == []
    assert all(r.rejection_reason == "OUT_OF_EFFECTIVE_DATE" for r in result.rejected if r.rejection_reason)


async def test_cross_tenant_is_isolated(session, qdrant, seeded):
    """越权 payload（篡改 tenant_id）检索不到本租户材料（硬过滤 + 回表双保险）。"""
    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(session, seeded["case_id"], tenant_id=uuid.uuid4())
    result = await retriever.retrieve(session, trusted, "资产负债率", COLLECTION, top_k=10)
    assert result.candidates == []


async def test_evidence_preview_renders_pdf_page(session, qdrant, object_store, seeded):
    """任一候选可打开原始 PDF 页（任务 13 闭环）。"""
    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(session, seeded["case_id"])
    result = await retriever.retrieve(session, trusted, "风险预警", COLLECTION, top_k=5)
    assert result.candidates
    ref = candidate_to_evidence_ref(result.candidates[0])
    preview = EvidencePreviewService(object_store)
    png = await preview.render_page(session, trusted, ref)
    assert png.startswith(b"\x89PNG")


async def test_blocked_quality_never_enters_candidates(session, qdrant, seeded):
    """质量 BLOCKED 的 Section 即使已在索引中，也会被回表复核拦截。"""
    section = (
        await session.scalars(
            select(DocumentSection).where(DocumentSection.section_type == "ARTICLE").limit(1)
        )
    ).one()
    section.quality_status = "BLOCKED"
    await session.flush()

    retriever = DenseRetriever(qdrant, seeded["embedder"])
    trusted = await _trusted_from_db(session, seeded["case_id"])
    result = await retriever.retrieve(session, trusted, section.text[:20], COLLECTION, top_k=30)
    assert all(c.section_id != section.id for c in result.candidates)
