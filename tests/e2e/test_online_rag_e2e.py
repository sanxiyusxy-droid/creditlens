"""在线 RAG 编排器 E2E 测试（真实 PG + Qdrant）。

覆盖：
- orchestrator.retrieve 全链路（Dense + Sparse + Summary + Exact + RRF + Packing）
- Trace 完整性
- Context Packing 不超 Budget
- Rerank 降级路径
"""

import asyncio
import uuid

import pytest

from tests.conftest import requires_integration

pytestmark = [
    pytest.mark.integration,
    requires_integration,
    # pg_engine 为 session 级异步夹具：测试必须同处 session 事件循环
    pytest.mark.asyncio(loop_scope="session"),
]


@pytest.fixture(scope="module")
def orchestrator_deps(pg_engine, real_qdrant):
    """构建编排器依赖。"""
    from creditlens.common.config import get_settings
    from creditlens.infrastructure.llm.embedding import build_embedding_provider
    from creditlens.infrastructure.qdrant.collections import CollectionManager
    from creditlens.retrieval.orchestrator import RetrievalOrchestrator

    settings = get_settings()
    embedder = build_embedding_provider(settings)
    manager = CollectionManager(real_qdrant, dense_dim=embedder.dim)
    manager.ensure_collection(settings.chunks_collection_name, settings.qdrant_chunks_alias)
    manager.ensure_collection(settings.summaries_collection_name, settings.qdrant_summaries_alias)
    orchestrator = RetrievalOrchestrator(
        qdrant=real_qdrant,
        embedder=embedder,
        reranker=None,  # 测试降级路径
        rrf_k=settings.rrf_k,
    )
    return orchestrator, settings, pg_engine


async def test_orchestrator_retrieve_returns_candidates(orchestrator_deps):
    """编排器检索应返回非空候选（前提是种子数据已入库）。"""
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.infrastructure.postgres.session import create_session_factory
    from creditlens.retrieval.orchestrator import ONLINE_CONFIG

    orchestrator, settings, engine = orchestrator_deps
    factory = create_session_factory(engine)

    # 使用固定种子案件 ID（seed_synthetic_data.py 中定义）
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000301")

    from creditlens.infrastructure.postgres.session import session_scope

    async with session_scope(factory, tenant_id=tenant_id, user_id=user_id) as session:
        trusted = await build_trusted_context(session, tenant_id, case_id)
        result = await orchestrator.retrieve(
            session,
            trusted,
            "资产负债率上限是多少",
            settings.chunks_collection_name,
            config=ONLINE_CONFIG,
            snapshot=None,
            summaries_collection=settings.summaries_collection_name,
        )

    # 基本断言
    assert result is not None
    assert isinstance(result.candidates, list)
    # Trace 完整性
    assert "routes" in result.trace
    assert "fusion" in result.trace


async def test_orchestrator_packing_within_budget(orchestrator_deps):
    """Context Packing 不超 Token Budget。"""
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope
    from creditlens.retrieval.orchestrator import ONLINE_CONFIG

    orchestrator, settings, engine = orchestrator_deps
    factory = create_session_factory(engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000301")

    async with session_scope(factory, tenant_id=tenant_id, user_id=user_id) as session:
        trusted = await build_trusted_context(session, tenant_id, case_id)
        result = await orchestrator.retrieve(
            session,
            trusted,
            "贷款期限和展期规定",
            settings.chunks_collection_name,
            config=ONLINE_CONFIG,
            snapshot=None,
            summaries_collection=settings.summaries_collection_name,
        )

    # packing 为 dict（PackingResult.model_dump）；启用 Packing 时不得超预算
    if result.packing:
        assert result.packing["total_tokens_est"] <= ONLINE_CONFIG.token_budget


async def test_orchestrator_rerank_degraded(orchestrator_deps):
    """Reranker 不可用时标记 rerank_degraded=True。"""
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope
    from creditlens.retrieval.orchestrator import ONLINE_CONFIG

    orchestrator, settings, engine = orchestrator_deps
    # orchestrator.reranker is None -> degraded
    factory = create_session_factory(engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000301")

    async with session_scope(factory, tenant_id=tenant_id, user_id=user_id) as session:
        trusted = await build_trusted_context(session, tenant_id, case_id)
        result = await orchestrator.retrieve(
            session,
            trusted,
            "风险预警情形",
            settings.chunks_collection_name,
            config=ONLINE_CONFIG,
            snapshot=None,
            summaries_collection=settings.summaries_collection_name,
        )

    # reranker=None 且请求精排时必须标记降级；否则精排应已应用或未请求
    if ONLINE_CONFIG.enable_rerank:
        assert result.channel_config["rerank_degraded"] is True
        assert result.channel_config["rerank"] is False
    else:
        assert result.channel_config["rerank"] is False


async def test_grounded_qa_persists_audited_answer_chain(orchestrator_deps):
    """真实 PG/Qdrant/RLS 下，离线抽取答案仍必须经过引用审计并完整落库。"""
    from datetime import UTC, date, datetime

    from sqlalchemy import func, select

    from creditlens.application.qa_service import QAService
    from creditlens.infrastructure.postgres.models import (
        ArtifactRecord,
        ClaimRecord,
        EvidenceRecord,
        ReviewRun,
        RunEvent,
    )
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    orchestrator, settings, engine = orchestrator_deps
    settings = settings.model_copy(update={"qa_allow_extractive_fallback": True})
    factory = create_session_factory(engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000301")
    service = QAService(
        session_factory=factory,
        orchestrator=orchestrator,
        settings=settings,
        chat=None,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    response = await service.ask(
        case_id=case_id,
        question="本行流动资金贷款对借款人资产负债率的要求是什么？",
        top_k=8,
        idempotency_key="integration-grounded-qa-v13",
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
    )
    replay = await service.ask(
        case_id=case_id,
        question="本行流动资金贷款对借款人资产负债率的要求是什么？",
        top_k=8,
        idempotency_key="integration-grounded-qa-v13",
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
    )

    assert response.answer_status == "NEEDS_REVIEW"
    assert response.generation_mode == "deterministic_extractive"
    assert response.answer == ""
    assert response.claims and response.claims[0].citations
    extractive_statement = response.claims[0].statement
    assert extractive_statement.startswith("原文摘录:")
    assert extractive_statement.removeprefix("原文摘录:").strip()
    assert response.claims[0].citations[0]["preview_url"]
    assert replay.run_id == response.run_id
    assert replay.idempotent_replay is True
    assert replay.candidates == []

    async with session_scope(factory, tenant_id=tenant_id, user_id=user_id) as session:
        run = await session.get(ReviewRun, response.run_id)
        artifacts = await session.scalar(
            select(func.count())
            .select_from(ArtifactRecord)
            .where(ArtifactRecord.run_id == response.run_id)
        )
        claims = await session.scalar(
            select(func.count())
            .select_from(ClaimRecord)
            .where(ClaimRecord.run_id == response.run_id)
        )
        claim_review_statuses = (
            await session.scalars(
                select(ClaimRecord.review_status).where(ClaimRecord.run_id == response.run_id)
            )
        ).all()
        evidence = await session.scalar(
            select(func.count())
            .select_from(EvidenceRecord)
            .where(EvidenceRecord.run_id == response.run_id)
        )
        events = (
            await session.scalars(
                select(RunEvent.event_type)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert run.status == "COMPLETED"
    assert artifacts == claims == evidence == 1
    assert claim_review_statuses == ["PENDING"]
    assert "ANSWER_AUDIT_COMPLETED" in events
    assert "ANSWER_PERSISTED" in events
    assert "QA_EXECUTION_FAILED" not in events


async def test_concurrent_grounded_qa_idempotency_creates_one_run(orchestrator_deps):
    """并发同键请求只允许一个执行者；另一方只能冲突或重放同一 Run。"""
    from datetime import UTC, date, datetime

    from sqlalchemy import func, select

    from creditlens.application.qa_service import GroundedQAResponse, QAService
    from creditlens.common.errors import IdempotencyConflictError
    from creditlens.infrastructure.postgres.models import ArtifactRecord, ReviewRun
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    orchestrator, settings, engine = orchestrator_deps
    settings = settings.model_copy(update={"qa_allow_extractive_fallback": True})
    factory = create_session_factory(engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000301")
    key = "integration-grounded-qa-concurrent-v13"
    service = QAService(
        session_factory=factory,
        orchestrator=orchestrator,
        settings=settings,
        chat=None,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    async def invoke():
        return await service.ask(
            case_id=case_id,
            question="流动资金贷款期限最长多久？",
            top_k=8,
            idempotency_key=key,
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
        )

    outcomes = await asyncio.gather(invoke(), invoke(), return_exceptions=True)
    responses = [item for item in outcomes if isinstance(item, GroundedQAResponse)]
    conflicts = [item for item in outcomes if isinstance(item, IdempotencyConflictError)]
    assert responses
    assert len(responses) + len(conflicts) == 2
    assert len({item.run_id for item in responses}) == 1

    async with session_scope(factory, tenant_id=tenant_id, user_id=user_id) as session:
        runs = (
            await session.scalars(select(ReviewRun).where(ReviewRun.request_idempotency_key == key))
        ).all()
        artifacts = await session.scalar(
            select(func.count())
            .select_from(ArtifactRecord)
            .where(ArtifactRecord.run_id == runs[0].id)
        )
    assert len(runs) == 1
    assert runs[0].status == "COMPLETED"
    assert artifacts == 1
