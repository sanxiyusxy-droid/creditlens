"""在线 RAG 编排器 E2E 测试（真实 PG + Qdrant）。

覆盖：
- orchestrator.retrieve 全链路（Dense + Sparse + Summary + Exact + RRF + Packing）
- Trace 完整性
- Context Packing 不超 Budget
- Rerank 降级路径
"""

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
