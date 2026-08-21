"""端到端 Multi-Agent 测试（任务 21-27）。

复用 e2e 入库夹具，追加合成财务事实，执行 Supervisor 固定 DAG：
- Policy/Financial 并行产出 Artifact；
- Challenger 反证；Auditor 确定性校验；
- HITL：HUMAN_REVIEW 暂停 -> 人工批准 -> COMPLETED；
- run_events 形成完整 Trace；
- 越权工具调用被 Gateway 拒绝。
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from creditlens.agents.wiring import build_supervisor
from creditlens.application.snapshot_service import freeze_snapshot
from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context
from creditlens.infrastructure.llm.embedding import HashEmbedding
from creditlens.infrastructure.postgres.models import (
    ClaimRecord,
    CreditCase,
    FinancialFact,
    HumanDecision,
    InvocationRecord,
    ReportVersion,
    ReviewRun,
    RunEvent,
    TelemetryOutbox,
)
from creditlens.tools.gateway import ToolCallDeniedError
from tests.e2e.test_ingest_retrieve_e2e import TENANT_ID, seeded  # noqa: F401  复用夹具

COLLECTION = "credit_chunks_v1"
PERIOD_END = date(2025, 12, 31)


@pytest.fixture
async def with_facts(session, seeded):  # noqa: F811
    """为黄金案件借款人补充合成财务事实。"""
    case = (await session.scalars(select(CreditCase))).one()
    values = {
        "total_assets": "100000000",
        "total_liabilities": "65000000",
        "current_assets": "42000000",
        "current_liabilities": "30000000",
    }
    for metric, value in values.items():
        session.add(
            FinancialFact(
                tenant_id=TENANT_ID,
                case_id=case.id,
                entity_id=case.borrower_entity_id,
                metric_code=metric,
                period_end=PERIOD_END,
                period_type="INSTANT",
                value=Decimal(value),
                canonical_value=Decimal(value),
                currency="CNY",
                consolidation_scope="CONSOLIDATED",
                extraction_method="SYNTHETIC",
                # P0-1：可获得时间必须早于审查截止（年报口径 2026-04-30 披露）
                source_available_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
    await session.flush()
    return {"case": case, **seeded}


async def _prepare(session, case):
    """v0.2：服务端派生可信上下文 + 冻结 Snapshot + 组装 Supervisor。"""
    trusted = await build_trusted_context(session, TENANT_ID, case.id)
    snapshot = await freeze_snapshot(
        session, trusted, chunks_collection=COLLECTION, acl_hash=acl_scope_hash(trusted)
    )
    return trusted, snapshot


async def test_full_review_dag(session, qdrant, with_facts):
    case = with_facts["case"]
    trusted, snapshot = await _prepare(session, case)
    supervisor, _gateway = build_supervisor(session, qdrant, HashEmbedding(), snapshot)
    outcome = await supervisor.execute_full_review(session, trusted, snapshot)

    # 固定 DAG 执行到 HUMAN_REVIEW（Challenger 发现冲突）或 COMPLETED
    assert outcome.status in {"HUMAN_REVIEW", "COMPLETED"}
    producers = {a.producer for a in outcome.artifacts}
    assert {"policy_analyst", "financial_analyst", "challenger"} <= producers

    # Financial Claim 携带可重放的 CalculationArtifact
    financial = next(a for a in outcome.artifacts if a.producer == "financial_analyst")
    assert financial.calculations
    assert all(c.status == "CALCULATED" for c in financial.calculations)
    debt = next(c for c in financial.calculations if c.metric_code == "debt_ratio")
    assert debt.result == Decimal("65.00")

    # Policy Claim 全部绑定原文证据
    policy = next(a for a in outcome.artifacts if a.producer == "policy_analyst")
    supported = [c for c in policy.claims if c.verdict == "SUPPORTED"]
    assert supported
    assert all(c.supporting_evidence_ids for c in supported)

    # Auditor：无证据 Claim 不得进入 accepted
    assert outcome.audit is not None
    assert not outcome.audit.replay_failures
    assert outcome.audit.accepted_claim_ids

    # Trace：run_events 覆盖完整状态迁移
    events = (
        await session.scalars(
            select(RunEvent).where(RunEvent.run_id == outcome.run_id).order_by(RunEvent.sequence_no)
        )
    ).all()
    types = [e.event_type for e in events]
    assert "STATE_CHANGED" in types
    assert "AUDIT_COMPLETED" in types
    tool_events = [event for event in events if event.event_type.startswith("TOOL_INVOCATION_")]
    invocations = (
        await session.scalars(
            select(InvocationRecord)
            .where(InvocationRecord.run_id == outcome.run_id)
            .order_by(InvocationRecord.ended_at, InvocationRecord.invocation_id)
        )
    ).all()
    deliveries = (
        await session.scalars(
            select(TelemetryOutbox).where(TelemetryOutbox.run_id == outcome.run_id)
        )
    ).all()
    assert tool_events == [], "v2 production path must not dual-write legacy RunEvents"
    assert len(invocations) == len(_gateway.calls)
    assert len(deliveries) == len(invocations)
    assert [event.sequence_no for event in events] == list(range(1, len(events) + 1))
    persisted_invocation_ids = {str(record.invocation_id) for record in invocations}
    assert persisted_invocation_ids == {
        str(call.invocation_id) for call in _gateway.calls if call.invocation_id is not None
    }
    assert all(record.tenant_id == trusted.tenant_id for record in invocations)
    assert all(record.case_id == trusted.case_id for record in invocations)
    assert {delivery.status for delivery in deliveries} == {"PENDING"}
    transitions = [
        (e.payload_redacted["from"], e.payload_redacted["to"])
        for e in events
        if e.event_type == "STATE_CHANGED"
    ]
    assert transitions[0] == ("RECEIVED", "AUTHORIZED")


async def test_hitl_resume(session, qdrant, with_facts):
    case = with_facts["case"]
    trusted, snapshot = await _prepare(session, case)
    supervisor, _ = build_supervisor(session, qdrant, HashEmbedding(), snapshot)
    outcome = await supervisor.execute_full_review(session, trusted, snapshot)
    if outcome.status != "HUMAN_REVIEW":
        # 未触发人工复核时走的是另一条合法路径：自动完成且报告已持久化。
        # 不 skip（skip 会掩盖回归），改为断言该路径本身正确。
        assert outcome.status == "COMPLETED", (
            f"非 HUMAN_REVIEW 时只允许 COMPLETED，实际 {outcome.status}"
        )
        report = await session.scalar(
            select(ReportVersion).where(ReportVersion.run_id == outcome.run_id)
        )
        assert report is not None, "COMPLETED 必须已持久化报告版本"
        return

    pending = outcome.audit.needs_human_review_claim_ids
    run = await session.get(ReviewRun, outcome.run_id)
    decision = HumanDecision(
        tenant_id=TENANT_ID,
        case_id=case.id,
        run_id=outcome.run_id,
        target_claim_ids=[str(c) for c in pending],
        action="APPROVE_CLAIM",
        reason_code="REVIEWED_OK",
        reason="正反证据已人工核对",
        # P1：幂等键 + 乐观锁为必填契约
        idempotency_key=f"e2e-approve-{outcome.run_id}",
        target_version=run.state_version,
    )
    status = await supervisor.resume_after_human_review(session, outcome.run_id, decision)
    assert status == "COMPLETED"
    for claim_id in pending:
        claim = await session.get(ClaimRecord, claim_id)
        assert claim.review_status == "HUMAN_APPROVED"


async def test_gateway_denies_unauthorized_tool(session, qdrant, with_facts):
    case = with_facts["case"]
    trusted, snapshot = await _prepare(session, case)
    _, gateway = build_supervisor(session, qdrant, HashEmbedding(), snapshot)
    # Policy Agent 不许调用财务计算工具（文档 §10.2）
    with pytest.raises(ToolCallDeniedError):
        await gateway.invoke(
            "policy_analyst",
            "compute_metric",
            trusted=trusted,
            metric_code="debt_ratio",
            formula_version="1.0",
            period_end=PERIOD_END,
        )
    denied = [c for c in gateway.calls if c.status == "DENIED"]
    assert denied, "被拒绝的调用必须进入审计记录"


async def test_missing_facts_yield_insufficient_evidence(session, qdrant, seeded):  # noqa: F811
    """不补充财务事实：Financial Agent 必须输出 MISSING_INPUT，不估算。"""
    case = (await session.scalars(select(CreditCase))).one()
    trusted, snapshot = await _prepare(session, case)
    supervisor, _ = build_supervisor(session, qdrant, HashEmbedding(), snapshot)
    outcome = await supervisor.execute_full_review(session, trusted, snapshot)
    financial = next(a for a in outcome.artifacts if a.producer == "financial_analyst")
    assert all(c.status == "MISSING_INPUT" for c in financial.calculations)
    assert all(c.verdict == "INSUFFICIENT_EVIDENCE" for c in financial.claims)
