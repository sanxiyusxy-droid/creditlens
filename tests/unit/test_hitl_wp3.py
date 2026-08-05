"""WP3 HITL 单测与故障注入：幂等/乐观锁/唯一约束/授权/409。

Supervisor 层：
- APPROVE_CLAIM 解决全部 blocking Claim -> REPORTING -> COMPLETED（APPROVED_DRAFT）；
- idempotency_key 重复提交幂等返回，不重复应用；
- expected_state_version 不匹配 -> ConcurrentReviewConflictError（API 映射 409）；
- (run_id, idempotency_key) 数据库唯一约束；
- 非 HUMAN_REVIEW/REWORK 状态拒绝处理。

API 层（ASGI 直测）：
- REVIEWER/OWNER 动作授权；VIEWER 返回 403；
- reviewer 服务端注入，客户端不可自报；
- 并发审批冲突返回 409；
- 未实现的 RERUN_TASK/OVERRIDE_WITH_REASON 不开放（422）。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from creditlens.agents.supervisor import resume_after_human_review
from creditlens.common.errors import ConcurrentReviewConflictError, InvalidStateTransitionError
from creditlens.infrastructure.postgres.models import (
    AppUser,
    ArtifactRecord,
    Base,
    CaseMembership,
    ClaimRecord,
    CreditCase,
    Entity,
    HumanDecision,
    ReportVersion,
    ReviewRun,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory

TENANT = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
CASE = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
ENTITY = uuid.UUID("00000000-0000-0000-0000-0000000000cc")
USER = uuid.UUID("00000000-0000-0000-0000-0000000000dd")
AS_OF = date(2026, 5, 1)
CUTOFF = datetime(2026, 5, 1, tzinfo=UTC)


@pytest.fixture
async def hitl_session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'hitl.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with factory() as session:
        session.add_all(
            [
                Tenant(id=TENANT, name="T"),
                AppUser(
                    id=USER,
                    tenant_id=TENANT,
                    external_subject="hitl-user",
                    display_name="复核员",
                ),
                Entity(
                    id=ENTITY,
                    tenant_id=TENANT,
                    entity_type="COMPANY",
                    canonical_name="示例公司",
                ),
                CreditCase(
                    id=CASE,
                    tenant_id=TENANT,
                    case_number="C-HITL",
                    borrower_entity_id=ENTITY,
                    product_code="working_capital",
                    requested_amount=Decimal("1000000"),
                    application_date=AS_OF,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                ),
            ]
        )
        await session.flush()
        yield session
        await session.commit()
    await engine.dispose()


async def _make_run(session, *, status="HUMAN_REVIEW", claim_status="PENDING"):
    run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        status=status,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    session.add(run)
    await session.flush()
    artifact = ArtifactRecord(
        tenant_id=TENANT,
        run_id=run.id,
        task_id="challenge",
        artifact_type="challenger",
        producer="challenger",
        payload={},
    )
    session.add(artifact)
    await session.flush()
    claim = ClaimRecord(
        tenant_id=TENANT,
        run_id=run.id,
        artifact_id=artifact.id,
        category="DATA_CONFLICT",
        statement="正反证据数值不一致。",
        verdict="PARTIALLY_SUPPORTED",
        as_of_date=AS_OF,
        review_status=claim_status,
        payload={},
    )
    session.add(claim)
    await session.flush()
    return run, claim


def _decision(run, *, action="APPROVE_CLAIM", claim_ids=None, idem=None, version=None):
    return HumanDecision(
        tenant_id=TENANT,
        case_id=CASE,
        run_id=run.id,
        target_claim_ids=[str(c) for c in (claim_ids or [])],
        action=action,
        reviewer_id=USER,
        idempotency_key=idem,
        target_version=version,
    )


# ====================== Supervisor 层 ======================


async def test_hitl_approve_completes_run_with_approved_draft(hitl_session):
    """APPROVE 解决全部 blocking Claim -> COMPLETED，报告 APPROVED_DRAFT。"""
    session = hitl_session
    run, claim = await _make_run(session)
    status = await resume_after_human_review(
        session, run.id, _decision(run, claim_ids=[claim.id], idem="k1")
    )
    assert status == "COMPLETED"
    await session.refresh(claim)
    assert claim.review_status == "HUMAN_APPROVED"
    report = (
        await session.scalars(select(ReportVersion).where(ReportVersion.run_id == run.id))
    ).one()
    assert report.status == "APPROVED_DRAFT", "人工批准后报告必须显式 APPROVED_DRAFT"


async def test_hitl_idempotency_key_prevents_reapply(hitl_session):
    """同一 idempotency_key 重复提交：幂等返回，不重复应用。"""
    session = hitl_session
    run, claim = await _make_run(session)
    first = await resume_after_human_review(
        session, run.id, _decision(run, claim_ids=[claim.id], idem="dup-key")
    )
    assert first == "COMPLETED"
    # 重复提交（即使 Run 已完成也应幂等返回而非报错）
    second = await resume_after_human_review(
        session, run.id, _decision(run, claim_ids=[claim.id], idem="dup-key")
    )
    assert second == "COMPLETED"
    count = await session.scalar(
        select(func.count())
        .select_from(HumanDecision)
        .where(HumanDecision.run_id == run.id, HumanDecision.idempotency_key == "dup-key")
    )
    assert count == 1, "幂等键重复提交不得产生第二条决定"


async def test_hitl_optimistic_lock_conflict(hitl_session):
    """expected_state_version 不匹配 -> ConcurrentReviewConflictError（409）。"""
    session = hitl_session
    run, claim = await _make_run(session)
    with pytest.raises(ConcurrentReviewConflictError):
        await resume_after_human_review(
            session,
            run.id,
            _decision(run, claim_ids=[claim.id], version=run.state_version + 99),
        )
    # 冲突后 Run 状态不变、Claim 未被处理
    await session.refresh(run)
    assert run.status == "HUMAN_REVIEW"
    await session.refresh(claim)
    assert claim.review_status == "PENDING"


async def test_hitl_unique_constraint_db_level(hitl_session):
    """(run_id, idempotency_key) 数据库唯一约束（防并发重复写入）。"""
    session = hitl_session
    run, _ = await _make_run(session)
    session.add(_decision(run, idem="uq-key"))
    await session.flush()
    session.add(_decision(run, idem="uq-key"))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_hitl_wrong_state_rejected(hitl_session):
    """非 HUMAN_REVIEW/REWORK 状态不得处理人工决定。"""
    session = hitl_session
    run, claim = await _make_run(session, status="COMPLETED")
    with pytest.raises(InvalidStateTransitionError):
        await resume_after_human_review(
            session, run.id, _decision(run, claim_ids=[claim.id], idem="late")
        )


# ====================== API 层（ASGI 直测） ======================

# API 固定演示身份（MVP 单租户）
API_TENANT = uuid.UUID("00000000-0000-0000-0000-000000000001")
API_USER = uuid.UUID("00000000-0000-0000-0000-000000000301")


@pytest.fixture
async def api_client(tmp_path, monkeypatch):
    import httpx
    from apps.api import main as api_main

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    monkeypatch.setattr(api_main, "session_factory", factory)

    async with factory() as session:
        session.add_all(
            [
                Tenant(id=API_TENANT, name="Demo"),
                AppUser(
                    id=API_USER,
                    tenant_id=API_TENANT,
                    external_subject="demo",
                    display_name="演示用户",
                ),
                Entity(
                    id=ENTITY,
                    tenant_id=API_TENANT,
                    entity_type="COMPANY",
                    canonical_name="示例公司",
                ),
                CreditCase(
                    id=CASE,
                    tenant_id=API_TENANT,
                    case_number="C-API",
                    borrower_entity_id=ENTITY,
                    product_code="working_capital",
                    requested_amount=Decimal("1000000"),
                    application_date=AS_OF,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                ),
            ]
        )
        await session.commit()

    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, factory
    await engine.dispose()


async def _api_run(factory, role: str):
    """建一个 HUMAN_REVIEW Run + blocking Claim，并授予演示用户案件角色。"""
    async with factory() as session:
        run = ReviewRun(
            tenant_id=API_TENANT,
            case_id=CASE,
            status="HUMAN_REVIEW",
            as_of_date=AS_OF,
            decision_cutoff_at=CUTOFF,
        )
        session.add(run)
        await session.flush()
        artifact = ArtifactRecord(
            tenant_id=API_TENANT,
            run_id=run.id,
            task_id="challenge",
            artifact_type="challenger",
            producer="challenger",
            payload={},
        )
        session.add(artifact)
        await session.flush()
        claim = ClaimRecord(
            tenant_id=API_TENANT,
            run_id=run.id,
            artifact_id=artifact.id,
            category="DATA_CONFLICT",
            statement="正反证据数值不一致。",
            verdict="PARTIALLY_SUPPORTED",
            as_of_date=AS_OF,
            review_status="PENDING",
            payload={},
        )
        session.add(claim)
        session.add(CaseMembership(case_id=CASE, user_id=API_USER, case_role=role))
        await session.commit()
        return run.id, claim.id, run.state_version


async def test_api_reviewer_can_approve_and_reviewer_injected(api_client):
    """REVIEWER 可审批；reviewer_id 由服务端注入。"""
    client, factory = api_client
    run_id, claim_id, _ = await _api_run(factory, "REVIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "idempotency_key": "api-approve-1",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "COMPLETED"
    async with factory() as session:
        decision = (
            await session.scalars(select(HumanDecision).where(HumanDecision.run_id == run_id))
        ).one()
        assert decision.reviewer_id == API_USER, "reviewer 必须服务端注入"


async def test_api_viewer_forbidden_403(api_client):
    """VIEWER 无审批权限 -> 403。"""
    client, factory = api_client
    run_id, claim_id, _ = await _api_run(factory, "VIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={"action": "APPROVE_CLAIM", "target_claim_ids": [str(claim_id)]},
    )
    assert response.status_code == 403


async def test_api_concurrent_conflict_returns_409(api_client):
    """expected_state_version 过期 -> 409 REVIEW_CONFLICT。"""
    client, factory = api_client
    run_id, claim_id, version = await _api_run(factory, "REVIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "expected_state_version": version + 10,
        },
    )
    assert response.status_code == 409


async def test_api_unimplemented_actions_closed(api_client):
    """WP3：未真正实现的复核动作从 API 删除（RERUN_TASK 等 -> 422）。"""
    client, factory = api_client
    run_id, claim_id, _ = await _api_run(factory, "REVIEWER")
    for action in ("RERUN_TASK", "OVERRIDE_WITH_REASON"):
        response = await client.post(
            f"/api/v1/runs/{run_id}/review-decisions",
            json={"action": action, "target_claim_ids": [str(claim_id)]},
        )
        assert response.status_code == 422, f"{action} 不应开放"
