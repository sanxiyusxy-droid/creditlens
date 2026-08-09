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
from creditlens.common.errors import (
    ConcurrentReviewConflictError,
    IdempotencyConflictError,
    InvalidReviewRequestError,
    InvalidStateTransitionError,
)
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


async def _add_real_artifact_claim(session, tenant_id, run, review_status):
    from creditlens.agents.contracts import AgentArtifact, AgentClaim
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )

    source_claim = AgentClaim(
        category="DATA_CONFLICT",
        statement="正反证据数值不一致。",
        verdict="PARTIALLY_SUPPORTED",
        as_of_date=AS_OF,
    )
    source_artifact = AgentArtifact(
        run_id=run.id,
        task_id="challenge",
        producer="challenger",
        lifecycle_status="VALIDATED",
        claims=[source_claim],
    )
    payload = source_artifact.model_dump(mode="json", exclude={"output_hash"})
    artifact = ArtifactRecord(
        id=source_artifact.artifact_id,
        tenant_id=tenant_id,
        run_id=run.id,
        task_id=source_artifact.task_id,
        artifact_type=source_artifact.producer,
        producer=source_artifact.producer,
        lifecycle_status=source_artifact.lifecycle_status,
        payload=payload,
        output_hash=canonical_artifact_payload_hash(payload),
    )
    session.add(artifact)
    await session.flush()
    claim = ClaimRecord(
        id=source_claim.claim_id,
        tenant_id=tenant_id,
        run_id=run.id,
        artifact_id=artifact.id,
        category=source_claim.category,
        statement=source_claim.statement,
        verdict=source_claim.verdict,
        as_of_date=source_claim.as_of_date,
        review_status=review_status,
        payload={
            "supporting_evidence_ids": [],
            "opposing_evidence_ids": [],
            "calculation_ids": [],
            "source_claim_id": None,
        },
    )
    session.add(claim)
    await session.flush()
    return claim


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
    claim = await _add_real_artifact_claim(session, TENANT, run, claim_status)
    return run, claim


def _decision(run, *, action="APPROVE_CLAIM", claim_ids=None, idem=None, version=None):
    """构造人工决定。

    P1：幂等键与乐观锁在服务层为必填契约，测试默认填入合法值；
    需要验证"缺失"行为的用例显式传 idem=""/version=None。
    """
    return HumanDecision(
        tenant_id=TENANT,
        case_id=CASE,
        run_id=run.id,
        target_claim_ids=[str(c) for c in (claim_ids or [])],
        action=action,
        reviewer_id=USER,
        idempotency_key=f"auto-{uuid.uuid4()}" if idem is None else idem,
        target_version=run.state_version if version is None else version,
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
    original_version = run.state_version
    first = await resume_after_human_review(
        session,
        run.id,
        _decision(run, claim_ids=[claim.id], idem="dup-key", version=original_version),
    )
    assert first == "COMPLETED"
    # 原请求原样重放（包括 expected version）；即使 Run 已完成也应幂等返回。
    second = await resume_after_human_review(
        session,
        run.id,
        _decision(run, claim_ids=[claim.id], idem="dup-key", version=original_version),
    )
    assert second == "COMPLETED"
    count = await session.scalar(
        select(func.count())
        .select_from(HumanDecision)
        .where(HumanDecision.run_id == run.id, HumanDecision.idempotency_key == "dup-key")
    )
    assert count == 1, "幂等键重复提交不得产生第二条决定"


async def test_hitl_same_key_with_different_payload_conflicts(hitl_session):
    """同一 key 只有请求体完全一致才是重试；复用 key 提交其他动作必须 409。"""
    session = hitl_session
    run, claim = await _make_run(session)
    original_version = run.state_version
    await resume_after_human_review(
        session,
        run.id,
        _decision(run, action="REQUEST_CHANGES", idem="reused", version=original_version),
    )

    with pytest.raises(IdempotencyConflictError):
        await resume_after_human_review(
            session,
            run.id,
            _decision(
                run,
                action="APPROVE_CLAIM",
                claim_ids=[claim.id],
                idem="reused",
                version=original_version,
            ),
        )


async def test_hitl_resolved_claim_cannot_be_silently_overwritten(hitl_session):
    """普通审批动作不得把已经批准的 Claim 再改为拒绝。"""
    session = hitl_session
    run, first_claim = await _make_run(session)
    second_claim = ClaimRecord(
        tenant_id=TENANT,
        run_id=run.id,
        artifact_id=first_claim.artifact_id,
        category="DATA_CONFLICT",
        statement="另一条仍待人工处理的冲突。",
        verdict="PARTIALLY_SUPPORTED",
        as_of_date=AS_OF,
        review_status="PENDING",
        payload={},
    )
    session.add(second_claim)
    await session.flush()

    await resume_after_human_review(
        session,
        run.id,
        _decision(run, claim_ids=[first_claim.id], idem="approve-first"),
    )
    await session.refresh(run)
    assert run.status == "HUMAN_REVIEW"
    assert second_claim.review_status == "PENDING"

    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(
            session,
            run.id,
            _decision(
                run,
                action="REJECT_CLAIM",
                claim_ids=[first_claim.id],
                idem="overwrite-first",
            ),
        )
    await session.refresh(first_claim)
    assert first_claim.review_status == "HUMAN_APPROVED"


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


# ====================== P1：并发安全契约 ======================


async def test_hitl_requires_idempotency_key(hitl_session):
    """缺 idempotency_key -> 422（并发/重试安全前提，不允许静默放行）。"""
    session = hitl_session
    run, claim = await _make_run(session)
    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(
            session, run.id, _decision(run, claim_ids=[claim.id], idem="")
        )


async def test_hitl_requires_expected_state_version(hitl_session):
    """缺 expected_state_version -> 422（否则乐观锁形同虚设）。"""
    session = hitl_session
    run, claim = await _make_run(session)
    decision = _decision(run, claim_ids=[claim.id], idem="no-version")
    decision.target_version = None
    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(session, run.id, decision)


async def test_hitl_unknown_claim_id_rejected(hitl_session):
    """不存在的 Claim ID -> 422，不再静默忽略（否则等于空批准）。"""
    session = hitl_session
    run, _ = await _make_run(session)
    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(
            session, run.id, _decision(run, claim_ids=[uuid.uuid4()], idem="ghost")
        )
    await session.rollback()


async def test_hitl_cross_run_claim_id_rejected(hitl_session):
    """跨 Run 的 Claim ID -> 422（防止在 A Run 上批准 B Run 的结论）。"""
    session = hitl_session
    run_a, _ = await _make_run(session)
    _run_b, claim_b = await _make_run(session)
    claim_b_id = claim_b.id
    await session.commit()  # 固化两个 Run，使回滚只撤销被拒的决定
    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(
            session, run_a.id, _decision(run_a, claim_ids=[claim_b_id], idem="cross")
        )
    await session.rollback()
    # B 的 Claim 未被 A 的决定改动
    reloaded = await session.get(ClaimRecord, claim_b_id)
    assert reloaded.review_status == "PENDING"


async def test_hitl_empty_claim_ids_rejected(hitl_session):
    """APPROVE/REJECT 必须指定目标 Claim -> 空列表 422（防止空批准推进状态）。"""
    session = hitl_session
    run, _ = await _make_run(session)
    with pytest.raises(InvalidReviewRequestError):
        await resume_after_human_review(session, run.id, _decision(run, claim_ids=[], idem="empty"))


async def test_blocked_report_submission_has_no_persisted_side_effects(hitl_session):
    """仍有 blocking Claim 时，失败的提交不能先写决定或推进版本。"""
    session = hitl_session
    run, _ = await _make_run(session)
    original_version = run.state_version

    with pytest.raises(InvalidStateTransitionError):
        await resume_after_human_review(
            session,
            run.id,
            _decision(run, action="SUBMIT_REPORT", idem="blocked-report"),
        )

    await session.refresh(run)
    decisions = await session.scalar(
        select(func.count()).select_from(HumanDecision).where(HumanDecision.run_id == run.id)
    )
    assert decisions == 0
    assert run.state_version == original_version


async def test_hitl_second_decision_sees_bumped_version(hitl_session):
    """第二个不同幂等键的决定必须带新版本号：用旧版本提交 -> 409。

    这是并发安全的核心不变量——即使两个请求各自幂等键不同，
    也不能都基于同一个 state_version 通过校验。
    """
    session = hitl_session
    run, claim = await _make_run(session)
    stale_version = run.state_version
    # 第一个决定（REQUEST_CHANGES：不终结 Run，便于验证后续提交）
    await resume_after_human_review(
        session,
        run.id,
        _decision(run, action="REQUEST_CHANGES", idem="first", version=stale_version),
    )
    await session.refresh(run)
    assert run.state_version > stale_version, "决定应推进 state_version"
    # 第二个决定沿用旧版本 -> 冲突
    with pytest.raises(ConcurrentReviewConflictError):
        await resume_after_human_review(
            session,
            run.id,
            _decision(run, claim_ids=[claim.id], idem="second", version=stale_version),
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
        claim = await _add_real_artifact_claim(session, API_TENANT, run, "PENDING")
        session.add(CaseMembership(case_id=CASE, user_id=API_USER, case_role=role))
        await session.commit()
        return run.id, claim.id, run.state_version


async def test_api_reviewer_can_approve_and_reviewer_injected(api_client):
    """REVIEWER 可审批；reviewer_id 由服务端注入。"""
    client, factory = api_client
    run_id, claim_id, version = await _api_run(factory, "REVIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "idempotency_key": "api-approve-1",
            "expected_state_version": version,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "COMPLETED"
    async with factory() as session:
        decision = (
            await session.scalars(select(HumanDecision).where(HumanDecision.run_id == run_id))
        ).one()
        assert decision.reviewer_id == API_USER, "reviewer 必须服务端注入"


async def test_api_missing_concurrency_fields_rejected(api_client):
    """P1：缺 idempotency_key / expected_state_version -> 422（契约必填）。"""
    client, factory = api_client
    run_id, claim_id, version = await _api_run(factory, "REVIEWER")
    bodies = [
        {"action": "APPROVE_CLAIM", "target_claim_ids": [str(claim_id)]},
        {
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "idempotency_key": "only-key",
        },
        {
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "expected_state_version": version,
        },
    ]
    for body in bodies:
        response = await client.post(f"/api/v1/runs/{run_id}/review-decisions", json=body)
        assert response.status_code == 422, f"{body} 应被拒绝: {response.text}"


async def test_api_cross_run_claim_returns_422(api_client):
    """P1：目标 Claim 不属于该 Run -> 422（不再静默忽略）。"""
    client, factory = api_client
    run_id, _, version = await _api_run(factory, "REVIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(uuid.uuid4())],
            "idempotency_key": "ghost-claim",
            "expected_state_version": version,
        },
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_REVIEW_REQUEST"


async def test_api_viewer_forbidden_403(api_client):
    """VIEWER 无审批权限 -> 403。"""
    client, factory = api_client
    run_id, claim_id, version = await _api_run(factory, "VIEWER")
    response = await client.post(
        f"/api/v1/runs/{run_id}/review-decisions",
        json={
            "action": "APPROVE_CLAIM",
            "target_claim_ids": [str(claim_id)],
            "idempotency_key": "viewer-try",
            "expected_state_version": version,
        },
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
            "idempotency_key": "stale-version",
            "expected_state_version": version + 10,
        },
    )
    assert response.status_code == 409


async def test_api_unimplemented_actions_closed(api_client):
    """WP3：未真正实现的复核动作从 API 删除（RERUN_TASK 等 -> 422）。"""
    client, factory = api_client
    run_id, claim_id, version = await _api_run(factory, "REVIEWER")
    for action in ("RERUN_TASK", "OVERRIDE_WITH_REASON"):
        response = await client.post(
            f"/api/v1/runs/{run_id}/review-decisions",
            # 并发字段填齐，确保 422 来自动作白名单而不是字段缺失
            json={
                "action": action,
                "target_claim_ids": [str(claim_id)],
                "idempotency_key": f"closed-{action}",
                "expected_state_version": version,
            },
        )
        assert response.status_code == 422, f"{action} 不应开放"
