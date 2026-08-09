"""HITL 真实并发安全集成测试（真实 PostgreSQL 行锁 + 唯一约束）。

背景（P1）：此前 HITL 只有"顺序幂等"——同一 Session 顺序提交时正确，
但两个**并发**请求（各自不同 idempotency_key、基于同一 state_version）
可能同时通过乐观锁校验，从而重复推进状态。

本测试用两个独立 Session 真正并发提交，验证：
- 恰好一个决定生效，另一个得到 REVIEW_CONFLICT（409）或幂等返回；
- 数据库中只有生效的那条 HumanDecision 落库；
- state_version 只前进一步（不被两次决定重复推进）；
- 同一 idempotency_key 并发提交由唯一约束兜底，不产生第二条记录。

行锁（SELECT ... FOR UPDATE）只在真实 PostgreSQL 生效，
SQLite 会被 SQLAlchemy 忽略——因此这些断言必须在集成阶段执行。
"""

import asyncio
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from tests.conftest import requires_integration

pytestmark = [
    pytest.mark.integration,
    requires_integration,
    pytest.mark.asyncio(loop_scope="session"),
]

TENANT = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
CASE = uuid.UUID("00000000-0000-0000-0000-0000000000c2")
ENTITY = uuid.UUID("00000000-0000-0000-0000-0000000000c3")
USER = uuid.UUID("00000000-0000-0000-0000-0000000000c4")
AS_OF = date(2026, 3, 31)
CUTOFF = datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)


async def _make_review_world(factory):
    """构造一个停在 HUMAN_REVIEW、带两条 blocking Claim 的 Run。"""
    from creditlens.agents.contracts import AgentArtifact, AgentClaim
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )
    from creditlens.infrastructure.postgres.models import (
        AppUser,
        ArtifactRecord,
        CaseMembership,
        ClaimRecord,
        CreditCase,
        Entity,
        ReviewRun,
        Tenant,
    )
    from creditlens.infrastructure.postgres.session import session_scope

    async with session_scope(factory, tenant_id=TENANT, user_id=USER) as session:
        if await session.get(Tenant, TENANT) is None:
            session.add(Tenant(id=TENANT, name="并发测试租户"))
            await session.flush()
        if await session.get(AppUser, USER) is None:
            session.add(
                AppUser(
                    id=USER,
                    tenant_id=TENANT,
                    external_subject="concurrency-reviewer",
                    display_name="并发复核员",
                )
            )
        if await session.get(Entity, ENTITY) is None:
            session.add(
                Entity(
                    id=ENTITY,
                    tenant_id=TENANT,
                    entity_type="COMPANY",
                    canonical_name="并发测试借款人",
                )
            )
        await session.flush()
        if await session.get(CreditCase, CASE) is None:
            session.add(
                CreditCase(
                    id=CASE,
                    tenant_id=TENANT,
                    case_number="CONCURRENCY-001",
                    borrower_entity_id=ENTITY,
                    product_code="SME_WORKING_CAPITAL",
                    requested_amount=Decimal("5000000.00"),
                    application_date=AS_OF,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                )
            )
            await session.flush()
            session.add(CaseMembership(case_id=CASE, user_id=USER, case_role="REVIEWER"))

        run = ReviewRun(
            tenant_id=TENANT,
            case_id=CASE,
            status="HUMAN_REVIEW",
            as_of_date=AS_OF,
            decision_cutoff_at=CUTOFF,
        )
        session.add(run)
        await session.flush()

        source_claims = [
            AgentClaim(
                category="DATA_CONFLICT",
                statement=f"并发测试待复核结论 {i}",
                verdict="PARTIALLY_SUPPORTED",
                as_of_date=AS_OF,
            )
            for i in range(2)
        ]
        source_artifact = AgentArtifact(
            run_id=run.id,
            task_id="challenge",
            producer="challenger",
            lifecycle_status="VALIDATED",
            claims=source_claims,
        )
        artifact_payload = source_artifact.model_dump(mode="json", exclude={"output_hash"})
        artifact = ArtifactRecord(
            id=source_artifact.artifact_id,
            tenant_id=TENANT,
            run_id=run.id,
            task_id=source_artifact.task_id,
            artifact_type=source_artifact.producer,
            contract_version=source_artifact.contract_version,
            producer=source_artifact.producer,
            lifecycle_status=source_artifact.lifecycle_status,
            execution_status=source_artifact.execution_status,
            input_hash=source_artifact.input_hash,
            payload=artifact_payload,
            output_hash=canonical_artifact_payload_hash(artifact_payload),
        )
        session.add(artifact)
        await session.flush()
        claims = []
        for source_claim in source_claims:
            claim = ClaimRecord(
                id=source_claim.claim_id,
                tenant_id=TENANT,
                run_id=run.id,
                artifact_id=artifact.id,
                category=source_claim.category,
                statement=source_claim.statement,
                verdict=source_claim.verdict,
                severity=source_claim.severity,
                as_of_date=source_claim.as_of_date,
                uncertainty_reason=source_claim.uncertainty_reason,
                review_status=source_claim.review_status,
                payload={
                    "supporting_evidence_ids": [
                        str(item) for item in source_claim.supporting_evidence_ids
                    ],
                    "opposing_evidence_ids": [
                        str(item) for item in source_claim.opposing_evidence_ids
                    ],
                    "calculation_ids": [str(item) for item in source_claim.calculation_ids],
                    "source_claim_id": (
                        str(source_claim.source_claim_id) if source_claim.source_claim_id else None
                    ),
                },
            )
            session.add(claim)
            claims.append(claim)
        # IDs 先由 Agent contract 生成，再以同一投影写入 Artifact/Claim 记录。
        await session.flush()
        claim_ids = [c.id for c in claims]
        return run.id, run.state_version, claim_ids


async def _submit(factory, run_id, claim_ids, idem, version):
    """在独立 Session 中提交一次人工决定，返回 (status, error_code)。"""
    from creditlens.agents.supervisor import resume_after_human_review
    from creditlens.common.errors import CreditLensError
    from creditlens.infrastructure.postgres.models import HumanDecision
    from creditlens.infrastructure.postgres.session import session_scope

    try:
        async with session_scope(factory, tenant_id=TENANT, user_id=USER) as session:
            decision = HumanDecision(
                tenant_id=TENANT,
                case_id=CASE,
                run_id=run_id,
                target_claim_ids=[str(c) for c in claim_ids],
                action="APPROVE_CLAIM",
                reviewer_id=USER,
                idempotency_key=idem,
                target_version=version,
            )
            status = await resume_after_human_review(session, run_id, decision)
            return status, None
    except CreditLensError as exc:
        # 只返回稳定错误码；失败时附带 message 便于定位（断言用错误码）
        return None, exc.error_code
    except Exception as exc:  # 数据库层冲突（唯一约束/序列化失败）
        return None, type(exc).__name__


async def test_concurrent_distinct_keys_only_one_wins(pg_engine):
    """两个不同幂等键、同一 state_version 的并发决定：只有一个能生效。"""
    from creditlens.infrastructure.postgres.models import HumanDecision, ReviewRun
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    run_id, version, claim_ids = await _make_review_world(factory)

    # 真正并发：两个独立 Session/事务同时提交
    results = await asyncio.gather(
        _submit(factory, run_id, claim_ids, f"c-a-{run_id}", version),
        _submit(factory, run_id, claim_ids, f"c-b-{run_id}", version),
    )
    succeeded = [r for r in results if r[0] is not None]
    rejected = [r for r in results if r[0] is None]

    assert len(succeeded) == 1, f"并发决定必须只有一个生效，实际 {results}"
    assert len(rejected) == 1, f"另一个必须被拒绝，实际 {results}"
    # 落败方的合法拒绝形态（三者都证明并发被正确串行化）：
    # - REVIEW_CONFLICT：行锁释放后读到新版本号，乐观锁拒绝；
    # - INVALID_STATE_TRANSITION：先手已把 Run 推进到终结状态；
    # - IntegrityError：数据库唯一约束兜底。
    assert rejected[0][1] in {
        "REVIEW_CONFLICT",
        "INVALID_STATE_TRANSITION",
        "IntegrityError",
    }, rejected[0][1]

    async with session_scope(factory, tenant_id=TENANT, user_id=USER) as session:
        count = await session.scalar(
            select(func.count()).select_from(HumanDecision).where(HumanDecision.run_id == run_id)
        )
        assert count == 1, "只有生效的那条决定可以落库"
        run = await session.get(ReviewRun, run_id)
        # 生效方推进状态；被拒方不得再次推进
        assert run.state_version > version
        assert run.status in {"COMPLETED", "HUMAN_REVIEW"}


async def test_concurrent_same_key_is_idempotent(pg_engine):
    """同一幂等键并发提交：唯一约束兜底，只落一条决定，不重复应用。"""
    from creditlens.infrastructure.postgres.models import HumanDecision
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    run_id, version, claim_ids = await _make_review_world(factory)
    same_key = f"same-{run_id}"

    results = await asyncio.gather(
        _submit(factory, run_id, claim_ids, same_key, version),
        _submit(factory, run_id, claim_ids, same_key, version),
    )
    # 允许的结果组合：一个成功 + 一个成功（幂等返回）或被冲突拒绝，
    # 但数据库里必须只有一条记录
    async with session_scope(factory, tenant_id=TENANT, user_id=USER) as session:
        count = await session.scalar(
            select(func.count())
            .select_from(HumanDecision)
            .where(HumanDecision.run_id == run_id, HumanDecision.idempotency_key == same_key)
        )
        assert count == 1, f"同一幂等键并发提交只能落一条决定，实际 {count}（{results}）"


async def test_stale_version_rejected_after_first_decision(pg_engine):
    """第一次决定推进版本后，用旧版本再提交必须 409（顺序场景的回归锁定）。"""
    from creditlens.infrastructure.postgres.session import create_session_factory

    factory = create_session_factory(pg_engine)
    run_id, version, claim_ids = await _make_review_world(factory)

    first_status, first_error = await _submit(
        factory, run_id, claim_ids[:1], f"seq-1-{run_id}", version
    )
    assert first_error is None, first_error
    assert first_status == "HUMAN_REVIEW", "仍有 blocking Claim 时不应完成"

    # 沿用旧版本提交第二个决定 -> 冲突
    _, second_error = await _submit(factory, run_id, claim_ids[1:], f"seq-2-{run_id}", version)
    assert second_error == "REVIEW_CONFLICT", second_error
