"""Checkpoint + RLS 集成测试（真实 PostgreSQL + NOSUPERUSER NOBYPASSRLS 业务角色）。

验证：
- checkpoint_commit 真实调用：中途提交后 RLS 上下文被重新注入，后续写读仍可见；
  （若未重新注入 SET LOCAL，业务角色在 NOBYPASSRLS 下查询将返回 0 行）
- 跨租户隔离：以租户 B 上下文实际查询租户 A 的 documents，断言不可见；
  且 WITH CHECK 阻止以租户 B 会话写入租户 A 的行；
- session_scope 异常自动回滚。

前置（CI integration job 已准备）：
- infra/postgres/rls_policies.sql 已应用；
- DATABASE_URL 使用 NOSUPERUSER NOBYPASSRLS 的业务角色连接。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from tests.conftest import requires_integration

pytestmark = [
    pytest.mark.integration,
    requires_integration,
    # pg_engine 为 session 级异步夹具：测试必须同处 session 事件循环，
    # 否则 asyncpg 连接跨循环报 "another operation is in progress"
    pytest.mark.asyncio(loop_scope="session"),
]

SEEDED_TENANT = uuid.UUID("00000000-0000-0000-0000-000000000001")
SEEDED_CASE = uuid.UUID("00000000-0000-0000-0000-000000000201")
SECOND_SEEDED_CASE = uuid.UUID("00000000-0000-0000-0000-000000000202")
SEEDED_MEMBER = uuid.UUID("00000000-0000-0000-0000-000000000301")
TENANT_A = SEEDED_TENANT
USER_A = SEEDED_MEMBER
TENANT_B = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
USER_B = uuid.UUID("00000000-0000-0000-0000-0000000000c4")
CASE_B = uuid.UUID("00000000-0000-0000-0000-0000000000c2")


def _assert_policy_or_privilege_denied(error: Exception) -> None:
    message = str(error).lower()
    assert any(
        marker in message
        for marker in ("row-level security", "policy", "permission denied", "insufficientprivilege")
    ), message


async def test_runtime_role_has_frozen_least_privilege_attributes(pg_engine):
    """CI/local provisioning must converge on the same role contract as preflight."""
    async with pg_engine.connect() as connection:
        role = (
            await connection.execute(
                text(
                    "SELECT r.rolsuper, r.rolbypassrls, r.rolcreatedb, r.rolcreaterole, "
                    "r.rolreplication, r.rolinherit, r.rolcanlogin, "
                    "d.datdba = r.oid AS owns_database "
                    "FROM pg_roles AS r "
                    "JOIN pg_database AS d ON d.datname = current_database() "
                    "WHERE r.rolname = current_user"
                )
            )
        ).one()
        assert role == (False, False, False, False, False, False, True, False)

        membership_count = await connection.scalar(
            text(
                "SELECT count(*) FROM pg_auth_members AS membership "
                "JOIN pg_roles AS child ON child.oid = membership.member "
                "WHERE child.rolname = current_user"
            )
        )
        assert membership_count == 0


async def test_business_role_cannot_mutate_append_only_audit_tables(pg_engine):
    """业务角色对审计/证据链表只有 SELECT+INSERT，没有 UPDATE/DELETE。"""
    protected = [
        "run_events",
        "human_decisions",
        "report_versions",
        "evidence",
        "artifacts",
        "invocation_records",
    ]
    async with pg_engine.connect() as connection:
        current_user = await connection.scalar(text("SELECT current_user"))
        assert current_user == "creditlens_app"
        for table in protected:
            qualified = f"public.{table}"
            assert await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, 'SELECT')"),
                {"table": qualified},
            )
            assert await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, 'INSERT')"),
                {"table": qualified},
            )
            assert not await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, 'UPDATE')"),
                {"table": qualified},
            )
            assert not await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, 'DELETE')"),
                {"table": qualified},
            )


async def test_invocation_outbox_privileges_and_parent_binding(pg_engine):
    """Invocation facts stay immutable; Outbox can mutate delivery state only."""
    from datetime import timedelta

    from creditlens.common.clock import utc_now
    from creditlens.infrastructure.postgres.models import (
        InvocationRecord,
        ReviewRun,
        TelemetryOutbox,
    )
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    async with pg_engine.connect() as connection:
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.review_runs', 'SELECT')")
        )
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.review_runs', 'INSERT')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.review_runs', 'UPDATE')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.review_runs', 'DELETE')")
        )
        for column in ("status", "state_version", "model_manifest", "completed_at"):
            assert await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.review_runs', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )
        for column in (
            "id",
            "tenant_id",
            "case_id",
            "run_type",
            "as_of_date",
            "decision_cutoff_at",
            "input_snapshot_id",
            "plan_version",
            "retrieval_config",
            "request_idempotency_key",
            "request_hash",
            "started_at",
            "created_by",
        ):
            assert not await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.review_runs', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )

        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.invocation_records', 'SELECT')")
        )
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.invocation_records', 'INSERT')")
        )
        for privilege in ("UPDATE", "DELETE"):
            assert not await connection.scalar(
                text(
                    "SELECT has_table_privilege(current_user, "
                    "'public.invocation_records', :privilege)"
                ),
                {"privilege": privilege},
            )

        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.telemetry_outbox', 'SELECT')")
        )
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.telemetry_outbox', 'INSERT')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.telemetry_outbox', 'UPDATE')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.telemetry_outbox', 'DELETE')")
        )
        for column in (
            "status",
            "attempts",
            "available_at",
            "locked_at",
            "locked_until",
            "last_error_code",
            "delivered_at",
            "dead_at",
        ):
            assert await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.telemetry_outbox', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )
        for column in ("tenant_id", "case_id", "run_id", "invocation_id", "topic"):
            assert not await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.telemetry_outbox', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )

    factory = create_session_factory(pg_engine)
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        run_a = ReviewRun(
            tenant_id=TENANT_A,
            case_id=SEEDED_CASE,
            status="RECEIVED",
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
        )
        session.add(run_a)
        await session.flush()
        invocation = InvocationRecord(
            tenant_id=TENANT_A,
            case_id=SEEDED_CASE,
            run_id=run_a.id,
            kind="MODEL",
            name="rls_test",
            status="SUCCESS",
            ended_at=utc_now(),
            payload_redacted={},
            payload_sha256="a" * 64,
        )
        second_invocation = InvocationRecord(
            tenant_id=TENANT_A,
            case_id=SEEDED_CASE,
            run_id=run_a.id,
            kind="TOOL",
            name="rls_binding_test",
            status="FAILED",
            ended_at=utc_now(),
            payload_redacted={"error_code": "TEST_FAILURE"},
            payload_sha256="b" * 64,
        )
        session.add_all([invocation, second_invocation])
        await session.flush()
        outbox = TelemetryOutbox(
            tenant_id=TENANT_A,
            case_id=SEEDED_CASE,
            run_id=run_a.id,
            invocation_id=invocation.invocation_id,
        )
        session.add(outbox)
        await session.flush()
        run_a_id = run_a.id
        invocation_id = invocation.invocation_id
        second_invocation_id = second_invocation.invocation_id
        outbox_id = outbox.id

    # Workflow state remains mutable, but the immutable case binding cannot be
    # reassigned even to another case that this same user is authorized to see.
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        result = await session.execute(
            text("UPDATE review_runs SET status = status WHERE id = :run_id"),
            {"run_id": run_a_id},
        )
        assert result.rowcount == 1

    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        with pytest.raises(Exception) as exc_info:
            await session.execute(
                text("UPDATE review_runs SET case_id = :case_id WHERE id = :run_id"),
                {"case_id": SECOND_SEEDED_CASE, "run_id": run_a_id},
            )
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    async with session_scope(factory, tenant_id=TENANT_B, user_id=USER_B) as session:
        run_b = ReviewRun(
            tenant_id=TENANT_B,
            case_id=CASE_B,
            status="RECEIVED",
            as_of_date=date(2026, 3, 31),
            decision_cutoff_at=datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC),
        )
        session.add(run_b)
        await session.flush()
        run_b_id = run_b.id
        assert await session.get(InvocationRecord, invocation_id) is None
        assert await session.get(TelemetryOutbox, outbox_id) is None

    outsider_id = uuid.uuid4()
    async with session_scope(factory, tenant_id=TENANT_A, user_id=outsider_id) as session:
        assert await session.get(InvocationRecord, invocation_id) is None
        assert await session.get(TelemetryOutbox, outbox_id) is None
        session.add(
            InvocationRecord(
                tenant_id=TENANT_A,
                case_id=SEEDED_CASE,
                run_id=run_a_id,
                kind="MODEL",
                name="outsider_write",
                status="SUCCESS",
                ended_at=utc_now(),
                payload_redacted={},
                payload_sha256="d" * 64,
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    async with session_scope(factory, tenant_id=TENANT_A, user_id=outsider_id) as session:
        session.add(
            TelemetryOutbox(
                tenant_id=TENANT_A,
                case_id=SEEDED_CASE,
                run_id=run_a_id,
                invocation_id=second_invocation_id,
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    # Authorized worker lifecycle update succeeds with only the granted columns.
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        row = await session.get(TelemetryOutbox, outbox_id)
        assert row is not None
        now = utc_now()
        row.status = "PROCESSING"
        row.attempts = 1
        row.locked_at = now
        row.locked_until = now + timedelta(seconds=30)
        await session.flush()

    # Identity/content mutation is denied even when it assigns the current value.
    mutation_attempts = (
        (
            text("UPDATE invocation_records SET status = status WHERE invocation_id = :target_id"),
            {"target_id": invocation_id},
        ),
        (
            text("UPDATE telemetry_outbox SET run_id = run_id WHERE id = :target_id"),
            {"target_id": outbox_id},
        ),
        (
            text("DELETE FROM telemetry_outbox WHERE id = :target_id"),
            {"target_id": outbox_id},
        ),
    )
    for statement, params in mutation_attempts:
        async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
            with pytest.raises(Exception) as exc_info:
                await session.execute(statement, params)
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)

    # Individually valid FK values still cannot forge a cross-tenant Run binding.
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        session.add(
            InvocationRecord(
                tenant_id=TENANT_A,
                case_id=SEEDED_CASE,
                run_id=run_b_id,
                kind="MODEL",
                name="forged_parent",
                status="SUCCESS",
                ended_at=utc_now(),
                payload_redacted={},
                payload_sha256="c" * 64,
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        session.add(
            TelemetryOutbox(
                tenant_id=TENANT_A,
                case_id=SEEDED_CASE,
                run_id=run_b_id,
                invocation_id=second_invocation_id,
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    # Sanity: the correctly bound rows remain visible after rejected mutations.
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        assert await session.get(ReviewRun, run_a_id) is not None
        assert await session.get(InvocationRecord, invocation_id) is not None
        assert await session.get(TelemetryOutbox, outbox_id) is not None


async def test_business_role_can_only_update_claim_review_status(pg_engine):
    """Claim facts stay append-only while the workflow may advance review_status."""
    from creditlens.agents.contracts import AgentArtifact, AgentClaim
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )
    from creditlens.infrastructure.postgres.models import ArtifactRecord, ClaimRecord, ReviewRun
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    async with pg_engine.connect() as connection:
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.claims', 'SELECT')")
        )
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.claims', 'INSERT')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.claims', 'UPDATE')")
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, 'public.claims', 'DELETE')")
        )
        assert await connection.scalar(
            text(
                "SELECT has_column_privilege("
                "current_user, 'public.claims', 'review_status', 'UPDATE')"
            )
        )
        for column in ("statement", "verdict", "payload", "severity", "as_of_date"):
            assert not await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.claims', :column, 'UPDATE')"
                ),
                {"column": column},
            )

    factory = create_session_factory(pg_engine)
    run = ReviewRun(
        tenant_id=TENANT_A,
        case_id=SEEDED_CASE,
        run_type="FULL_REVIEW",
        status="HUMAN_REVIEW",
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
    )
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        session.add(run)
        await session.flush()
        run_id = run.id

    source_claim = AgentClaim(
        category="ELIGIBILITY",
        statement="immutable grounded fact",
        verdict="SUPPORTED",
        as_of_date=run.as_of_date,
        supporting_evidence_ids=[uuid.uuid4()],
    )
    source_artifact = AgentArtifact(
        run_id=run_id,
        task_id="claim-integrity",
        producer="policy_analyst",
        claims=[source_claim],
    )
    payload = source_artifact.model_dump(mode="json", exclude={"output_hash"})
    payload["lifecycle_status"] = "VALIDATED"
    artifact = ArtifactRecord(
        id=source_artifact.artifact_id,
        tenant_id=TENANT_A,
        run_id=run_id,
        task_id=source_artifact.task_id,
        artifact_type=source_artifact.producer,
        producer=source_artifact.producer,
        lifecycle_status="VALIDATED",
        payload=payload,
        output_hash=canonical_artifact_payload_hash(payload),
    )
    claim = ClaimRecord(
        id=source_claim.claim_id,
        tenant_id=TENANT_A,
        run_id=run_id,
        artifact_id=artifact.id,
        category=source_claim.category,
        statement=source_claim.statement,
        verdict=source_claim.verdict,
        severity=source_claim.severity,
        as_of_date=source_claim.as_of_date,
        review_status="PENDING",
        payload={
            "supporting_evidence_ids": [str(item) for item in source_claim.supporting_evidence_ids],
            "opposing_evidence_ids": [],
            "calculation_ids": [],
            "source_claim_id": None,
        },
    )
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        session.add(artifact)
        session.add(claim)
        await session.flush()
        claim.review_status = "AUDITED"
        await session.flush()
        claim_id = claim.id

    mutation_attempts = (
        text("UPDATE claims SET statement = 'tampered' WHERE id = :claim_id"),
        text("UPDATE claims SET verdict = 'CONTRADICTED' WHERE id = :claim_id"),
        text("UPDATE claims SET payload = '{}' WHERE id = :claim_id"),
        text("UPDATE claims SET severity = 'CRITICAL' WHERE id = :claim_id"),
    )
    for statement in mutation_attempts:
        async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
            with pytest.raises(Exception) as exc_info:
                await session.execute(statement, {"claim_id": claim_id})
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)

    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        persisted = await session.get(ClaimRecord, claim_id)
        assert persisted is not None
        assert persisted.review_status == "AUDITED"
        assert persisted.statement == "immutable grounded fact"
        assert persisted.verdict == "SUPPORTED"


async def test_business_role_cannot_self_grant_or_revoke_case_membership(pg_engine):
    """授权根由管理身份维护；运行角色不能给自己授权、撤销或删除授权。"""
    from creditlens.infrastructure.postgres.models import CaseMembership
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    async with pg_engine.connect() as connection:
        qualified = "public.case_memberships"
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, :table, 'SELECT')"),
            {"table": qualified},
        )
        for privilege in ("INSERT", "UPDATE", "DELETE"):
            assert not await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                {"table": qualified, "privilege": privilege},
            )

    statements = (
        text(
            """
            INSERT INTO case_memberships (case_id, user_id, case_role, granted_at)
            VALUES (:case_id, :user_id, 'OWNER', now())
            """
        ),
        text(
            """
            UPDATE case_memberships SET revoked_at = now()
            WHERE case_id = :case_id AND user_id = :user_id
            """
        ),
        text(
            """
            DELETE FROM case_memberships
            WHERE case_id = :case_id AND user_id = :user_id
            """
        ),
    )
    params = {"case_id": SEEDED_CASE, "user_id": SEEDED_MEMBER}
    for statement in statements:
        async with pg_engine.connect() as connection:
            with pytest.raises(Exception) as exc_info:
                await connection.execute(statement, params)
            await connection.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)

    factory = create_session_factory(pg_engine)
    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=uuid.uuid4(),
    ) as session:
        leaked = (
            await session.scalars(
                select(CaseMembership).where(CaseMembership.case_id == SEEDED_CASE)
            )
        ).all()
        assert leaked == [], "业务角色不得枚举其他用户的授权根记录"


async def test_identity_roots_and_global_catalogs_are_read_only(pg_engine):
    """Tenant/User 仅暴露当前身份；授权根与全局目录都不能被业务角色改写。"""
    from creditlens.infrastructure.postgres.models import AppUser, Tenant
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    readonly_tables = (
        "tenants",
        "app_users",
        "financial_metric_definitions",
        "search_index_versions",
        "alembic_version",
    )
    async with pg_engine.connect() as connection:
        for table in readonly_tables:
            qualified = f"public.{table}"
            assert await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, 'SELECT')"),
                {"table": qualified},
            )
            for privilege in ("INSERT", "UPDATE", "DELETE"):
                assert not await connection.scalar(
                    text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                    {"table": qualified, "privilege": privilege},
                ), f"{qualified} must not grant {privilege} to creditlens_app"
            # 运行时仍可读取必要的全局/当前身份元数据。
            await connection.execute(text(f"SELECT 1 FROM {qualified} LIMIT 1"))

    factory = create_session_factory(pg_engine)
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        assert await session.get(Tenant, TENANT_A) is not None
        assert await session.get(AppUser, USER_A) is not None
        assert await session.get(Tenant, TENANT_B) is None
        assert await session.get(AppUser, USER_B) is None
        assert (await session.scalars(select(Tenant.id))).all() == [TENANT_A]
        assert (await session.scalars(select(AppUser.id))).all() == [USER_A]

    async with session_scope(factory, tenant_id=TENANT_B, user_id=USER_B) as session:
        assert await session.get(Tenant, TENANT_B) is not None
        assert await session.get(AppUser, USER_B) is not None
        assert await session.get(Tenant, TENANT_A) is None
        assert await session.get(AppUser, USER_A) is None

    mutation_attempts = (
        (
            text(
                """
                INSERT INTO tenants (id, name, status, data_isolation_mode, created_at)
                VALUES (:new_id, 'unauthorized tenant', 'ACTIVE', 'SHARED_COLLECTION', now())
                """
            ),
            {"new_id": uuid.uuid4()},
        ),
        (
            text(
                """
                INSERT INTO app_users
                  (id, tenant_id, external_subject, display_name, status, created_at)
                VALUES
                  (:new_id, :tenant_id, :subject, 'unauthorized user', 'ACTIVE', now())
                """
            ),
            {
                "new_id": uuid.uuid4(),
                "tenant_id": TENANT_A,
                "subject": f"unauthorized-{uuid.uuid4()}",
            },
        ),
        (
            text("UPDATE tenants SET name = 'cross-tenant-write' WHERE id = :target_id"),
            {"target_id": TENANT_B},
        ),
        (
            text("UPDATE app_users SET status = 'DISABLED' WHERE id = :target_id"),
            {"target_id": USER_B},
        ),
    )
    for statement, params in mutation_attempts:
        async with pg_engine.connect() as connection:
            await connection.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": str(TENANT_A)},
            )
            await connection.execute(
                text("SELECT set_config('app.user_id', :user_id, true)"),
                {"user_id": str(USER_A)},
            )
            with pytest.raises(Exception) as exc_info:
                await connection.execute(statement, params)
            await connection.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)


async def test_checkpoint_commit_restores_rls(pg_engine):
    """真实调用 checkpoint_commit：中途提交后 RLS 上下文必须被重新注入。"""
    from creditlens.infrastructure.postgres.models import Document
    from creditlens.infrastructure.postgres.session import (
        checkpoint_commit,
        create_session_factory,
        session_scope,
    )

    factory = create_session_factory(pg_engine)
    doc_before = Document(
        tenant_id=TENANT_A,
        logical_key=f"chk-before-{uuid.uuid4().hex[:8]}",
        title="Checkpoint 前写入",
        document_type="INTERNAL_POLICY",
    )
    doc_after = Document(
        tenant_id=TENANT_A,
        logical_key=f"chk-after-{uuid.uuid4().hex[:8]}",
        title="Checkpoint 后写入",
        document_type="INTERNAL_POLICY",
    )

    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        # 阶段一：写入并真实执行阶段 Checkpoint 提交
        session.add(doc_before)
        await session.flush()
        await checkpoint_commit(session)

        # 阶段二：SET LOCAL 已随上一事务失效——checkpoint_commit 必须重新注入，
        # 否则业务角色（NOBYPASSRLS）在本事务读回阶段一数据将得到 0 行
        rows = (await session.scalars(select(Document).where(Document.id == doc_before.id))).all()
        assert len(rows) == 1, "checkpoint_commit 后 RLS 上下文未恢复：阶段一数据不可见"

        # 阶段二继续写入并通过 WITH CHECK（tenant_id = app_current_tenant()）
        session.add(doc_after)
        await session.flush()

    # 新事务复核：两次 Checkpoint 间的数据均已持久化且对同租户可见
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        assert await session.get(Document, doc_before.id) is not None
        assert await session.get(Document, doc_after.id) is not None


async def test_cross_tenant_isolation(pg_engine):
    """租户 B 会话实际访问租户 A 的 documents：RLS 下必须不可见且不可写。"""
    from creditlens.infrastructure.postgres.models import Document
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    # 租户 A 写入一条受 RLS 租户隔离保护的 document
    secret_doc_id = uuid.uuid4()
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        session.add(
            Document(
                id=secret_doc_id,
                tenant_id=TENANT_A,
                logical_key=f"secret-{uuid.uuid4().hex[:8]}",
                title="租户 A 私有材料",
                document_type="ANNUAL_REPORT",
            )
        )

    # 租户 B 上下文：按主键直取、条件查询、计数三重断言不可见
    async with session_scope(factory, tenant_id=TENANT_B, user_id=USER_B) as session:
        assert await session.get(Document, secret_doc_id) is None, "跨租户主键读取必须被 RLS 拒绝"
        rows = (await session.scalars(select(Document).where(Document.id == secret_doc_id))).all()
        assert rows == [], "跨租户条件查询必须返回 0 行"
        count = await session.scalar(select(Document.id).where(Document.tenant_id == TENANT_A))
        assert count is None, "租户 B 不得看到租户 A 的任何 document"

        # WITH CHECK：以租户 B 会话写入 tenant_id=A 的行必须被拒绝
        session.add(
            Document(
                tenant_id=TENANT_A,
                logical_key=f"cross-write-{uuid.uuid4().hex[:8]}",
                title="越权写入",
                document_type="INTERNAL_POLICY",
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        # flush 失败后 session 处于待回滚状态：先回滚，避免退出 scope 时
        # commit 抛 PendingRollbackError
        await session.rollback()
    # 确认拒绝原因是 RLS 策略而非其他错误
    message = str(exc_info.value).lower()
    assert "row-level security" in message or "policy" in message


async def test_session_rollback_on_error(pg_engine):
    """session_scope 异常时应自动回滚。"""
    from creditlens.infrastructure.postgres.models import Document
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    document_id = uuid.uuid4()

    with pytest.raises(ValueError):
        async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
            session.add(
                Document(
                    id=document_id,
                    tenant_id=TENANT_A,
                    logical_key=f"rollback-{uuid.uuid4().hex}",
                    title="将被回滚的文档",
                    document_type="INTERNAL_POLICY",
                )
            )
            await session.flush()
            raise ValueError("模拟异常")

    # 验证数据未持久化
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_A) as session:
        assert await session.get(Document, document_id) is None


async def test_snapshot_parent_and_children_require_case_membership(pg_engine):
    """同租户但无案件 Membership 的用户不可读取 Snapshot 及其三个子表。"""
    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.common.config import get_settings
    from creditlens.infrastructure.postgres.models import (
        CaseSnapshot,
        SnapshotDocument,
        SnapshotFact,
        SnapshotIndex,
    )
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    case_id = uuid.UUID("00000000-0000-0000-0000-000000000201")
    member_id = uuid.UUID("00000000-0000-0000-0000-000000000301")
    outsider_id = uuid.uuid4()

    async with session_scope(factory, tenant_id=tenant_id, user_id=member_id) as session:
        trusted = await build_trusted_context(
            session,
            tenant_id,
            case_id,
            user_id=member_id,
        )
        frozen = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=get_settings().chunks_collection_name,
        )
        snapshot_id = frozen.snapshot_id
        assert frozen.allowed_parse_run_ids

    async with session_scope(factory, tenant_id=tenant_id, user_id=outsider_id) as session:
        assert await session.get(CaseSnapshot, snapshot_id) is None
        for model in (SnapshotDocument, SnapshotIndex, SnapshotFact):
            rows = (
                await session.scalars(select(model).where(model.snapshot_id == snapshot_id))
            ).all()
            assert rows == []


async def test_snapshot_tables_are_strictly_append_only_for_runtime_role(pg_engine):
    """冻结根和成员只能 SELECT/INSERT；冻结路径本身不依赖任何 UPDATE。"""
    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.common.config import get_settings
    from creditlens.infrastructure.postgres.models import CaseSnapshot
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    snapshot_tables = (
        "case_snapshots",
        "snapshot_documents",
        "snapshot_indexes",
        "snapshot_facts",
    )
    async with pg_engine.connect() as connection:
        for table in snapshot_tables:
            qualified = f"public.{table}"
            for privilege in ("SELECT", "INSERT"):
                assert await connection.scalar(
                    text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                    {"table": qualified, "privilege": privilege},
                )
            for privilege in ("UPDATE", "DELETE"):
                assert not await connection.scalar(
                    text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                    {"table": qualified, "privilege": privilege},
                )
            updateable_columns = (
                (
                    await connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema = 'public' AND table_name = :table "
                            "AND has_column_privilege(current_user, "
                            "format('public.%I', table_name), column_name, 'UPDATE')"
                        ),
                        {"table": table},
                    )
                )
                .scalars()
                .all()
            )
            assert updateable_columns == []

        policy_rows = (
            await connection.execute(
                text(
                    "SELECT tablename, policyname, cmd FROM pg_policies "
                    "WHERE schemaname = 'public' "
                    "AND tablename = ANY(CAST(:tables AS text[]))"
                ),
                {"tables": list(snapshot_tables)},
            )
        ).all()
        assert {(row.tablename, row.policyname, row.cmd) for row in policy_rows} == {
            ("case_snapshots", "case_snapshot_select", "SELECT"),
            ("case_snapshots", "case_snapshot_insert", "INSERT"),
            ("snapshot_documents", "snapshot_parent_select", "SELECT"),
            ("snapshot_documents", "snapshot_parent_insert", "INSERT"),
            ("snapshot_indexes", "snapshot_parent_select", "SELECT"),
            ("snapshot_indexes", "snapshot_parent_insert", "INSERT"),
            ("snapshot_facts", "snapshot_parent_select", "SELECT"),
            ("snapshot_facts", "snapshot_parent_insert", "INSERT"),
        }

    factory = create_session_factory(pg_engine)
    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=SEEDED_MEMBER,
    ) as session:
        trusted = await build_trusted_context(
            session,
            SEEDED_TENANT,
            SEEDED_CASE,
            user_id=SEEDED_MEMBER,
        )
        frozen = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=get_settings().chunks_collection_name,
        )
        snapshot = await session.get(CaseSnapshot, frozen.snapshot_id)
        assert snapshot is not None
        assert len(snapshot.snapshot_hash) == 64
        snapshot_id = snapshot.id
        snapshot_hash = snapshot.snapshot_hash

    mutation_attempts = (
        text("UPDATE case_snapshots SET snapshot_hash = repeat('0', 64) WHERE id = :snapshot_id"),
        text("DELETE FROM case_snapshots WHERE id = :snapshot_id"),
        text(
            "UPDATE snapshot_documents SET parse_run_id = parse_run_id "
            "WHERE snapshot_id = :snapshot_id"
        ),
        text("DELETE FROM snapshot_documents WHERE snapshot_id = :snapshot_id"),
        text(
            "UPDATE snapshot_indexes SET physical_collection_name = 'tampered' "
            "WHERE snapshot_id = :snapshot_id"
        ),
        text("DELETE FROM snapshot_indexes WHERE snapshot_id = :snapshot_id"),
        text("UPDATE snapshot_facts SET fact_id = fact_id WHERE snapshot_id = :snapshot_id"),
        text("DELETE FROM snapshot_facts WHERE snapshot_id = :snapshot_id"),
    )
    for statement in mutation_attempts:
        async with session_scope(
            factory,
            tenant_id=SEEDED_TENANT,
            user_id=SEEDED_MEMBER,
        ) as session:
            with pytest.raises(Exception) as exc_info:
                await session.execute(statement, {"snapshot_id": snapshot_id})
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)

    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=SEEDED_MEMBER,
    ) as session:
        persisted = await session.get(CaseSnapshot, snapshot_id)
        assert persisted is not None
        assert persisted.snapshot_hash == snapshot_hash


async def test_same_tenant_outsider_cannot_insert_case_scoped_rows(pg_engine):
    """仅持有 tenant_id 不构成写权限；无 Membership 不得创建 Run/Upload。"""
    from creditlens.infrastructure.postgres.models import ReviewRun, UploadSession
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    outsider_id = uuid.uuid4()
    attempts = (
        ReviewRun(
            tenant_id=SEEDED_TENANT,
            case_id=SEEDED_CASE,
            run_type="SIMPLE_QA",
            status="RECEIVED",
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
        ),
        UploadSession(
            tenant_id=SEEDED_TENANT,
            case_id=SEEDED_CASE,
            object_key=f"unauthorized/{uuid.uuid4()}",
        ),
    )
    for row in attempts:
        async with session_scope(
            factory,
            tenant_id=SEEDED_TENANT,
            user_id=outsider_id,
        ) as session:
            session.add(row)
            with pytest.raises(Exception) as exc_info:
                await session.flush()
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)


async def test_credit_case_creation_is_admin_only_and_root_fields_are_immutable(pg_engine):
    """业务角色只能更新案件业务白名单，身份、产品、金额和时间根均不可变。"""
    from creditlens.infrastructure.postgres.models import CreditCase, Entity
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    async with pg_engine.connect() as connection:
        qualified = "public.credit_cases"
        assert await connection.scalar(
            text("SELECT has_table_privilege(current_user, :table, 'SELECT')"),
            {"table": qualified},
        )
        assert not await connection.scalar(
            text("SELECT has_table_privilege(current_user, :table, 'UPDATE')"),
            {"table": qualified},
        )
        for privilege in ("INSERT", "DELETE"):
            assert not await connection.scalar(
                text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                {"table": qualified, "privilege": privilege},
            )
        mutable_columns = {
            "loan_purpose",
            "industry_code",
            "region_code",
            "status",
            "current_report_id",
            "updated_at",
            "version",
        }
        immutable_columns = {
            "id",
            "tenant_id",
            "case_number",
            "borrower_entity_id",
            "product_code",
            "requested_amount",
            "currency",
            "application_date",
            "as_of_date",
            "decision_cutoff_at",
            "created_by",
            "created_at",
        }
        for column in mutable_columns:
            assert await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.credit_cases', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )
        for column in immutable_columns:
            assert not await connection.scalar(
                text(
                    "SELECT has_column_privilege(current_user, 'public.credit_cases', "
                    ":column, 'UPDATE')"
                ),
                {"column": column},
            )
        policies = (
            await connection.execute(
                text(
                    "SELECT policyname, cmd FROM pg_policies "
                    "WHERE schemaname = 'public' AND tablename = 'credit_cases'"
                )
            )
        ).all()
        assert {(row.policyname, row.cmd) for row in policies} == {
            ("case_tenant_select", "SELECT"),
            ("case_tenant_update", "UPDATE"),
        }

    factory = create_session_factory(pg_engine)
    outsider_id = uuid.uuid4()
    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=outsider_id,
    ) as session:
        session.add(
            CreditCase(
                tenant_id=SEEDED_TENANT,
                case_number=f"UNAUTHORIZED-{uuid.uuid4().hex[:8]}",
                borrower_entity_id=uuid.UUID("00000000-0000-0000-0000-000000000101"),
                product_code="working_capital",
                requested_amount=Decimal("1"),
                application_date=date(2026, 6, 30),
                as_of_date=date(2026, 6, 30),
                decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        await session.rollback()
    _assert_policy_or_privilege_denied(exc_info.value)

    foreign_tenant = TENANT_B
    foreign_borrower = uuid.uuid4()
    async with session_scope(factory, tenant_id=foreign_tenant, user_id=USER_B) as session:
        session.add(
            Entity(
                id=foreign_borrower,
                tenant_id=foreign_tenant,
                entity_type="COMPANY",
                canonical_name="另一租户借款人",
            )
        )

    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=SEEDED_MEMBER,
    ) as session:
        case = await session.get(CreditCase, SEEDED_CASE)
        assert case is not None
        case.loan_purpose = "合法同租户更新"
        await session.flush()
        await session.rollback()

    immutable_mutations = (
        ("tenant_id", str(SEEDED_TENANT)),
        ("case_number", "tampered-case-number"),
        ("borrower_entity_id", str(foreign_borrower)),
        ("product_code", "tampered-product"),
        ("requested_amount", Decimal("1")),
        ("currency", "USD"),
        ("application_date", date(2026, 7, 1)),
        ("as_of_date", date(2026, 7, 1)),
        ("decision_cutoff_at", datetime(2026, 7, 1, tzinfo=UTC)),
        ("created_by", str(uuid.uuid4())),
    )
    for column, value in immutable_mutations:
        async with session_scope(
            factory,
            tenant_id=SEEDED_TENANT,
            user_id=SEEDED_MEMBER,
        ) as session:
            with pytest.raises(Exception) as exc_info:
                await session.execute(
                    text(f"UPDATE credit_cases SET {column} = :value WHERE id = :case_id"),
                    {"value": value, "case_id": SEEDED_CASE},
                )
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)


async def test_snapshot_children_reject_cross_tenant_sources(pg_engine):
    """授权案件成员也不能把另一租户的文档/Fact 或不可归属索引挂进 Snapshot。"""
    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.common.config import get_settings
    from creditlens.infrastructure.postgres.models import (
        Document,
        DocumentSection,
        DocumentVersion,
        Entity,
        FinancialFact,
        ParseRun,
        SnapshotDocument,
        SnapshotFact,
        SnapshotIndex,
        SummaryNode,
        SummaryNodeSource,
    )
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=SEEDED_MEMBER,
    ) as session:
        trusted = await build_trusted_context(
            session,
            SEEDED_TENANT,
            SEEDED_CASE,
            user_id=SEEDED_MEMBER,
        )
        frozen = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=get_settings().chunks_collection_name,
        )
        snapshot_id = frozen.snapshot_id

    foreign_tenant = TENANT_B
    foreign_entity_id = uuid.uuid4()
    foreign_document_id = uuid.uuid4()
    foreign_version_id = uuid.uuid4()
    foreign_parse_id = uuid.uuid4()
    foreign_section_id = uuid.uuid4()
    foreign_fact_id = uuid.uuid4()
    async with session_scope(factory, tenant_id=foreign_tenant, user_id=USER_B) as session:
        session.add_all(
            [
                Entity(
                    id=foreign_entity_id,
                    tenant_id=foreign_tenant,
                    entity_type="COMPANY",
                    canonical_name="跨租户污染源",
                ),
                Document(
                    id=foreign_document_id,
                    tenant_id=foreign_tenant,
                    logical_key=f"foreign-{uuid.uuid4().hex}",
                    title="另一租户文档",
                    document_type="ANNUAL_REPORT",
                ),
            ]
        )
        await session.flush()
        version = DocumentVersion(
            id=foreign_version_id,
            tenant_id=foreign_tenant,
            document_id=foreign_document_id,
            version_label="1",
            source_available_at=datetime(2026, 1, 1, tzinfo=UTC),
            object_uri="local://foreign.pdf",
            source_filename="foreign.pdf",
            mime_type="application/pdf",
            file_size=1,
            content_hash="f" * 64,
        )
        session.add(version)
        await session.flush()
        parse_run = ParseRun(
            id=foreign_parse_id,
            tenant_id=foreign_tenant,
            document_version_id=foreign_version_id,
            generation_no=1,
            status="SUCCEEDED",
            activation_status="ACTIVE",
            parser_name="rls-test",
            parser_version="1",
            config_hash="e" * 64,
        )
        session.add(parse_run)
        await session.flush()
        version.active_parse_run_id = foreign_parse_id
        session.add_all(
            [
                DocumentSection(
                    id=foreign_section_id,
                    tenant_id=foreign_tenant,
                    document_version_id=foreign_version_id,
                    parse_run_id=foreign_parse_id,
                    section_type="PARAGRAPH",
                    ordinal=1,
                    page_start=1,
                    page_end=1,
                    text="另一租户原文",
                    text_hash="d" * 64,
                ),
                FinancialFact(
                    id=foreign_fact_id,
                    tenant_id=foreign_tenant,
                    case_id=None,
                    entity_id=foreign_entity_id,
                    metric_code="revenue",
                    period_end=date(2025, 12, 31),
                    value=Decimal("1"),
                    canonical_value=Decimal("1"),
                    source_available_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ]
        )

    async with session_scope(
        factory,
        tenant_id=SEEDED_TENANT,
        user_id=SEEDED_MEMBER,
    ) as session:
        summary_node_id = await session.scalar(select(SummaryNode.id).limit(1))
        assert summary_node_id is not None

    attempts = (
        SnapshotDocument(
            snapshot_id=snapshot_id,
            document_version_id=foreign_version_id,
            parse_run_id=foreign_parse_id,
        ),
        SnapshotFact(snapshot_id=snapshot_id, fact_id=foreign_fact_id),
        SnapshotIndex(
            snapshot_id=snapshot_id,
            index_family="SUMMARIES",
            index_version_id=uuid.uuid4(),
            physical_collection_name="unattestable-foreign-index",
        ),
        SummaryNodeSource(
            summary_node_id=summary_node_id,
            section_id=foreign_section_id,
            ordinal=999999,
        ),
    )
    for child in attempts:
        async with session_scope(
            factory,
            tenant_id=SEEDED_TENANT,
            user_id=SEEDED_MEMBER,
        ) as session:
            session.add(child)
            with pytest.raises(Exception) as exc_info:
                await session.flush()
            await session.rollback()
        _assert_policy_or_privilege_denied(exc_info.value)
