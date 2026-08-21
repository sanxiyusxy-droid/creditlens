import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from creditlens.infrastructure.llm.chat import ModelInvocationTrace
from creditlens.infrastructure.postgres.models import (
    CreditCase,
    Entity,
    InvocationRecord,
    ReviewRun,
    TelemetryOutbox,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory
from creditlens.observability.invocation import InvocationEnvelope, InvocationKind, InvocationStatus
from creditlens.observability.writer import (
    InvocationAuditPersistError,
    InvocationIdentityConflict,
    InvocationWriter,
)

_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000921")
_CASE_ID = uuid.UUID("00000000-0000-0000-0000-000000000922")
_RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000923")
_HASH_A = "a" * 64
_HASH_B = "b" * 64


async def _seed_run(session) -> None:
    entity_id = uuid.UUID("00000000-0000-0000-0000-000000000924")
    session.add_all(
        [
            Tenant(id=_TENANT_ID, name="Invocation writer tenant"),
            Entity(
                id=entity_id,
                tenant_id=_TENANT_ID,
                entity_type="COMPANY",
                canonical_name="Invocation writer borrower",
            ),
            CreditCase(
                id=_CASE_ID,
                tenant_id=_TENANT_ID,
                case_number="INVOCATION-WRITER-CASE",
                borrower_entity_id=entity_id,
                product_code="working_capital",
                requested_amount=Decimal("1"),
                application_date=date(2026, 8, 21),
                as_of_date=date(2026, 8, 21),
                decision_cutoff_at=datetime(2026, 8, 21, tzinfo=UTC),
            ),
            ReviewRun(
                id=_RUN_ID,
                tenant_id=_TENANT_ID,
                case_id=_CASE_ID,
                run_type="SIMPLE_QA",
                status="GENERATING",
                as_of_date=date(2026, 8, 21),
                decision_cutoff_at=datetime(2026, 8, 21, tzinfo=UTC),
            ),
        ]
    )
    await session.commit()


def _envelope(*, invocation_id: uuid.UUID | None = None, name: str = "lookup"):
    return InvocationEnvelope(
        invocation_id=invocation_id or uuid.uuid4(),
        kind=InvocationKind.TOOL,
        name=name,
        provider="internal",
        version="v1",
        started_at=datetime(2026, 8, 21, 1, tzinfo=UTC),
        ended_at=datetime(2026, 8, 21, 1, 0, 1, tzinfo=UTC),
        latency_ms=1000,
        status=InvocationStatus.SUCCESS,
        request_sha256=_HASH_A,
        response_sha256=_HASH_B,
    )


async def test_writer_atomically_creates_record_and_outbox_and_replays_idempotently(session):
    await _seed_run(session)
    writer = InvocationWriter(
        session,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
        actor_role="policy_analyst",
        task_id="policy_review",
    )
    envelope = _envelope()

    created = await writer.record(envelope)
    replay = await writer.record(envelope)

    assert created.created is True
    assert replay.created is False
    assert replay.record.invocation_id == created.record.invocation_id
    assert created.envelope.actor_role == "policy_analyst"
    assert created.envelope.task_id == "policy_review"
    assert created.record.payload_redacted["actor_role"] == "policy_analyst"
    assert created.record.payload_sha256 == replay.record.payload_sha256
    assert created.outbox.status == "PENDING"
    assert created.outbox.attempts == 0
    assert (await session.scalar(select(func.count()).select_from(InvocationRecord))) == 1
    assert (await session.scalar(select(func.count()).select_from(TelemetryOutbox))) == 1


async def test_writer_rejects_same_invocation_id_with_different_payload(session):
    await _seed_run(session)
    writer = InvocationWriter(
        session,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
    )
    invocation_id = uuid.uuid4()
    await writer.record(_envelope(invocation_id=invocation_id))

    with pytest.raises(InvocationIdentityConflict) as captured:
        await writer.record(_envelope(invocation_id=invocation_id, name="different_tool"))
    assert captured.value.error_code == "INVOCATION_ID_CONFLICT"


async def test_writer_rejects_tampered_persisted_payload_even_when_hash_column_is_unchanged(
    session,
):
    await _seed_run(session)
    writer = InvocationWriter(
        session,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
    )
    envelope = _envelope()
    created = await writer.record(envelope)
    created.record.payload_redacted = {"contract_version": "invocation_v2"}
    await session.flush()

    with pytest.raises(InvocationIdentityConflict) as captured:
        await writer.record(envelope)
    assert captured.value.error_code == "INVOCATION_ID_CONFLICT"


async def test_writer_rejects_envelope_context_that_disagrees_with_bound_context(session):
    await _seed_run(session)
    writer = InvocationWriter(
        session,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
        actor_role="policy_analyst",
    )
    conflicting = InvocationEnvelope.model_validate(
        {**_envelope().model_dump(mode="python"), "actor_role": "financial_analyst"}
    )
    with pytest.raises(InvocationAuditPersistError) as captured:
        await writer.record(conflicting)
    assert captured.value.error_code == "INVOCATION_AUDIT_PERSIST_FAILED"


async def test_writer_adapts_model_trace_with_safe_diagnostics(session):
    await _seed_run(session)
    writer = InvocationWriter(
        session,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
        actor_role="grounded_qa",
        task_id="answer_generation",
    )
    trace = ModelInvocationTrace(
        provider="openai_compatible",
        model="model-v1",
        prompt_version="grounded-qa-v1",
        prompt_sha256=_HASH_A,
        request_sha256=_HASH_B,
        latency_ms=2,
        attempts=2,
        status="FAILED",
        error_type="ValidationError",
        schema_error_fingerprint=_HASH_A,
        schema_error_counts={"MISSING_FIELD": 2},
    )

    result = await writer.record_model_trace(
        trace,
        ended_at=datetime(2026, 8, 21, 1, tzinfo=UTC),
    )

    assert result.record.kind == "MODEL"
    assert result.record.status == "FAILED"
    assert result.envelope.schema_diagnostics is not None
    assert result.envelope.schema_diagnostics.error_counts == {"MISSING_FIELD": 2}


async def test_outer_rollback_removes_record_and_outbox_together(engine):
    factory = create_session_factory(engine)
    async with factory() as session:
        await _seed_run(session)
        writer = InvocationWriter(
            session,
            tenant_id=_TENANT_ID,
            case_id=_CASE_ID,
            run_id=_RUN_ID,
        )
        await writer.record(_envelope())
        await session.rollback()

    async with factory() as session:
        assert (await session.scalar(select(func.count()).select_from(InvocationRecord))) == 0
        assert (await session.scalar(select(func.count()).select_from(TelemetryOutbox))) == 0
