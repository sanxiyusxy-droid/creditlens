import asyncio
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

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
from creditlens.observability.outbox_worker import (
    LocalDirectoryTelemetryExporter,
    NoopTelemetryExporter,
    TelemetryDelivery,
    TelemetryOutboxWorker,
    TelemetryPayloadInvalid,
)
from creditlens.observability.writer import INVOCATION_OUTBOX_TOPIC, InvocationWriter

_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000931")
_CASE_ID = uuid.UUID("00000000-0000-0000-0000-000000000932")
_RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000933")
_NOW = datetime(2026, 8, 22, 2, tzinfo=UTC)


class _Clock:
    def __init__(self, now: datetime = _NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _RecordingExporter:
    def __init__(self, failures: list[BaseException] | None = None):
        self.failures = list(failures or [])
        self.calls: list[tuple[InvocationEnvelope, TelemetryDelivery, str]] = []

    async def export(
        self,
        envelope: InvocationEnvelope,
        *,
        delivery: TelemetryDelivery,
        idempotency_key: str,
    ) -> None:
        self.calls.append((envelope, delivery, idempotency_key))
        if self.failures:
            raise self.failures.pop(0)


class _DestinationUnavailable(Exception):
    error_code = "DESTINATION_UNAVAILABLE"


async def _seed_pending(factory) -> uuid.UUID:
    entity_id = uuid.UUID("00000000-0000-0000-0000-000000000934")
    invocation_id = uuid.UUID("00000000-0000-0000-0000-000000000935")
    async with factory() as session:
        session.add_all(
            [
                Tenant(id=_TENANT_ID, name="Telemetry outbox tenant"),
                Entity(
                    id=entity_id,
                    tenant_id=_TENANT_ID,
                    entity_type="COMPANY",
                    canonical_name="Telemetry outbox borrower",
                ),
                CreditCase(
                    id=_CASE_ID,
                    tenant_id=_TENANT_ID,
                    case_number="TELEMETRY-OUTBOX-CASE",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1"),
                    application_date=date(2026, 8, 21),
                    as_of_date=date(2026, 8, 21),
                    decision_cutoff_at=_NOW,
                ),
                ReviewRun(
                    id=_RUN_ID,
                    tenant_id=_TENANT_ID,
                    case_id=_CASE_ID,
                    run_type="SIMPLE_QA",
                    status="GENERATING",
                    as_of_date=date(2026, 8, 21),
                    decision_cutoff_at=_NOW,
                ),
            ]
        )
        await session.commit()
        writer = InvocationWriter(
            session,
            tenant_id=_TENANT_ID,
            case_id=_CASE_ID,
            run_id=_RUN_ID,
            actor_role="grounded_qa",
            task_id="answer_generation",
        )
        await writer.record(
            InvocationEnvelope(
                invocation_id=invocation_id,
                kind=InvocationKind.MODEL,
                name="structured_generation",
                provider="openai_compatible",
                model="model-v1",
                version="grounded-qa-v1",
                started_at=_NOW,
                ended_at=_NOW + timedelta(seconds=1),
                latency_ms=1000,
                status=InvocationStatus.SUCCESS,
                request_sha256="a" * 64,
                response_sha256="b" * 64,
            )
        )
        outbox = await session.scalar(select(TelemetryOutbox))
        outbox.available_at = _NOW
        await session.commit()
    return invocation_id


async def _outbox(factory) -> TelemetryOutbox:
    async with factory() as session:
        return await session.scalar(select(TelemetryOutbox))


def test_postgres_claim_uses_skip_locked_and_sqlite_does_not():
    postgres_stmt = TelemetryOutboxWorker._claim_due_entries_stmt(_NOW, 8, "postgresql")
    sqlite_stmt = TelemetryOutboxWorker._claim_due_entries_stmt(_NOW, 8, "sqlite")
    tenant_stmt = TelemetryOutboxWorker._claim_due_entries_stmt(
        _NOW,
        8,
        "postgresql",
        tenant_id=_TENANT_ID,
    )
    postgres_sql = str(postgres_stmt.compile(dialect=postgresql.dialect()))
    sqlite_sql = str(sqlite_stmt.compile(dialect=sqlite.dialect()))
    tenant_sql = str(tenant_stmt.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in postgres_sql
    assert "SKIP LOCKED" in postgres_sql
    assert "FOR UPDATE" not in sqlite_sql
    assert "telemetry_outbox.tenant_id =" in tenant_sql


async def test_worker_delivers_once_and_clears_lease(engine):
    factory = create_session_factory(engine)
    invocation_id = await _seed_pending(factory)
    exporter = _RecordingExporter()
    worker = TelemetryOutboxWorker(factory, exporter, clock=_Clock())

    stats = await worker.process_batch()

    assert stats.claimed == 1
    assert stats.delivered == 1
    assert stats.retried == stats.dead == stats.lost_leases == 0
    assert len(exporter.calls) == 1
    _, delivery, idempotency_key = exporter.calls[0]
    assert idempotency_key == str(invocation_id)
    assert delivery == TelemetryDelivery(
        invocation_id=invocation_id,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
        topic=INVOCATION_OUTBOX_TOPIC,
    )
    row = await _outbox(factory)
    assert row.status == "DELIVERED"
    assert row.attempts == 1
    assert row.locked_at is row.locked_until is None
    assert row.delivered_at == _NOW
    assert row.dead_at is None


async def test_worker_retries_with_backoff_then_delivers_without_error_text(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    secret = "provider response contains customer-secret"
    exporter = _RecordingExporter([RuntimeError(secret)])
    clock = _Clock()
    worker = TelemetryOutboxWorker(
        factory,
        exporter,
        base_backoff_seconds=5,
        clock=clock,
    )

    first = await worker.process_batch()
    assert first.retried == 1
    row = await _outbox(factory)
    assert row.status == "PENDING"
    assert row.attempts == 1
    assert row.available_at == _NOW + timedelta(seconds=5)
    assert row.last_error_code == "TELEMETRY_EXPORT_FAILED"
    assert secret not in str(row.__dict__)
    assert (await worker.process_batch()).claimed == 0

    clock.advance(5)
    second = await worker.process_batch()
    assert second.delivered == 1
    row = await _outbox(factory)
    assert row.status == "DELIVERED"
    assert row.attempts == 2
    assert row.last_error_code is None


async def test_worker_exhaustion_moves_row_to_dead_with_declared_safe_code(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    exporter = _RecordingExporter([_DestinationUnavailable(), _DestinationUnavailable()])
    clock = _Clock()
    worker = TelemetryOutboxWorker(
        factory,
        exporter,
        max_attempts=2,
        base_backoff_seconds=1,
        clock=clock,
    )

    assert (await worker.process_batch()).retried == 1
    clock.advance(1)
    assert (await worker.process_batch()).dead == 1
    row = await _outbox(factory)
    assert row.status == "DEAD"
    assert row.attempts == 2
    assert row.last_error_code == "DESTINATION_UNAVAILABLE"
    assert row.dead_at == clock.now
    assert row.delivered_at is None
    assert row.locked_at is row.locked_until is None


async def test_expired_processing_lease_is_reclaimed_with_attempt_fencing(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    async with factory() as session:
        row = await session.scalar(select(TelemetryOutbox))
        row.status = "PROCESSING"
        row.attempts = 1
        row.locked_at = _NOW - timedelta(seconds=120)
        row.locked_until = _NOW - timedelta(seconds=60)
        await session.commit()

    exporter = _RecordingExporter()
    worker = TelemetryOutboxWorker(factory, exporter, clock=_Clock())
    stats = await worker.process_batch()

    assert stats.claimed == stats.reclaimed == stats.delivered == 1
    row = await _outbox(factory)
    assert row.status == "DELIVERED"
    assert row.attempts == 2


async def test_worker_cancellation_leaves_processing_lease_for_recovery(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    exporter = _RecordingExporter([asyncio.CancelledError()])
    worker = TelemetryOutboxWorker(factory, exporter, clock=_Clock())

    with pytest.raises(asyncio.CancelledError):
        await worker.process_batch()

    row = await _outbox(factory)
    assert row.status == "PROCESSING"
    assert row.attempts == 1
    assert row.locked_at == _NOW
    assert row.locked_until == _NOW + timedelta(seconds=60)


async def test_cancelled_final_attempt_expires_to_dead_without_duplicate_export(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    exporter = _RecordingExporter([asyncio.CancelledError()])
    clock = _Clock()
    worker = TelemetryOutboxWorker(
        factory,
        exporter,
        max_attempts=1,
        lease_seconds=60,
        clock=clock,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.process_batch()
    assert len(exporter.calls) == 1

    clock.advance(60)
    stats = await worker.process_batch()

    assert stats.claimed == 0
    assert stats.dead == 1
    assert len(exporter.calls) == 1
    row = await _outbox(factory)
    assert row.status == "DEAD"
    assert row.attempts == 1
    assert row.last_error_code == "TELEMETRY_ATTEMPTS_EXHAUSTED"
    assert row.locked_at is row.locked_until is None
    assert row.dead_at == clock.now


async def test_invalid_persisted_payload_is_never_exported_and_uses_safe_code(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    async with factory() as session:
        record = await session.scalar(select(InvocationRecord))
        record.payload_redacted = {"raw_prompt": "customer-secret"}
        await session.commit()

    exporter = _RecordingExporter()
    worker = TelemetryOutboxWorker(factory, exporter, max_attempts=1, clock=_Clock())
    stats = await worker.process_batch()

    assert stats.dead == 1
    assert exporter.calls == []
    row = await _outbox(factory)
    assert row.last_error_code == "TELEMETRY_PAYLOAD_INVALID"


async def test_noop_exporter_fails_closed_instead_of_acknowledging(engine):
    factory = create_session_factory(engine)
    await _seed_pending(factory)
    worker = TelemetryOutboxWorker(
        factory,
        NoopTelemetryExporter(),
        max_attempts=1,
        clock=_Clock(),
    )

    assert (await worker.process_batch()).dead == 1
    row = await _outbox(factory)
    assert row.status == "DEAD"
    assert row.last_error_code == "TELEMETRY_EXPORTER_NOT_CONFIGURED"


async def test_local_directory_exporter_is_durable_and_idempotent(tmp_path):
    invocation_id = uuid.UUID("00000000-0000-0000-0000-000000000935")
    envelope = InvocationEnvelope(
        invocation_id=invocation_id,
        kind=InvocationKind.MODEL,
        name="structured_generation",
        provider="deterministic_local",
        model="extractive-fallback",
        version="grounded-qa-v1",
        started_at=_NOW,
        ended_at=_NOW + timedelta(seconds=1),
        latency_ms=1000,
        status=InvocationStatus.SUCCESS,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
    )
    delivery = TelemetryDelivery(
        invocation_id=invocation_id,
        tenant_id=_TENANT_ID,
        case_id=_CASE_ID,
        run_id=_RUN_ID,
        topic=INVOCATION_OUTBOX_TOPIC,
    )
    exporter = LocalDirectoryTelemetryExporter(tmp_path)

    await exporter.export(envelope, delivery=delivery, idempotency_key=str(invocation_id))
    await exporter.export(envelope, delivery=delivery, idempotency_key=str(invocation_id))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "creditlens.local-telemetry-delivery.v1"
    assert payload["idempotency_key"] == str(invocation_id)
    assert payload["delivery"]["run_id"] == str(_RUN_ID)
    assert "raw_prompt" not in files[0].read_text(encoding="utf-8")

    files[0].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(TelemetryPayloadInvalid):
        await exporter.export(envelope, delivery=delivery, idempotency_key=str(invocation_id))
