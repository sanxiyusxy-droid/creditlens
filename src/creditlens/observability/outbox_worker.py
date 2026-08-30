"""Lease-based, at-least-once delivery for invocation telemetry outbox rows."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from creditlens.common.clock import utc_now
from creditlens.common.errors import CreditLensError
from creditlens.infrastructure.postgres.models import InvocationRecord, TelemetryOutbox
from creditlens.infrastructure.postgres.session import session_scope
from creditlens.observability.invocation import (
    InvocationEnvelope,
    hash_invocation_envelope,
    safe_error_code,
)


class TelemetryExporter(Protocol):
    """An idempotent destination; ``idempotency_key`` is the invocation UUID."""

    async def export(
        self,
        envelope: InvocationEnvelope,
        *,
        delivery: TelemetryDelivery,
        idempotency_key: str,
    ) -> None: ...


class TelemetryExporterNotConfigured(CreditLensError):
    error_code = "TELEMETRY_EXPORTER_NOT_CONFIGURED"


class TelemetryPayloadInvalid(CreditLensError):
    error_code = "TELEMETRY_PAYLOAD_INVALID"


class NoopTelemetryExporter:
    """Fail closed instead of silently acknowledging and discarding telemetry."""

    async def export(
        self,
        envelope: InvocationEnvelope,
        *,
        delivery: TelemetryDelivery,
        idempotency_key: str,
    ) -> None:
        del envelope, delivery, idempotency_key
        raise TelemetryExporterNotConfigured()


class LocalDirectoryTelemetryExporter:
    """Durably deliver redacted local-demo telemetry as one file per invocation.

    The filename is the invocation UUID (the outbox idempotency key).  A retry
    either observes identical canonical bytes or fails closed; it never appends
    a duplicate record.  This exporter is intentionally only wired by the API
    for local/dev/test profiles.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory).resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    async def export(
        self,
        envelope: InvocationEnvelope,
        *,
        delivery: TelemetryDelivery,
        idempotency_key: str,
    ) -> None:
        await asyncio.to_thread(
            self._export_sync,
            envelope,
            delivery=delivery,
            idempotency_key=idempotency_key,
        )

    def _export_sync(
        self,
        envelope: InvocationEnvelope,
        *,
        delivery: TelemetryDelivery,
        idempotency_key: str,
    ) -> None:
        try:
            invocation_id = uuid.UUID(idempotency_key)
        except (TypeError, ValueError, AttributeError):
            raise TelemetryPayloadInvalid() from None
        if (
            str(invocation_id) != idempotency_key
            or envelope.invocation_id != invocation_id
            or delivery.invocation_id != invocation_id
        ):
            raise TelemetryPayloadInvalid()

        payload = {
            "schema_version": "creditlens.local-telemetry-delivery.v1",
            "idempotency_key": idempotency_key,
            "delivery": {
                "invocation_id": str(delivery.invocation_id),
                "tenant_id": str(delivery.tenant_id),
                "case_id": str(delivery.case_id),
                "run_id": str(delivery.run_id),
                "topic": delivery.topic,
            },
            "envelope": envelope.model_dump(mode="json"),
        }
        rendered = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / f"{idempotency_key}.json"
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == rendered:
                return
            raise TelemetryPayloadInvalid()

        temporary = self._directory / f".{idempotency_key}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            # A crash after this atomic replace but before the database ACK is
            # harmless: the retry verifies these same canonical bytes.
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True, slots=True)
class TelemetryWorkerStats:
    claimed: int = 0
    reclaimed: int = 0
    delivered: int = 0
    retried: int = 0
    dead: int = 0
    lost_leases: int = 0


@dataclass(frozen=True, slots=True)
class TelemetryDelivery:
    """Verified immutable routing context supplied to an exporter."""

    invocation_id: uuid.UUID
    tenant_id: uuid.UUID
    case_id: uuid.UUID
    run_id: uuid.UUID
    topic: str


@dataclass(frozen=True, slots=True)
class _Claim:
    outbox_id: uuid.UUID
    invocation_id: uuid.UUID
    attempt: int
    reclaimed: bool


@dataclass(frozen=True, slots=True)
class _ClaimBatch:
    claims: list[_Claim]
    exhausted_leases: int = 0


class TelemetryOutboxWorker:
    """Claim due rows with leases, export, then finalize with attempt fencing.

    When ``tenant_id`` is supplied each short database transaction restores the
    normal application RLS context.  When omitted, the supplied factory must be
    backed by a dedicated worker/service role; an ordinary RLS application role
    intentionally cannot scan across tenants without context.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        exporter: TelemetryExporter,
        *,
        max_attempts: int = 5,
        lease_seconds: float = 60,
        base_backoff_seconds: float = 5,
        max_backoff_seconds: float = 300,
        worker_id: str | None = None,
        tenant_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if base_backoff_seconds <= 0 or max_backoff_seconds <= 0:
            raise ValueError("backoff must be positive")
        if base_backoff_seconds > max_backoff_seconds:
            raise ValueError("base backoff cannot exceed maximum backoff")
        self._session_factory = session_factory
        self._exporter = exporter
        self._max_attempts = max_attempts
        self._lease = timedelta(seconds=lease_seconds)
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._worker_id = worker_id or f"telemetry-{uuid.uuid4().hex}"
        self._tenant_id = uuid.UUID(str(tenant_id)) if tenant_id is not None else None
        self._user_id = uuid.UUID(str(user_id)) if user_id is not None else None
        self._clock = clock

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def process_batch(self, batch_size: int = 32) -> TelemetryWorkerStats:
        if not 1 <= batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        batch = await self._claim_batch(batch_size)
        claims = batch.claims
        delivered = retried = lost_leases = 0
        dead = batch.exhausted_leases
        for claim in claims:
            try:
                envelope, delivery = await self._load_envelope(claim)
                await self._exporter.export(
                    envelope,
                    delivery=delivery,
                    idempotency_key=str(claim.invocation_id),
                )
            except asyncio.CancelledError:
                # Do not guess whether the destination observed the call.  The
                # durable PROCESSING lease will be reclaimed for an idempotent retry.
                raise
            except Exception as error:
                outcome = await self._mark_failure(claim, error)
                if outcome == "RETRIED":
                    retried += 1
                elif outcome == "DEAD":
                    dead += 1
                else:
                    lost_leases += 1
            else:
                if await self._mark_delivered(claim):
                    delivered += 1
                else:
                    lost_leases += 1
        return TelemetryWorkerStats(
            claimed=len(claims),
            reclaimed=sum(claim.reclaimed for claim in claims),
            delivered=delivered,
            retried=retried,
            dead=dead,
            lost_leases=lost_leases,
        )

    async def _claim_batch(self, batch_size: int) -> _ClaimBatch:
        now = self._aware_now()
        async with session_scope(
            self._session_factory,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        ) as session:
            dialect_name = session.bind.dialect.name if session.bind is not None else ""
            entries = (
                await session.scalars(
                    self._claim_due_entries_stmt(
                        now,
                        batch_size,
                        dialect_name,
                        tenant_id=self._tenant_id,
                    )
                )
            ).all()
            claims: list[_Claim] = []
            exhausted_leases = 0
            for entry in entries:
                reclaimed = entry.status == "PROCESSING"
                if reclaimed and entry.attempts >= self._max_attempts:
                    entry.status = "DEAD"
                    entry.locked_at = None
                    entry.locked_until = None
                    entry.last_error_code = "TELEMETRY_ATTEMPTS_EXHAUSTED"
                    entry.delivered_at = None
                    entry.dead_at = now
                    exhausted_leases += 1
                    continue
                entry.status = "PROCESSING"
                entry.attempts += 1
                entry.locked_at = now
                entry.locked_until = now + self._lease
                entry.last_error_code = None
                entry.delivered_at = None
                entry.dead_at = None
                claims.append(
                    _Claim(
                        outbox_id=entry.id,
                        invocation_id=entry.invocation_id,
                        attempt=entry.attempts,
                        reclaimed=reclaimed,
                    )
                )
            await session.flush()
            return _ClaimBatch(claims=claims, exhausted_leases=exhausted_leases)

    @staticmethod
    def _claim_due_entries_stmt(
        now: datetime,
        batch_size: int,
        dialect_name: str,
        *,
        tenant_id: uuid.UUID | None = None,
    ):
        due = and_(TelemetryOutbox.status == "PENDING", TelemetryOutbox.available_at <= now)
        expired = and_(
            TelemetryOutbox.status == "PROCESSING",
            TelemetryOutbox.locked_until.is_not(None),
            TelemetryOutbox.locked_until <= now,
        )
        statement = (
            select(TelemetryOutbox)
            .where(or_(due, expired))
            .order_by(TelemetryOutbox.available_at, TelemetryOutbox.created_at)
            .limit(batch_size)
        )
        if tenant_id is not None:
            # RLS remains the authorization boundary in PostgreSQL.  The
            # explicit predicate also keeps tenant-sharded workers scoped under
            # SQLite/dev databases where set_rls_context is intentionally a no-op.
            statement = statement.where(TelemetryOutbox.tenant_id == tenant_id)
        if dialect_name == "postgresql":
            return statement.with_for_update(skip_locked=True, of=TelemetryOutbox)
        return statement

    async def _load_envelope(self, claim: _Claim) -> tuple[InvocationEnvelope, TelemetryDelivery]:
        async with session_scope(
            self._session_factory,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        ) as session:
            outbox = await session.get(TelemetryOutbox, claim.outbox_id)
            if not self._owns_claim(outbox, claim):
                raise TelemetryPayloadInvalid()
            record = await session.get(InvocationRecord, claim.invocation_id)
            if record is None or not self._bindings_match(record, outbox):
                raise TelemetryPayloadInvalid()
            try:
                envelope = InvocationEnvelope.model_validate(record.payload_redacted)
            except Exception:
                raise TelemetryPayloadInvalid() from None
            if (
                envelope.invocation_id != claim.invocation_id
                or hash_invocation_envelope(envelope) != record.payload_sha256
            ):
                raise TelemetryPayloadInvalid()
            return envelope, TelemetryDelivery(
                invocation_id=outbox.invocation_id,
                tenant_id=outbox.tenant_id,
                case_id=outbox.case_id,
                run_id=outbox.run_id,
                topic=outbox.topic,
            )

    async def _mark_delivered(self, claim: _Claim) -> bool:
        now = self._aware_now()
        async with session_scope(
            self._session_factory,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        ) as session:
            outbox = await session.scalar(
                select(TelemetryOutbox)
                .where(TelemetryOutbox.id == claim.outbox_id)
                .with_for_update()
            )
            if not self._owns_claim(outbox, claim):
                return False
            outbox.status = "DELIVERED"
            outbox.locked_at = None
            outbox.locked_until = None
            outbox.last_error_code = None
            outbox.delivered_at = now
            outbox.dead_at = None
            await session.flush()
            return True

    async def _mark_failure(self, claim: _Claim, error: Exception) -> str:
        now = self._aware_now()
        code = safe_error_code(error, fallback="TELEMETRY_EXPORT_FAILED")
        async with session_scope(
            self._session_factory,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        ) as session:
            outbox = await session.scalar(
                select(TelemetryOutbox)
                .where(TelemetryOutbox.id == claim.outbox_id)
                .with_for_update()
            )
            if not self._owns_claim(outbox, claim):
                return "LOST_LEASE"
            outbox.locked_at = None
            outbox.locked_until = None
            outbox.last_error_code = code
            outbox.delivered_at = None
            if claim.attempt >= self._max_attempts:
                outbox.status = "DEAD"
                outbox.dead_at = now
                outcome = "DEAD"
            else:
                delay = min(
                    self._base_backoff_seconds * (2 ** (claim.attempt - 1)),
                    self._max_backoff_seconds,
                )
                outbox.status = "PENDING"
                outbox.available_at = now + timedelta(seconds=delay)
                outbox.dead_at = None
                outcome = "RETRIED"
            await session.flush()
            return outcome

    @staticmethod
    def _owns_claim(outbox: TelemetryOutbox | None, claim: _Claim) -> bool:
        return bool(
            outbox is not None
            and outbox.status == "PROCESSING"
            and outbox.attempts == claim.attempt
            and outbox.invocation_id == claim.invocation_id
            and outbox.locked_at is not None
            and outbox.locked_until is not None
        )

    @staticmethod
    def _bindings_match(record: InvocationRecord, outbox: TelemetryOutbox) -> bool:
        return (
            record.invocation_id == outbox.invocation_id
            and record.tenant_id == outbox.tenant_id
            and record.case_id == outbox.case_id
            and record.run_id == outbox.run_id
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RuntimeError("TELEMETRY_CLOCK_MUST_BE_TIMEZONE_AWARE")
        return value
