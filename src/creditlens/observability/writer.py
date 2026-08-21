"""Transactional persistence for redacted model and tool invocations.

The writer deliberately owns no transaction boundary.  Callers add an
``InvocationRecord`` and its ``TelemetryOutbox`` row to the same transaction as
the run checkpoint they describe.  Durability therefore begins only after that
outer transaction commits.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.common.clock import utc_now
from creditlens.common.errors import CreditLensError
from creditlens.infrastructure.postgres.models import InvocationRecord, TelemetryOutbox
from creditlens.observability.invocation import (
    InvocationEnvelope,
    PricingCatalog,
    adapt_model_invocation_trace,
    hash_invocation_envelope,
    invocation_run_event_payload,
)

INVOCATION_OUTBOX_TOPIC = "INVOCATION_TERMINATED"


class InvocationIdentityConflict(CreditLensError):
    """One invocation id was presented with different immutable content."""

    error_code = "INVOCATION_ID_CONFLICT"


class InvocationAuditPersistError(CreditLensError):
    """Stable public wrapper for a persistence failure at a workflow boundary."""

    error_code = "INVOCATION_AUDIT_PERSIST_FAILED"


@dataclass(frozen=True, slots=True)
class InvocationRecordResult:
    record: InvocationRecord
    outbox: TelemetryOutbox
    envelope: InvocationEnvelope
    created: bool


class InvocationWriter:
    """Write one immutable invocation and one retryable delivery intent."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        case_id: uuid.UUID,
        run_id: uuid.UUID,
        actor_role: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self._session = session
        self._tenant_id = uuid.UUID(str(tenant_id))
        self._case_id = uuid.UUID(str(case_id))
        self._run_id = uuid.UUID(str(run_id))
        self._actor_role = actor_role
        self._task_id = task_id

    async def record(self, envelope: InvocationEnvelope) -> InvocationRecordResult:
        """Persist ``envelope`` idempotently without committing the outer transaction.

        A nested transaction contains the unique-key race so a concurrent,
        identical insert does not poison the caller's business transaction.
        """

        resolved = self._with_context(envelope)
        payload = invocation_run_event_payload(resolved)
        payload_sha256 = hash_invocation_envelope(resolved)

        existing = await self._load_existing(resolved.invocation_id)
        if existing is not None:
            return self._validate_existing(existing, resolved, payload_sha256)

        record = InvocationRecord(
            invocation_id=resolved.invocation_id,
            tenant_id=self._tenant_id,
            case_id=self._case_id,
            run_id=self._run_id,
            contract_version=resolved.contract_version,
            kind=resolved.kind.value,
            name=resolved.name,
            provider=resolved.provider,
            model=resolved.model,
            version=resolved.version,
            actor_role=resolved.actor_role,
            task_id=resolved.task_id,
            status=resolved.status.value,
            ended_at=resolved.ended_at,
            payload_redacted=payload,
            payload_sha256=payload_sha256,
        )
        outbox = TelemetryOutbox(
            tenant_id=self._tenant_id,
            case_id=self._case_id,
            run_id=self._run_id,
            invocation_id=resolved.invocation_id,
            topic=INVOCATION_OUTBOX_TOPIC,
            status="PENDING",
            attempts=0,
            available_at=utc_now(),
        )
        dialect_name = self._session.bind.dialect.name if self._session.bind is not None else ""
        try:
            if dialect_name == "sqlite":
                # SQLite can make a SAVEPOINT the physical outer transaction
                # when no write has begun yet; releasing it would survive a
                # later SQLAlchemy rollback.  Local SQLite has no production
                # multi-worker claim guarantee, so preserve atomic rollback and
                # rely on the pre-read for sequential idempotency instead.
                self._session.add_all((record, outbox))
                await self._session.flush()
            else:
                async with self._session.begin_nested():
                    self._session.add_all((record, outbox))
                    await self._session.flush()
        except IntegrityError:
            if dialect_name == "sqlite":
                # The outer transaction is failed and cannot safely perform an
                # idempotency lookup without discarding the caller's work.
                raise
            # The savepoint rollback leaves the outer workflow transaction usable.
            existing = await self._load_existing(resolved.invocation_id)
            if existing is None:
                raise
            return self._validate_existing(existing, resolved, payload_sha256)
        return InvocationRecordResult(record, outbox, resolved, True)

    async def record_model_trace(
        self,
        trace: object,
        *,
        name: str = "structured_generation",
        actor_role: str | None = None,
        task_id: str | None = None,
        ended_at: datetime | None = None,
        pricing: PricingCatalog | None = None,
    ) -> InvocationRecordResult:
        envelope = adapt_model_invocation_trace(
            trace,
            name=name,
            actor_role=actor_role if actor_role is not None else self._actor_role,
            task_id=task_id if task_id is not None else self._task_id,
            ended_at=ended_at,
            pricing=pricing,
        )
        return await self.record(envelope)

    def _with_context(self, envelope: InvocationEnvelope) -> InvocationEnvelope:
        actor_role = envelope.actor_role or self._actor_role
        task_id = envelope.task_id or self._task_id
        if self._actor_role is not None and envelope.actor_role not in {None, self._actor_role}:
            raise InvocationAuditPersistError()
        if self._task_id is not None and envelope.task_id not in {None, self._task_id}:
            raise InvocationAuditPersistError()
        if actor_role == envelope.actor_role and task_id == envelope.task_id:
            return envelope
        payload: dict[str, Any] = envelope.model_dump(mode="python")
        payload.update({"actor_role": actor_role, "task_id": task_id})
        return InvocationEnvelope.model_validate(payload)

    async def _load_existing(
        self, invocation_id: uuid.UUID
    ) -> tuple[InvocationRecord, TelemetryOutbox | None] | None:
        record = await self._session.scalar(
            select(InvocationRecord).where(InvocationRecord.invocation_id == invocation_id)
        )
        if record is None:
            return None
        outbox = await self._session.scalar(
            select(TelemetryOutbox).where(TelemetryOutbox.invocation_id == invocation_id)
        )
        return record, outbox

    def _validate_existing(
        self,
        existing: tuple[InvocationRecord, TelemetryOutbox | None],
        envelope: InvocationEnvelope,
        payload_sha256: str,
    ) -> InvocationRecordResult:
        record, outbox = existing
        persisted_payload_matches = False
        try:
            persisted = InvocationEnvelope.model_validate(record.payload_redacted)
            persisted_payload_matches = (
                persisted.invocation_id == record.invocation_id
                and persisted.kind.value == record.kind
                and persisted.name == record.name
                and persisted.provider == record.provider
                and persisted.model == record.model
                and persisted.version == record.version
                and persisted.actor_role == record.actor_role
                and persisted.task_id == record.task_id
                and persisted.status.value == record.status
                and persisted.ended_at == record.ended_at
                and hash_invocation_envelope(persisted) == record.payload_sha256
            )
        except Exception:
            pass
        record_matches = (
            record.invocation_id == envelope.invocation_id
            and record.tenant_id == self._tenant_id
            and record.case_id == self._case_id
            and record.run_id == self._run_id
            and record.contract_version == envelope.contract_version
            and record.payload_sha256 == payload_sha256
            and persisted_payload_matches
        )
        outbox_matches = (
            outbox is not None
            and outbox.tenant_id == self._tenant_id
            and outbox.case_id == self._case_id
            and outbox.run_id == self._run_id
            and outbox.invocation_id == envelope.invocation_id
            and outbox.topic == INVOCATION_OUTBOX_TOPIC
        )
        if not record_matches:
            raise InvocationIdentityConflict()
        if not outbox_matches:
            raise InvocationAuditPersistError()
        return InvocationRecordResult(record, outbox, envelope, False)
