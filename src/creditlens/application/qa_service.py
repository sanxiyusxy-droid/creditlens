"""可审计 Grounded QA 应用服务（v1.3）。

边界：
- 事务 A 冻结 Snapshot 并创建 Run；事务 B 执行检索、生成、审计与持久化；
- Run 状态与答案状态分离，技术故障只能得到 FAILED Run，不能伪装成拒答；
- 仅持久化最终被 Claim 引用且通过确定性回表审计的 Evidence；
- RunEvent 只保存脱敏 Hash、版本、计数、延迟和错误类型，不保存 Prompt/模型原文。
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from creditlens.agents.auditor import (
    GROUNDING_AUDIT_IMPLEMENTATION_VERSION,
    EvidenceAuditor,
)
from creditlens.agents.contracts import (
    GROUNDED_ANSWER_CONTRACT_VERSION,
    GroundedAnswerArtifact,
    RefusalReasonCode,
)
from creditlens.agents.grounded_qa import (
    GroundedQAAgent,
    GroundedQAAuditFeedback,
    GroundedQAOutputRejected,
    grounded_qa_repair_hint,
)
from creditlens.application.snapshot_service import freeze_snapshot, load_snapshot_context
from creditlens.application.trusted_context import (
    acl_scope_hash,
    build_trusted_context,
)
from creditlens.common.clock import utc_now
from creditlens.common.errors import IdempotencyConflictError
from creditlens.common.hashing import sha256_text
from creditlens.formulas.engine import FormulaRegistry
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    CaseSnapshot,
    ClaimRecord,
    DocumentVersion,
    EvidenceRecord,
    ReviewRun,
    RunEvent,
)
from creditlens.infrastructure.postgres.session import checkpoint_commit, session_scope
from creditlens.observability.writer import (
    InvocationAuditPersistError,
    InvocationIdentityConflict,
    InvocationWriter,
)
from creditlens.retrieval.context_packing import PackedContext
from creditlens.retrieval.orchestrator import OrchestratorConfig, RetrievalOrchestrator
from creditlens.retrieval.query_spec import QuerySpec

_QA_TRANSITIONS: dict[str, set[str]] = {
    "RECEIVED": {"AUTHORIZED", "FAILED"},
    "AUTHORIZED": {"RETRIEVING", "FAILED"},
    "RETRIEVING": {"GENERATING", "AUDITING", "FAILED"},
    "GENERATING": {"AUDITING", "FAILED"},
    "AUDITING": {"GENERATING", "COMPLETED", "FAILED"},
}

_QA_TASK_ID = "grounded_qa"
_QA_PRODUCER = "grounded_qa"
_QA_REQUEST_HASH_VERSION = "grounded_qa_request_v2"
_SAFE_VIOLATION_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_REPLAY_INTEGRITY_ERROR = "IDEMPOTENT_REPLAY_INTEGRITY_FAILED"
_ANSWER_AUDIT_FAILED = "ANSWER_AUDIT_FAILED"
_QA_CALL_CANCELLED = "QA_CALL_CANCELLED"
_INVOCATION_ID_CONFLICT = "INVOCATION_ID_CONFLICT"
_INVOCATION_AUDIT_PERSIST_FAILED = "INVOCATION_AUDIT_PERSIST_FAILED"
_UNHANDLED_EXECUTION_ERROR = "UnhandledExecutionError"
_REPLAYABLE_FAILURE_ERROR_TYPES = frozenset(
    {
        _ANSWER_AUDIT_FAILED,
        _QA_CALL_CANCELLED,
        _INVOCATION_ID_CONFLICT,
        _INVOCATION_AUDIT_PERSIST_FAILED,
        "GroundedQAOutputRejected",
        "LLMCallError",
        _UNHANDLED_EXECUTION_ERROR,
    }
)
_SAFE_QA_CLAIM_CATEGORIES = frozenset(
    {
        "ELIGIBILITY",
        "FINANCIAL",
        "CASH_FLOW",
        "CONCENTRATION",
        "RELATED_PARTY",
        "DATA_CONFLICT",
        "MISSING_MATERIAL",
        "EXCEPTION",
    }
)


def _canonical_persisted_payload_hash(payload: dict[str, Any]) -> str:
    """Hash the exact JSON payload persisted in ``artifact_records.payload``."""
    return sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


class QAServiceError(Exception):
    """Grounded QA 已创建 Run 后发生的技术失败。"""

    def __init__(self, run_id: uuid.UUID, error_type: str):
        super().__init__(f"Grounded QA failed: {error_type}")
        self.run_id = run_id
        self.error_type = error_type


class _AnswerAuditFailed(RuntimeError):
    """Internal sentinel whose public failure type is stable and non-sensitive."""

    error_type = _ANSWER_AUDIT_FAILED

    def __init__(self) -> None:
        super().__init__(self.error_type)


class GroundedQAClaimResponse(BaseModel):
    claim_id: uuid.UUID
    category: str
    statement: str
    verdict: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    opposing_citations: list[dict[str, Any]] = Field(default_factory=list)


class GroundedQAResponse(BaseModel):
    schema_version: str = "1.0"
    question: str
    answer_status: str
    answer: str
    claims: list[GroundedQAClaimResponse] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    refusal_reason_code: RefusalReasonCode | None = None
    run_id: uuid.UUID
    snapshot_id: uuid.UUID
    state_version: int
    as_of_date: date
    generation_mode: str
    model_invocation_ids: list[uuid.UUID]
    trace_url: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    query_spec: dict[str, Any] = Field(default_factory=dict)
    channel_config: dict[str, Any] = Field(default_factory=dict)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    packing: dict[str, Any] | None = None
    idempotent_replay: bool = False


@dataclass(frozen=True)
class _QARequestReservation:
    run_id: uuid.UUID
    snapshot_id: uuid.UUID
    created: bool


@dataclass(frozen=True)
class _ReplayRetrieval:
    candidates: list[Any] = field(default_factory=list)
    query_spec: dict[str, Any] = field(default_factory=dict)
    channel_config: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    packing: dict[str, Any] | None = None


@dataclass
class QAService:
    session_factory: async_sessionmaker[AsyncSession]
    orchestrator: RetrievalOrchestrator
    settings: Any
    chat: Any = None
    tenant_id: uuid.UUID = field(
        default_factory=lambda: uuid.UUID("00000000-0000-0000-0000-000000000001")
    )
    user_id: uuid.UUID = field(
        default_factory=lambda: uuid.UUID("00000000-0000-0000-0000-000000000301")
    )
    auditor: EvidenceAuditor | None = None

    def __post_init__(self) -> None:
        if self.auditor is None:
            self.auditor = EvidenceAuditor(FormulaRegistry())
        prompt_path = (
            Path(__file__).resolve().parents[3]
            / "config"
            / "prompts"
            / (f"{self.settings.qa_prompt_version}.yaml")
        )
        self._qa_prompt_sha256 = sha256_text(prompt_path.read_text(encoding="utf-8"))
        self._audit_implementation_version = GROUNDING_AUDIT_IMPLEMENTATION_VERSION
        self._grounded_answer_contract_version = GROUNDED_ANSWER_CONTRACT_VERSION
        self.agent = GroundedQAAgent(
            self.chat,
            prompt_path=prompt_path,
            prompt_version=self.settings.qa_prompt_version,
            max_claims=self.settings.qa_max_claims,
            max_tokens=self.settings.qa_max_generation_tokens,
            allow_extractive_fallback=self.settings.qa_allow_extractive_fallback,
        )

    async def ask(
        self,
        *,
        case_id: uuid.UUID,
        question: str,
        top_k: int,
        idempotency_key: str,
        as_of_date: date | None = None,
        decision_cutoff_at: datetime | None = None,
    ) -> GroundedQAResponse:
        """冻结、检索、生成、审计并持久化一个 SIMPLE_QA Run。"""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not 8 <= len(idempotency_key) <= 128:
            raise ValueError("idempotency_key length must be between 8 and 128")
        if decision_cutoff_at is not None and (
            decision_cutoff_at.tzinfo is None or decision_cutoff_at.utcoffset() is None
        ):
            raise ValueError("decision_cutoff_at must include a timezone")
        request_hash = self._request_hash(
            case_id=case_id,
            question=normalized_question,
            top_k=top_k,
            as_of_date=as_of_date,
            decision_cutoff_at=decision_cutoff_at,
        )
        legacy_request_hash = self._request_hash(
            case_id=case_id,
            question=normalized_question,
            top_k=top_k,
            as_of_date=as_of_date,
            decision_cutoff_at=decision_cutoff_at,
            request_hash_version=None,
        )
        reservation = await self._create_run(
            case_id=case_id,
            question=normalized_question,
            top_k=top_k,
            as_of_date=as_of_date,
            decision_cutoff_at=decision_cutoff_at,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            legacy_request_hash=legacy_request_hash,
        )
        if not reservation.created:
            return await self._replay_completed(
                reservation.run_id,
                normalized_question,
            )
        try:
            return await self._execute(
                run_id=reservation.run_id,
                snapshot_id=reservation.snapshot_id,
                case_id=case_id,
                question=normalized_question,
                top_k=top_k,
            )
        except asyncio.CancelledError:
            # Cancellation must remain observable to the caller, but a Run that
            # has already checkpointed RETRIEVING/GENERATING must not be left in
            # a permanent non-terminal state when the database remains available.
            # Shield only the bounded terminal transition. A failed best-effort
            # cleanup must never replace the caller's cancellation control flow;
            # crash-safe reconciliation still requires a durable worker lease.
            timeout = float(
                getattr(self.settings, "invocation_cancel_persist_timeout_seconds", 2.0)
            )
            with contextlib.suppress(BaseException):
                await _persist_terminal_with_cancellation_drain(
                    self._mark_failed(reservation.run_id, _QA_CALL_CANCELLED),
                    timeout_seconds=timeout,
                    bound_normal_wait=True,
                )
            raise
        except Exception as exc:
            error_type = _execution_error_type(exc)
            await self._mark_failed(reservation.run_id, error_type)
            raise QAServiceError(reservation.run_id, error_type) from exc

    async def _create_run(
        self,
        *,
        case_id: uuid.UUID,
        question: str,
        top_k: int,
        as_of_date: date | None,
        decision_cutoff_at: datetime | None,
        idempotency_key: str,
        request_hash: str,
        legacy_request_hash: str,
    ) -> _QARequestReservation:
        config = self._orchestrator_config(top_k)
        try:
            async with session_scope(
                self.session_factory,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
            ) as session:
                existing = await session.scalar(
                    select(ReviewRun).where(
                        ReviewRun.tenant_id == self.tenant_id,
                        ReviewRun.request_idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return await self._existing_reservation(
                        session,
                        existing,
                        request_hash,
                        legacy_request_hash,
                    )

                trusted = await build_trusted_context(
                    session,
                    tenant_id=self.tenant_id,
                    case_id=case_id,
                    user_id=self.user_id,
                    purpose="grounded_qa",
                )
                update: dict[str, Any] = {}
                if as_of_date is not None:
                    update["as_of_date"] = as_of_date
                if decision_cutoff_at is not None:
                    cutoff = decision_cutoff_at
                    if cutoff.tzinfo is None:
                        cutoff = cutoff.replace(tzinfo=UTC)
                    update["decision_cutoff_at"] = cutoff.astimezone(UTC)
                if update:
                    trusted = trusted.model_copy(update=update)

                snapshot = await freeze_snapshot(
                    session,
                    trusted,
                    chunks_collection=self.settings.chunks_collection_name,
                    summaries_collection=self.settings.summaries_collection_name,
                    acl_hash=acl_scope_hash(trusted),
                )
                run = ReviewRun(
                    tenant_id=trusted.tenant_id,
                    case_id=trusted.case_id,
                    run_type="SIMPLE_QA",
                    status="RECEIVED",
                    as_of_date=trusted.as_of_date,
                    decision_cutoff_at=trusted.decision_cutoff_at,
                    input_snapshot_id=snapshot.snapshot_id,
                    retrieval_config=config.model_dump(mode="json"),
                    model_manifest={
                        "workflow": "grounded_qa_v1",
                        "request_hash_version": _QA_REQUEST_HASH_VERSION,
                        "prompt_version": self.settings.qa_prompt_version,
                        "provider": self.settings.llm_provider,
                        "model": self.settings.llm_model or None,
                        "audit": self._audit_implementation_version,
                        "audit_contract_version": self._grounded_answer_contract_version,
                        "semantic_entailment_verified": False,
                        "model_invocation_ids": [],
                    },
                    request_idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    created_by=self.user_id,
                )
                session.add(run)
                await session.flush()
                session.add(
                    RunEvent(
                        run_id=run.id,
                        tenant_id=run.tenant_id,
                        case_id=run.case_id,
                        sequence_no=1,
                        event_type="RUN_CREATED",
                        payload_redacted={
                            "run_type": "SIMPLE_QA",
                            "question_hash": sha256_text(question),
                            "idempotency_key_hash": sha256_text(idempotency_key),
                            "snapshot_id": str(snapshot.snapshot_id),
                            "request_hash_version": _QA_REQUEST_HASH_VERSION,
                        },
                    )
                )
                await session.flush()
                return _QARequestReservation(run.id, snapshot.snapshot_id, True)
        except IntegrityError:
            # Concurrent requests may both miss the initial read.  The database
            # unique constraint is authoritative; after rollback, resolve the
            # winning Run and never execute the loser a second time.
            return await self._reservation_after_unique_conflict(
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                legacy_request_hash=legacy_request_hash,
            )

    async def _existing_reservation(
        self,
        session: AsyncSession,
        run: ReviewRun,
        request_hash: str,
        legacy_request_hash: str,
    ) -> _QARequestReservation:
        manifest = run.model_manifest if isinstance(run.model_manifest, dict) else {}
        request_hash_version = manifest.get("request_hash_version")
        current_request_matches = (
            request_hash_version == _QA_REQUEST_HASH_VERSION
            and isinstance(run.request_hash, str)
            and hmac.compare_digest(run.request_hash, request_hash)
        )
        legacy_completed_candidate = (
            run.status == "COMPLETED"
            and request_hash_version is None
            and isinstance(run.request_hash, str)
            and hmac.compare_digest(run.request_hash, legacy_request_hash)
        )
        legacy_completed_matches = legacy_completed_candidate and await self._is_v13_run_contract(
            session,
            run,
        )
        if run.run_type != "SIMPLE_QA" or not (current_request_matches or legacy_completed_matches):
            raise IdempotencyConflictError(
                "同一 idempotency_key 已用于不同的问答请求",
                {"run_id": str(run.id), "status": run.status},
            )
        if run.input_snapshot_id is None:
            raise QAServiceError(run.id, "IDEMPOTENT_RUN_WITHOUT_SNAPSHOT")
        if run.status == "COMPLETED":
            return _QARequestReservation(run.id, run.input_snapshot_id, False)
        if run.status == "FAILED":
            failures = (
                await session.scalars(
                    select(RunEvent)
                    .where(
                        RunEvent.run_id == run.id,
                        RunEvent.event_type == "QA_EXECUTION_FAILED",
                    )
                    .order_by(RunEvent.sequence_no)
                )
            ).all()
            last_sequence = await session.scalar(
                select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run.id)
            )
            if len(failures) != 1:
                raise QAServiceError(run.id, _REPLAY_INTEGRITY_ERROR)
            failed = failures[0]
            payload = failed.payload_redacted
            failure_from = payload.get("from") if isinstance(payload, dict) else None
            error_type = (
                _stable_failure_error_type(payload.get("error_type"))
                if isinstance(payload, dict)
                else None
            )
            failure_event_matches = (
                failed.tenant_id == run.tenant_id
                and failed.case_id == run.case_id
                and failed.sequence_no == last_sequence
                and isinstance(payload, dict)
                and set(payload) == {"from", "error_type"}
                and isinstance(failure_from, str)
                and "FAILED" in _QA_TRANSITIONS.get(failure_from, set())
                and error_type is not None
            )
            if not failure_event_matches:
                raise QAServiceError(run.id, _REPLAY_INTEGRITY_ERROR)
            raise QAServiceError(run.id, error_type)
        raise IdempotencyConflictError(
            "同一幂等请求正在执行，请使用原 run_id 查询状态",
            {"run_id": str(run.id), "status": run.status, "retryable": True},
        )

    async def _is_v13_run_contract(self, session: AsyncSession, run: ReviewRun) -> bool:
        """Recognize an intact v1.3 Run rather than a downgraded v2 manifest."""
        created_events = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == run.id,
                    RunEvent.event_type == "RUN_CREATED",
                )
            )
        ).all()
        if len(created_events) != 1:
            return False
        created = created_events[0]
        payload = created.payload_redacted
        if not isinstance(payload, dict) or set(payload) != {
            "run_type",
            "question_hash",
            "idempotency_key_hash",
            "snapshot_id",
        }:
            return False
        return (
            created.sequence_no == 1
            and created.tenant_id == run.tenant_id
            and created.case_id == run.case_id
            and payload.get("run_type") == "SIMPLE_QA"
            and payload.get("snapshot_id") == str(run.input_snapshot_id)
            and isinstance(run.request_idempotency_key, str)
            and payload.get("idempotency_key_hash") == sha256_text(run.request_idempotency_key)
        )

    async def _reservation_after_unique_conflict(
        self,
        *,
        idempotency_key: str,
        request_hash: str,
        legacy_request_hash: str,
    ) -> _QARequestReservation:
        async with session_scope(
            self.session_factory,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        ) as session:
            existing = await session.scalar(
                select(ReviewRun).where(
                    ReviewRun.tenant_id == self.tenant_id,
                    ReviewRun.request_idempotency_key == idempotency_key,
                )
            )
            if existing is None:
                raise RuntimeError("QA_IDEMPOTENCY_RESOLUTION_FAILED")
            return await self._existing_reservation(
                session,
                existing,
                request_hash,
                legacy_request_hash,
            )

    def _request_hash(
        self,
        *,
        case_id: uuid.UUID,
        question: str,
        top_k: int,
        as_of_date: date | None,
        decision_cutoff_at: datetime | None,
        request_hash_version: str | None = _QA_REQUEST_HASH_VERSION,
    ) -> str:
        cutoff = decision_cutoff_at
        if cutoff is not None:
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=UTC)
            cutoff = cutoff.astimezone(UTC)
        payload = {
            "case_id": str(case_id),
            "question": question,
            "top_k": top_k,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "decision_cutoff_at": cutoff.isoformat() if cutoff else None,
            "prompt_version": self.settings.qa_prompt_version,
            "prompt_sha256": self._qa_prompt_sha256,
            "llm_provider": self.settings.llm_provider,
            "llm_model": self.settings.llm_model or None,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_version": self.settings.effective_embedding_version,
            "rerank_provider": self.settings.rerank_provider,
            "rerank_model": self.settings.rerank_model or None,
            "chunks_collection": self.settings.chunks_collection_name,
            "summaries_collection": self.settings.summaries_collection_name,
            "qa_max_claims": self.settings.qa_max_claims,
            "qa_max_generation_tokens": self.settings.qa_max_generation_tokens,
            "qa_max_audit_repairs": self.settings.qa_max_audit_repairs,
            "qa_allow_extractive_fallback": self.settings.qa_allow_extractive_fallback,
            "audit_implementation_version": self._audit_implementation_version,
            "grounded_answer_contract_version": self._grounded_answer_contract_version,
            "sparse_encoder_version": getattr(self.settings, "sparse_encoder_version", None),
            "orchestrator_runtime": {
                "rrf_k": getattr(
                    self.orchestrator,
                    "rrf_k",
                    getattr(self.settings, "rrf_k", None),
                ),
                "route_weights": getattr(self.orchestrator, "route_weights", None),
                "embedding_version": getattr(
                    getattr(self.orchestrator, "embedder", None),
                    "version",
                    self.settings.effective_embedding_version,
                ),
                "reranker_version": getattr(
                    getattr(self.orchestrator, "reranker", None),
                    "version",
                    self.settings.rerank_model or None,
                ),
            },
            "retrieval": self._orchestrator_config(top_k).model_dump(mode="json"),
        }
        if request_hash_version is not None:
            payload["request_hash_version"] = request_hash_version
        return sha256_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    async def _replay_completed(
        self,
        run_id: uuid.UUID,
        question: str,
    ) -> GroundedQAResponse:
        async with session_scope(
            self.session_factory,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        ) as session:
            run = await session.get(ReviewRun, run_id)
            if run is None or run.status != "COMPLETED" or run.input_snapshot_id is None:
                raise QAServiceError(run_id, "IDEMPOTENT_REPLAY_NOT_AVAILABLE")
            record = await session.scalar(
                select(ArtifactRecord)
                .where(
                    ArtifactRecord.run_id == run.id,
                    ArtifactRecord.artifact_type == "GROUNDED_ANSWER",
                )
                .order_by(ArtifactRecord.created_at.desc())
            )
            if record is None:
                raise QAServiceError(run_id, "IDEMPOTENT_REPLAY_ARTIFACT_MISSING")
            try:
                if not isinstance(record.payload, dict):
                    raise ValueError("invalid persisted payload")
                payload = dict(record.payload)

                # Verify the untouched persisted object before removing service
                # metadata or asking Pydantic to coerce any values.
                actual_hash = _canonical_persisted_payload_hash(payload)
                expected_hash = record.output_hash
                if not isinstance(expected_hash, str) or not hmac.compare_digest(
                    actual_hash, expected_hash
                ):
                    raise ValueError("persisted payload hash mismatch")

                artifact_payload = dict(payload)
                generation_mode = str(artifact_payload.pop("generation_mode", "unknown"))
                replay_query_spec = artifact_payload.pop("retrieval_query_spec", {})
                if not isinstance(replay_query_spec, dict):
                    raise ValueError("invalid persisted query spec")
                replay_query_spec = _canonical_query_spec(replay_query_spec)
                artifact = GroundedAnswerArtifact.model_validate(artifact_payload)
                self._validate_replayed_artifact(
                    run=run,
                    record=record,
                    artifact=artifact,
                    generation_mode=generation_mode,
                )
            except QAServiceError:
                raise
            except Exception:
                # Payload/metadata may contain untrusted model-authored text.
                # Never expose a validation message or the stored answer.
                raise QAServiceError(run_id, _REPLAY_INTEGRITY_ERROR) from None
            return _build_response(
                run=run,
                question=question,
                snapshot_id=run.input_snapshot_id,
                artifact=artifact,
                generation_mode=generation_mode,
                retrieval=_ReplayRetrieval(
                    query_spec=replay_query_spec,
                    channel_config={"idempotent_replay": True},
                ),
                idempotent_replay=True,
            )

    def _validate_replayed_artifact(
        self,
        *,
        run: ReviewRun,
        record: ArtifactRecord,
        artifact: GroundedAnswerArtifact,
        generation_mode: str,
    ) -> None:
        """Bind replayed JSON to its immutable record and originating Run."""
        manifest = run.model_manifest if isinstance(run.model_manifest, dict) else {}
        raw_manifest_ids = manifest.get("model_invocation_ids")
        if not isinstance(raw_manifest_ids, list):
            raise ValueError("invalid model invocation manifest")
        try:
            manifest_ids = {uuid.UUID(str(value)) for value in raw_manifest_ids}
        except (AttributeError, TypeError, ValueError):
            raise ValueError("invalid model invocation manifest") from None

        expected_model = self.settings.llm_model or None
        checks = (
            run.tenant_id == self.tenant_id,
            run.run_type == "SIMPLE_QA",
            record.tenant_id == run.tenant_id,
            record.run_id == run.id,
            record.id == artifact.artifact_id,
            record.artifact_type == "GROUNDED_ANSWER",
            record.task_id == artifact.task_id == _QA_TASK_ID,
            record.producer == artifact.producer == _QA_PRODUCER,
            record.lifecycle_status == artifact.lifecycle_status == "VERIFIED",
            record.execution_status == artifact.execution_status,
            record.contract_version == artifact.contract_version,
            record.input_hash == artifact.input_hash,
            artifact.run_id == run.id,
            artifact.prompt_version == self.settings.qa_prompt_version,
            manifest.get("workflow") == "grounded_qa_v1",
            manifest.get("prompt_version") == artifact.prompt_version,
            manifest.get("provider") == self.settings.llm_provider,
            manifest.get("model") == expected_model,
            manifest.get("audit") == self._audit_implementation_version,
            manifest.get("audit_contract_version") == self._grounded_answer_contract_version,
            artifact.contract_version == self._grounded_answer_contract_version,
            set(artifact.model_invocation_ids).issubset(manifest_ids),
            generation_mode in {"llm", "deterministic_extractive", "abstained_empty_context"},
        )
        if not all(checks):
            raise ValueError("replayed artifact metadata mismatch")

    async def _execute(
        self,
        *,
        run_id: uuid.UUID,
        snapshot_id: uuid.UUID,
        case_id: uuid.UUID,
        question: str,
        top_k: int,
    ) -> GroundedQAResponse:
        config = self._orchestrator_config(top_k)
        async with session_scope(
            self.session_factory, tenant_id=self.tenant_id, user_id=self.user_id
        ) as session:
            run = await session.scalar(
                select(ReviewRun).where(ReviewRun.id == run_id).with_for_update()
            )
            if run is None:
                raise RuntimeError("QA_RUN_NOT_FOUND")
            if run.status != "RECEIVED":
                raise RuntimeError(f"QA_RUN_ALREADY_CLAIMED:{run.status}")
            snapshot_record = await session.get(CaseSnapshot, snapshot_id)
            if snapshot_record is None:
                raise RuntimeError("QA_SNAPSHOT_NOT_FOUND")
            if (
                run.tenant_id != self.tenant_id
                or run.case_id != case_id
                or run.input_snapshot_id != snapshot_record.id
                or snapshot_record.id != snapshot_id
                or snapshot_record.tenant_id != run.tenant_id
                or snapshot_record.case_id != run.case_id
            ):
                raise RuntimeError("QA_SNAPSHOT_BINDING_MISMATCH")
            trusted = await build_trusted_context(
                session,
                tenant_id=self.tenant_id,
                case_id=case_id,
                user_id=self.user_id,
                purpose="grounded_qa",
            )
            trusted = trusted.model_copy(
                update={
                    "as_of_date": run.as_of_date,
                    "decision_cutoff_at": run.decision_cutoff_at,
                }
            )
            snapshot = await load_snapshot_context(session, snapshot_record.id)
            events = _QAEventWriter(session, run)
            run.model_manifest = {
                **(run.model_manifest or {}),
                "invocation_contract_version": "invocation_v2",
            }
            await events.transition("AUTHORIZED")
            await checkpoint_commit(session)
            await events.transition("RETRIEVING")
            await checkpoint_commit(session)

            retrieval = await self.orchestrator.retrieve(
                session,
                trusted,
                question,
                snapshot.chunks_collection,
                config=config,
                snapshot=snapshot,
                summaries_collection=snapshot.summaries_collection,
            )
            packed = _packed_context(retrieval.packing, config.token_budget)
            await events.emit(
                "RETRIEVAL_COMPLETED",
                {
                    "question_hash": sha256_text(question),
                    "candidate_count": len(retrieval.candidates),
                    "packed_count": len(packed.sections),
                    "packed_section_ids": [str(section.section_id) for section in packed.sections],
                    "rerank_degraded": retrieval.rerank_degraded,
                    "rerank_degraded_reason": retrieval.rerank_degraded_reason,
                    "routes": retrieval.channel_config.get("routes", []),
                },
            )
            await checkpoint_commit(session)

            if packed.sections:
                await events.transition("GENERATING")
                await checkpoint_commit(session)
            else:
                await events.transition("AUDITING")
                await checkpoint_commit(session)

            generation, generation_repair_count = await self._generate_with_output_repairs(
                events,
                question=question,
                run_id=run.id,
                as_of_date=trusted.as_of_date,
                packed=packed,
            )
            if packed.sections:
                await events.transition("AUDITING")
            await checkpoint_commit(session)

            allowed_ids = {
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"evidence:{section.section_id}:{section.text_hash}",
                )
                for section in packed.sections
            }
            artifact = generation.artifact
            audit = await self.auditor.verify_grounded_answer(
                session,
                trusted,
                artifact,
                allowed_evidence_ids=allowed_ids,
                snapshot=snapshot,
            )
            artifact, review_normalized = _normalize_non_blocking_review(artifact, audit)

            audit_repair_count = 0
            while not audit.ok and audit_repair_count < self.settings.qa_max_audit_repairs:
                audit_feedback = _audit_repair_feedback(audit.violations, artifact)
                await events.emit(
                    "ANSWER_AUDIT_REJECTED",
                    {
                        "repair_attempt": audit_repair_count + 1,
                        "violation_codes": _violation_codes(audit.violations),
                    },
                )
                await events.transition("GENERATING")
                await checkpoint_commit(session)
                generation, output_repairs = await self._generate_with_output_repairs(
                    events,
                    question=question,
                    run_id=run.id,
                    as_of_date=trusted.as_of_date,
                    packed=packed,
                    audit_feedback=audit_feedback,
                )
                generation_repair_count += output_repairs
                await events.transition("AUDITING")
                await checkpoint_commit(session)
                artifact = generation.artifact
                audit = await self.auditor.verify_grounded_answer(
                    session,
                    trusted,
                    artifact,
                    allowed_evidence_ids=allowed_ids,
                    snapshot=snapshot,
                )
                artifact, normalized_after_repair = _normalize_non_blocking_review(artifact, audit)
                review_normalized = review_normalized or normalized_after_repair
                audit_repair_count += 1

            if not audit.ok:
                await events.emit(
                    "ANSWER_AUDIT_REJECTED",
                    {
                        "repair_attempt": audit_repair_count,
                        "violation_codes": _violation_codes(audit.violations),
                        "terminal": True,
                    },
                )
                await checkpoint_commit(session)
                raise _AnswerAuditFailed()

            await events.emit(
                "ANSWER_AUDIT_COMPLETED",
                {
                    "answer_status": audit.derived_answer_status,
                    "claim_count": len(artifact.claims),
                    "repair_count": audit_repair_count,
                    "review_normalized": review_normalized,
                    "violation_codes": _violation_codes(audit.violations),
                },
            )
            await self._persist_answer(
                session,
                run,
                artifact,
                generation.generation_mode,
                generation.model_traces,
                retrieval.query_spec,
            )
            run.model_manifest = {
                **(run.model_manifest or {}),
                "generation_mode": generation.generation_mode,
                "generation_repairs": generation_repair_count,
                "audit_repairs": audit_repair_count,
                "review_normalized": review_normalized,
            }
            await events.emit(
                "ANSWER_PERSISTED",
                {
                    "artifact_id": str(artifact.artifact_id),
                    "answer_status": audit.derived_answer_status,
                    "claim_count": len(artifact.claims),
                    "citation_count": len(artifact.evidence),
                },
            )
            await events.transition("COMPLETED")
            run.completed_at = utc_now()
            await session.flush()

            return _build_response(
                run=run,
                question=question,
                snapshot_id=snapshot_id,
                artifact=artifact,
                generation_mode=generation.generation_mode,
                retrieval=retrieval,
            )

    async def _persist_answer(
        self,
        session: AsyncSession,
        run: ReviewRun,
        artifact: GroundedAnswerArtifact,
        generation_mode: str,
        model_traces: list[Any],
        query_spec: dict[str, Any],
    ) -> None:
        self._validate_artifact_provenance(run, artifact, model_traces)
        persisted_payload = artifact.model_dump(mode="json", exclude={"output_hash"})
        persisted_payload["lifecycle_status"] = "VERIFIED"
        persisted_payload["generation_mode"] = generation_mode
        persisted_payload["retrieval_query_spec"] = _canonical_query_spec(query_spec)
        output_hash = _canonical_persisted_payload_hash(persisted_payload)
        session.add(
            ArtifactRecord(
                id=artifact.artifact_id,
                tenant_id=run.tenant_id,
                run_id=run.id,
                task_id=_QA_TASK_ID,
                artifact_type="GROUNDED_ANSWER",
                contract_version=artifact.contract_version,
                producer=_QA_PRODUCER,
                lifecycle_status=persisted_payload["lifecycle_status"],
                execution_status=artifact.execution_status,
                payload=persisted_payload,
                input_hash=artifact.input_hash,
                output_hash=output_hash,
            )
        )
        answer_status = getattr(artifact.answer_status, "value", artifact.answer_status)
        claim_review_status = "PENDING" if answer_status == "NEEDS_REVIEW" else "AUDITED"
        for evidence in artifact.evidence:
            source_available_at = evidence.source_available_at
            if evidence.document_version_id is not None:
                version = await session.get(DocumentVersion, evidence.document_version_id)
                if version is not None and version.source_available_at is not None:
                    source_available_at = version.source_available_at
            session.add(
                EvidenceRecord(
                    evidence_key=evidence.evidence_id,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type=evidence.evidence_type,
                    source_id=evidence.source_id,
                    document_version_id=evidence.document_version_id,
                    section_id=evidence.section_id,
                    page_number=evidence.page_number,
                    locator={
                        "document_version_id": str(evidence.document_version_id)
                        if evidence.document_version_id
                        else None,
                        "parse_run_id": str(evidence.parse_run_id)
                        if evidence.parse_run_id
                        else None,
                        "section_id": str(evidence.section_id) if evidence.section_id else None,
                        "page_number": evidence.page_number,
                    },
                    content_hash=evidence.content_hash,
                    snapshot={
                        "snapshot_id": str(run.input_snapshot_id),
                        "as_of_date": run.as_of_date.isoformat(),
                    },
                    valid_from=evidence.valid_from,
                    valid_to=evidence.valid_to,
                    source_available_at=source_available_at,
                )
            )
        for claim in artifact.claims:
            session.add(
                ClaimRecord(
                    id=claim.claim_id,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    artifact_id=artifact.artifact_id,
                    category=claim.category,
                    statement=claim.statement,
                    verdict=claim.verdict,
                    severity=claim.severity,
                    confidence_level="MEDIUM",
                    as_of_date=claim.as_of_date,
                    uncertainty_reason=claim.uncertainty_reason,
                    review_status=claim_review_status,
                    payload={
                        "supporting_evidence_ids": [
                            str(item) for item in claim.supporting_evidence_ids
                        ],
                        "opposing_evidence_ids": [
                            str(item) for item in claim.opposing_evidence_ids
                        ],
                        "calculation_ids": [str(item) for item in claim.calculation_ids],
                        "answer_status": answer_status,
                    },
                )
            )
        await session.flush()

    def _validate_artifact_provenance(
        self,
        run: ReviewRun,
        artifact: GroundedAnswerArtifact,
        model_traces: list[Any],
    ) -> None:
        """Reject agent-owned identity/provenance before any answer rows are added."""
        if artifact.run_id != run.id:
            raise RuntimeError("QA_ARTIFACT_RUN_ID_MISMATCH")
        if artifact.task_id != _QA_TASK_ID:
            raise RuntimeError("QA_ARTIFACT_TASK_ID_MISMATCH")
        if artifact.producer != _QA_PRODUCER:
            raise RuntimeError("QA_ARTIFACT_PRODUCER_MISMATCH")
        if artifact.prompt_version != self.settings.qa_prompt_version:
            raise RuntimeError("QA_ARTIFACT_PROVENANCE_MISMATCH")
        if artifact.contract_version != self._grounded_answer_contract_version:
            raise RuntimeError("QA_ARTIFACT_CONTRACT_VERSION_MISMATCH")

        answer_status = getattr(artifact.answer_status, "value", artifact.answer_status)
        if (answer_status == "ABSTAINED") != (artifact.refusal_reason_code is not None):
            # model_copy(update=...) can bypass Pydantic validators.  Recheck at
            # the persistence boundary so an invalid reason/status pair cannot
            # enter the audit record or later be replayed.
            raise RuntimeError("QA_ARTIFACT_REFUSAL_REASON_MISMATCH")

        trace_ids = _model_invocation_ids(model_traces)
        if len(trace_ids) != len(model_traces) or artifact.model_invocation_ids != trace_ids:
            raise RuntimeError("QA_ARTIFACT_PROVENANCE_MISMATCH")
        trace_prompt_versions = [_trace_value(trace, "prompt_version") for trace in model_traces]
        if any(value != artifact.prompt_version for value in trace_prompt_versions):
            raise RuntimeError("QA_ARTIFACT_PROVENANCE_MISMATCH")

    async def _record_model_traces(
        self,
        run: ReviewRun,
        traces: list[Any],
    ) -> None:
        if not traces:
            return

        async def persist_all() -> None:
            for trace in traces:
                async with session_scope(
                    self.session_factory,
                    tenant_id=run.tenant_id,
                    user_id=self.user_id,
                ) as audit_session:
                    writer = InvocationWriter(
                        audit_session,
                        tenant_id=run.tenant_id,
                        case_id=run.case_id,
                        run_id=run.id,
                        actor_role=_QA_PRODUCER,
                        task_id=_QA_TASK_ID,
                    )
                    await writer.record_model_trace(
                        trace,
                        name="grounded_qa_generation",
                        actor_role=_QA_PRODUCER,
                        task_id=_QA_TASK_ID,
                    )

        timeout = float(getattr(self.settings, "invocation_cancel_persist_timeout_seconds", 2.0))
        try:
            await _persist_terminal_with_cancellation_drain(
                persist_all(),
                timeout_seconds=timeout,
                bound_normal_wait=any(
                    getattr(trace, "status", None) == "CANCELLED" for trace in traces
                ),
            )
        except asyncio.CancelledError:
            raise
        except (InvocationIdentityConflict, InvocationAuditPersistError):
            raise
        except Exception:
            # Never expose SQL/provider details and never continue a bank
            # answer path whose invocation ledger could not be persisted.
            raise InvocationAuditPersistError() from None

    async def _generate(
        self,
        events: _QAEventWriter,
        **kwargs,
    ):
        """调用 Agent，并保证 Provider 失败 Trace 在异常返回前持久化。"""
        try:
            return await self.agent.generate(**kwargs)
        except asyncio.CancelledError as exc:
            trace = getattr(exc, "trace", None)
            if trace is not None:
                with contextlib.suppress(BaseException):
                    await self._record_model_traces(events.run, [trace])
            raise
        except Exception as exc:
            trace = getattr(exc, "trace", None)
            if trace is not None:
                await self._record_model_traces(events.run, [trace])
                self._append_model_invocations(events.run, [trace])
                await checkpoint_commit(events.session)
            raise

    async def _generate_with_output_repairs(
        self,
        events: _QAEventWriter,
        **kwargs,
    ) -> tuple[Any, int]:
        """Retry bounded, allow-listed model-output rejections without abstaining."""
        feedback = list(kwargs.pop("audit_feedback", None) or [])
        repair_count = 0
        while True:
            try:
                generation = await self._generate(
                    events,
                    **kwargs,
                    audit_feedback=feedback or None,
                )
            except GroundedQAOutputRejected as exc:
                code = _safe_error_code(exc.error_code)
                terminal = repair_count >= self.settings.qa_max_audit_repairs
                await events.emit(
                    "ANSWER_GENERATION_REJECTED",
                    {
                        "repair_attempt": repair_count if terminal else repair_count + 1,
                        "error_code": code,
                        "terminal": terminal,
                    },
                )
                await checkpoint_commit(events.session)
                if terminal:
                    raise
                repair_count += 1
                if code not in feedback:
                    feedback.append(code)
                continue

            await self._record_model_traces(events.run, generation.model_traces)
            self._append_model_invocations(events.run, generation.model_traces)
            return generation, repair_count

    @staticmethod
    def _append_model_invocations(run: ReviewRun, traces: list[Any]) -> None:
        manifest = dict(run.model_manifest or {})
        invocation_ids = list(manifest.get("model_invocation_ids") or [])
        for invocation_id in _model_invocation_ids(traces):
            value = str(invocation_id)
            if value not in invocation_ids:
                invocation_ids.append(value)
        manifest["model_invocation_ids"] = invocation_ids
        run.model_manifest = manifest

    async def _mark_failed(self, run_id: uuid.UUID, error_type: str) -> None:
        async with session_scope(
            self.session_factory, tenant_id=self.tenant_id, user_id=self.user_id
        ) as session:
            run = await session.scalar(
                select(ReviewRun).where(ReviewRun.id == run_id).with_for_update()
            )
            if run is None or run.status == "COMPLETED":
                return
            if run.status == "FAILED":
                if run.completed_at is None:
                    run.completed_at = utc_now()
                    await session.flush()
                return
            old = run.status
            run.status = "FAILED"
            run.state_version += 1
            run.completed_at = utc_now()
            last_sequence = await session.scalar(
                select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run.id)
            )
            session.add(
                RunEvent(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    case_id=run.case_id,
                    sequence_no=(last_sequence or 0) + 1,
                    event_type="QA_EXECUTION_FAILED",
                    payload_redacted={
                        "from": old,
                        "error_type": error_type,
                    },
                )
            )
            await session.flush()

    def _orchestrator_config(self, top_k: int) -> OrchestratorConfig:
        return OrchestratorConfig(
            final_limit=top_k,
            enable_rerank=self.settings.orchestrator_enable_rerank,
            enable_summary=self.settings.orchestrator_enable_summary,
            enable_exact=self.settings.orchestrator_enable_exact,
            enable_packing=True,
            token_budget=self.settings.context_token_budget,
            max_per_document_ratio=self.settings.context_max_per_document_ratio,
            expand_adjacent=self.settings.context_expand_adjacent,
        )


class _QAEventWriter:
    def __init__(self, session: AsyncSession, run: ReviewRun):
        self.session = session
        self.run = run

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        last_sequence = await self.session.scalar(
            select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == self.run.id)
        )
        self.session.add(
            RunEvent(
                run_id=self.run.id,
                tenant_id=self.run.tenant_id,
                case_id=self.run.case_id,
                sequence_no=(last_sequence or 0) + 1,
                event_type=event_type,
                payload_redacted=payload,
            )
        )
        await self.session.flush()

    async def transition(self, new_status: str) -> None:
        if new_status not in _QA_TRANSITIONS.get(self.run.status, set()):
            raise RuntimeError(f"INVALID_QA_TRANSITION:{self.run.status}:{new_status}")
        old = self.run.status
        self.run.status = new_status
        self.run.state_version += 1
        await self.emit("STATE_CHANGED", {"from": old, "to": new_status})


def _packed_context(raw: dict[str, Any] | None, budget: int) -> PackedContext:
    if raw:
        return PackedContext.model_validate(raw)
    return PackedContext(sections=[], total_tokens_est=0, budget=budget)


async def _persist_terminal_with_cancellation_drain(
    persistence,
    *,
    timeout_seconds: float,
    bound_normal_wait: bool = False,
):
    """Finish a started short audit transaction before propagating cancellation."""

    persistence_task = asyncio.create_task(persistence)
    persistence_task.add_done_callback(_consume_background_task_result)
    try:
        if bound_normal_wait:
            return await asyncio.wait_for(
                asyncio.shield(persistence_task),
                timeout=timeout_seconds,
            )
        return await asyncio.shield(persistence_task)
    except asyncio.CancelledError:
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(
                asyncio.shield(persistence_task),
                timeout=timeout_seconds,
            )
        raise


def _consume_background_task_result(task: asyncio.Task) -> None:
    """Observe a timed-out shielded audit task without leaking its exception."""

    with contextlib.suppress(BaseException):
        task.result()


def _trace_value(trace: Any, field_name: str) -> Any:
    if isinstance(trace, dict):
        return trace.get(field_name)
    return getattr(trace, field_name, None)


def _model_invocation_ids(traces: list[Any]) -> list[uuid.UUID]:
    invocation_ids: list[uuid.UUID] = []
    for trace in traces:
        value = _trace_value(trace, "invocation_id") or _trace_value(trace, "id")
        try:
            invocation_id = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        except (AttributeError, TypeError, ValueError):
            continue
        if invocation_id not in invocation_ids:
            invocation_ids.append(invocation_id)
    return invocation_ids


def _safe_error_code(raw: Any) -> str:
    code = str(raw).strip()
    if not _SAFE_VIOLATION_CODE.fullmatch(code):
        raise RuntimeError("INVALID_QA_REPAIR_CODE")
    return code


def _execution_error_type(exc: Exception) -> str:
    """Map internal sentinels to a stable public type without exposing messages."""
    if isinstance(exc, _AnswerAuditFailed):
        return _stable_failure_error_type(exc.error_type) or _UNHANDLED_EXECUTION_ERROR

    safe_error_code = _stable_failure_error_type(getattr(exc, "error_code", None))
    if safe_error_code is not None:
        return safe_error_code
    return _stable_failure_error_type(type(exc).__name__) or _UNHANDLED_EXECUTION_ERROR


def _stable_failure_error_type(raw: Any) -> str | None:
    """Return only an explicitly supported, stable public failure classification."""
    value = raw if isinstance(raw, str) else ""
    return value if value in _REPLAYABLE_FAILURE_ERROR_TYPES else None


def _stable_violation_code(raw: Any) -> str | None:
    for component in str(raw).split(":"):
        if _SAFE_VIOLATION_CODE.fullmatch(component):
            return component
    return None


def _violation_codes(violations: dict[str, list[str]]) -> list[str]:
    """Extract the stable CODE component, excluding ids/tokens/messages."""
    safe: set[str] = set()
    for values in violations.values():
        for violation in values:
            code = _stable_violation_code(violation)
            if code is not None:
                safe.add(code)
    return sorted(safe)


def _audit_repair_feedback(
    violations: dict[str, list[str]],
    artifact: GroundedAnswerArtifact,
) -> list[dict[str, Any]]:
    """Build prompt-only locators from fields that were already visible to the model.

    Claim statements, violation payloads and exception text are deliberately
    dropped.  Ambiguous category/evidence locators apply to every matching
    Claim.  RunEvent persistence continues to use ``_violation_codes``.
    """
    evidence_ids = {
        str(evidence.evidence_id): evidence.evidence_id for evidence in artifact.evidence
    }

    def safe_evidence_ids(values: list[Any]) -> tuple[uuid.UUID, ...]:
        safe: set[uuid.UUID] = set()
        for value in values:
            try:
                normalized = str(uuid.UUID(str(value)))
            except (AttributeError, TypeError, ValueError):
                continue
            if normalized in evidence_ids:
                safe.add(evidence_ids[normalized])
        return tuple(sorted(safe, key=str))

    claim_locators: dict[
        str,
        tuple[str, tuple[uuid.UUID, ...], tuple[uuid.UUID, ...]],
    ] = {}
    locator_counts: dict[
        tuple[str, tuple[uuid.UUID, ...], tuple[uuid.UUID, ...]],
        int,
    ] = {}
    for claim in artifact.claims:
        category = str(claim.category)
        if category not in _SAFE_QA_CLAIM_CATEGORIES:
            continue
        locator = (
            category,
            safe_evidence_ids(list(claim.supporting_evidence_ids)),
            safe_evidence_ids(list(claim.opposing_evidence_ids)),
        )
        claim_locators[str(claim.claim_id)] = locator
        locator_counts[locator] = locator_counts.get(locator, 0) + 1

    feedback: dict[str, dict[str, Any]] = {}
    for subject, values in violations.items():
        for violation in values:
            code = _stable_violation_code(violation)
            if code is None:
                continue
            locator = claim_locators.get(subject)
            if locator is not None:
                category, supporting_ids, opposing_ids = locator
                item = GroundedQAAuditFeedback(
                    scope="CLAIM",
                    code=code,
                    repair_hint=grounded_qa_repair_hint(code),
                    category=category,
                    supporting_evidence_ids=list(supporting_ids),
                    opposing_evidence_ids=list(opposing_ids),
                    apply_to=(
                        "ALL_MATCHING_CLAIMS" if locator_counts[locator] > 1 else "MATCHING_CLAIM"
                    ),
                )
            elif subject in evidence_ids:
                item = GroundedQAAuditFeedback(
                    scope="EVIDENCE",
                    code=code,
                    repair_hint=grounded_qa_repair_hint(code),
                    evidence_id=evidence_ids[subject],
                )
            else:
                item = GroundedQAAuditFeedback(
                    scope="ANSWER",
                    code=code,
                    repair_hint=grounded_qa_repair_hint(code),
                )
            payload = item.model_dump(mode="json", exclude_none=True)
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            feedback[canonical] = payload
    return [feedback[key] for key in sorted(feedback)]


def _normalize_non_blocking_review(
    artifact: GroundedAnswerArtifact,
    audit: Any,
) -> tuple[GroundedAnswerArtifact, bool]:
    """Apply the auditor-owned ANSWERED -> NEEDS_REVIEW downgrade server-side."""
    if not (
        getattr(audit, "review_normalization_required", False)
        and getattr(audit, "ok", False)
        and getattr(audit, "requires_human_review", False)
        and getattr(audit, "derived_answer_status", None) == "NEEDS_REVIEW"
        and getattr(artifact.answer_status, "value", artifact.answer_status) == "ANSWERED"
        and artifact.execution_status != "FAILED"
    ):
        return artifact, False
    payload = artifact.model_dump(mode="python")
    payload.update(
        {
            "answer_status": "NEEDS_REVIEW",
            "direct_answer": None,
            "execution_status": "PARTIAL",
            "abstention_reason": None,
            "refusal_reason_code": None,
        }
    )
    return GroundedAnswerArtifact.model_validate(payload), True


def _build_response(
    *,
    run: ReviewRun,
    question: str,
    snapshot_id: uuid.UUID,
    artifact: GroundedAnswerArtifact,
    generation_mode: str,
    retrieval,
    idempotent_replay: bool = False,
) -> GroundedQAResponse:
    query_spec = _canonical_query_spec(retrieval.query_spec)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in artifact.evidence}

    def locator(evidence_id: uuid.UUID) -> dict[str, Any] | None:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            return None
        preview_url = None
        if all(
            value is not None
            for value in (
                evidence.section_id,
                evidence.document_version_id,
                evidence.parse_run_id,
                evidence.page_number,
            )
        ):
            preview_url = (
                "/api/v1/evidence/preview"
                f"?case_id={run.case_id}&section_id={evidence.section_id}"
                f"&document_version_id={evidence.document_version_id}"
                f"&parse_run_id={evidence.parse_run_id}&page_number={evidence.page_number}"
                f"&text_hash={evidence.content_hash}"
            )
        return {
            "evidence_id": str(evidence.evidence_id),
            "evidence_type": evidence.evidence_type,
            "document_version_id": str(evidence.document_version_id)
            if evidence.document_version_id
            else None,
            "parse_run_id": str(evidence.parse_run_id) if evidence.parse_run_id else None,
            "section_id": str(evidence.section_id) if evidence.section_id else None,
            "page_number": evidence.page_number,
            "content_hash": evidence.content_hash,
            "preview_url": preview_url,
        }

    claims = []
    for claim in artifact.claims:
        supporting = [locator(item) for item in claim.supporting_evidence_ids]
        opposing = [locator(item) for item in claim.opposing_evidence_ids]
        claims.append(
            GroundedQAClaimResponse(
                claim_id=claim.claim_id,
                category=claim.category,
                statement=claim.statement,
                verdict=claim.verdict,
                citations=[item for item in supporting if item is not None],
                opposing_citations=[item for item in opposing if item is not None],
            )
        )

    candidates = [
        {
            "section_id": str(candidate.section_id),
            "document_version_id": str(candidate.document_version_id),
            "parse_run_id": str(candidate.parse_run_id),
            "heading_path": candidate.heading_path,
            "page": candidate.page_start,
            "text": candidate.text,
            "text_hash": candidate.text_hash,
        }
        for candidate in retrieval.candidates
    ]
    return GroundedQAResponse(
        question=question,
        answer_status=getattr(artifact.answer_status, "value", artifact.answer_status),
        answer=artifact.direct_answer or "",
        claims=claims,
        missing_information=artifact.missing_information,
        conflicts=artifact.conflicts,
        abstention_reason=artifact.abstention_reason,
        refusal_reason_code=artifact.refusal_reason_code,
        run_id=run.id,
        snapshot_id=snapshot_id,
        state_version=run.state_version,
        as_of_date=run.as_of_date,
        generation_mode=generation_mode,
        model_invocation_ids=list(artifact.model_invocation_ids),
        trace_url=f"/api/v1/runs/{run.id}/trace",
        candidates=candidates,
        query_spec=query_spec,
        channel_config=retrieval.channel_config,
        retrieval_trace=retrieval.trace,
        packing=retrieval.packing,
        idempotent_replay=idempotent_replay,
    )


def _canonical_query_spec(value: dict[str, Any]) -> dict[str, Any]:
    """Use one JSON representation for live responses, persistence and replay."""

    if not value:
        return {}
    return QuerySpec.model_validate(value).model_dump(mode="json")
