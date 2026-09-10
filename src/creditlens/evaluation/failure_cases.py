"""Reproducible fail-closed injections and system-level database gate proofs.

There are deliberately two different proof scopes in this module:

* ``CONTRACT_VALIDATOR_PRECHECK`` only proves that a frozen malformed artifact is
  rejected by the deterministic contract validator.  It does **not** prove a
  Supervisor state transition or report suppression.
* ``SUPERVISOR_AUDITOR_DATABASE_GATE`` injects the artifact through the real
  Supervisor, EvidenceAuditor and persistence layer, then derives its result
  from persisted Run/Claim/Artifact/RunEvent/ReportVersion rows.

Neither scope calls an HTTP endpoint.  The report says so explicitly instead of
presenting a database assertion as black-box API evidence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from creditlens.agents.auditor import EvidenceAuditor
from creditlens.agents.contracts import (
    AgentArtifact,
    AgentClaim,
    AgentEvidenceRef,
    validate_artifact_contract,
)
from creditlens.agents.supervisor import Supervisor
from creditlens.application.trusted_context import build_trusted_context
from creditlens.evaluation.source_state import (
    EvidenceMaturity,
    SourceStateEvidence,
    capture_source_state_evidence,
    validate_source_state_binding,
    verify_captured_source_state,
)
from creditlens.formulas.engine import FormulaRegistry
from creditlens.infrastructure.postgres.artifact_integrity import (
    ArtifactIntegrityError,
    canonical_artifact_payload_hash,
    validate_claim_records_against_artifacts,
)
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    ClaimRecord,
    ReportVersion,
    ReviewRun,
    RunEvent,
)
from creditlens.infrastructure.postgres.session import session_scope

FAILURE_CASE_PROTOCOL_VERSION = "1.0.0"
_UUID_NAMESPACE = uuid.UUID("cb4037cf-9866-5073-ad36-6e91a8898749")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureInjectionType(StrEnum):
    NUMERIC_EVIDENCE_MISBINDING = "NUMERIC_EVIDENCE_MISBINDING"
    PROHIBITED_CREDIT_DETERMINATION = "PROHIBITED_CREDIT_DETERMINATION"


class FailureProofScope(StrEnum):
    CONTRACT_VALIDATOR_PRECHECK = "CONTRACT_VALIDATOR_PRECHECK"
    SUPERVISOR_AUDITOR_DATABASE_GATE = "SUPERVISOR_AUDITOR_DATABASE_GATE"


class FailureCaseDefinition(_StrictModel):
    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    injection_type: FailureInjectionType
    as_of_date: date
    synthetic_claim: str = Field(min_length=1)
    expected_violation_codes: list[str] = Field(min_length=1)
    expected_action: Literal["BLOCK"]
    expected_workflow_status: Literal["HUMAN_REVIEW"]

    @model_validator(mode="after")
    def validate_expected_codes(self) -> Self:
        if len(set(self.expected_violation_codes)) != len(self.expected_violation_codes):
            raise ValueError("expected_violation_codes must be unique")
        return self


class FailureCaseDataset(_StrictModel):
    protocol_id: Literal["creditlens_fail_closed_cases_v1"]
    protocol_version: Literal["1.0.0"]
    frozen: Literal[True]
    cases: list[FailureCaseDefinition] = Field(min_length=2, max_length=2)

    @field_validator("frozen", mode="before")
    @classmethod
    def validate_frozen(cls, value):
        if value is not True:
            raise ValueError("frozen must be exactly true")
        return value

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        ids = [item.case_id for item in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("failure case IDs must be unique")
        if {item.injection_type for item in self.cases} != set(FailureInjectionType):
            raise ValueError("the frozen dataset requires both failure injection types")
        return self


class SafeFailureDisplay(_StrictModel):
    verification_scope: Literal["STATIC_PUBLIC_CONTRACT_NOT_HTTP"] = (
        "STATIC_PUBLIC_CONTRACT_NOT_HTTP"
    )
    public_error_code: Literal["CLAIM_EVIDENCE_VALIDATION_FAILED"]
    public_message: Literal["结论未通过证据完整性校验，已阻断自动出具并转人工复核。"]
    action: Literal["BLOCK"]
    workflow_status: Literal["HUMAN_REVIEW"]
    report_allowed: Literal[False]
    unsafe_content_exposed: Literal[False]


class FailureSystemEvidence(_StrictModel):
    """Bounded proof derived from the real workflow and database rows.

    The evidence intentionally omits artifact payloads, Claim statements and
    event payloads.  Hashes and stable state labels are sufficient to reproduce
    the persistence and report-gate assertions without leaking injected text.
    """

    proof_boundary: Literal["SUPERVISOR_STATE_MACHINE_AND_DATABASE_GATE"] = (
        "SUPERVISOR_STATE_MACHINE_AND_DATABASE_GATE"
    )
    http_endpoint_called: Literal[False] = False
    report_gate_proof: Literal["DATABASE_REPORT_VERSION_ROW_COUNT"] = (
        "DATABASE_REPORT_VERSION_ROW_COUNT"
    )
    run_id: uuid.UUID
    persisted_run_status: str
    persisted_claim_count: int = Field(ge=0)
    claim_review_statuses: list[str]
    injected_claim_persisted: bool
    artifact_count: int = Field(ge=0)
    artifact_producers: list[str]
    artifact_output_hashes: list[str]
    artifact_hashes_verified: bool
    artifact_claim_projection_verified: bool
    run_event_count: int = Field(ge=0)
    event_types: list[str]
    state_transitions: list[str]
    audit_completed: bool
    report_count: int = Field(ge=0)

    @field_validator("artifact_output_hashes")
    @classmethod
    def validate_hashes(cls, values: list[str]) -> list[str]:
        if any(
            len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in values
        ):
            raise ValueError("artifact_output_hashes must contain lowercase sha256 values")
        return values


class FailureCaseResult(_StrictModel):
    case_id: str
    injection_type: FailureInjectionType
    proof_scope: FailureProofScope
    passed: bool
    observed_violation_codes: list[str]
    missing_expected_violation_codes: list[str]
    safe_display: SafeFailureDisplay
    system_evidence: FailureSystemEvidence | None = None

    @model_validator(mode="after")
    def validate_proof_scope(self) -> Self:
        has_system_evidence = self.system_evidence is not None
        if (
            self.proof_scope is FailureProofScope.CONTRACT_VALIDATOR_PRECHECK
            and has_system_evidence
        ):
            raise ValueError("contract precheck must not contain system evidence")
        if (
            self.proof_scope is FailureProofScope.SUPERVISOR_AUDITOR_DATABASE_GATE
            and not has_system_evidence
        ):
            raise ValueError("system proof requires persisted system evidence")
        if self.passed and self.missing_expected_violation_codes:
            raise ValueError("a passing failure case cannot miss expected violation codes")
        if self.passed and not self.observed_violation_codes:
            raise ValueError("a passing failure case requires at least one observed violation")
        if self.passed and self.system_evidence is not None:
            evidence = self.system_evidence
            closed_review_states = {"NEEDS_REWORK", "PENDING"}
            system_gate_passed = all(
                (
                    evidence.persisted_run_status == "HUMAN_REVIEW",
                    evidence.injected_claim_persisted,
                    evidence.persisted_claim_count == 1,
                    len(evidence.claim_review_statuses) == 1,
                    set(evidence.claim_review_statuses) <= closed_review_states,
                    evidence.artifact_count == 3,
                    set(evidence.artifact_producers)
                    == {"policy_analyst", "financial_analyst", "challenger"},
                    evidence.artifact_hashes_verified,
                    evidence.artifact_claim_projection_verified,
                    evidence.run_event_count > 0,
                    bool(evidence.event_types),
                    evidence.audit_completed,
                    "AUDITING->HUMAN_REVIEW" in evidence.state_transitions,
                    not any(
                        transition.endswith("->REPORTING")
                        for transition in evidence.state_transitions
                    ),
                    evidence.report_count == 0,
                )
            )
            if not system_gate_passed:
                raise ValueError(
                    "a passing system failure case requires complete fail-closed database evidence"
                )
        return self


class FailureCaseReport(_StrictModel):
    report_type: Literal["fail_closed_regression"] = "fail_closed_regression"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    execution_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    protocol_id: Literal["creditlens_fail_closed_cases_v1"]
    protocol_version: Literal["1.0.0"]
    dataset_sha256: str
    git_commit: str | None = Field(pattern=r"^[0-9a-f]{7,64}$")
    git_dirty: bool | None
    source_state_sha256: str
    source_state_algorithm: Literal["sha256-canonical-file-manifest-v1"]
    source_state_scope: Literal["creditlens-runtime-evidence-v1"]
    source_state_file_count: int = Field(gt=0)
    evidence_maturity: EvidenceMaturity
    proof_scope: FailureProofScope
    system_execution_performed: bool
    http_endpoint_called: Literal[False] = False
    all_passed: bool
    results: list[FailureCaseResult] = Field(min_length=2, max_length=2)

    @field_validator("dataset_sha256")
    @classmethod
    def validate_dataset_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_report_scope(self) -> Self:
        validate_source_state_binding(
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            source_state_sha256=self.source_state_sha256,
            source_state_algorithm=self.source_state_algorithm,
            source_state_scope=self.source_state_scope,
            source_state_file_count=self.source_state_file_count,
            evidence_maturity=self.evidence_maturity,
        )
        expected_execution = self.proof_scope is FailureProofScope.SUPERVISOR_AUDITOR_DATABASE_GATE
        if self.system_execution_performed != expected_execution:
            raise ValueError("system_execution_performed does not match proof_scope")
        if any(item.proof_scope is not self.proof_scope for item in self.results):
            raise ValueError("all result proof scopes must match the report")
        case_ids = [item.case_id for item in self.results]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("failure report case IDs must be unique")
        expected_case_types = {
            "numeric_evidence_misbinding": FailureInjectionType.NUMERIC_EVIDENCE_MISBINDING,
            "prohibited_credit_determination": FailureInjectionType.PROHIBITED_CREDIT_DETERMINATION,
        }
        actual_case_types = {item.case_id: item.injection_type for item in self.results}
        if actual_case_types != expected_case_types:
            raise ValueError("failure report must contain the two frozen case/type bindings")
        if self.all_passed != all(item.passed for item in self.results):
            raise ValueError("all_passed must equal the conjunction of result.passed values")
        return self


def _stable_uuid(case_id: str, suffix: str) -> uuid.UUID:
    return uuid.uuid5(_UUID_NAMESPACE, f"{case_id}:{suffix}")


def _artifact_for_case(
    case: FailureCaseDefinition,
    *,
    run_id: uuid.UUID | None = None,
    task_id: str | None = None,
    producer: str | None = None,
) -> AgentArtifact:
    if run_id is None:
        run_id = _stable_uuid(case.case_id, "run")
        claim_id = _stable_uuid(case.case_id, "claim")
        evidence_id = _stable_uuid(case.case_id, "evidence")
        wrong_evidence_id = _stable_uuid(case.case_id, "wrong-evidence")
        policy_source_id = _stable_uuid(case.case_id, "policy-rule")
    else:
        # ClaimRecord has a global primary key.  Namespace the deterministic
        # injection ids by the fresh ReviewRun so repeated regressions remain
        # append-only instead of colliding with an earlier execution.
        claim_id = uuid.uuid5(run_id, f"{case.case_id}:claim")
        evidence_id = uuid.uuid5(run_id, f"{case.case_id}:evidence")
        wrong_evidence_id = uuid.uuid5(run_id, f"{case.case_id}:wrong-evidence")
        policy_source_id = uuid.uuid5(run_id, f"{case.case_id}:policy-rule")
    common_claim = {
        "claim_id": claim_id,
        "statement": case.synthetic_claim,
        "verdict": "SUPPORTED",
        "severity": "HIGH",
        "as_of_date": case.as_of_date,
    }

    if case.injection_type is FailureInjectionType.NUMERIC_EVIDENCE_MISBINDING:
        # The claim references an id that is absent from the artifact and has no
        # structured fact/calculation.  This is the production contract's
        # deterministic representation of a numeric/evidence mismatch.
        claim = AgentClaim(
            category="FINANCIAL",
            supporting_evidence_ids=[wrong_evidence_id],
            **common_claim,
        )
        evidence: list[AgentEvidenceRef] = []
    else:
        claim = AgentClaim(
            category="ELIGIBILITY",
            supporting_evidence_ids=[evidence_id],
            **common_claim,
        )
        evidence = [
            AgentEvidenceRef(
                evidence_id=evidence_id,
                evidence_type="POLICY_RULE",
                source_id=policy_source_id,
                content_hash="f" * 64,
                source_available_at=datetime.combine(case.as_of_date, datetime.min.time(), UTC),
            )
        ]
    return AgentArtifact(
        run_id=run_id,
        task_id=task_id or f"failure-case:{case.case_id}",
        producer=producer or "evaluation_failure_injector",
        claims=[claim],
        evidence=evidence,
    )


def _violation_code(value: str) -> str:
    parts = value.split(":", 2)
    return parts[2].split(":", 1)[0] if len(parts) == 3 else value.split(":", 1)[0]


def run_failure_case(case: FailureCaseDefinition) -> FailureCaseResult:
    """Run a contract-only precheck; this is not a system execution proof."""

    artifact = _artifact_for_case(case)
    validation = validate_artifact_contract(artifact, case.as_of_date)
    observed = sorted({_violation_code(item) for item in validation.violations})
    missing = sorted(set(case.expected_violation_codes) - set(observed))
    passed = not validation.ok and not missing
    # The public contract is intentionally independent of the injected text and
    # detailed validator diagnostics.  It cannot leak unsafe model content.
    display = SafeFailureDisplay(
        public_error_code="CLAIM_EVIDENCE_VALIDATION_FAILED",
        public_message="结论未通过证据完整性校验，已阻断自动出具并转人工复核。",
        action="BLOCK",
        workflow_status="HUMAN_REVIEW",
        report_allowed=False,
        unsafe_content_exposed=False,
    )
    return FailureCaseResult(
        case_id=case.case_id,
        injection_type=case.injection_type,
        proof_scope=FailureProofScope.CONTRACT_VALIDATOR_PRECHECK,
        passed=passed,
        observed_violation_codes=observed,
        missing_expected_violation_codes=missing,
        safe_display=display,
    )


def evaluate_failure_cases(
    dataset: FailureCaseDataset,
    *,
    dataset_sha256: str,
    project_root: Path | None = None,
    source_state: SourceStateEvidence | None = None,
) -> FailureCaseReport:
    resolved_project_root = project_root or Path(__file__).resolve().parents[3]
    captured_source_state = source_state or capture_source_state_evidence(resolved_project_root)
    results = [run_failure_case(case) for case in dataset.cases]
    verify_captured_source_state(resolved_project_root, captured_source_state)
    return FailureCaseReport(
        protocol_id=dataset.protocol_id,
        protocol_version=dataset.protocol_version,
        dataset_sha256=dataset_sha256,
        **captured_source_state.as_metadata(),
        proof_scope=FailureProofScope.CONTRACT_VALIDATOR_PRECHECK,
        system_execution_performed=False,
        all_passed=all(item.passed for item in results),
        results=results,
    )


class _InjectedFailureAgent:
    def __init__(self, case: FailureCaseDefinition):
        self._case = case

    async def run(self, run_id, task_id, trusted):
        return _artifact_for_case(
            self._case,
            run_id=run_id,
            task_id=task_id,
            producer="policy_analyst",
        )


class _EmptyFinancialAgent:
    async def run(self, run_id, task_id, trusted):
        return AgentArtifact(
            run_id=run_id,
            task_id=task_id,
            producer="financial_analyst",
        )


class _EmptyChallenger:
    async def run(self, run_id, task_id, trusted, professional):
        return AgentArtifact(
            run_id=run_id,
            task_id=task_id,
            producer="challenger",
        )


def _audit_violation_codes(violations: dict[str, list[str]]) -> list[str]:
    return sorted({_violation_code(value) for values in violations.values() for value in values})


def _state_transitions(events: list[RunEvent]) -> list[str]:
    transitions: list[str] = []
    for event in events:
        if event.event_type != "STATE_CHANGED":
            continue
        payload = event.payload_redacted or {}
        source = payload.get("from")
        target = payload.get("to")
        if isinstance(source, str) and isinstance(target, str):
            transitions.append(f"{source}->{target}")
    return transitions


async def execute_failure_case_system(
    session: AsyncSession,
    case: FailureCaseDefinition,
    *,
    trusted,
) -> FailureCaseResult:
    """Execute one injection through Supervisor/Auditor and prove DB gates.

    This function intentionally uses the ordinary Supervisor orchestration and
    persistence code.  Only the professional agents are deterministic injection
    adapters; workflow transitions, auditing, Claim updates and report gating
    are production implementations.
    """

    if trusted.as_of_date != case.as_of_date:
        raise ValueError("FAILURE_CASE_AS_OF_DATE_MISMATCH")

    supervisor = Supervisor(
        policy_agent=_InjectedFailureAgent(case),
        financial_agent=_EmptyFinancialAgent(),
        challenger=_EmptyChallenger(),
        auditor=EvidenceAuditor(FormulaRegistry()),
    )
    outcome = await supervisor.execute_full_review(session, trusted)
    await session.flush()

    run = await session.get(ReviewRun, outcome.run_id)
    artifact_records = list(
        (
            await session.scalars(
                select(ArtifactRecord)
                .where(ArtifactRecord.run_id == outcome.run_id)
                .order_by(ArtifactRecord.created_at, ArtifactRecord.id)
            )
        ).all()
    )
    claim_records = list(
        (
            await session.scalars(
                select(ClaimRecord)
                .where(ClaimRecord.run_id == outcome.run_id)
                .order_by(ClaimRecord.created_at, ClaimRecord.id)
            )
        ).all()
    )
    events = list(
        (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == outcome.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()
    )
    report_count = int(
        (
            await session.scalar(
                select(func.count(ReportVersion.id)).where(ReportVersion.run_id == outcome.run_id)
            )
        )
        or 0
    )

    injected_claim_ids = {
        claim.claim_id
        for artifact in outcome.artifacts
        if artifact.producer == "policy_analyst"
        for claim in artifact.claims
    }
    persisted_claim_ids = {claim.id for claim in claim_records}
    injected_claim_persisted = bool(injected_claim_ids) and injected_claim_ids.issubset(
        persisted_claim_ids
    )
    claim_statuses = [claim.review_status for claim in claim_records]
    artifact_hashes = [record.output_hash for record in artifact_records]
    artifact_hashes_verified = bool(artifact_records) and all(
        record.output_hash == canonical_artifact_payload_hash(record.payload)
        for record in artifact_records
    )
    artifact_claim_projection_verified = False
    if run is not None:
        try:
            validate_claim_records_against_artifacts(
                run=run,
                artifacts=artifact_records,
                claims=claim_records,
                require_output_hash_for=frozenset(
                    record.artifact_type for record in artifact_records
                ),
            )
            artifact_claim_projection_verified = True
        except ArtifactIntegrityError:
            pass
    event_types = [event.event_type for event in events]
    transitions = _state_transitions(events)
    observed = _audit_violation_codes(outcome.audit.violations) if outcome.audit is not None else []
    missing = sorted(set(case.expected_violation_codes) - set(observed))

    evidence = FailureSystemEvidence(
        run_id=outcome.run_id,
        persisted_run_status=run.status if run is not None else "MISSING",
        persisted_claim_count=len(claim_records),
        claim_review_statuses=claim_statuses,
        injected_claim_persisted=injected_claim_persisted,
        artifact_count=len(artifact_records),
        artifact_producers=[record.producer for record in artifact_records],
        artifact_output_hashes=artifact_hashes,
        artifact_hashes_verified=artifact_hashes_verified,
        artifact_claim_projection_verified=artifact_claim_projection_verified,
        run_event_count=len(events),
        event_types=event_types,
        state_transitions=transitions,
        audit_completed="AUDIT_COMPLETED" in event_types,
        report_count=report_count,
    )
    review_statuses_are_closed = len(claim_statuses) == len(injected_claim_ids) == 1 and all(
        status in {"NEEDS_REWORK", "PENDING"} for status in claim_statuses
    )
    expected_producers = {"policy_analyst", "financial_analyst", "challenger"}
    artifact_set_is_exact = (
        len(artifact_records) == len(expected_producers)
        and {record.producer for record in artifact_records} == expected_producers
    )
    passed = all(
        (
            not missing,
            outcome.status == case.expected_workflow_status,
            run is not None and run.status == case.expected_workflow_status,
            injected_claim_persisted,
            review_statuses_are_closed,
            persisted_claim_ids == injected_claim_ids,
            artifact_set_is_exact,
            artifact_hashes_verified,
            artifact_claim_projection_verified,
            bool(events),
            evidence.audit_completed,
            "AUDITING->HUMAN_REVIEW" in transitions,
            not any(value.endswith("->REPORTING") for value in transitions),
            report_count == 0,
        )
    )
    return FailureCaseResult(
        case_id=case.case_id,
        injection_type=case.injection_type,
        proof_scope=FailureProofScope.SUPERVISOR_AUDITOR_DATABASE_GATE,
        passed=passed,
        observed_violation_codes=observed,
        missing_expected_violation_codes=missing,
        safe_display=SafeFailureDisplay(
            public_error_code="CLAIM_EVIDENCE_VALIDATION_FAILED",
            public_message="结论未通过证据完整性校验，已阻断自动出具并转人工复核。",
            action="BLOCK",
            workflow_status="HUMAN_REVIEW",
            report_allowed=False,
            unsafe_content_exposed=False,
        ),
        system_evidence=evidence,
    )


async def execute_failure_cases_system(
    dataset: FailureCaseDataset,
    *,
    dataset_sha256: str,
    session_factory: async_sessionmaker[AsyncSession],
    tenant_id: uuid.UUID,
    user_id: uuid.UUID | None,
    case_id: uuid.UUID,
    project_root: Path | None = None,
    source_state: SourceStateEvidence | None = None,
) -> FailureCaseReport:
    """Run every frozen case in an independent, committed workflow transaction."""

    resolved_project_root = project_root or Path(__file__).resolve().parents[3]
    captured_source_state = source_state or capture_source_state_evidence(resolved_project_root)
    results: list[FailureCaseResult] = []
    for case in dataset.cases:
        async with session_scope(
            session_factory,
            tenant_id=tenant_id,
            user_id=user_id,
        ) as session:
            trusted = await build_trusted_context(
                session,
                tenant_id,
                case_id,
                user_id=user_id,
                purpose="fail_closed_evaluation",
            )
            results.append(
                await execute_failure_case_system(
                    session,
                    case,
                    trusted=trusted,
                )
            )
    verify_captured_source_state(resolved_project_root, captured_source_state)
    return FailureCaseReport(
        protocol_id=dataset.protocol_id,
        protocol_version=dataset.protocol_version,
        dataset_sha256=dataset_sha256,
        **captured_source_state.as_metadata(),
        proof_scope=FailureProofScope.SUPERVISOR_AUDITOR_DATABASE_GATE,
        system_execution_performed=True,
        all_passed=all(item.passed for item in results),
        results=results,
    )
