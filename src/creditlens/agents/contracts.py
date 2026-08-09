"""Artifact Contract（任务 20，文档 §10.9）。

- Agent 之间只通过结构化 Artifact 交换结果；
- Contract Validator 是确定性检查，语义检查不能覆盖其失败；
- suggested_tasks 只是建议，专业 Agent 无权直接启动任务。
"""

import unicodedata
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creditlens.formulas.engine import CalculationArtifact


class AgentEvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: uuid.UUID
    evidence_type: Literal["DOCUMENT_SPAN", "TABLE_CELL", "SQL_FACT", "CALCULATION", "POLICY_RULE"]
    source_id: uuid.UUID
    content_hash: str
    document_version_id: uuid.UUID | None = None
    section_id: uuid.UUID | None = None
    # P1：parse_run_id 是回原文的必要定位信息——同一 Section 在不同解析批次
    # 的页码/坐标可能不同，缺失时无法证明引用的是 Snapshot 冻结的那一版
    parse_run_id: uuid.UUID | None = None
    page_number: int | None = None
    fact_id: uuid.UUID | None = None
    calculation_id: uuid.UUID | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    source_available_at: datetime


AgentClaimCategory = Literal[
    "ELIGIBILITY",
    "FINANCIAL",
    "CASH_FLOW",
    "CONCENTRATION",
    "RELATED_PARTY",
    "DATA_CONFLICT",
    "MISSING_MATERIAL",
    "EXCEPTION",
]
AgentClaimVerdict = Literal[
    "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
]


class AgentClaim(BaseModel):
    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    category: AgentClaimCategory
    statement: str
    verdict: AgentClaimVerdict
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = "INFO"
    supporting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    opposing_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    calculation_ids: list[uuid.UUID] = Field(default_factory=list)
    as_of_date: date
    uncertainty_reason: str | None = None
    recommended_follow_up: list[str] = Field(default_factory=list)
    review_status: str = "PENDING"
    # v1.1：Challenger 反证时记录被质疑的原始 Claim ID
    source_claim_id: uuid.UUID | None = None


class AgentArtifact(BaseModel):
    contract_version: str = "1.0"
    artifact_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    task_id: str
    producer: str
    input_hash: str = ""
    lifecycle_status: Literal[
        "CREATED", "VALIDATED", "VERIFIED", "ACCEPTED", "REJECTED", "STALE"
    ] = "CREATED"
    # DEGRADED：Agent 产出了结果但依赖的工具/检索部分失败（与 PARTIAL 区分：
    # PARTIAL 指证据不足导致结论不完整，DEGRADED 指执行链路本身发生了失败）
    execution_status: Literal[
        "SUCCESS", "PARTIAL", "DEGRADED", "INSUFFICIENT_EVIDENCE", "FAILED"
    ] = "SUCCESS"
    claims: list[AgentClaim] = Field(default_factory=list)
    evidence: list[AgentEvidenceRef] = Field(default_factory=list)
    calculations: list[CalculationArtifact] = Field(default_factory=list)
    unresolved_issues: list[dict] = Field(default_factory=list)
    suggested_tasks: list[dict] = Field(default_factory=list)
    output_hash: str = ""


class AnswerStatus(StrEnum):
    """Business answer state; technical failures belong to the enclosing Run."""

    ANSWERED = "ANSWERED"
    ABSTAINED = "ABSTAINED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class RefusalReasonCode(StrEnum):
    """Stable, non-sensitive reason taxonomy for business abstentions."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSPECIFIED_REFUSAL = "UNSPECIFIED_REFUSAL"
    MISSING_PERSONAL_CREDIT = "MISSING_PERSONAL_CREDIT"
    MISSING_EXTERNAL_CREDIT = "MISSING_EXTERNAL_CREDIT"
    MISSING_BANK_STATEMENTS = "MISSING_BANK_STATEMENTS"
    PRIVACY_AND_MISSING_EVIDENCE = "PRIVACY_AND_MISSING_EVIDENCE"
    SENSITIVE_DATA_UNAVAILABLE = "SENSITIVE_DATA_UNAVAILABLE"
    MISSING_PERSONAL_ASSETS = "MISSING_PERSONAL_ASSETS"
    MISSING_FUTURE_DATA = "MISSING_FUTURE_DATA"
    MISSING_CREDIT_REPORT = "MISSING_CREDIT_REPORT"
    NOT_APPLICABLE_NON_PUBLIC_COMPANY = "NOT_APPLICABLE_NON_PUBLIC_COMPANY"
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    MISSING_CORPORATE_IDENTITY_DATA = "MISSING_CORPORATE_IDENTITY_DATA"
    MISSING_FINANCIAL_DATA = "MISSING_FINANCIAL_DATA"


GROUNDING_EXECUTION_STATUS_BY_ANSWER: dict[AnswerStatus, str] = {
    AnswerStatus.ANSWERED: "SUCCESS",
    AnswerStatus.NEEDS_REVIEW: "PARTIAL",
    AnswerStatus.ABSTAINED: "INSUFFICIENT_EVIDENCE",
}


class DraftAnswerClaim(BaseModel):
    """Untrusted model draft without server-owned ids, locators or answer text."""

    model_config = ConfigDict(extra="forbid")

    category: AgentClaimCategory
    statement: str = Field(min_length=1, max_length=1000)
    verdict: AgentClaimVerdict
    supporting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    opposing_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    uncertainty_reason: str | None = Field(default=None, max_length=500)

    @field_validator("statement", mode="before")
    @classmethod
    def normalize_statement(cls, value):
        """Normalize untrusted text before applying length/non-empty constraints."""
        if not isinstance(value, str):
            return value
        normalized = unicodedata.normalize("NFKC", value).strip()
        if not normalized:
            raise ValueError("statement must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> "DraftAnswerClaim":
        if self.verdict == "SUPPORTED" and not self.supporting_evidence_ids:
            raise ValueError("SUPPORTED draft claim requires supporting evidence")
        if self.verdict == "INSUFFICIENT_EVIDENCE" and not self.uncertainty_reason:
            raise ValueError("INSUFFICIENT_EVIDENCE draft claim requires uncertainty_reason")
        return self


class GroundedAnswerDraft(BaseModel):
    """Only structure the model may produce; status and direct answer are server-owned."""

    model_config = ConfigDict(extra="forbid")

    claims: list[DraftAnswerClaim] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    abstention_reason: str | None = Field(default=None, max_length=500)
    refusal_reason_code: RefusalReasonCode | None = None

    @field_validator("refusal_reason_code", mode="before")
    @classmethod
    def discard_unknown_refusal_reason(cls, value):
        """Treat an untrusted model's unknown code as absent; never infer a replacement."""
        if value is None or isinstance(value, RefusalReasonCode):
            return value
        if not isinstance(value, str):
            return None
        try:
            return RefusalReasonCode(value.strip())
        except ValueError:
            return None

    @model_validator(mode="after")
    def validate_refusal_reason_shape(self) -> "GroundedAnswerDraft":
        """A trusted refusal code may only accompany a draft that will abstain."""
        if self.refusal_reason_code is None:
            return self
        insufficient_only = bool(self.claims) and all(
            claim.verdict == "INSUFFICIENT_EVIDENCE" for claim in self.claims
        )
        will_abstain = (
            bool(self.abstention_reason)
            or insufficient_only
            or (not self.claims and bool(self.missing_information))
        )
        if not will_abstain:
            raise ValueError("refusal_reason_code is only valid for an abstaining draft")
        return self


class GroundedAnswerArtifact(AgentArtifact):
    """Server-materialized grounded answer with deterministic evidence bindings."""

    model_config = ConfigDict(extra="forbid")

    answer_status: AnswerStatus
    direct_answer: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    refusal_reason_code: RefusalReasonCode | None = None
    prompt_version: str
    model_invocation_ids: list[uuid.UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answer_state(self) -> "GroundedAnswerArtifact":
        expected_execution_status = GROUNDING_EXECUTION_STATUS_BY_ANSWER[self.answer_status]
        if self.execution_status != expected_execution_status:
            raise ValueError(
                f"{self.answer_status.value} requires execution_status={expected_execution_status}"
            )
        if self.answer_status == AnswerStatus.ANSWERED:
            if not self.direct_answer or not self.claims:
                raise ValueError("ANSWERED artifact requires direct_answer and grounded claims")
        elif self.direct_answer is not None:
            raise ValueError("non-ANSWERED artifact must not expose direct_answer")
        if self.answer_status == AnswerStatus.ABSTAINED and not self.abstention_reason:
            raise ValueError("ABSTAINED artifact requires abstention_reason")
        if self.answer_status == AnswerStatus.ABSTAINED:
            if self.refusal_reason_code is None:
                raise ValueError("ABSTAINED artifact requires refusal_reason_code")
        elif self.refusal_reason_code is not None:
            raise ValueError("non-ABSTAINED artifact must not expose refusal_reason_code")
        return self


@dataclass
class ContractValidationResult:
    ok: bool
    violations: list[str]


_NUMERIC_HINT = ("率", "倍", "金额", "元", "%", "百分之")


def validate_artifact_contract(
    artifact: AgentArtifact,
    as_of_date: date,
    requested_amount: Decimal | None = None,
    *,
    allow_grounded_document_numeric: bool = False,
) -> ContractValidationResult:
    """Contract Validator（文档 §10.9）。"""
    violations: list[str] = []
    evidence_ids = {e.evidence_id for e in artifact.evidence}
    calculation_ids = {c.calculation_id for c in artifact.calculations}

    for claim in artifact.claims:
        prefix = f"claim:{claim.claim_id}"

        # SUPPORTED 至少一条正式证据
        if claim.verdict == "SUPPORTED" and not claim.supporting_evidence_ids:
            violations.append(f"{prefix}:SUPPORTED_WITHOUT_EVIDENCE")

        # 引用的 Evidence/Calculation 必须存在于 Artifact 中（不可伪造 ID）
        for evidence_id in claim.supporting_evidence_ids + claim.opposing_evidence_ids:
            if evidence_id not in evidence_ids:
                violations.append(f"{prefix}:UNKNOWN_EVIDENCE:{evidence_id}")
        for calculation_id in claim.calculation_ids:
            if calculation_id not in calculation_ids:
                violations.append(f"{prefix}:UNKNOWN_CALCULATION:{calculation_id}")

        # INSUFFICIENT_EVIDENCE 必须描述缺失项
        if claim.verdict == "INSUFFICIENT_EVIDENCE" and not claim.uncertainty_reason:
            violations.append(f"{prefix}:MISSING_UNCERTAINTY_REASON")

        # 数字类 Claim（含数字且提及指标）必须绑定 Fact/Calculation
        has_digit = any(ch.isdigit() for ch in claim.statement)
        mentions_numeric = any(hint in claim.statement for hint in _NUMERIC_HINT)
        if (
            has_digit
            and mentions_numeric
            and claim.category in {"FINANCIAL", "CASH_FLOW"}
            and not claim.calculation_ids
        ):
            fact_backed = any(
                e.evidence_type in {"SQL_FACT", "CALCULATION", "TABLE_CELL"}
                for e in artifact.evidence
                if e.evidence_id in claim.supporting_evidence_ids
            )
            # Grounded QA may quote a number directly from an immutable document
            # span, but only its dedicated auditor is allowed to open this gate.
            # That caller subsequently proves the locator/hash and requires every
            # numeric token to occur verbatim in the cited section.  Generic Agent
            # artifacts retain the stricter structured-fact requirement.
            grounded_document_backed = (
                allow_grounded_document_numeric
                and isinstance(artifact, GroundedAnswerArtifact)
                and any(
                    e.evidence_type == "DOCUMENT_SPAN"
                    for e in artifact.evidence
                    if e.evidence_id in claim.supporting_evidence_ids
                )
            )
            if not fact_backed and not grounded_document_backed:
                violations.append(f"{prefix}:NUMERIC_CLAIM_WITHOUT_FACT")

        # 时点必须与硬约束一致
        if claim.as_of_date != as_of_date:
            violations.append(f"{prefix}:AS_OF_DATE_MISMATCH")

        # 不得自动定性欺诈/拒贷（文档 §9.4）
        for banned in ("欺诈", "拒贷", "拒绝贷款", "批准贷款"):
            if banned in claim.statement:
                violations.append(f"{prefix}:FORBIDDEN_DETERMINATION:{banned}")

    return ContractValidationResult(ok=not violations, violations=violations)
