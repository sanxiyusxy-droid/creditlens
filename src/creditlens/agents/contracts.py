"""Artifact Contract（任务 20，文档 §10.9）。

- Agent 之间只通过结构化 Artifact 交换结果；
- Contract Validator 是确定性检查，语义检查不能覆盖其失败；
- suggested_tasks 只是建议，专业 Agent 无权直接启动任务。
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creditlens.formulas.engine import CalculationArtifact


class AgentEvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: uuid.UUID
    evidence_type: Literal[
        "DOCUMENT_SPAN", "TABLE_CELL", "SQL_FACT", "CALCULATION", "POLICY_RULE"
    ]
    source_id: uuid.UUID
    content_hash: str
    document_version_id: uuid.UUID | None = None
    section_id: uuid.UUID | None = None
    page_number: int | None = None
    fact_id: uuid.UUID | None = None
    calculation_id: uuid.UUID | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    source_available_at: datetime


class AgentClaim(BaseModel):
    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    category: Literal[
        "ELIGIBILITY",
        "FINANCIAL",
        "CASH_FLOW",
        "CONCENTRATION",
        "RELATED_PARTY",
        "DATA_CONFLICT",
        "MISSING_MATERIAL",
        "EXCEPTION",
    ]
    statement: str
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = "INFO"
    supporting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    opposing_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    calculation_ids: list[uuid.UUID] = Field(default_factory=list)
    as_of_date: date
    uncertainty_reason: str | None = None
    recommended_follow_up: list[str] = Field(default_factory=list)
    review_status: str = "PENDING"


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
    execution_status: Literal["SUCCESS", "PARTIAL", "INSUFFICIENT_EVIDENCE", "FAILED"] = "SUCCESS"
    claims: list[AgentClaim] = Field(default_factory=list)
    evidence: list[AgentEvidenceRef] = Field(default_factory=list)
    calculations: list[CalculationArtifact] = Field(default_factory=list)
    unresolved_issues: list[dict] = Field(default_factory=list)
    suggested_tasks: list[dict] = Field(default_factory=list)
    output_hash: str = ""


@dataclass
class ContractValidationResult:
    ok: bool
    violations: list[str]


_NUMERIC_HINT = ("率", "倍", "金额", "元", "%", "百分之")


def validate_artifact_contract(
    artifact: AgentArtifact,
    as_of_date: date,
    requested_amount: Decimal | None = None,
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
            if not fact_backed:
                violations.append(f"{prefix}:NUMERIC_CLAIM_WITHOUT_FACT")

        # 时点必须与硬约束一致
        if claim.as_of_date != as_of_date:
            violations.append(f"{prefix}:AS_OF_DATE_MISMATCH")

        # 不得自动定性欺诈/拒贷（文档 §9.4）
        for banned in ("欺诈", "拒贷", "拒绝贷款", "批准贷款"):
            if banned in claim.statement:
                violations.append(f"{prefix}:FORBIDDEN_DETERMINATION:{banned}")

    return ContractValidationResult(ok=not violations, violations=violations)
