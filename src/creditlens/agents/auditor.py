"""Evidence Auditor（任务 25，文档 §8.15/§10.10；v1.1 确定性证据一致性审计）。

说明：v1.1 未接入 LLM 语义裁判，本 Agent 为“确定性证据一致性审计”：
只验证可机器核验的绑定/哈希/时点/权限一致性，不做语义判断。

先执行确定性检查，语义判断不得覆盖权限、时点或哈希失败：
1. Contract Validator（Claim-Evidence 绑定完整性）；
2. DOCUMENT_SPAN 证据回表：Section 存在、text_hash 一致、租户匹配；
3. CALCULATION 证据重放：公式重算结果与 trace_hash 一致；
4. 真冲突（DATA_CONFLICT）保留双方，标记需人工复核；
   补充材料（非冲突）不阻断流程、不送 HITL；
5. v1.1：强制验证 DocumentVersion 时点（valid_from/valid_to）与 ParseRun 激活状态；
6. v1.1 WP3：Case 存在性/租户复核、cutoff 可获得性复核、Snapshot ParseRun 集合复核；
7. v1.1：核心 Agent（Policy/Financial）失败 -> 阻断；非核心（Risk）-> DEGRADED。
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC

from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.agents.contracts import (
    AgentArtifact,
    validate_artifact_contract,
)
from creditlens.formulas.engine import FormulaRegistry, replay_calculation
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    CreditCase,
    DocumentSection,
    DocumentVersion,
    FinancialFact,
    ParseRun,
)
from creditlens.retrieval.contracts import TrustedRequestContext

AGENT_ROLE = "evidence_auditor"

# 核心 Agent：失败时阻断报告
_CORE_PRODUCERS = {"policy_analyst", "financial_analyst"}

# Claim category 与 Evidence heading_path 关键词的确定性蕴含映射
_CATEGORY_KEYWORDS = {
    "ELIGIBILITY": ["准入", "条件", "要求", "资格", "成立"],
    "FINANCIAL": ["财务", "负债", "利润", "资产", "比率"],
    "CASH_FLOW": ["现金流", "回款", "经营"],
    "CONCENTRATION": ["集中", "客户", "占比"],
    "EXCEPTION": ["例外", "担保", "特殊"],
    "MISSING_MATERIAL": ["材料", "提交", "申请"],
}


@dataclass
class AuditResult:
    accepted_claim_ids: list[uuid.UUID] = field(default_factory=list)
    rejected_claim_ids: list[uuid.UUID] = field(default_factory=list)
    needs_human_review_claim_ids: list[uuid.UUID] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    replay_failures: list[str] = field(default_factory=list)
    degraded: bool = False  # v1.1：非核心 Agent 失败时标记

    @property
    def requires_human_review(self) -> bool:
        return bool(self.needs_human_review_claim_ids)

    @property
    def ok(self) -> bool:
        return not self.rejected_claim_ids and not self.replay_failures


def _reject_evidence(
    result: AuditResult,
    bad_evidence: set[uuid.UUID],
    evidence_id: uuid.UUID,
    code: str,
) -> None:
    """Record one deterministic violation without duplicating error codes."""
    bad_evidence.add(evidence_id)
    violations = result.violations.setdefault(str(evidence_id), [])
    if code not in violations:
        violations.append(code)


class EvidenceAuditor:
    def __init__(self, registry: FormulaRegistry):
        self._registry = registry

    async def verify(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        artifacts: list[AgentArtifact],
        snapshot=None,
    ) -> AuditResult:
        result = AuditResult()

        # WP3：Case 复核——案件存在且租户一致，否则全部 Claim 拒绝
        case = await session.get(CreditCase, trusted.case_id)
        case_ok = case is not None and str(case.tenant_id) == str(trusted.tenant_id)
        if not case_ok:
            for artifact in artifacts:
                for claim in artifact.claims:
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(str(claim.claim_id), []).append(
                        "CASE_VALIDATION_FAILED"
                    )
            return result

        for artifact in artifacts:
            # v1.1：核心 Agent 执行失败 -> 阻断
            if artifact.execution_status == "FAILED":
                if artifact.producer in _CORE_PRODUCERS:
                    # 核心 Agent 失败：全部 Claim 拒绝
                    for claim in artifact.claims:
                        result.rejected_claim_ids.append(claim.claim_id)
                        result.violations.setdefault(str(claim.claim_id), []).append(
                            f"CORE_AGENT_FAILED:{artifact.producer}"
                        )
                    continue
                else:
                    # 非核心 Agent 失败：标记 DEGRADED，跳过其 Claim
                    result.degraded = True
                    continue

            # 1. Contract 校验
            contract = validate_artifact_contract(artifact, trusted.as_of_date)
            contract_failed_claims = {
                v.split(":", 2)[1] for v in contract.violations if v.startswith("claim:")
            }

            # 2/3. 证据级确定性校验
            bad_evidence: set[uuid.UUID] = set()
            calculations_by_id = {
                calculation.calculation_id: calculation for calculation in artifact.calculations
            }
            replay_failed: set[str] = set()
            for calc in artifact.calculations:
                consistent, _ = replay_calculation(self._registry, calc)
                if not consistent:
                    calc_id = str(calc.calculation_id)
                    replay_failed.add(calc_id)
                    result.replay_failures.append(calc_id)

            for evidence in artifact.evidence:
                if evidence.evidence_type == "CALCULATION":
                    if evidence.calculation_id is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "MISSING_CALCULATION_ID",
                        )
                        continue
                    if evidence.source_id != evidence.calculation_id:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "CALCULATION_SOURCE_MISMATCH",
                        )
                    calculation = calculations_by_id.get(evidence.calculation_id)
                    if calculation is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "CALCULATION_NOT_FOUND",
                        )
                    else:
                        if evidence.content_hash != calculation.trace_hash:
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "CALCULATION_CONTENT_HASH_MISMATCH",
                            )
                        if str(calculation.calculation_id) in replay_failed:
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "CALCULATION_REPLAY_FAILED",
                            )
                    continue

                if evidence.evidence_type == "SQL_FACT":
                    if evidence.fact_id is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "MISSING_FACT_ID",
                        )
                        continue
                    if evidence.source_id != evidence.fact_id:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "FACT_SOURCE_MISMATCH",
                        )
                    fact = await session.get(FinancialFact, evidence.fact_id)
                    if fact is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "FACT_NOT_FOUND",
                        )
                    else:
                        if str(fact.tenant_id) != str(trusted.tenant_id):
                            _reject_evidence(
                                result, bad_evidence, evidence.evidence_id, "ACL_DENIED"
                            )
                        if fact.case_id is not None and fact.case_id != trusted.case_id:
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "FACT_CASE_MISMATCH",
                            )
                        if fact.entity_id != case.borrower_entity_id:
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "FACT_ENTITY_MISMATCH",
                            )
                        available_at = fact.source_available_at
                        if available_at.tzinfo is None:
                            available_at = available_at.replace(tzinfo=UTC)
                        if available_at > trusted.decision_cutoff_at.astimezone(UTC):
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "FACT_NOT_AVAILABLE_AT_CUTOFF",
                            )
                        if fact.verification_status == "REJECTED":
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "FACT_REJECTED",
                            )
                    if snapshot is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "SNAPSHOT_REQUIRED",
                        )
                    elif evidence.fact_id not in set(snapshot.allowed_fact_ids):
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "FACT_NOT_IN_SNAPSHOT",
                        )
                    continue

                has_document_locator = any(
                    value is not None
                    for value in (
                        evidence.section_id,
                        evidence.document_version_id,
                        evidence.parse_run_id,
                        evidence.page_number,
                    )
                )
                is_document_evidence = evidence.evidence_type == "DOCUMENT_SPAN" or (
                    evidence.evidence_type == "TABLE_CELL" and has_document_locator
                )
                if evidence.evidence_type == "TABLE_CELL" and not has_document_locator:
                    _reject_evidence(
                        result,
                        bad_evidence,
                        evidence.evidence_id,
                        "MISSING_TABLE_CELL_LOCATOR",
                    )
                    continue
                if not is_document_evidence:
                    continue

                # DOCUMENT_SPAN has a mandatory, typed locator. Missing fields
                # must never cause the deterministic checks to be skipped.
                required_locator = {
                    "section_id": evidence.section_id,
                    "document_version_id": evidence.document_version_id,
                    "parse_run_id": evidence.parse_run_id,
                    "page_number": evidence.page_number,
                }
                missing = [name for name, value in required_locator.items() if value is None]
                if not evidence.content_hash:
                    missing.append("content_hash")
                if missing:
                    for field_name in missing:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            f"MISSING_DOCUMENT_LOCATOR:{field_name}",
                        )
                    continue

                section = await session.get(DocumentSection, evidence.section_id)
                if section is None:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "SECTION_NOT_FOUND"
                    )
                    continue
                if section.text_hash != evidence.content_hash:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "CONTENT_HASH_MISMATCH"
                    )
                if str(section.tenant_id) != str(trusted.tenant_id):
                    _reject_evidence(result, bad_evidence, evidence.evidence_id, "ACL_DENIED")
                if evidence.source_id != section.id:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "SOURCE_SECTION_MISMATCH"
                    )
                if evidence.document_version_id != section.document_version_id:
                    _reject_evidence(
                        result,
                        bad_evidence,
                        evidence.evidence_id,
                        "DOCUMENT_VERSION_MISMATCH",
                    )
                if evidence.parse_run_id != section.parse_run_id:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "PARSE_RUN_MISMATCH"
                    )
                if not section.page_start <= evidence.page_number <= section.page_end:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "PAGE_OUTSIDE_SECTION"
                    )

                # Resolve the ids declared by the citation itself, then prove
                # their ownership and cross-object consistency with the section.
                version = await session.get(DocumentVersion, evidence.document_version_id)
                if version is None:
                    _reject_evidence(
                        result,
                        bad_evidence,
                        evidence.evidence_id,
                        "DOCUMENT_VERSION_NOT_FOUND",
                    )
                else:
                    if str(version.tenant_id) != str(trusted.tenant_id):
                        _reject_evidence(result, bad_evidence, evidence.evidence_id, "ACL_DENIED")
                    if (version.valid_from and trusted.as_of_date < version.valid_from) or (
                        version.valid_to and trusted.as_of_date >= version.valid_to
                    ):
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "OUT_OF_EFFECTIVE_DATE",
                        )
                    available_at = version.source_available_at
                    if available_at is not None:
                        if available_at.tzinfo is None:
                            available_at = available_at.replace(tzinfo=UTC)
                        if available_at > trusted.decision_cutoff_at.astimezone(UTC):
                            _reject_evidence(
                                result,
                                bad_evidence,
                                evidence.evidence_id,
                                "NOT_AVAILABLE_AT_CUTOFF",
                            )

                    case_document = await session.get(
                        CaseDocument,
                        {
                            "case_id": trusted.case_id,
                            "document_version_id": version.id,
                        },
                    )
                    if case_document is None:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "DOCUMENT_NOT_BOUND_TO_CASE",
                        )

                parse_run = await session.get(ParseRun, evidence.parse_run_id)
                if parse_run is None:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "PARSE_RUN_NOT_FOUND"
                    )
                else:
                    if str(parse_run.tenant_id) != str(trusted.tenant_id):
                        _reject_evidence(result, bad_evidence, evidence.evidence_id, "ACL_DENIED")
                    if parse_run.document_version_id != evidence.document_version_id:
                        _reject_evidence(
                            result,
                            bad_evidence,
                            evidence.evidence_id,
                            "PARSE_RUN_VERSION_MISMATCH",
                        )
                    if parse_run.activation_status in {"REVOKED", "TOMBSTONED"}:
                        _reject_evidence(
                            result, bad_evidence, evidence.evidence_id, "PARSE_RUN_REVOKED"
                        )

                # A citation without an immutable input snapshot is not
                # reproducible; missing and out-of-snapshot runs both fail closed.
                if snapshot is None:
                    _reject_evidence(
                        result, bad_evidence, evidence.evidence_id, "SNAPSHOT_REQUIRED"
                    )
                elif evidence.parse_run_id not in set(snapshot.allowed_parse_run_ids):
                    _reject_evidence(
                        result,
                        bad_evidence,
                        evidence.evidence_id,
                        "PARSE_RUN_NOT_IN_SNAPSHOT",
                    )

            # 4. 按 Claim 汇总
            for claim in artifact.claims:
                claim_key = str(claim.claim_id)
                if claim_key in contract_failed_claims:
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(claim_key, []).extend(
                        v for v in contract.violations if claim_key in v
                    )
                    continue
                if any(eid in bad_evidence for eid in claim.supporting_evidence_ids):
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(claim_key, []).append(
                        "INVALID_SUPPORTING_EVIDENCE"
                    )
                    continue
                if any(eid in bad_evidence for eid in claim.opposing_evidence_ids):
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(claim_key, []).append("INVALID_OPPOSING_EVIDENCE")
                    continue
                if any(str(cid) in replay_failed for cid in claim.calculation_ids):
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(claim_key, []).append("REPLAY_FAILED")
                    continue
                if claim.category == "DATA_CONFLICT" or (
                    claim.opposing_evidence_ids and claim.verdict == "PARTIALLY_SUPPORTED"
                ):
                    # WP3：真冲突不投票裁决，保留双方进入人工复核；
                    # 补充材料（非冲突）不在此列，不阻断流程、不送 HITL
                    result.needs_human_review_claim_ids.append(claim.claim_id)
                    continue
                result.accepted_claim_ids.append(claim.claim_id)

        return result
