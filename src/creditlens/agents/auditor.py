"""Evidence Auditor（任务 25，文档 §8.15/§10.10；结构化证据验证）。

说明：本 Agent 输出的是 structural verification（结构验证），不是语义蕴含结论：
只验证可机器核验的绑定/哈希/时点/权限一致性，并以轻量词项覆盖将明显无关的
非数字 Claim 转人工复核；它不会宣称 Claim 被证据在语义上蕴含。

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

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC

from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.agents.contracts import (
    GROUNDING_EXECUTION_STATUS_BY_ANSWER,
    AgentArtifact,
    AnswerStatus,
    GroundedAnswerArtifact,
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

# 仅用于轻量 structural coverage；不是语义蕴含映射或语义裁判。
_STRUCTURAL_CATEGORY_TERMS = {
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
    blocking_failures: list[str] = field(default_factory=list)
    degraded: bool = False  # v1.1：非核心 Agent 失败时标记
    verification_scope: str = "STRUCTURAL_VERIFICATION"
    semantic_entailment_verified: bool = False
    # v1.3：答案级约束可能在没有 Claim（例如非法拒答）时失败，不能只依赖
    # rejected_claim_ids 推断整体结果。
    grounded_answer_ok: bool = True
    derived_answer_status: str | None = None

    @property
    def requires_human_review(self) -> bool:
        return bool(self.needs_human_review_claim_ids)

    @property
    def ok(self) -> bool:
        return (
            not self.rejected_claim_ids
            and not self.replay_failures
            and not self.blocking_failures
            and self.grounded_answer_ok
        )


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
        *,
        allow_grounded_document_numeric: bool = False,
    ) -> AuditResult:
        result = AuditResult()

        # WP3：Case 复核——案件存在且租户一致，否则全部 Claim 拒绝
        case = await session.get(CreditCase, trusted.case_id)
        case_ok = case is not None and str(case.tenant_id) == str(trusted.tenant_id)
        if not case_ok:
            result.blocking_failures.append("CASE_VALIDATION_FAILED")
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
                if isinstance(artifact, GroundedAnswerArtifact):
                    result.blocking_failures.append("GROUNDING_EXECUTION_FAILED")
                    for claim in artifact.claims:
                        result.rejected_claim_ids.append(claim.claim_id)
                        result.violations.setdefault(str(claim.claim_id), []).append(
                            "GROUNDING_EXECUTION_FAILED"
                        )
                    continue
                if artifact.producer in _CORE_PRODUCERS:
                    # 核心 Agent 失败：全部 Claim 拒绝
                    result.blocking_failures.append(f"CORE_AGENT_FAILED:{artifact.producer}")
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
            contract = validate_artifact_contract(
                artifact,
                trusted.as_of_date,
                allow_grounded_document_numeric=allow_grounded_document_numeric,
            )
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

    async def verify_grounded_answer(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        artifact: GroundedAnswerArtifact,
        *,
        allowed_evidence_ids: set[uuid.UUID],
        snapshot=None,
    ) -> AuditResult:
        """在既有回表审计之上增加答案层 structural verification 门禁。

        该方法仍不冒充语义蕴含裁判；它验证的是：引用来自本次已打包上下文、
        数字/日期能在所引原文中逐字找到、答案逐句等于已审计 Claim，以及答案状态
        只能由证据形态推导。轻量词项覆盖只能发现明显无关反例，不能证明语义蕴含；
        语义 Faithfulness 继续由冻结答案集独立评估。
        """
        actual_status = getattr(artifact.answer_status, "value", artifact.answer_status)
        try:
            normalized_status = AnswerStatus(actual_status)
        except ValueError:
            normalized_status = None
        expected_execution_status = (
            GROUNDING_EXECUTION_STATUS_BY_ANSWER.get(normalized_status)
            if normalized_status is not None
            else None
        )

        # Pydantic's model_copy(update=...) and old persisted payloads can bypass
        # model validators.  Audit the state matrix independently and fail before
        # deriving any business answer state from a failed generation.
        if artifact.execution_status == "FAILED":
            return AuditResult(
                blocking_failures=["GROUNDING_EXECUTION_FAILED"],
                violations={"grounded_answer": ["GROUNDING_EXECUTION_FAILED"]},
                grounded_answer_ok=False,
            )
        if expected_execution_status != artifact.execution_status:
            code = f"GROUNDING_STATUS_MATRIX_MISMATCH:{artifact.execution_status}:{actual_status}"
            return AuditResult(
                blocking_failures=[code],
                violations={"grounded_answer": [code]},
                grounded_answer_ok=False,
            )

        result = await self.verify(
            session,
            trusted,
            [artifact],
            snapshot=snapshot,
            allow_grounded_document_numeric=True,
        )
        if result.blocking_failures:
            result.grounded_answer_ok = False
            return result
        claims_by_id = {claim.claim_id: claim for claim in artifact.claims}
        evidence_by_id = {evidence.evidence_id: evidence for evidence in artifact.evidence}

        cited_ids = {
            evidence_id
            for claim in artifact.claims
            for evidence_id in (claim.supporting_evidence_ids + claim.opposing_evidence_ids)
        }
        artifact_key = "grounded_answer"

        def reject_artifact(code: str) -> None:
            result.grounded_answer_ok = False
            violations = result.violations.setdefault(artifact_key, [])
            if code not in violations:
                violations.append(code)

        def reject_claim(claim_id: uuid.UUID, code: str) -> None:
            result.grounded_answer_ok = False
            if claim_id not in result.rejected_claim_ids:
                result.rejected_claim_ids.append(claim_id)
            result.accepted_claim_ids = [
                item for item in result.accepted_claim_ids if item != claim_id
            ]
            result.needs_human_review_claim_ids = [
                item for item in result.needs_human_review_claim_ids if item != claim_id
            ]
            violations = result.violations.setdefault(str(claim_id), [])
            if code not in violations:
                violations.append(code)

        def flag_claim_for_review(claim_id: uuid.UUID, code: str) -> None:
            if claim_id in result.rejected_claim_ids:
                return
            result.accepted_claim_ids = [
                item for item in result.accepted_claim_ids if item != claim_id
            ]
            if claim_id not in result.needs_human_review_claim_ids:
                result.needs_human_review_claim_ids.append(claim_id)
            violations = result.violations.setdefault(str(claim_id), [])
            if code not in violations:
                violations.append(code)

        unknown_context_ids = set(evidence_by_id) - allowed_evidence_ids
        if unknown_context_ids:
            for evidence_id in sorted(unknown_context_ids, key=str):
                reject_artifact(f"EVIDENCE_NOT_IN_PACKED_CONTEXT:{evidence_id}")

        if set(evidence_by_id) != cited_ids:
            reject_artifact("ARTIFACT_EVIDENCE_NOT_EXACTLY_CITED")

        for claim in artifact.claims:
            claim_evidence_ids = claim.supporting_evidence_ids + claim.opposing_evidence_ids
            if any(evidence_id not in allowed_evidence_ids for evidence_id in claim_evidence_ids):
                reject_claim(claim.claim_id, "CITATION_NOT_IN_PACKED_CONTEXT")

            cited_texts: list[str] = []
            for evidence_id in claim.supporting_evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None or evidence.section_id is None:
                    continue
                section = await session.get(DocumentSection, evidence.section_id)
                if section is not None:
                    cited_texts.append(section.text)

            # 数字、百分比和 ISO/中文日期只能来自该 Claim 的正式引用原文。
            # 比较的是带符号、币种、量级和单位的规范化 span，而不是裸数字
            # 子串；否则 ``-10%`` 会被 ``+10%`` 支撑，``50000万元`` 也会被
            # ``50000元`` 错误支撑。格式层仅兼容 NFKC、空格和千分位，不做
            # 金额、比例或期间的单位换算。
            numeric_spans = _answer_numeric_spans(claim.statement)
            if numeric_spans:
                cited_span_keys = {
                    span.key
                    for cited_text in cited_texts
                    for span in _answer_numeric_spans(cited_text)
                }
                for span in numeric_spans:
                    if span.key not in cited_span_keys:
                        reject_claim(
                            claim.claim_id,
                            f"NUMERIC_TOKEN_NOT_IN_CITATION:{span.display}",
                        )

            # This is deliberately a structural overlap guard, not entailment.
            # A clearly unrelated Claim must not remain auto-answered.  This guard
            # intentionally still runs after numeric validation: a copied number
            # alone cannot make an otherwise unrelated statement grounded.
            if (
                claim.verdict == "SUPPORTED"
                and claim.claim_id not in result.rejected_claim_ids
                and _has_obvious_polarity_conflict(claim.statement, cited_texts)
            ):
                flag_claim_for_review(
                    claim.claim_id,
                    "POLARITY_CONFLICT_WITH_CITATION",
                )

            # Ambiguous cases are routed to review instead of being hard rejected.
            if (
                claim.verdict == "SUPPORTED"
                and claim.claim_id not in result.rejected_claim_ids
                and not _has_structural_coverage(
                    claim.statement,
                    cited_texts,
                    category=claim.category,
                )
            ):
                flag_claim_for_review(
                    claim.claim_id,
                    "STRUCTURAL_COVERAGE_INSUFFICIENT",
                )

        has_review_signal = bool(
            artifact.conflicts
            or artifact.missing_information
            or result.needs_human_review_claim_ids
        ) or any(
            claim.category == "DATA_CONFLICT"
            or bool(claim.opposing_evidence_ids)
            or claim.verdict != "SUPPORTED"
            for claim in artifact.claims
        )
        if has_review_signal and artifact.claims:
            derived_status = "NEEDS_REVIEW"
        elif artifact.claims:
            derived_status = "ANSWERED"
        else:
            derived_status = "ABSTAINED"
        result.derived_answer_status = derived_status

        if actual_status != derived_status:
            reject_artifact(f"ANSWER_STATUS_MISMATCH:{actual_status}:{derived_status}")

        expected_answer = " ".join(
            claim.statement.strip() for claim in artifact.claims if claim.verdict == "SUPPORTED"
        ).strip()
        if derived_status == "ABSTAINED":
            if artifact.direct_answer:
                reject_artifact("ABSTENTION_MUST_NOT_HAVE_DIRECT_ANSWER")
            if not artifact.abstention_reason:
                reject_artifact("ABSTENTION_REASON_REQUIRED")
        elif derived_status == "NEEDS_REVIEW":
            if artifact.direct_answer:
                reject_artifact("NEEDS_REVIEW_MUST_NOT_HAVE_DIRECT_ANSWER")
        elif (artifact.direct_answer or "").strip() != expected_answer:
            reject_artifact("DIRECT_ANSWER_NOT_EXACT_CLAIM_RENDERING")

        # 防止模型把不存在于 Artifact 的句子或 Claim 偷渡进答案对象。
        if len(claims_by_id) != len(artifact.claims):
            reject_artifact("DUPLICATE_CLAIM_ID")
        return result


@dataclass(frozen=True)
class _NumericSpan:
    """One comparison-safe numeric/date mention.

    ``display`` deliberately stays compatible with the public violation format;
    ``key`` carries the stricter sign/currency/unit semantics used for matching.
    """

    display: str
    key: tuple[str, ...]


_ANSWER_DATE_RE = re.compile(
    r"(?<!\d)\d{4}\s*(?:"
    r"[-/]\s*\d{1,2}(?:\s*[-/]\s*\d{1,2})?"
    r"|年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r")(?!\d)"
)
_ANSWER_NUMBER_RE = re.compile(
    r"(?<![\d.])"
    r"(?P<sign_before>[+-])?\s*"
    r"(?P<currency_prefix>人民币|rmb|cny|美元|usd|港元|hkd|欧元|eur|日元|jpy|[¥$])?\s*"
    r"(?P<sign_after>[+-])?\s*"
    r"(?P<value>(?:\d{1,3}(?:[\s,]\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?:"
    r"(?P<ratio_unit>个百分点|百分点|基点|bps?|%)"
    r"|(?P<period_unit>年度|年|个月|月|季度|季|星期|周|天|日)"
    r"|(?P<scale>万亿|千亿|百亿|亿|千万|百万|十万|万|千|百)?\s*"
    r"(?P<currency_suffix>人民币|rmb|cny|美元|usd|港元|hkd|欧元|eur|日元|jpy|元)"
    r"|(?P<scale_only>万亿|千亿|百亿|亿|千万|百万|十万|万|千|百)"
    r")?"
    r"(?![\d.])",
    re.IGNORECASE,
)
_STRUCTURAL_TEXT_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_STRUCTURAL_PREFIXES = (
    "根据现有证据",
    "现有证据表明",
    "现有证据显示",
    "根据材料",
    "材料显示",
    "证据显示",
    "原文显示",
)
_MIN_STRUCTURAL_COVERAGE = 0.55

_CURRENCY_ALIASES = {
    "人民币": "CNY",
    "rmb": "CNY",
    "cny": "CNY",
    "元": "CNY",
    "¥": "CNY",
    "美元": "USD",
    "usd": "USD",
    "$": "USD",
    "港元": "HKD",
    "hkd": "HKD",
    "欧元": "EUR",
    "eur": "EUR",
    "日元": "JPY",
    "jpy": "JPY",
}

# Only high-confidence lexical oppositions live here.  In particular, generic
# negation and threshold words are excluded so “不得超过 70%” and “上限 70%”
# are not treated as contradictory merely because one uses “不得”。
_POLARITY_PAIRS = (
    ("不存在", "存在"),
    ("不满足", "满足"),
    ("不符合", "符合"),
    ("未通过", "通过"),
    ("上升", "下降"),
    ("增加", "减少"),
    ("增长", "下降"),
)


def _canonical_decimal(raw: str) -> str:
    """Normalize harmless numeric formatting without changing the unit."""
    compact = re.sub(r"[\s,]", "", raw)
    integer, dot, fraction = compact.partition(".")
    integer = integer.lstrip("0") or "0"
    if not dot:
        return integer
    fraction = fraction.rstrip("0")
    return integer if not fraction else f"{integer}.{fraction}"


def _canonical_currency(raw: str | None) -> str:
    if not raw:
        return ""
    return _CURRENCY_ALIASES.get(raw.casefold(), raw.casefold())


def _answer_numeric_spans(statement: str) -> list[_NumericSpan]:
    """Extract comparison-safe number/date spans from model-authored text."""
    normalized = unicodedata.normalize("NFKC", statement).casefold()
    spans: list[_NumericSpan] = []
    occupied: list[tuple[int, int]] = []

    for match in _ANSWER_DATE_RE.finditer(normalized):
        digits = tuple(str(int(item)) for item in re.findall(r"\d+", match.group(0)))
        display = re.sub(r"\s+", "", match.group(0))
        spans.append(_NumericSpan(display=display, key=("date", *digits)))
        occupied.append(match.span())

    for match in _ANSWER_NUMBER_RE.finditer(normalized):
        start, end = match.span()
        if any(
            start < occupied_end and end > occupied_start
            for occupied_start, occupied_end in occupied
        ):
            continue

        sign_before = match.group("sign_before") or ""
        sign_after = match.group("sign_after") or ""
        # Two explicit signs are malformed rather than silently simplified.
        sign = sign_before + sign_after
        value = _canonical_decimal(match.group("value"))
        prefix_currency = _canonical_currency(match.group("currency_prefix"))
        suffix_currency = _canonical_currency(match.group("currency_suffix"))
        currency = prefix_currency or suffix_currency
        if prefix_currency and suffix_currency and prefix_currency != suffix_currency:
            currency = f"{prefix_currency}/{suffix_currency}"

        ratio_unit = (match.group("ratio_unit") or "").casefold()
        if ratio_unit in {"百分点", "个百分点"}:
            ratio_unit = "个百分点"
        elif ratio_unit in {"bp", "bps"}:
            ratio_unit = "bp"
        period_unit = match.group("period_unit") or ""
        scale = match.group("scale") or match.group("scale_only") or ""

        if ratio_unit:
            kind = "ratio"
            unit_key = ratio_unit
            display = f"{sign}{value}{ratio_unit}"
        elif period_unit:
            kind = "period"
            unit_key = period_unit
            display = f"{sign}{value}{period_unit}"
        elif currency or scale:
            kind = "amount"
            unit_key = f"{currency}:{scale}"
            # Preserve the established violation payload (bare amount value)
            # while the internal key still binds currency and scale.
            display = f"{sign}{value}"
        else:
            kind = "number"
            unit_key = ""
            display = f"{sign}{value}"

        spans.append(
            _NumericSpan(
                display=display,
                key=(kind, sign, value, unit_key),
            )
        )

    # Keep source order and suppress identical repeated mentions.
    ordered = sorted(spans, key=lambda span: normalized.find(span.display.casefold()))
    return list(dict.fromkeys(ordered))


def _answer_numeric_tokens(statement: str) -> list[str]:
    """Backward-compatible violation tokens for all normalized numeric spans."""
    return list(dict.fromkeys(span.display for span in _answer_numeric_spans(statement)))


def _normalize_structural_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    for prefix in _STRUCTURAL_PREFIXES:
        normalized_prefix = unicodedata.normalize("NFKC", prefix).casefold()
        if normalized.startswith(normalized_prefix):
            normalized = normalized[len(normalized_prefix) :].lstrip(" ：:，,。")
            break
    return normalized


def _structural_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _STRUCTURAL_TEXT_RE.findall(text):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", token):
            if len(token) == 1:
                terms.add(token)
            else:
                terms.update(token[index : index + 2] for index in range(len(token) - 1))
        elif len(token) >= 2:
            terms.add(token)
    return terms


def _polarity_markers(text: str, first: str, second: str) -> set[str]:
    """Return explicit markers while avoiding substring negation mistakes."""
    compact = "".join(_STRUCTURAL_TEXT_RE.findall(_normalize_structural_text(text)))
    markers: set[str] = set()
    if first in compact:
        markers.add(first)
    # “存在” is a substring of “不存在” (likewise 满足/不满足 and
    # 通过/未通过).  Mask the longer marker before looking for its opposite.
    without_first = compact.replace(first, "")
    if second in without_first:
        markers.add(second)
    return markers


def _has_obvious_polarity_conflict(statement: str, cited_texts: list[str]) -> bool:
    """Detect only high-confidence lexical reversals and route them to review."""
    if not cited_texts:
        return False
    source = "\n".join(cited_texts)
    for first, second in _POLARITY_PAIRS:
        claim_markers = _polarity_markers(statement, first, second)
        source_markers = _polarity_markers(source, first, second)
        if (first in claim_markers and second in source_markers) or (
            second in claim_markers and first in source_markers
        ):
            return True
    return False


def _has_structural_coverage(
    statement: str,
    cited_texts: list[str],
    *,
    category: str,
) -> bool:
    """Conservative lexical guard; a True result is not semantic entailment."""
    claim_text = _normalize_structural_text(statement)
    source_text = _normalize_structural_text("\n".join(cited_texts))
    claim_compact = "".join(_STRUCTURAL_TEXT_RE.findall(claim_text))
    source_compact = "".join(_STRUCTURAL_TEXT_RE.findall(source_text))
    if not claim_compact or not source_compact:
        return False
    if claim_compact in source_compact:
        return True

    category_terms = _STRUCTURAL_CATEGORY_TERMS.get(category, [])
    claim_category_terms = {term for term in category_terms if term.casefold() in claim_compact}
    if claim_category_terms and not any(
        term.casefold() in source_compact for term in claim_category_terms
    ):
        return False

    claim_terms = _structural_terms(claim_text)
    if not claim_terms:
        return False
    source_terms = _structural_terms(source_text)
    coverage = len(claim_terms & source_terms) / len(claim_terms)
    return coverage >= _MIN_STRUCTURAL_COVERAGE
