"""Evidence Auditor（任务 25，文档 §8.15/§10.10）。

先执行确定性检查，语义判断不得覆盖权限、时点或哈希失败：
1. Contract Validator（Claim-Evidence 绑定完整性）；
2. DOCUMENT_SPAN 证据回表：Section 存在、text_hash 一致、租户匹配；
3. CALCULATION 证据重放：公式重算结果与 trace_hash 一致；
4. 存在反证的 Claim 保留双方，标记需人工复核。
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.agents.contracts import (
    AgentArtifact,
    validate_artifact_contract,
)
from creditlens.formulas.engine import FormulaRegistry, replay_calculation
from creditlens.infrastructure.postgres.models import DocumentSection
from creditlens.retrieval.contracts import TrustedRequestContext

AGENT_ROLE = "evidence_auditor"


@dataclass
class AuditResult:
    accepted_claim_ids: list[uuid.UUID] = field(default_factory=list)
    rejected_claim_ids: list[uuid.UUID] = field(default_factory=list)
    needs_human_review_claim_ids: list[uuid.UUID] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    replay_failures: list[str] = field(default_factory=list)

    @property
    def requires_human_review(self) -> bool:
        return bool(self.needs_human_review_claim_ids)

    @property
    def ok(self) -> bool:
        return not self.rejected_claim_ids and not self.replay_failures


class EvidenceAuditor:
    def __init__(self, registry: FormulaRegistry):
        self._registry = registry

    async def verify(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        artifacts: list[AgentArtifact],
    ) -> AuditResult:
        result = AuditResult()

        for artifact in artifacts:
            # 1. Contract 校验
            contract = validate_artifact_contract(artifact, trusted.as_of_date)
            contract_failed_claims = {
                v.split(":", 2)[1] for v in contract.violations if v.startswith("claim:")
            }

            # 2/3. 证据级确定性校验
            bad_evidence: set[uuid.UUID] = set()
            for evidence in artifact.evidence:
                if evidence.evidence_type == "DOCUMENT_SPAN" and evidence.section_id:
                    section = await session.get(DocumentSection, evidence.section_id)
                    if section is None or section.text_hash != evidence.content_hash:
                        bad_evidence.add(evidence.evidence_id)
                        result.violations.setdefault(str(evidence.evidence_id), []).append(
                            "INVALID_REFERENCE"
                        )
                    elif str(section.tenant_id) != str(trusted.tenant_id):
                        bad_evidence.add(evidence.evidence_id)
                        result.violations.setdefault(str(evidence.evidence_id), []).append(
                            "ACL_DENIED"
                        )
            for calc in artifact.calculations:
                consistent, _ = replay_calculation(self._registry, calc)
                if not consistent:
                    result.replay_failures.append(str(calc.calculation_id))

            replay_failed = set(result.replay_failures)

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
                    result.violations.setdefault(claim_key, []).append("INVALID_REFERENCE")
                    continue
                if any(str(cid) in replay_failed for cid in claim.calculation_ids):
                    result.rejected_claim_ids.append(claim.claim_id)
                    result.violations.setdefault(claim_key, []).append("REPLAY_FAILED")
                    continue
                if claim.opposing_evidence_ids or claim.category == "DATA_CONFLICT":
                    # 冲突不投票裁决，保留双方进入人工复核
                    result.needs_human_review_claim_ids.append(claim.claim_id)
                    continue
                result.accepted_claim_ids.append(claim.claim_id)

        return result
