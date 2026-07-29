"""Risk Challenger（任务 24，文档 §8.16）。

- 只围绕现有 Claim 和硬约束生成反向 Query，不引入新的企业、日期或案件；
- 每个高影响 Claim 最多一轮；
- 找到反证后保留双方 Evidence，不做多数投票；
- 找不到反证不等于 Claim 自动正确。
"""

import uuid

from creditlens.agents.contracts import AgentArtifact, AgentClaim
from creditlens.agents.policy_agent import _candidate_to_evidence
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.tools.gateway import ToolGateway

AGENT_ROLE = "challenger"

# 类别 -> 反向检索模板（确定性；LLM 版本仍须通过同一约束）
_COUNTER_TEMPLATES = {
    "ELIGIBILITY": "例外 特殊情形 不适用 另有规定",
    "FINANCIAL": "下降 恶化 不满足 低于 高于 预警",
    "CASH_FLOW": "现金流 下降 为负 回款延迟",
    "EXCEPTION": "禁止 不得突破 限制",
    "CONCENTRATION": "集中度 占比 上升 回款延迟",
}

_HIGH_IMPACT = {"HIGH", "CRITICAL"}


class Challenger:
    def __init__(self, gateway: ToolGateway, challenge_all_supported: bool = True):
        self._gateway = gateway
        # MVP 案件小，对全部 SUPPORTED Claim 反证；规模化后只挑高影响
        self._challenge_all = challenge_all_supported

    async def run(
        self,
        run_id: uuid.UUID,
        task_id: str,
        trusted: TrustedRequestContext,
        provisional: list[AgentArtifact],
    ) -> AgentArtifact:
        artifact = AgentArtifact(run_id=run_id, task_id=task_id, producer=AGENT_ROLE)
        challenged = 0

        for source in provisional:
            evidence_by_id = {e.evidence_id: e for e in source.evidence}
            for claim in source.claims:
                if claim.verdict != "SUPPORTED":
                    continue
                if not self._challenge_all and claim.severity not in _HIGH_IMPACT:
                    continue
                template = _COUNTER_TEMPLATES.get(claim.category)
                if template is None:
                    continue
                counter_query = f"{claim.statement[:40]} {template}"
                result = await self._gateway.invoke(
                    AGENT_ROLE, "search_counter_evidence", trusted=trusted, query=counter_query
                )
                challenged += 1

                supporting_sections = {
                    evidence_by_id[eid].section_id
                    for eid in claim.supporting_evidence_ids
                    if eid in evidence_by_id
                }
                counter = [
                    c for c in result.candidates[:3] if c.section_id not in supporting_sections
                ]
                if not counter:
                    continue
                opposing_ids = []
                for candidate in counter:
                    ref = _candidate_to_evidence(candidate)
                    if all(e.evidence_id != ref.evidence_id for e in artifact.evidence):
                        artifact.evidence.append(ref)
                    opposing_ids.append(ref.evidence_id)
                artifact.claims.append(
                    AgentClaim(
                        category="DATA_CONFLICT",
                        statement=(
                            f"对既有结论「{claim.statement[:40]}…」存在需要核对的反向材料，"
                            "支持与反对证据均已保留，待审计与人工裁决。"
                        ),
                        verdict="PARTIALLY_SUPPORTED",
                        severity="MEDIUM",
                        opposing_evidence_ids=opposing_ids,
                        as_of_date=trusted.as_of_date,
                        uncertainty_reason="反证检索发现潜在冲突材料",
                        recommended_follow_up=["由审查员核对正反证据的时点与口径"],
                    )
                )

        artifact.unresolved_issues.append({"challenged_claims": challenged})
        return artifact
