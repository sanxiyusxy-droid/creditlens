"""Financial Agent（任务 22，文档 §10.2/§10.10）。

约束：
- 不得自行计算或估算数字，所有数字来自 FinancialFact 或 CalculationArtifact；
- 必须说明期间、单位、口径和公式版本；
- 缺输入返回 MISSING_INPUT；
- 异常只是风险信号，不定性为欺诈。
"""

import uuid
from datetime import UTC, date

from creditlens.agents.contracts import AgentArtifact, AgentClaim, AgentEvidenceRef
from creditlens.common.clock import utc_now
from creditlens.formulas.engine import CalculationArtifact
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.tools.gateway import ToolGateway

AGENT_ROLE = "financial_analyst"

# 合成政策口径的核查项：(指标, 公式版本, 展示名)
_CHECKS: list[tuple[str, str, str]] = [
    ("debt_ratio", "1.0", "资产负债率"),
    ("current_ratio", "1.0", "流动比率"),
]


def _calc_evidence(calc: CalculationArtifact) -> AgentEvidenceRef:
    return AgentEvidenceRef(
        evidence_id=uuid.uuid5(uuid.NAMESPACE_URL, f"calc:{calc.trace_hash}"),
        evidence_type="CALCULATION",
        source_id=calc.calculation_id,
        content_hash=calc.trace_hash,
        calculation_id=calc.calculation_id,
        source_available_at=utc_now().astimezone(UTC),
    )


class FinancialAgent:
    def __init__(self, gateway: ToolGateway, period_end: date | None = None):
        self._gateway = gateway
        self._period_end = period_end

    async def run(
        self, run_id: uuid.UUID, task_id: str, trusted: TrustedRequestContext
    ) -> AgentArtifact:
        artifact = AgentArtifact(run_id=run_id, task_id=task_id, producer=AGENT_ROLE)
        period_end = self._period_end or date(trusted.as_of_date.year - 1, 12, 31)

        for metric_code, version, display in _CHECKS:
            calc: CalculationArtifact = await self._gateway.invoke(
                AGENT_ROLE,
                "compute_metric",
                task_id=f"{run_id}:{task_id}",
                trusted=trusted,
                metric_code=metric_code,
                formula_version=version,
                period_end=period_end,
            )
            artifact.calculations.append(calc)
            if calc.status == "CALCULATED":
                ref = _calc_evidence(calc)
                artifact.evidence.append(ref)
                unit = "%" if calc.result_unit == "percent" else ""
                artifact.claims.append(
                    AgentClaim(
                        category="FINANCIAL",
                        statement=(
                            f"{display}（{period_end.isoformat()}，公式 {metric_code}@{version}）"
                            f"为 {calc.result}{unit}。"
                        ),
                        verdict="SUPPORTED",
                        severity="INFO",
                        supporting_evidence_ids=[ref.evidence_id],
                        calculation_ids=[calc.calculation_id],
                        as_of_date=trusted.as_of_date,
                    )
                )
            else:
                artifact.claims.append(
                    AgentClaim(
                        category="FINANCIAL",
                        statement=f"{display}无法计算（状态 {calc.status}）。",
                        verdict="INSUFFICIENT_EVIDENCE",
                        severity="MEDIUM",
                        as_of_date=trusted.as_of_date,
                        uncertainty_reason=f"公式 {metric_code}@{version} 输入缺失或冲突：{calc.status}",
                        recommended_follow_up=[f"补充 {metric_code} 所需财务科目事实"],
                    )
                )
                artifact.execution_status = "PARTIAL"
        return artifact
