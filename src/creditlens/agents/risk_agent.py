"""Risk Agent（v1.1，文档 §10.2 扩展）。

约束：
- 只分析已有材料中的风险信号，不引入外部数据；
- 不得自行计算或估算数字，所有数字来自 FinancialFact 或 CalculationArtifact；
- 异常只是风险信号，不定性为欺诈或自动拒贷；
- 输出仍为 Claim + Evidence，走统一 Contract Validator。

分析维度（确定性规则 + 检索辅助）：
- 收入/利润/现金流异常变化（同比变动 > 阈值）
- 客户/供应商集中度（第一大客户占比）
- 材料间金额/日期/主体冲突
- 异常交易聚合（应收暴增、经营现金流为负）
"""

import uuid
from datetime import UTC, date

from creditlens.agents.contracts import AgentArtifact, AgentClaim, AgentEvidenceRef
from creditlens.agents.policy_agent import _candidate_to_evidence, consume_evidence_sections
from creditlens.agents.risk_thresholds import load_risk_thresholds
from creditlens.common.clock import utc_now
from creditlens.formulas.engine import CalculationArtifact
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.tools.gateway import ToolGateway

AGENT_ROLE = "risk_analyst"

# 风险检索子问题（确定性；LLM 版本仍须通过同一约束）
_RISK_QUERIES: list[tuple[str, str, str]] = [
    ("客户集中度 第一大客户 占比 依赖", "CONCENTRATION", "客户集中度风险"),
    ("应收账款 大幅增长 回收 坏账", "FINANCIAL", "应收账款异常"),
    ("经营现金流 下降 为负 回款延迟", "CASH_FLOW", "现金流异常"),
    ("关联交易 担保 关联方 利益输送", "RELATED_PARTY", "关联方与担保风险"),
]

# WP3：风险阈值改为版本化配置（risk_thresholds.py），写入 Manifest 可追溯


def _calc_evidence(calc: CalculationArtifact) -> AgentEvidenceRef:
    return AgentEvidenceRef(
        evidence_id=uuid.uuid5(uuid.NAMESPACE_URL, f"risk-calc:{calc.trace_hash}"),
        evidence_type="CALCULATION",
        source_id=calc.calculation_id,
        content_hash=calc.trace_hash,
        calculation_id=calc.calculation_id,
        source_available_at=utc_now().astimezone(UTC),
    )


class RiskAgent:
    """风险分析 Agent：检索年报风险段落 + 重算财务指标阈值。"""

    def __init__(self, gateway: ToolGateway, chat=None, threshold_version: str | None = None):
        self._gateway = gateway
        self._chat = chat
        # WP3：版本化阈值配置（写入 Manifest，评测口径可对齐）
        self.threshold_version, self._thresholds = load_risk_thresholds(threshold_version)

    async def run(
        self, run_id: uuid.UUID, task_id: str, trusted: TrustedRequestContext
    ) -> AgentArtifact:
        artifact = AgentArtifact(run_id=run_id, task_id=task_id, producer=AGENT_ROLE)
        seen_evidence: dict[uuid.UUID, AgentEvidenceRef] = {}
        period_end = date(trusted.as_of_date.year - 1, 12, 31)

        # 1. 财务指标阈值检查（复用 compute_metric 工具）
        for metric_code, thresholds in self._thresholds.items():
            try:
                calc: CalculationArtifact = await self._gateway.invoke(
                    AGENT_ROLE,
                    "compute_metric",
                    task_id=f"{run_id}:{task_id}",
                    trusted=trusted,
                    metric_code=metric_code,
                    formula_version="1.0",
                    period_end=period_end,
                )
            except Exception as exc:
                # WP3：工具异常必须记录降级，不能静默吞掉
                artifact.unresolved_issues.append(
                    {
                        "degraded": True,
                        "tool": "compute_metric",
                        "metric_code": metric_code,
                        "error": type(exc).__name__,
                    }
                )
                # P1：工具部分失败体现在 Artifact 执行状态上（供 Supervisor 标 DEGRADED）
                artifact.execution_status = "DEGRADED"
                continue

            if calc.status != "CALCULATED":
                continue

            artifact.calculations.append(calc)
            ref = _calc_evidence(calc)
            seen_evidence[ref.evidence_id] = ref
            artifact.evidence.append(ref)

            display = thresholds["display"]
            unit = thresholds["unit"]
            value = float(calc.result)

            # 判断是否触发风险信号
            risk_triggered = False
            direction = ""
            if "warn_above" in thresholds and value > thresholds["warn_above"]:
                risk_triggered = True
                direction = f"高于预警线 {thresholds['warn_above']}{unit}"
            elif "warn_below" in thresholds and value < thresholds["warn_below"]:
                risk_triggered = True
                direction = f"低于预警线 {thresholds['warn_below']}{unit}"

            if risk_triggered:
                artifact.claims.append(
                    AgentClaim(
                        category="FINANCIAL",
                        statement=(
                            f"{display}（{period_end.isoformat()}）为 {calc.result}{unit}，"
                            f"{direction}，需关注偿债能力变化。"
                        ),
                        verdict="SUPPORTED",
                        severity="HIGH",
                        supporting_evidence_ids=[ref.evidence_id],
                        calculation_ids=[calc.calculation_id],
                        as_of_date=trusted.as_of_date,
                    )
                )

        # 2. 年报风险段落检索
        for query, category, topic in _RISK_QUERIES:
            try:
                result = await self._gateway.invoke(
                    AGENT_ROLE,
                    "search_risk_evidence",
                    task_id=f"{run_id}:{task_id}",
                    trusted=trusted,
                    query=query,
                )
            except Exception as exc:
                # WP3：检索工具异常同样记录降级
                artifact.unresolved_issues.append(
                    {
                        "degraded": True,
                        "tool": "search_risk_evidence",
                        "query_topic": topic,
                        "error": type(exc).__name__,
                    }
                )
                # P1：工具部分失败必须体现在 Artifact 执行状态上，
                # 否则 Supervisor/报告层无法得知本次风险分析是降级产出
                artifact.execution_status = "DEGRADED"
                continue

            # WP2：消费 Packed Sections（含相邻复核与预算控制）
            top = consume_evidence_sections(result, limit=2)
            if not top:
                continue

            evidence_ids = []
            for candidate in top:
                ref = _candidate_to_evidence(candidate)
                if ref.evidence_id not in seen_evidence:
                    seen_evidence[ref.evidence_id] = ref
                    artifact.evidence.append(ref)
                evidence_ids.append(ref.evidence_id)

            heading = " / ".join((c.heading_path[-1] if c.heading_path else "段落") for c in top)
            artifact.claims.append(
                AgentClaim(
                    category=category,
                    statement=(
                        f"年报中「{topic}」相关披露（{heading}）存在需要审查员关注的风险信号。"
                    ),
                    verdict="SUPPORTED",
                    severity="MEDIUM",
                    supporting_evidence_ids=evidence_ids,
                    as_of_date=trusted.as_of_date,
                    recommended_follow_up=[f"由审查员核对{topic}的具体情况"],
                )
            )

        if not artifact.claims:
            artifact.claims.append(
                AgentClaim(
                    category="FINANCIAL",
                    statement="未发现明显风险信号（基于现有材料和阈值规则）。",
                    verdict="INSUFFICIENT_EVIDENCE",
                    severity="INFO",
                    as_of_date=trusted.as_of_date,
                    uncertainty_reason="材料有限，未覆盖全部风险维度",
                )
            )
            # 证据不足是 PARTIAL；但若已因工具失败标记 DEGRADED，保留更严重的降级语义
            if artifact.execution_status != "DEGRADED":
                artifact.execution_status = "PARTIAL"

        # WP3：阈值配置版本随 Artifact 可追溯（Supervisor 另写入 Manifest）
        artifact.unresolved_issues.append({"risk_threshold_version": self.threshold_version})
        return artifact
