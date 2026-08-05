"""Multi-Agent 评测指标（文档 §16.8）。

- Challenger Counter-Evidence Recall：反证检索召回率
- Auditor Unsupported Claim 拦截率：无证据支撑 Claim 被正确拒绝的比率
"""

from dataclasses import dataclass, field


@dataclass
class ChallengerMetrics:
    """Challenger 反证检索评测。"""

    total_claims_challenged: int = 0
    counter_evidence_found: int = 0
    true_conflicts_detected: int = 0
    false_conflicts: int = 0

    @property
    def counter_evidence_recall(self) -> float:
        """对需要反证的 Claim，实际找到反证的比例。"""
        if self.total_claims_challenged == 0:
            return 0.0
        return self.counter_evidence_found / self.total_claims_challenged

    @property
    def conflict_precision(self) -> float:
        """标记为 CONFLICT 的反证中真正存在数字矛盾的比例。"""
        total_conflicts = self.true_conflicts_detected + self.false_conflicts
        if total_conflicts == 0:
            return 0.0
        return self.true_conflicts_detected / total_conflicts

    def summary(self) -> dict:
        return {
            "total_claims_challenged": self.total_claims_challenged,
            "counter_evidence_found": self.counter_evidence_found,
            "counter_evidence_recall": round(self.counter_evidence_recall, 4),
            "true_conflicts_detected": self.true_conflicts_detected,
            "false_conflicts": self.false_conflicts,
            "conflict_precision": round(self.conflict_precision, 4),
        }


@dataclass
class AuditorMetrics:
    """Auditor 验证拦截评测。"""

    total_claims_audited: int = 0
    claims_approved: int = 0
    claims_rejected: int = 0
    unsupported_claims_total: int = 0
    unsupported_claims_rejected: int = 0
    core_agent_failures_blocked: int = 0
    non_core_degraded: int = 0

    @property
    def unsupported_rejection_rate(self) -> float:
        """无证据支撑 Claim 被正确拦截的比率。"""
        if self.unsupported_claims_total == 0:
            return 0.0
        return self.unsupported_claims_rejected / self.unsupported_claims_total

    @property
    def approval_rate(self) -> float:
        """Claim 通过率。"""
        if self.total_claims_audited == 0:
            return 0.0
        return self.claims_approved / self.total_claims_audited

    def summary(self) -> dict:
        return {
            "total_claims_audited": self.total_claims_audited,
            "claims_approved": self.claims_approved,
            "claims_rejected": self.claims_rejected,
            "approval_rate": round(self.approval_rate, 4),
            "unsupported_claims_total": self.unsupported_claims_total,
            "unsupported_claims_rejected": self.unsupported_claims_rejected,
            "unsupported_rejection_rate": round(self.unsupported_rejection_rate, 4),
            "core_agent_failures_blocked": self.core_agent_failures_blocked,
            "non_core_degraded": self.non_core_degraded,
        }


@dataclass
class AgentEvalReport:
    """Multi-Agent 综合评测报告。"""

    dataset_id: str = ""
    case_key: str = ""
    challenger: ChallengerMetrics = field(default_factory=ChallengerMetrics)
    auditor: AuditorMetrics = field(default_factory=AuditorMetrics)

    def summary(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "case_key": self.case_key,
            "challenger": self.challenger.summary(),
            "auditor": self.auditor.summary(),
        }
