"""Supervisor（任务 23/26/27，文档 §10.6/§10.7/§10.8）。

- Supervisor 是唯一控制中心；专业 Agent 不直接互相通信；
- 固定 DAG：AUTHORIZED -> VALIDATING_CASE -> PLANNING -> EXECUTING
  (Policy || Financial) -> SYNTHESIZING -> CHALLENGING -> AUDITING
  -> HUMAN_REVIEW | REPORTING -> COMPLETED；
- 每次状态迁移写 run_events（Trace，任务 27）；
- Auditor 要求人工复核时 Run 停在 HUMAN_REVIEW，等待 HumanDecision（任务 26）；
- Supervisor 自身不生成业务结论。
"""

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.agents.auditor import AuditResult, EvidenceAuditor
from creditlens.agents.challenger import Challenger
from creditlens.agents.contracts import AgentArtifact
from creditlens.agents.financial_agent import FinancialAgent
from creditlens.agents.policy_agent import PolicyAgent
from creditlens.common.clock import utc_now
from creditlens.common.errors import InvalidStateTransitionError
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    CaseDocument,
    ClaimRecord,
    CreditCase,
    HumanDecision,
    ReviewRun,
    RunEvent,
)
from creditlens.retrieval.contracts import TrustedRequestContext

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext

# 权威状态机（文档 §10.7 简化为 MVP 固定路径）
_TRANSITIONS: dict[str, set[str]] = {
    "RECEIVED": {"AUTHORIZED", "DENIED"},
    "AUTHORIZED": {"VALIDATING_CASE"},
    "VALIDATING_CASE": {"PLANNING", "NEED_MORE_INFO", "DATA_QUALITY_BLOCKED"},
    "PLANNING": {"EXECUTING"},
    "EXECUTING": {"SYNTHESIZING", "FAILED"},
    "SYNTHESIZING": {"CHALLENGING"},
    "CHALLENGING": {"AUDITING"},
    "AUDITING": {"HUMAN_REVIEW", "REPORTING", "NEED_MORE_INFO"},
    "HUMAN_REVIEW": {"REPORTING", "REWORK"},
    "REWORK": {"HUMAN_REVIEW", "PLANNING"},  # P0-3：REWORK 有恢复路径
    "REPORTING": {"COMPLETED"},
}


@dataclass
class RunOutcome:
    run_id: uuid.UUID
    status: str
    artifacts: list[AgentArtifact] = field(default_factory=list)
    audit: AuditResult | None = None


class Supervisor:
    def __init__(
        self,
        policy_agent: PolicyAgent,
        financial_agent: FinancialAgent,
        challenger: Challenger,
        auditor: EvidenceAuditor,
    ):
        self._policy = policy_agent
        self._financial = financial_agent
        self._challenger = challenger
        self._auditor = auditor

    async def execute_full_review(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        snapshot: "SnapshotContext | None" = None,
        run: ReviewRun | None = None,
        commit_each_stage: bool = False,
    ) -> RunOutcome:
        """执行固定 DAG。run 为 None 时自建；API 异步路径可传入预创建的
        RECEIVED 状态 Run（先返回 run_id，后台继续执行）。

        commit_each_stage=True（P0-4）：每次状态迁移后提交事务——SSE 可见
        实时进度，且中途异常不回滚已完成阶段的 Artifact/Trace。"""
        self._commit_each_stage = commit_each_stage
        if run is None:
            run = ReviewRun(
                tenant_id=trusted.tenant_id,
                case_id=trusted.case_id,
                run_type="FULL_REVIEW",
                status="RECEIVED",
                as_of_date=trusted.as_of_date,
                decision_cutoff_at=trusted.decision_cutoff_at,
                input_snapshot_id=snapshot.snapshot_id if snapshot else None,
                model_manifest={"mode": "deterministic-mvp"},
            )
            session.add(run)
            await session.flush()
        elif run.status != "RECEIVED":
            raise InvalidStateTransitionError("预建 Run 必须处于 RECEIVED 状态")
        seq = _EventWriter(session, run)

        await self._transition(session, run, seq, "AUTHORIZED")

        # 校验案件可审查
        case = await session.get(CreditCase, trusted.case_id)
        await self._transition(session, run, seq, "VALIDATING_CASE")
        if case is None:
            await self._transition(session, run, seq, "DATA_QUALITY_BLOCKED", force=True)
            return RunOutcome(run.id, run.status)
        bound = await session.scalar(
            select(CaseDocument.case_id).where(CaseDocument.case_id == case.id).limit(1)
        )
        if bound is None:
            await self._transition(session, run, seq, "NEED_MORE_INFO", force=True)
            return RunOutcome(run.id, run.status)

        # 固定 DAG：MVP 由代码生成，不经 LLM Planner
        await self._transition(session, run, seq, "PLANNING")
        await self._transition(session, run, seq, "EXECUTING")

        # P0-4：专业 Agent 串行执行——多个 Agent 共享同一 AsyncSession，
        # asyncio.gather 并发使用同一 Session 不安全。真正并行需要 per-Agent
        # 独立 Session + 持久任务队列（Celery），列入偏差 D9。
        professional: list[AgentArtifact] = []
        for name, agent, task_key in (
            ("policy", self._policy, "policy_review"),
            ("financial", self._financial, "financial_analysis"),
        ):
            try:
                result = await agent.run(run.id, task_key, trusted)
                professional.append(result)
                await seq.emit("TASK_COMPLETED", {"task": name, "claims": len(result.claims)})
            except Exception as exc:
                await seq.emit("TASK_FAILED", {"task": name, "error": type(exc).__name__})
        if not professional:
            await self._transition(session, run, seq, "FAILED", force=True)
            return RunOutcome(run.id, run.status)

        await self._transition(session, run, seq, "SYNTHESIZING")
        await self._persist_artifacts(session, run, professional)

        await self._transition(session, run, seq, "CHALLENGING")
        challenge = await self._challenger.run(run.id, "challenger", trusted, professional)
        await self._persist_artifacts(session, run, [challenge])
        all_artifacts = [*professional, challenge]

        await self._transition(session, run, seq, "AUDITING")
        audit = await self._auditor.verify(session, trusted, all_artifacts)
        await seq.emit(
            "AUDIT_COMPLETED",
            {
                "accepted": len(audit.accepted_claim_ids),
                "rejected": len(audit.rejected_claim_ids),
                "human_review": len(audit.needs_human_review_claim_ids),
            },
        )
        await self._apply_audit_to_claims(session, run, audit)

        # P0-3：Auditor 拒绝的 Claim 同样是 blocking——不得进入 REPORTING，
        # 必须由人工处理（重做或显式否决）后才可能完成
        if audit.requires_human_review or audit.rejected_claim_ids:
            await self._transition(session, run, seq, "HUMAN_REVIEW")
            return RunOutcome(run.id, run.status, all_artifacts, audit)

        await self._transition(session, run, seq, "REPORTING")
        # P0-3：报告版本成功持久化是 COMPLETED 的前置条件
        await _persist_report_version(session, run)
        await self._transition(session, run, seq, "COMPLETED")
        run.completed_at = utc_now()
        return RunOutcome(run.id, run.status, all_artifacts, audit)

    async def resume_after_human_review(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        decision: HumanDecision,
    ) -> str:
        return await resume_after_human_review(session, run_id, decision)

    async def _transition(
        self,
        session: AsyncSession,
        run: ReviewRun,
        seq: "_EventWriter",
        new_status: str,
        force: bool = False,
    ) -> None:
        allowed = _TRANSITIONS.get(run.status, set())
        if not force and new_status not in allowed:
            raise InvalidStateTransitionError(f"{run.status} -> {new_status} 不被状态机允许")
        old = run.status
        run.status = new_status
        run.state_version += 1
        await seq.emit("STATE_CHANGED", {"from": old, "to": new_status})
        await session.flush()
        # P0-4：阶段 Checkpoint——提交后 SSE 立即可见，异常不回滚已完成阶段；
        # checkpoint_commit 会恢复 SET LOCAL 的 RLS 上下文（commit 即失效）
        if getattr(self, "_commit_each_stage", False):
            from creditlens.infrastructure.postgres.session import checkpoint_commit

            await checkpoint_commit(session)

    async def _persist_artifacts(
        self, session: AsyncSession, run: ReviewRun, artifacts: list[AgentArtifact]
    ) -> None:
        """Artifact/Claim Append-only 持久化。"""
        for artifact in artifacts:
            record = ArtifactRecord(
                id=artifact.artifact_id,
                tenant_id=run.tenant_id,
                run_id=run.id,
                task_id=artifact.task_id,
                artifact_type=artifact.producer,
                producer=artifact.producer,
                lifecycle_status="VALIDATED",
                execution_status=artifact.execution_status,
                payload=artifact.model_dump(mode="json"),
            )
            session.add(record)
            for claim in artifact.claims:
                session.add(
                    ClaimRecord(
                        id=claim.claim_id,
                        tenant_id=run.tenant_id,
                        run_id=run.id,
                        artifact_id=artifact.artifact_id,
                        category=claim.category,
                        statement=claim.statement,
                        verdict=claim.verdict,
                        severity=claim.severity,
                        as_of_date=claim.as_of_date,
                        uncertainty_reason=claim.uncertainty_reason,
                        payload={
                            "supporting_evidence_ids": [
                                str(e) for e in claim.supporting_evidence_ids
                            ],
                            "opposing_evidence_ids": [str(e) for e in claim.opposing_evidence_ids],
                            "calculation_ids": [str(c) for c in claim.calculation_ids],
                        },
                    )
                )
        await session.flush()

    @staticmethod
    async def _apply_audit_to_claims(
        session: AsyncSession, run: ReviewRun, audit: AuditResult
    ) -> None:
        async def mark(claim_ids: list[uuid.UUID], status: str) -> None:
            for claim_id in claim_ids:
                claim = await session.get(ClaimRecord, claim_id)
                if claim is not None:
                    claim.review_status = status

        await mark(audit.accepted_claim_ids, "AUDITED")
        await mark(audit.rejected_claim_ids, "NEEDS_REWORK")
        await mark(audit.needs_human_review_claim_ids, "PENDING")
        await session.flush()


class _EventWriter:
    """run_events 追加写（任务 27 Trace 的事实源）。"""

    def __init__(self, session: AsyncSession, run: ReviewRun, resume: bool = False):
        self._session = session
        self._run = run
        self._next_seq = 1
        self._resume = resume
        self._initialized = not resume

    async def emit(self, event_type: str, payload: dict) -> None:
        if not self._initialized:
            from sqlalchemy import func

            last = await self._session.scalar(
                select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == self._run.id)
            )
            self._next_seq = (last or 0) + 1
            self._initialized = True
        self._session.add(
            RunEvent(
                run_id=self._run.id,
                tenant_id=self._run.tenant_id,
                case_id=self._run.case_id,
                sequence_no=self._next_seq,
                event_type=event_type,
                payload_redacted=payload,
                occurred_at=utc_now(),
            )
        )
        self._next_seq += 1


async def _transition_run(
    session: AsyncSession, run: ReviewRun, seq: "_EventWriter", new_status: str
) -> None:
    """模块级状态迁移（与 Supervisor._transition 同一白名单）。"""
    if new_status not in _TRANSITIONS.get(run.status, set()):
        raise InvalidStateTransitionError(f"{run.status} -> {new_status} 不被状态机允许")
    old = run.status
    run.status = new_status
    run.state_version += 1
    await seq.emit("STATE_CHANGED", {"from": old, "to": new_status})
    await session.flush()


async def _persist_report_version(session: AsyncSession, run: ReviewRun) -> None:
    """P0-3：报告版本持久化（文档 §6.4 report_versions）。
    只渲染已通过审计/人工批准的 Claim；APPROVED_DRAFT 不代表真实授信批准。"""
    import json

    from creditlens.common.hashing import sha256_text
    from creditlens.infrastructure.postgres.models import ReportVersion

    claims = (
        await session.scalars(select(ClaimRecord).where(ClaimRecord.run_id == run.id))
    ).all()
    accepted = [
        {
            "claim_id": str(c.id),
            "category": c.category,
            "statement": c.statement,
            "verdict": c.verdict,
            "review_status": c.review_status,
            "evidence": c.payload,
        }
        for c in claims
        if c.review_status in {"AUDITED", "HUMAN_APPROVED"}
    ]
    content = {
        "run_id": str(run.id),
        "as_of_date": run.as_of_date.isoformat(),
        "claims": accepted,
        "excluded_claims": len(claims) - len(accepted),
        "disclaimer": "本报告为授信预审辅助草稿，不构成授信批准或拒绝决定。",
    }
    from sqlalchemy import func

    last_no = await session.scalar(
        select(func.max(ReportVersion.version_no)).where(ReportVersion.run_id == run.id)
    )
    session.add(
        ReportVersion(
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
            version_no=(last_no or 0) + 1,
            status="APPROVED_DRAFT",
            content_json=content,
            content_hash=sha256_text(json.dumps(content, ensure_ascii=False, sort_keys=True)),
        )
    )
    await session.flush()


async def _blocking_claims_remain(session: AsyncSession, run_id: uuid.UUID) -> int:
    """blocking = 待人工（PENDING）或被审计打回（NEEDS_REWORK）的 Claim 数。"""
    from sqlalchemy import func

    return (
        await session.scalar(
            select(func.count())
            .select_from(ClaimRecord)
            .where(
                ClaimRecord.run_id == run_id,
                ClaimRecord.review_status.in_(["PENDING", "NEEDS_REWORK"]),
            )
        )
    ) or 0


async def resume_after_human_review(
    session: AsyncSession,
    run_id: uuid.UUID,
    decision: HumanDecision,
) -> str:
    """HITL（任务 26）：人工决定追加写，不覆盖 Agent Claim。

    P0-3 收口：
    - 幂等键：同一 run + idempotency_key 重复提交直接返回当前状态；
    - APPROVE/REJECT 只作用于目标 Claim；仅当所有 blocking Claim
      （PENDING/NEEDS_REWORK）全部解决后，Run 才 REPORTING→COMPLETED，
      且报告版本持久化是 COMPLETED 前置条件；
    - REQUEST_CHANGES/REQUEST_MORE_INFORMATION 进入 REWORK，可再回 HUMAN_REVIEW。
    安全拒绝（ACL/越权）不走此路径（文档 §11.5）。
    """
    run = await session.get(ReviewRun, run_id)
    if run is None:
        raise InvalidStateTransitionError("Run 不存在")

    # 幂等键先于状态检查：已完成 Run 的重复提交应幂等返回，而非报错
    if decision.idempotency_key:
        duplicate = await session.scalar(
            select(HumanDecision.id).where(
                HumanDecision.run_id == run_id,
                HumanDecision.idempotency_key == decision.idempotency_key,
            )
        )
        if duplicate is not None:
            return run.status  # 幂等：不重复应用

    if run.status not in {"HUMAN_REVIEW", "REWORK"}:
        raise InvalidStateTransitionError("Run 不在可人工处理状态")

    session.add(decision)
    seq = _EventWriter(session, run, resume=True)
    await seq.emit("HUMAN_DECISION", {"action": decision.action})

    if decision.action in {"APPROVE_CLAIM", "REJECT_CLAIM"}:
        new_status = "HUMAN_APPROVED" if decision.action == "APPROVE_CLAIM" else "HUMAN_REJECTED"
        for claim_id in decision.target_claim_ids:
            claim = await session.get(ClaimRecord, uuid.UUID(str(claim_id)))
            if claim is not None and claim.run_id == run.id:
                claim.review_status = new_status
        await session.flush()
        remaining = await _blocking_claims_remain(session, run_id)
        if remaining > 0:
            await seq.emit("HUMAN_REVIEW_PENDING", {"blocking_claims": remaining})
            await session.flush()
            return run.status  # 仍有 blocking Claim，保持人工处理状态
        if run.status == "REWORK":
            await _transition_run(session, run, seq, "HUMAN_REVIEW")
        await _transition_run(session, run, seq, "REPORTING")
        await _persist_report_version(session, run)
        await _transition_run(session, run, seq, "COMPLETED")
        run.completed_at = utc_now()
    elif decision.action in {"SUBMIT_REPORT", "APPROVE_REPORT_DRAFT"}:
        remaining = await _blocking_claims_remain(session, run_id)
        if remaining > 0:
            raise InvalidStateTransitionError(
                f"仍有 {remaining} 条 blocking Claim，未解决前不得提交报告"
            )
        if run.status == "REWORK":
            await _transition_run(session, run, seq, "HUMAN_REVIEW")
        await _transition_run(session, run, seq, "REPORTING")
        await _persist_report_version(session, run)
        await _transition_run(session, run, seq, "COMPLETED")
        run.completed_at = utc_now()
    else:
        # REQUEST_CHANGES / REQUEST_MORE_INFORMATION / RERUN_TASK / OVERRIDE_WITH_REASON
        if run.status != "REWORK":
            await _transition_run(session, run, seq, "REWORK")
    await session.flush()
    return run.status
