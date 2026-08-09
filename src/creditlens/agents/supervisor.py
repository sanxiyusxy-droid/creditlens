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
from creditlens.agents.report_agent import ReportAgent
from creditlens.agents.risk_agent import RiskAgent
from creditlens.common.clock import utc_now
from creditlens.common.errors import (
    ConcurrentReviewConflictError,
    IdempotencyConflictError,
    InvalidReviewRequestError,
    InvalidStateTransitionError,
)
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    CaseDocument,
    ClaimRecord,
    CreditCase,
    EvidenceRecord,
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
    # v1.1：Report Agent 生成失败时 Run 不得 COMPLETED（WP1 失败门禁）
    "REPORTING": {"COMPLETED", "FAILED"},
}

# v1.1 失败门禁：关键 Agent 失败禁止生成报告；Risk 失败降级继续（WP1）
_CRITICAL_AGENTS = {"policy", "financial"}
_DEGRADABLE_AGENTS = {"risk"}

_ALLOWED_HUMAN_ACTIONS = {
    "APPROVE_CLAIM",
    "REJECT_CLAIM",
    "REQUEST_CHANGES",
    "REQUEST_MORE_INFORMATION",
    "SUBMIT_REPORT",
    "APPROVE_REPORT_DRAFT",
}


def _same_human_decision(existing: HumanDecision, incoming: HumanDecision) -> bool:
    """判断同一幂等键是否真的是同一个请求，而非 key 被误复用。"""

    def claim_ids(value: list | None) -> tuple[str, ...]:
        return tuple(sorted(str(item) for item in (value or [])))

    return (
        existing.tenant_id == incoming.tenant_id
        and existing.case_id == incoming.case_id
        and existing.run_id == incoming.run_id
        and existing.action == incoming.action
        and claim_ids(existing.target_claim_ids) == claim_ids(incoming.target_claim_ids)
        and existing.target_version == incoming.target_version
        and (existing.reason_code or "") == (incoming.reason_code or "")
        and (existing.reason or "") == (incoming.reason or "")
        and existing.reviewer_id == incoming.reviewer_id
    )


def _ensure_same_idempotent_request(existing: HumanDecision, incoming: HumanDecision) -> None:
    if not _same_human_decision(existing, incoming):
        raise IdempotencyConflictError(
            "同一 idempotency_key 已用于不同的人工复核请求",
            {"idempotency_key": incoming.idempotency_key, "run_id": str(incoming.run_id)},
        )


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
        risk_agent: RiskAgent | None = None,
        report_agent: ReportAgent | None = None,
    ):
        self._policy = policy_agent
        self._financial = financial_agent
        self._challenger = challenger
        self._auditor = auditor
        self._risk = risk_agent
        self._report = report_agent or ReportAgent()

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
        # v1.1：Policy || Financial || Risk 三路（仍串行）
        professional: list[AgentArtifact] = []
        agent_tasks = [
            ("policy", self._policy, "policy_review", "policy_analyst"),
            ("financial", self._financial, "financial_analysis", "financial_analyst"),
        ]
        if self._risk is not None:
            agent_tasks.append(("risk", self._risk, "risk_analysis", "risk_analyst"))
        degraded_agents: list[str] = []
        for name, agent, task_key, producer in agent_tasks:
            try:
                result = await agent.run(run.id, task_key, trusted)
                professional.append(result)
                # 每个 Agent 一完成就落库。这样后续核心 Agent 异常时，已经成功的
                # 阶段不会出现“有 TASK_COMPLETED 事件但没有 Artifact”的断链。
                await self._persist_artifacts(session, run, [result])
                if result.execution_status == "FAILED":
                    await seq.emit(
                        "TASK_FAILED",
                        {"task": name, "reason": "FAILED_ARTIFACT"},
                    )
                    if name in _CRITICAL_AGENTS:
                        await self._transition(session, run, seq, "FAILED", force=True)
                        return RunOutcome(run.id, run.status, professional)
                    degraded_agents.append(name)
                    run.model_manifest = {
                        **(run.model_manifest or {}),
                        "degraded_agents": degraded_agents,
                        "degraded": True,
                    }
                    await session.flush()
                    continue
                await seq.emit("TASK_COMPLETED", {"task": name, "claims": len(result.claims)})
                # P1：Agent 产出但工具部分失败（execution_status=DEGRADED）时，
                # Run Manifest 必须同样标记降级——报告读者需要知道结论是降级产出
                if result.execution_status == "DEGRADED":
                    degraded_agents.append(name)
                    run.model_manifest = {
                        **(run.model_manifest or {}),
                        "degraded_agents": degraded_agents,
                        "degraded": True,
                    }
                    await seq.emit(
                        "TASK_DEGRADED",
                        {
                            "task": name,
                            "reason": "TOOL_PARTIAL_FAILURE",
                            "issues": [
                                issue for issue in result.unresolved_issues if issue.get("degraded")
                            ],
                        },
                    )
                    await session.flush()
                if name == "risk":
                    # WP3：Risk 阈值配置版本写入 Manifest，评测/审计口径可追溯
                    threshold_version = getattr(agent, "threshold_version", None)
                    if threshold_version:
                        run.model_manifest = {
                            **(run.model_manifest or {}),
                            "risk_threshold_version": threshold_version,
                        }
                        await session.flush()
            except Exception as exc:
                # WP1：Agent 异常必须生成 FAILED Artifact 并持久化，不能只写事件后继续
                failed = AgentArtifact(
                    run_id=run.id,
                    task_id=task_key,
                    producer=producer,
                    execution_status="FAILED",
                    unresolved_issues=[{"error": type(exc).__name__, "stage": name}],
                )
                await self._persist_artifacts(session, run, [failed])
                await seq.emit("TASK_FAILED", {"task": name, "error": type(exc).__name__})
                if name in _CRITICAL_AGENTS:
                    # Policy/Financial 失败：禁止生成报告，Run 直接 FAILED
                    await self._transition(session, run, seq, "FAILED", force=True)
                    return RunOutcome(run.id, run.status, [*professional, failed])
                if name in _DEGRADABLE_AGENTS:
                    # Risk 失败：允许继续，但必须标记 DEGRADED 写入 Manifest
                    degraded_agents.append(name)
                    run.model_manifest = {
                        **(run.model_manifest or {}),
                        "degraded_agents": degraded_agents,
                        "degraded": True,
                    }
                    await session.flush()
        if not professional:
            await self._transition(session, run, seq, "FAILED", force=True)
            return RunOutcome(run.id, run.status)

        await self._transition(session, run, seq, "SYNTHESIZING")

        await self._transition(session, run, seq, "CHALLENGING")
        # P1：Challenger 异常同样必须落 FAILED Artifact + RunEvent，不允许整个 Run
        # 以未捕获异常方式中断（否则 Trace 里看不到发生了什么）。反证缺失不阻断
        # 报告，但 Run 必须标记 DEGRADED——审查员需知道本次未经过反证检验。
        try:
            challenge = await self._challenger.run(run.id, "challenger", trusted, professional)
        except Exception as exc:
            challenge = AgentArtifact(
                run_id=run.id,
                task_id="challenger",
                producer="challenger",
                execution_status="FAILED",
                unresolved_issues=[{"error": type(exc).__name__, "stage": "challenger"}],
            )
            degraded_agents.append("challenger")
            run.model_manifest = {
                **(run.model_manifest or {}),
                "degraded_agents": degraded_agents,
                "degraded": True,
            }
            await seq.emit("TASK_FAILED", {"task": "challenger", "error": type(exc).__name__})
        await self._persist_artifacts(session, run, [challenge])
        all_artifacts = [*professional, challenge]

        await self._transition(session, run, seq, "AUDITING")
        # WP3：Auditor 同步复核 Case/Snapshot/cutoff
        # P1：Auditor 异常属安全关键——审计不可用时不得放行报告，Run 转 FAILED
        try:
            audit = await self._auditor.verify(session, trusted, all_artifacts, snapshot=snapshot)
        except Exception as exc:
            audit_failed = AgentArtifact(
                run_id=run.id,
                task_id="auditor",
                producer="auditor",
                execution_status="FAILED",
                unresolved_issues=[{"error": type(exc).__name__, "stage": "auditor"}],
            )
            await self._persist_artifacts(session, run, [audit_failed])
            await seq.emit("AUDIT_FAILED", {"error": type(exc).__name__})
            await self._transition(session, run, seq, "FAILED", force=True)
            return RunOutcome(run.id, run.status, [*all_artifacts, audit_failed])
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
        # v1.1：Report Agent 生成结构化报告（替代内部 JSON 渲染）；
        # WP1：Report 失败时 Run 不得 COMPLETED，转 FAILED
        try:
            report_content = await self._report.generate(session, run)
            await self._report.persist(session, run, report_content)
        except Exception as exc:
            await seq.emit("REPORT_FAILED", {"error": type(exc).__name__})
            await self._transition(session, run, seq, "FAILED", force=True)
            return RunOutcome(run.id, run.status, all_artifacts, audit)
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
        """Artifact/Claim/Evidence Append-only 持久化。"""
        from creditlens.infrastructure.postgres.artifact_integrity import (
            canonical_artifact_payload_hash,
        )

        for artifact in artifacts:
            persisted_payload = artifact.model_dump(mode="json", exclude={"output_hash"})
            persisted_payload["lifecycle_status"] = "VALIDATED"
            record = ArtifactRecord(
                id=artifact.artifact_id,
                tenant_id=run.tenant_id,
                run_id=run.id,
                task_id=artifact.task_id,
                artifact_type=artifact.producer,
                producer=artifact.producer,
                lifecycle_status="VALIDATED",
                execution_status=artifact.execution_status,
                payload=persisted_payload,
                input_hash=artifact.input_hash,
                output_hash=canonical_artifact_payload_hash(persisted_payload),
            )
            session.add(record)
            # P1：EvidenceRecord 独立落库——报告/审计可直接从 evidence 表
            # 追溯到冻结 Snapshot 中的原始段落，不必解析 Artifact payload
            await self._persist_evidence(session, run, artifact)
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
                            # WP3：source_claim_id 持久化，API/报告可追踪反证来源
                            "source_claim_id": str(claim.source_claim_id)
                            if claim.source_claim_id
                            else None,
                        },
                    )
                )
        await session.flush()

    @staticmethod
    async def _persist_evidence(
        session: AsyncSession, run: ReviewRun, artifact: AgentArtifact
    ) -> None:
        """P1：把 Agent 引用的证据落成独立 EvidenceRecord（同 Run 内幂等去重）。

        evidence_id 由 (section_id, text_hash) 派生，天然稳定；同一 Run 内多个
        Agent 引用同一段落只保留一条记录。locator 含 parse_run_id，使报告能证明
        引用的是 Snapshot 冻结的那个解析批次。
        """
        for ref in artifact.evidence:
            existing = await session.scalar(
                select(EvidenceRecord.id).where(
                    EvidenceRecord.run_id == run.id,
                    EvidenceRecord.evidence_key == ref.evidence_id,
                )
            )
            if existing is not None:
                continue
            session.add(
                EvidenceRecord(
                    evidence_key=ref.evidence_id,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type=ref.evidence_type,
                    source_id=ref.source_id,
                    document_version_id=ref.document_version_id,
                    section_id=ref.section_id,
                    page_number=ref.page_number,
                    locator={
                        "document_version_id": str(ref.document_version_id)
                        if ref.document_version_id
                        else None,
                        "section_id": str(ref.section_id) if ref.section_id else None,
                        # 回原文必需：定位到具体解析批次
                        "parse_run_id": str(ref.parse_run_id) if ref.parse_run_id else None,
                        "page_number": ref.page_number,
                        "fact_id": str(ref.fact_id) if ref.fact_id else None,
                        "calculation_id": str(ref.calculation_id) if ref.calculation_id else None,
                    },
                    content_hash=ref.content_hash,
                    snapshot={
                        "snapshot_id": str(run.input_snapshot_id)
                        if run.input_snapshot_id
                        else None,
                        "as_of_date": run.as_of_date.isoformat() if run.as_of_date else None,
                    },
                    valid_from=ref.valid_from,
                    valid_to=ref.valid_to,
                    source_available_at=ref.source_available_at,
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


async def _persist_report_via_agent(
    session: AsyncSession, run: ReviewRun, status: str = "APPROVED_DRAFT"
) -> None:
    """v1.1：通过 Report Agent 生成并持久化报告版本。

    WP3：人工批准后的 resume 路径默认 APPROVED_DRAFT；
    自动 REPORTING 路径（Supervisor 内）使用默认 VERIFIED_DRAFT。"""
    agent = ReportAgent()
    content = await agent.generate(session, run)
    await agent.persist(session, run, content, status=status)


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

    P1 并发安全收口（顺序幂等 -> 真正并发安全）：
    - 以 `SELECT ... FOR UPDATE` 锁定 Run 行，使同一 Run 的并发决定串行化；
      两个不同 idempotency_key 的并发请求不再能同时通过版本校验；
    - idempotency_key 与 expected_state_version 强制提供（缺失 -> 422），
      否则乐观锁形同虚设；
    - 唯一约束冲突（并发同键）视为重复提交，返回既有结果而非 500；
    - 目标 Claim 不存在或不属于本 Run -> 422，不再静默忽略。
    安全拒绝（ACL/越权）不走此路径（文档 §11.5）。
    """
    from sqlalchemy.exc import IntegrityError

    if not decision.idempotency_key:
        raise InvalidReviewRequestError("必须提供 idempotency_key（并发/重试安全前提）")
    if decision.target_version is None:
        raise InvalidReviewRequestError("必须提供 expected_state_version（乐观锁前提）")
    if decision.action not in _ALLOWED_HUMAN_ACTIONS:
        raise InvalidReviewRequestError("不支持的人工复核动作", {"action": decision.action})

    # 行锁：并发请求在此串行化（SQLite 无 FOR UPDATE，由 SQLAlchemy 自动忽略，
    # 单测仍走同一代码路径；真实并发保护在 PostgreSQL 生效）
    run = await session.scalar(select(ReviewRun).where(ReviewRun.id == run_id).with_for_update())
    if run is None:
        raise InvalidStateTransitionError("Run 不存在")

    # 幂等键先于状态检查：已完成 Run 的重复提交应幂等返回，而非报错
    duplicate = await session.scalar(
        select(HumanDecision).where(
            HumanDecision.run_id == run_id,
            HumanDecision.idempotency_key == decision.idempotency_key,
        )
    )
    if duplicate is not None:
        _ensure_same_idempotent_request(duplicate, decision)
        return run.status  # 幂等：不重复应用

    if run.status not in {"HUMAN_REVIEW", "REWORK"}:
        raise InvalidStateTransitionError("Run 不在可人工处理状态")

    # WP3 乐观锁：expected_state_version 不匹配 -> 并发审批冲突（API 映射 409）
    if decision.target_version != run.state_version:
        raise ConcurrentReviewConflictError(
            "Run 状态版本已变更，请刷新后重试",
            {"expected": decision.target_version, "actual": run.state_version},
        )

    # 目标 Claim 校验：不存在或跨 Run 一律拒绝请求，不静默跳过
    if decision.action in {"APPROVE_CLAIM", "REJECT_CLAIM"}:
        if not decision.target_claim_ids:
            raise InvalidReviewRequestError(f"{decision.action} 必须指定 target_claim_ids")
        for raw_id in decision.target_claim_ids:
            try:
                claim_uuid = uuid.UUID(str(raw_id))
            except (ValueError, AttributeError, TypeError) as exc:
                raise InvalidReviewRequestError(f"非法 Claim ID: {raw_id}") from exc
            claim = await session.get(ClaimRecord, claim_uuid)
            if claim is None or claim.run_id != run.id:
                raise InvalidReviewRequestError(
                    "目标 Claim 不存在或不属于该 Run",
                    {"claim_id": str(raw_id), "run_id": str(run.id)},
                )
            if claim.review_status not in {"PENDING", "NEEDS_REWORK"}:
                raise InvalidReviewRequestError(
                    "目标 Claim 已完成裁决；如需改判必须走显式 supersede 流程",
                    {
                        "claim_id": str(raw_id),
                        "review_status": claim.review_status,
                    },
                )
    elif decision.action in {"SUBMIT_REPORT", "APPROVE_REPORT_DRAFT"}:
        # 在写 HumanDecision / bump version 前完成前置校验。否则直接调用服务层且
        # 捕获异常的调用方若未回滚，可能把本应失败的决定误提交。
        remaining = await _blocking_claims_remain(session, run_id)
        if remaining > 0:
            raise InvalidStateTransitionError(
                f"仍有 {remaining} 条 blocking Claim，未解决前不得提交报告"
            )

    session.add(decision)
    try:
        # 提前 flush：让唯一约束 (run_id, idempotency_key) 在此处判定，
        # 并发同键的落败方按"重复提交"处理而不是 500
        await session.flush()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(
            select(HumanDecision).where(
                HumanDecision.run_id == run_id,
                HumanDecision.idempotency_key == decision.idempotency_key,
            )
        )
        if existing is None:
            # 不是 run + idempotency_key 唯一约束导致的冲突，不能伪装成成功。
            raise
        _ensure_same_idempotent_request(existing, decision)
        refreshed = await session.get(ReviewRun, run_id)
        return refreshed.status if refreshed else "UNKNOWN"

    # 决定一旦生效立即推进 state_version：否则"部分批准"这类不改变 Run 状态的
    # 决定会让版本号停滞，第二个不同幂等键的请求可以用同一版本再次通过乐观锁
    run.state_version += 1
    await session.flush()

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
        await _persist_report_via_agent(session, run)
        await _transition_run(session, run, seq, "COMPLETED")
        run.completed_at = utc_now()
    elif decision.action in {"SUBMIT_REPORT", "APPROVE_REPORT_DRAFT"}:
        if run.status == "REWORK":
            await _transition_run(session, run, seq, "HUMAN_REVIEW")
        await _transition_run(session, run, seq, "REPORTING")
        await _persist_report_via_agent(session, run)
        await _transition_run(session, run, seq, "COMPLETED")
        run.completed_at = utc_now()
    else:
        # REQUEST_CHANGES / REQUEST_MORE_INFORMATION
        if run.status != "REWORK":
            await _transition_run(session, run, seq, "REWORK")
    await session.flush()
    return run.status
