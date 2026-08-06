"""P0 修复防回归测试（v0.9 审核意见）。

- P0-1：SnapshotFact 冻结——Snapshot 后新增/未来 Fact 不影响历史计算；
- P0-2：同租户跨案件访问证据被拒绝（不只是跨租户）；
- P0-3：Auditor rejected 阻断完成；部分批准不 COMPLETED；报告版本持久化；幂等键。
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from creditlens.application.snapshot_service import freeze_snapshot
from creditlens.application.trusted_context import build_trusted_context
from creditlens.common.errors import AclDeniedError, InvalidStateTransitionError
from creditlens.evidence.preview import EvidencePreviewService, candidate_to_evidence_ref
from creditlens.formulas.engine import FormulaRegistry
from creditlens.infrastructure.postgres.models import (
    AppUser,
    CaseMembership,
    ClaimRecord,
    CreditCase,
    FinancialFact,
    HumanDecision,
    ReportVersion,
    ReviewRun,
)
from creditlens.retrieval.dense import DenseRetriever
from creditlens.tools.finance_tools import compute_metric_for_entity
from tests.e2e.test_ingest_retrieve_e2e import TENANT_ID, seeded  # noqa: F401  复用夹具
from tests.e2e.test_multi_agent_e2e import PERIOD_END, with_facts  # noqa: F401  复用夹具

COLLECTION = "credit_chunks_v1"
CUTOFF = datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC)


class TestP01SnapshotFact:
    async def test_snapshot_freezes_fact_world(self, session, with_facts):  # noqa: F811
        """Snapshot 后修改底层 Fact（新增替代值），历史计算结果不变。"""
        case = with_facts["case"]
        trusted = await build_trusted_context(session, TENANT_ID, case.id)
        snapshot = await freeze_snapshot(session, trusted, chunks_collection=COLLECTION)
        assert snapshot.allowed_fact_ids, "冻结集合应包含已 seed 的财务事实"
        registry = FormulaRegistry()

        before = await compute_metric_for_entity(
            session,
            registry,
            TENANT_ID,
            case.borrower_entity_id,
            "debt_ratio",
            "1.0",
            PERIOD_END,
            case_id=case.id,
            decision_cutoff_at=CUTOFF,
            allowed_fact_ids=snapshot.allowed_fact_ids,
        )
        assert before.status == "CALCULATED"

        # 底层世界变化：新增一条"更大负债"的事实（模拟重述/补录）
        session.add(
            FinancialFact(
                tenant_id=TENANT_ID,
                case_id=case.id,
                entity_id=case.borrower_entity_id,
                metric_code="total_liabilities",
                period_end=PERIOD_END,
                period_type="INSTANT",
                value=Decimal("99000000"),
                canonical_value=Decimal("99000000"),
                consolidation_scope="CONSOLIDATED",
                extraction_method="MANUAL",
            )
        )
        await session.flush()

        after = await compute_metric_for_entity(
            session,
            registry,
            TENANT_ID,
            case.borrower_entity_id,
            "debt_ratio",
            "1.0",
            PERIOD_END,
            case_id=case.id,
            decision_cutoff_at=CUTOFF,
            allowed_fact_ids=snapshot.allowed_fact_ids,
        )
        assert after.result == before.result, "历史 Snapshot 的计算不得随底层 Fact 变化"
        assert after.trace_hash == before.trace_hash

    async def test_future_fact_excluded_by_cutoff(self, session, with_facts):  # noqa: F811
        """审查截止后才可获得的 Fact 不进入新 Snapshot（历史时点不泄漏）。"""
        case = with_facts["case"]
        session.add(
            FinancialFact(
                tenant_id=TENANT_ID,
                case_id=case.id,
                entity_id=case.borrower_entity_id,
                metric_code="total_liabilities",
                period_end=PERIOD_END,
                period_type="INSTANT",
                value=Decimal("1"),
                canonical_value=Decimal("1"),
                consolidation_scope="CONSOLIDATED",
                extraction_method="MANUAL",
                source_available_at=CUTOFF + timedelta(days=30),
            )
        )
        await session.flush()
        trusted = await build_trusted_context(session, TENANT_ID, case.id)
        snapshot = await freeze_snapshot(session, trusted, chunks_collection=COLLECTION)
        future_facts = (
            await session.scalars(
                select(FinancialFact.id).where(FinancialFact.value == Decimal("1"))
            )
        ).all()
        assert all(f not in snapshot.allowed_fact_ids for f in future_facts)


class TestP02CrossCaseIsolation:
    async def test_same_tenant_other_case_preview_denied(
        self,
        session,
        qdrant,
        object_store,
        seeded,  # noqa: F811
    ):
        """同租户、另一个案件的授权用户不能预览本案件证据。"""
        # 案件 B（同租户）+ 用户 B
        other_case = CreditCase(
            tenant_id=TENANT_ID,
            case_number="other-001",
            borrower_entity_id=(await session.scalars(select(CreditCase)))
            .first()
            .borrower_entity_id,
            product_code="working_capital",
            requested_amount=Decimal("1000000"),
            application_date=date(2026, 6, 30),
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=CUTOFF,
        )
        session.add(other_case)
        await session.flush()
        user_b = AppUser(tenant_id=TENANT_ID, external_subject="analyst-b", display_name="B")
        session.add(user_b)
        await session.flush()
        session.add(CaseMembership(case_id=other_case.id, user_id=user_b.id, case_role="ANALYST"))
        await session.flush()

        # 用案件 A 的授权拿到一个证据 ref
        trusted_a = await build_trusted_context(session, TENANT_ID, seeded["case_id"])
        retriever = DenseRetriever(qdrant, seeded["embedder"])
        result = await retriever.retrieve(session, trusted_a, "资产负债率", COLLECTION, top_k=3)
        ref = candidate_to_evidence_ref(result.candidates[0])

        # 用户 B 以案件 B 的上下文访问该证据 -> 拒绝（文档不绑定案件 B）
        trusted_b = await build_trusted_context(
            session, TENANT_ID, other_case.id, user_id=user_b.id
        )
        preview = EvidencePreviewService(object_store)
        with pytest.raises(AclDeniedError):
            await preview.render_page(session, trusted_b, ref)


class TestP03StateMachine:
    async def _setup_run(self, session, case, review_statuses):
        run = ReviewRun(
            tenant_id=TENANT_ID,
            case_id=case.id,
            run_type="FULL_REVIEW",
            status="HUMAN_REVIEW",
            as_of_date=date(2026, 6, 30),
            decision_cutoff_at=CUTOFF,
        )
        session.add(run)
        await session.flush()
        from creditlens.infrastructure.postgres.models import ArtifactRecord

        artifact = ArtifactRecord(
            tenant_id=TENANT_ID, run_id=run.id, task_id="t", artifact_type="test", producer="test"
        )
        session.add(artifact)
        await session.flush()
        claim_ids = []
        for status in review_statuses:
            claim = ClaimRecord(
                tenant_id=TENANT_ID,
                run_id=run.id,
                artifact_id=artifact.id,
                category="ELIGIBILITY",
                statement="测试",
                verdict="SUPPORTED",
                as_of_date=date(2026, 6, 30),
                review_status=status,
            )
            session.add(claim)
            await session.flush()
            claim_ids.append(claim.id)
        return run, claim_ids

    async def test_partial_approval_does_not_complete(self, session, seeded):  # noqa: F811
        from creditlens.agents.supervisor import resume_after_human_review

        case = (await session.scalars(select(CreditCase))).first()
        run, claim_ids = await self._setup_run(session, case, ["PENDING", "PENDING"])
        decision = HumanDecision(
            tenant_id=TENANT_ID,
            case_id=case.id,
            run_id=run.id,
            target_claim_ids=[str(claim_ids[0])],
            action="APPROVE_CLAIM",
            # P1：幂等键 + 乐观锁为必填契约
            idempotency_key="partial-approve",
            target_version=run.state_version,
        )
        status = await resume_after_human_review(session, run.id, decision)
        assert status == "HUMAN_REVIEW", "仍有 blocking Claim 时不得完成"

    async def test_rejected_by_auditor_blocks_until_resolved(self, session, seeded):  # noqa: F811
        """NEEDS_REWORK（审计打回）的 Claim 未处理前，SUBMIT_REPORT 被拒。"""
        from creditlens.agents.supervisor import resume_after_human_review

        case = (await session.scalars(select(CreditCase))).first()
        run, _claim_ids = await self._setup_run(session, case, ["NEEDS_REWORK"])
        decision = HumanDecision(
            tenant_id=TENANT_ID,
            case_id=case.id,
            run_id=run.id,
            target_claim_ids=[],
            action="SUBMIT_REPORT",
            idempotency_key="submit-blocked",
            target_version=run.state_version,
        )
        with pytest.raises(InvalidStateTransitionError):
            await resume_after_human_review(session, run.id, decision)

    async def test_full_approval_completes_and_persists_report(self, session, seeded):  # noqa: F811
        from creditlens.agents.supervisor import resume_after_human_review

        case = (await session.scalars(select(CreditCase))).first()
        run, claim_ids = await self._setup_run(session, case, ["PENDING", "NEEDS_REWORK"])
        request_version = run.state_version
        decision = HumanDecision(
            tenant_id=TENANT_ID,
            case_id=case.id,
            run_id=run.id,
            target_claim_ids=[str(c) for c in claim_ids],
            action="APPROVE_CLAIM",
            idempotency_key="idem-1",
            target_version=request_version,
        )
        status = await resume_after_human_review(session, run.id, decision)
        assert status == "COMPLETED"
        report = await session.scalar(select(ReportVersion).where(ReportVersion.run_id == run.id))
        assert report is not None, "COMPLETED 前必须持久化报告版本"
        assert report.status == "APPROVED_DRAFT"
        assert len(report.content_json["claims"]) == 2

        # 幂等键：重复提交返回当前状态，不产生第二份决定/报告
        dup = HumanDecision(
            tenant_id=TENANT_ID,
            case_id=case.id,
            run_id=run.id,
            target_claim_ids=[str(c) for c in claim_ids],
            action="APPROVE_CLAIM",
            idempotency_key="idem-1",
            target_version=request_version,
        )
        assert await resume_after_human_review(session, run.id, dup) == "COMPLETED"
        reports = (
            await session.scalars(select(ReportVersion).where(ReportVersion.run_id == run.id))
        ).all()
        assert len(reports) == 1
