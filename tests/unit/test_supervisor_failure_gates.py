"""Supervisor 关键 Agent 失败门禁与 Artifact/Trace 连续性。"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import select

from creditlens.agents.contracts import AgentArtifact
from creditlens.agents.supervisor import Supervisor
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    CaseDocument,
    CreditCase,
    Document,
    DocumentVersion,
    Entity,
    ReportVersion,
    RunEvent,
    Tenant,
)
from creditlens.retrieval.contracts import TrustedRequestContext


class _ArtifactAgent:
    def __init__(self, producer: str, status: str = "SUCCESS"):
        self.producer = producer
        self.status = status

    async def run(self, run_id, task_id, trusted):
        return AgentArtifact(
            run_id=run_id,
            task_id=task_id,
            producer=self.producer,
            execution_status=self.status,
        )


class _RaisingAgent:
    async def run(self, run_id, task_id, trusted):
        raise RuntimeError("injected failure")


async def _reviewable_world(session):
    tenant_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    case_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    as_of = date(2026, 5, 1)
    cutoff = datetime(2026, 5, 1, tzinfo=UTC)
    session.add_all(
        [
            Tenant(id=tenant_id, name="failure-gate"),
            Entity(
                id=entity_id,
                tenant_id=tenant_id,
                entity_type="COMPANY",
                canonical_name="故障注入企业",
            ),
            CreditCase(
                id=case_id,
                tenant_id=tenant_id,
                case_number=f"FAIL-{case_id.hex[:8]}",
                borrower_entity_id=entity_id,
                product_code="working_capital",
                requested_amount=Decimal("1000000"),
                application_date=as_of,
                as_of_date=as_of,
                decision_cutoff_at=cutoff,
            ),
            Document(
                id=document_id,
                tenant_id=tenant_id,
                logical_key=f"failure-doc-{document_id}",
                title="故障注入材料",
                document_type="ANNUAL_REPORT",
            ),
        ]
    )
    await session.flush()
    session.add(
        DocumentVersion(
            id=version_id,
            tenant_id=tenant_id,
            document_id=document_id,
            version_label="2025",
            source_available_at=cutoff,
            object_uri="obj://failure-gate",
            source_filename="failure.pdf",
            mime_type="application/pdf",
            file_size=1,
            content_hash="f" * 64,
        )
    )
    await session.flush()
    session.add(
        CaseDocument(
            case_id=case_id,
            document_version_id=version_id,
            document_role="BORROWER_PROVIDED",
        )
    )
    await session.flush()
    trusted = TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=tenant_id,
        case_id=case_id,
        borrower_entity_id=entity_id,
        as_of_date=as_of,
        decision_cutoff_at=cutoff,
    )
    return trusted


async def test_returned_failed_policy_artifact_blocks_report(session):
    """核心 Agent 即使是返回 FAILED 而非抛异常，也必须 fail closed。"""
    trusted = await _reviewable_world(session)
    supervisor = Supervisor(
        policy_agent=_ArtifactAgent("policy_analyst", "FAILED"),
        financial_agent=_ArtifactAgent("financial_analyst"),
        challenger=object(),
        auditor=object(),
    )

    outcome = await supervisor.execute_full_review(session, trusted)

    assert outcome.status == "FAILED"
    artifacts = (
        await session.scalars(select(ArtifactRecord).where(ArtifactRecord.run_id == outcome.run_id))
    ).all()
    assert [(row.producer, row.execution_status) for row in artifacts] == [
        ("policy_analyst", "FAILED")
    ]
    assert not (
        await session.scalars(select(ReportVersion).where(ReportVersion.run_id == outcome.run_id))
    ).all()
    events = (
        await session.scalars(select(RunEvent).where(RunEvent.run_id == outcome.run_id))
    ).all()
    assert "TASK_FAILED" in {event.event_type for event in events}


async def test_prior_success_artifact_survives_later_core_exception(session):
    """Financial 异常时，已完成 Policy 的 Artifact 仍应与事件一起落库。"""
    trusted = await _reviewable_world(session)
    supervisor = Supervisor(
        policy_agent=_ArtifactAgent("policy_analyst"),
        financial_agent=_RaisingAgent(),
        challenger=object(),
        auditor=object(),
    )

    outcome = await supervisor.execute_full_review(session, trusted)

    assert outcome.status == "FAILED"
    artifacts = (
        await session.scalars(
            select(ArtifactRecord)
            .where(ArtifactRecord.run_id == outcome.run_id)
            .order_by(ArtifactRecord.created_at)
        )
    ).all()
    assert {(row.producer, row.execution_status) for row in artifacts} == {
        ("policy_analyst", "SUCCESS"),
        ("financial_analyst", "FAILED"),
    }
    assert {artifact.producer for artifact in outcome.artifacts} == {
        "policy_analyst",
        "financial_analyst",
    }
