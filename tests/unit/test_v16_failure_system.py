"""SQLite system proofs for the v1.6 fail-closed demonstrations."""

import hashlib
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from creditlens.evaluation.failure_cases import (
    FailureCaseDataset,
    FailureProofScope,
    execute_failure_cases_system,
)
from creditlens.infrastructure.postgres.models import (
    AppUser,
    ArtifactRecord,
    CaseDocument,
    CaseMembership,
    ClaimRecord,
    CreditCase,
    Document,
    DocumentVersion,
    Entity,
    ReportVersion,
    ReviewRun,
    RunEvent,
    Tenant,
)
from creditlens.infrastructure.postgres.session import (
    create_session_factory,
    session_scope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "fail_closed_cases_v1.json"


async def _seed_reviewable_world(factory):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    case_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    as_of = date(2026, 6, 30)
    cutoff = datetime(2026, 6, 30, 16, tzinfo=UTC)

    async with session_scope(factory) as session:
        session.add_all(
            [
                Tenant(id=tenant_id, name="fail-closed-system-test"),
                AppUser(
                    id=user_id,
                    tenant_id=tenant_id,
                    external_subject=f"fail-closed-{user_id}",
                    display_name="安全回归用户",
                ),
                Entity(
                    id=entity_id,
                    tenant_id=tenant_id,
                    entity_type="COMPANY",
                    canonical_name="合成安全回归企业",
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
                    logical_key=f"failure-proof-{document_id}",
                    title="合成安全回归材料",
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
                version_label="2026",
                source_available_at=cutoff,
                object_uri="obj://fail-closed-system-test",
                source_filename="failure.pdf",
                mime_type="application/pdf",
                file_size=1,
                content_hash="f" * 64,
            )
        )
        await session.flush()
        session.add_all(
            [
                CaseDocument(
                    case_id=case_id,
                    document_version_id=version_id,
                    document_role="BORROWER_PROVIDED",
                ),
                CaseMembership(
                    case_id=case_id,
                    user_id=user_id,
                    case_role="REVIEWER",
                ),
            ]
        )

    return tenant_id, user_id, case_id


async def test_frozen_failures_execute_through_real_supervisor_and_db_gate(engine) -> None:
    factory = create_session_factory(engine)
    tenant_id, user_id, case_id = await _seed_reviewable_world(factory)
    dataset_payload = DATASET_PATH.read_bytes()
    dataset = FailureCaseDataset.model_validate_json(dataset_payload)

    report = await execute_failure_cases_system(
        dataset,
        dataset_sha256=hashlib.sha256(dataset_payload).hexdigest(),
        session_factory=factory,
        tenant_id=tenant_id,
        user_id=user_id,
        case_id=case_id,
    )

    assert report.all_passed is True
    assert report.proof_scope is FailureProofScope.SUPERVISOR_AUDITOR_DATABASE_GATE
    assert report.system_execution_performed is True
    assert report.http_endpoint_called is False
    assert len(report.results) == 2
    for result in report.results:
        assert result.passed is True
        assert result.missing_expected_violation_codes == []
        assert result.system_evidence is not None
        evidence = result.system_evidence
        assert evidence.http_endpoint_called is False
        assert evidence.report_gate_proof == "DATABASE_REPORT_VERSION_ROW_COUNT"
        assert evidence.persisted_run_status == "HUMAN_REVIEW"
        assert evidence.claim_review_statuses == ["NEEDS_REWORK"]
        assert evidence.injected_claim_persisted is True
        assert evidence.artifact_count == 3
        assert set(evidence.artifact_producers) == {
            "policy_analyst",
            "financial_analyst",
            "challenger",
        }
        assert evidence.artifact_hashes_verified is True
        assert evidence.artifact_claim_projection_verified is True
        assert evidence.audit_completed is True
        assert "AUDITING->HUMAN_REVIEW" in evidence.state_transitions
        assert not any(
            transition.endswith("->REPORTING") for transition in evidence.state_transitions
        )
        assert evidence.report_count == 0

    rendered = report.model_dump_json()
    for case in dataset.cases:
        assert case.synthetic_claim not in rendered

    async with session_scope(factory) as session:
        runs = list(
            (await session.scalars(select(ReviewRun).where(ReviewRun.case_id == case_id))).all()
        )
        assert len(runs) == 2
        assert {run.status for run in runs} == {"HUMAN_REVIEW"}
        run_ids = {run.id for run in runs}
        assert (
            await session.scalar(
                select(func.count(ClaimRecord.id)).where(ClaimRecord.run_id.in_(run_ids))
            )
            == 2
        )
        assert (
            await session.scalar(
                select(func.count(ArtifactRecord.id)).where(ArtifactRecord.run_id.in_(run_ids))
            )
            == 6
        )
        assert (
            await session.scalar(
                select(func.count(RunEvent.id)).where(RunEvent.run_id.in_(run_ids))
            )
            > 0
        )
        assert (
            await session.scalar(
                select(func.count(ReportVersion.id)).where(ReportVersion.run_id.in_(run_ids))
            )
            == 0
        )
