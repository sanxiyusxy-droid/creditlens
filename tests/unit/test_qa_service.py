"""Grounded QA service transaction, state and persistence tests."""

import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from creditlens.agents.auditor import AuditResult
from creditlens.agents.contracts import (
    AgentClaim,
    AnswerStatus,
    DraftAnswerClaim,
    GroundedAnswerArtifact,
    GroundedAnswerDraft,
    RefusalReasonCode,
)
from creditlens.agents.grounded_qa import (
    DEFAULT_PROMPT_PATH,
    GroundedQAAgent,
    GroundedQAGeneration,
    GroundedQAOutputRejected,
    evidence_ref_from_packed,
    grounded_qa_repair_hint,
)
from creditlens.application import qa_service as qa_module
from creditlens.application.qa_service import QAService, QAServiceError
from creditlens.application.snapshot_service import SnapshotContext
from creditlens.common.errors import IdempotencyConflictError
from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.llm.chat import LLMCallError, ModelInvocationTrace
from creditlens.infrastructure.postgres.models import (
    AppUser,
    ArtifactRecord,
    CaseMembership,
    CaseSnapshot,
    ClaimRecord,
    CreditCase,
    Entity,
    EvidenceRecord,
    ReviewRun,
    RunEvent,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory
from creditlens.retrieval.context_packing import PackedContext, PackedSection
from creditlens.retrieval.contracts import RetrievedCandidate, TrustedRequestContext
from creditlens.retrieval.orchestrator import OrchestratedResult

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")
AS_OF = date(2026, 6, 30)
CUTOFF = datetime(2026, 6, 30, 12, tzinfo=UTC)


class _FakeOrchestrator:
    def __init__(self, packed: PackedContext):
        self.packed = packed
        self.calls: list[dict] = []

    async def retrieve(
        self,
        _session,
        _trusted,
        query,
        _collection,
        *,
        config,
        snapshot,
        summaries_collection,
    ) -> OrchestratedResult:
        self.calls.append(
            {
                "query": query,
                "config": config,
                "snapshot": snapshot,
                "summaries_collection": summaries_collection,
            }
        )
        candidates = [
            RetrievedCandidate(
                section_id=section.section_id,
                document_id=section.document_id,
                document_version_id=section.document_version_id,
                parse_run_id=section.parse_run_id,
                page_start=section.page_start,
                page_end=section.page_end,
                heading_path=section.heading_path,
                text=section.text,
                text_hash=section.text_hash,
                channel="FUSED",
                rank=section.rank,
                raw_score=1.0,
            )
            for section in self.packed.sections
        ]
        return OrchestratedResult(
            query=query,
            candidates=candidates,
            rejected=[],
            trace={"routes": [{"route": "fake", "candidates_count": len(candidates)}]},
            packing=self.packed.model_dump(mode="json") if self.packed.sections else None,
            channel_config={"routes": ["fake"], "packing_tokens": self.packed.total_tokens_est},
        )


class _AcceptingAuditor:
    async def verify_grounded_answer(self, _session, _trusted, artifact, **_kwargs):
        return AuditResult(
            accepted_claim_ids=[claim.claim_id for claim in artifact.claims],
            grounded_answer_ok=True,
            derived_answer_status=artifact.answer_status.value,
        )


class _AnsweredAgent:
    def __init__(self):
        self.calls = 0

    async def generate(self, question, run_id, as_of_date, packed, **_kwargs):
        self.calls += 1
        section = packed.sections[0]
        evidence = evidence_ref_from_packed(section, as_of_date)
        statement = "现有材料要求提交最近一期经审计财务报表。"
        claim = AgentClaim(
            category="MISSING_MATERIAL",
            statement=statement,
            verdict="SUPPORTED",
            supporting_evidence_ids=[evidence.evidence_id],
            as_of_date=as_of_date,
        )
        trace = ModelInvocationTrace(
            provider="fake",
            model="fake-grounded-model",
            prompt_version="grounded_qa_v1",
            prompt_sha256="a" * 64,
            request_sha256="b" * 64,
            response_sha256="c" * 64,
            input_tokens=20,
            output_tokens=8,
            total_tokens=28,
            latency_ms=2.5,
            attempts=1,
            status="SUCCESS",
        )
        artifact = GroundedAnswerArtifact(
            run_id=run_id,
            task_id="grounded_qa",
            producer="grounded_qa",
            input_hash="d" * 64,
            lifecycle_status="CREATED",
            claims=[claim],
            evidence=[evidence],
            answer_status=AnswerStatus.ANSWERED,
            direct_answer=statement,
            prompt_version="grounded_qa_v1",
            model_invocation_ids=[trace.invocation_id],
            output_hash="f" * 64,
        )
        return GroundedQAGeneration(
            artifact=artifact,
            model_traces=[trace],
            generation_mode="llm",
        )


class _AbstainingAgent:
    async def generate(self, question, run_id, as_of_date, packed, **_kwargs):
        del question, packed
        artifact = GroundedAnswerArtifact(
            run_id=run_id,
            task_id="grounded_qa",
            producer="grounded_qa",
            execution_status="INSUFFICIENT_EVIDENCE",
            claims=[],
            evidence=[],
            answer_status=AnswerStatus.ABSTAINED,
            direct_answer=None,
            missing_information=["缺少可支持回答的案件材料"],
            abstention_reason="当前冻结材料中没有可支持回答的证据。",
            refusal_reason_code=RefusalReasonCode.MISSING_FINANCIAL_DATA,
            prompt_version="grounded_qa_v1",
        )
        return GroundedQAGeneration(
            artifact=artifact,
            model_traces=[],
            generation_mode="abstained_empty_context",
        )


class _FailingAgent:
    async def generate(self, *_args, **_kwargs):
        trace = ModelInvocationTrace(
            provider="fake",
            model="fake-grounded-model",
            prompt_version="grounded_qa_v1",
            prompt_sha256="1" * 64,
            request_sha256="2" * 64,
            response_sha256="3" * 64,
            latency_ms=3.0,
            attempts=1,
            status="FAILED",
            error_type="ReadTimeout",
        )
        raise LLMCallError("provider response secret must not persist", trace)


class _SequencedChat:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


class _RepairFeedbackAgent:
    def __init__(self):
        self.delegate = _AnsweredAgent()
        self.feedback: list[list[object]] = []

    async def generate(self, question, run_id, as_of_date, packed, audit_feedback=None):
        self.feedback.append(list(audit_feedback or []))
        return await self.delegate.generate(question, run_id, as_of_date, packed)


class _RejectingAgent:
    def __init__(self, reject_times: int):
        self.reject_times = reject_times
        self.calls = 0
        self.feedback: list[list[str]] = []
        self.rejected_invocation_ids: list[uuid.UUID] = []
        self.delegate = _AnsweredAgent()

    async def generate(self, question, run_id, as_of_date, packed, audit_feedback=None):
        self.calls += 1
        self.feedback.append(list(audit_feedback or []))
        if self.calls <= self.reject_times:
            trace = ModelInvocationTrace(
                provider="fake",
                model="fake-grounded-model",
                prompt_version="grounded_qa_v1",
                prompt_sha256="4" * 64,
                request_sha256="5" * 64,
                response_sha256="6" * 64,
                latency_ms=1.0,
                attempts=1,
                status="SUCCESS",
            )
            self.rejected_invocation_ids.append(trace.invocation_id)
            raise GroundedQAOutputRejected("UNKNOWN_EVIDENCE_ID", trace)
        return await self.delegate.generate(question, run_id, as_of_date, packed)


class _RejectThenAcceptAuditor:
    def __init__(self):
        self.calls = 0
        self.rejected_claim_id: uuid.UUID | None = None

    async def verify_grounded_answer(self, _session, _trusted, artifact, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            claim_id = artifact.claims[0].claim_id
            self.rejected_claim_id = claim_id
            return AuditResult(
                rejected_claim_ids=[claim_id],
                grounded_answer_ok=False,
                derived_answer_status=artifact.answer_status.value,
                violations={
                    str(claim_id): [
                        f"claim:{claim_id}:SUPPORTED_WITHOUT_EVIDENCE",
                        "NUMERIC_TOKEN_NOT_IN_CITATION:70%",
                    ]
                },
            )
        return AuditResult(
            accepted_claim_ids=[claim.claim_id for claim in artifact.claims],
            grounded_answer_ok=True,
            derived_answer_status=artifact.answer_status.value,
        )


class _ReviewOnlyAuditor:
    async def verify_grounded_answer(self, _session, _trusted, artifact, **_kwargs):
        claim_id = artifact.claims[0].claim_id
        return AuditResult(
            needs_human_review_claim_ids=[claim_id],
            violations={str(claim_id): ["STRUCTURAL_COVERAGE_INSUFFICIENT"]},
            grounded_answer_ok=True,
            derived_answer_status="NEEDS_REVIEW",
            review_normalization_required=True,
        )


class _AlwaysRejectingAuditor:
    async def verify_grounded_answer(self, _session, _trusted, artifact, **_kwargs):
        claim_id = artifact.claims[0].claim_id
        return AuditResult(
            rejected_claim_ids=[claim_id],
            violations={str(claim_id): ["NUMERIC_TOKEN_NOT_IN_CITATION:70%"]},
            grounded_answer_ok=False,
            derived_answer_status="ANSWERED",
        )


class _NeedsReviewAgent:
    async def generate(self, question, run_id, as_of_date, packed, **_kwargs):
        del question
        evidence = evidence_ref_from_packed(packed.sections[0], as_of_date)
        claim = AgentClaim(
            category="DATA_CONFLICT",
            statement="材料之间存在冲突，需人工复核。",
            verdict="PARTIALLY_SUPPORTED",
            supporting_evidence_ids=[evidence.evidence_id],
            opposing_evidence_ids=[evidence.evidence_id],
            as_of_date=as_of_date,
        )
        trace = ModelInvocationTrace(
            provider="fake",
            model="fake-grounded-model",
            prompt_version="grounded_qa_v1",
            prompt_sha256="7" * 64,
            request_sha256="8" * 64,
            response_sha256="9" * 64,
            latency_ms=1.0,
            attempts=1,
            status="SUCCESS",
        )
        artifact = GroundedAnswerArtifact(
            run_id=run_id,
            task_id="grounded_qa",
            producer="grounded_qa",
            execution_status="PARTIAL",
            claims=[claim],
            evidence=[evidence],
            answer_status=AnswerStatus.NEEDS_REVIEW,
            direct_answer=None,
            conflicts=["材料之间存在冲突"],
            prompt_version="grounded_qa_v1",
            model_invocation_ids=[trace.invocation_id],
        )
        return GroundedQAGeneration(
            artifact=artifact,
            model_traces=[trace],
            generation_mode="llm",
        )


class _TamperedAgent:
    def __init__(self, update: dict):
        self.update = update
        self.delegate = _AnsweredAgent()

    async def generate(self, question, run_id, as_of_date, packed, **_kwargs):
        generation = await self.delegate.generate(question, run_id, as_of_date, packed)
        return generation.model_copy(
            update={"artifact": generation.artifact.model_copy(update=self.update)}
        )


def _settings():
    return SimpleNamespace(
        qa_prompt_version="grounded_qa_v1",
        qa_max_claims=6,
        qa_max_generation_tokens=1024,
        qa_max_audit_repairs=1,
        qa_allow_extractive_fallback=True,
        chunks_collection_name="chunks_test",
        summaries_collection_name="summaries_test",
        llm_provider="fake",
        llm_model="fake-grounded-model",
        embedding_provider="fake",
        effective_embedding_version="fake-embedding-v1",
        rerank_provider="fake",
        rerank_model="fake-reranker-v1",
        orchestrator_enable_rerank=True,
        orchestrator_enable_summary=True,
        orchestrator_enable_exact=True,
        context_token_budget=2048,
        context_max_per_document_ratio=0.6,
        context_expand_adjacent=True,
    )


def _section() -> PackedSection:
    text = "申请材料应包括最近一期经审计财务报表。"
    return PackedSection(
        section_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        parse_run_id=uuid.uuid4(),
        heading_path=["申请材料", "财务报表"],
        text=text,
        text_hash="e" * 64,
        page_start=3,
        page_end=3,
        tokens_est=len(text),
        rank=1,
    )


def _packed(*sections: PackedSection) -> PackedContext:
    return PackedContext(
        sections=list(sections),
        total_tokens_est=sum(section.tokens_est for section in sections),
        budget=2048,
    )


@pytest.fixture
async def qa_world(engine, monkeypatch):
    factory = create_session_factory(engine)
    case_id = uuid.uuid4()
    other_case_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    async with factory() as session:
        session.add_all(
            [
                Tenant(id=TENANT_ID, name="QA unit tenant"),
                AppUser(
                    id=USER_ID,
                    tenant_id=TENANT_ID,
                    external_subject=f"qa-user-{case_id}",
                    display_name="QA unit user",
                ),
                Entity(
                    id=entity_id,
                    tenant_id=TENANT_ID,
                    entity_type="COMPANY",
                    canonical_name="QA 测试企业",
                ),
                CreditCase(
                    id=case_id,
                    tenant_id=TENANT_ID,
                    case_number=f"QA-{case_id}",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1000000"),
                    application_date=AS_OF,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                    created_by=USER_ID,
                ),
                CreditCase(
                    id=other_case_id,
                    tenant_id=TENANT_ID,
                    case_number=f"QA-OTHER-{other_case_id}",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1"),
                    application_date=AS_OF,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                    created_by=USER_ID,
                ),
                CaseSnapshot(
                    id=snapshot_id,
                    tenant_id=TENANT_ID,
                    case_id=case_id,
                    case_version=1,
                    as_of_date=AS_OF,
                    decision_cutoff_at=CUTOFF,
                    borrower_entity_id=entity_id,
                    acl_scope_hash="a" * 64,
                    snapshot_hash="b" * 64,
                ),
                CaseMembership(case_id=case_id, user_id=USER_ID, case_role="ANALYST"),
            ]
        )
        await session.commit()

    trusted = TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        role_codes=["ANALYST"],
        purpose="grounded_qa",
        case_id=case_id,
        borrower_entity_id=entity_id,
        product_code="working_capital",
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    snapshot = SnapshotContext(
        snapshot_id=snapshot_id,
        chunks_collection="chunks_test",
        summaries_collection="summaries_test",
    )

    async def fake_build_trusted_context(*_args, purpose="grounded_qa", **_kwargs):
        return trusted.model_copy(update={"purpose": purpose, "request_id": uuid.uuid4()})

    async def fake_freeze_snapshot(*_args, **_kwargs):
        return snapshot

    snapshot_load_calls: list[uuid.UUID] = []

    async def fake_load_snapshot_context(_session, loaded_snapshot_id):
        snapshot_load_calls.append(loaded_snapshot_id)
        return snapshot

    monkeypatch.setattr(qa_module, "build_trusted_context", fake_build_trusted_context)
    monkeypatch.setattr(qa_module, "freeze_snapshot", fake_freeze_snapshot)
    monkeypatch.setattr(qa_module, "load_snapshot_context", fake_load_snapshot_context)
    return SimpleNamespace(
        factory=factory,
        case_id=case_id,
        other_case_id=other_case_id,
        snapshot=snapshot,
        snapshot_load_calls=snapshot_load_calls,
    )


def _service(qa_world, packed: PackedContext, agent, auditor=None) -> QAService:
    service = QAService(
        session_factory=qa_world.factory,
        orchestrator=_FakeOrchestrator(packed),
        settings=_settings(),
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        auditor=auditor or _AcceptingAuditor(),
    )
    service.agent = agent
    return service


async def test_answered_run_persists_complete_audited_chain(qa_world, monkeypatch):
    section = _section()
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(section), agent)

    response = await service.ask(
        case_id=qa_world.case_id,
        question="需要提交哪些财务报表？",
        top_k=5,
        idempotency_key="unit-answered-request",
    )

    assert response.answer_status == "ANSWERED"
    assert response.answer == "现有材料要求提交最近一期经审计财务报表。"
    assert response.claims[0].citations[0]["section_id"] == str(section.section_id)
    assert response.claims[0].citations[0]["preview_url"]
    assert agent.calls == 1

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        artifact = await session.scalar(
            select(ArtifactRecord).where(ArtifactRecord.run_id == response.run_id)
        )
        claim = await session.scalar(
            select(ClaimRecord).where(ClaimRecord.run_id == response.run_id)
        )
        evidence = await session.scalar(
            select(EvidenceRecord).where(EvidenceRecord.run_id == response.run_id)
        )
        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert run.status == "COMPLETED"
    assert run.completed_at is not None
    assert run.state_version == response.state_version
    assert artifact.artifact_type == "GROUNDED_ANSWER"
    assert artifact.task_id == "grounded_qa"
    assert artifact.producer == "grounded_qa"
    assert artifact.lifecycle_status == "VERIFIED"
    assert artifact.payload["lifecycle_status"] == "VERIFIED"
    assert "output_hash" not in artifact.payload
    assert artifact.output_hash != "f" * 64
    assert artifact.output_hash == sha256_text(
        json.dumps(
            artifact.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert artifact.payload["answer_status"] == "ANSWERED"
    assert run.model_manifest["model_invocation_ids"] == artifact.payload["model_invocation_ids"]
    assert [str(item) for item in response.model_invocation_ids] == artifact.payload[
        "model_invocation_ids"
    ]
    assert claim.review_status == "AUDITED"
    assert claim.artifact_id == artifact.id
    assert evidence.evidence_key == uuid.UUID(
        artifact.payload["claims"][0]["supporting_evidence_ids"][0]
    )
    assert evidence.locator["parse_run_id"] == str(section.parse_run_id)
    event_types = [event.event_type for event in events]
    assert event_types == [
        "RUN_CREATED",
        "STATE_CHANGED",
        "STATE_CHANGED",
        "RETRIEVAL_COMPLETED",
        "STATE_CHANGED",
        "MODEL_INVOCATION_COMPLETED",
        "STATE_CHANGED",
        "ANSWER_AUDIT_COMPLETED",
        "ANSWER_PERSISTED",
        "STATE_CHANGED",
    ]
    assert [event.sequence_no for event in events] == list(range(1, len(events) + 1))

    # The existing GET projection must expose the persisted QA artifact.
    from apps.api import main as api_main

    monkeypatch.setattr(api_main, "session_factory", qa_world.factory)
    run_view = await api_main.get_run(response.run_id)
    assert run_view["grounded_answer"]["answer_status"] == "ANSWERED"
    assert run_view["grounded_answer"]["answer"] == response.answer
    assert run_view["grounded_answer"]["generation_mode"] == "llm"


async def test_agent_failure_commits_failed_run_and_safe_trace(qa_world):
    service = _service(qa_world, _packed(_section()), _FailingAgent())

    with pytest.raises(QAServiceError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="provider secret question",
            top_k=5,
            idempotency_key="unit-provider-failure",
        )

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, captured.value.run_id)
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.sequence_no)
            )
        ).all()
        artifact_count = await session.scalar(
            select(func.count()).select_from(ArtifactRecord).where(ArtifactRecord.run_id == run.id)
        )
        claim_count = await session.scalar(
            select(func.count()).select_from(ClaimRecord).where(ClaimRecord.run_id == run.id)
        )
        evidence_count = await session.scalar(
            select(func.count()).select_from(EvidenceRecord).where(EvidenceRecord.run_id == run.id)
        )

    assert captured.value.error_type == "LLMCallError"
    assert run.status == "FAILED"
    assert run.completed_at is not None
    assert run.model_manifest["model_invocation_ids"] == [
        events[-2].payload_redacted["invocation_id"]
    ]
    assert artifact_count == claim_count == evidence_count == 0
    event_types = [event.event_type for event in events]
    assert "MODEL_INVOCATION_FAILED" in event_types
    assert event_types[-1] == "QA_EXECUTION_FAILED"
    assert "ANSWER_PERSISTED" not in event_types
    persisted_trace = json.dumps(
        [event.payload_redacted for event in events], ensure_ascii=False, sort_keys=True
    )
    assert "provider response secret must not persist" not in persisted_trace
    assert "provider secret question" not in persisted_trace
    assert "ABSTAINED" not in persisted_trace
    failure_payload = events[-1].payload_redacted
    assert failure_payload == {"from": "GENERATING", "error_type": "LLMCallError"}


async def test_empty_packed_context_completes_as_business_abstention(qa_world):
    service = _service(qa_world, _packed(), _AbstainingAgent())

    request = dict(
        case_id=qa_world.case_id,
        question="材料中没有的信息是什么？",
        top_k=5,
        idempotency_key="unit-abstained-request",
    )
    response = await service.ask(**request)
    replay = await service.ask(**request)

    assert response.answer_status == "ABSTAINED"
    assert response.answer == ""
    assert response.abstention_reason
    assert response.refusal_reason_code == RefusalReasonCode.MISSING_FINANCIAL_DATA
    assert response.model_dump(mode="json")["refusal_reason_code"] == "MISSING_FINANCIAL_DATA"
    assert response.claims == []
    assert response.model_invocation_ids == []
    assert replay.idempotent_replay is True
    assert replay.refusal_reason_code == response.refusal_reason_code
    assert replay.model_invocation_ids == response.model_invocation_ids
    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        artifact = await session.scalar(
            select(ArtifactRecord).where(ArtifactRecord.run_id == response.run_id)
        )
        claim_count = await session.scalar(
            select(func.count())
            .select_from(ClaimRecord)
            .where(ClaimRecord.run_id == response.run_id)
        )
        evidence_count = await session.scalar(
            select(func.count())
            .select_from(EvidenceRecord)
            .where(EvidenceRecord.run_id == response.run_id)
        )
        states = (
            await session.scalars(
                select(RunEvent.event_type)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert run.status == "COMPLETED"
    assert artifact.execution_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.payload["answer_status"] == "ABSTAINED"
    assert artifact.payload["refusal_reason_code"] == "MISSING_FINANCIAL_DATA"
    assert claim_count == evidence_count == 0
    assert "ANSWER_PERSISTED" in states
    assert "QA_EXECUTION_FAILED" not in states


async def test_audit_repair_receives_stable_codes_and_manifest_accumulates(qa_world):
    agent = _RepairFeedbackAgent()
    auditor = _RejectThenAcceptAuditor()
    service = _service(qa_world, _packed(_section()), agent, auditor)

    response = await service.ask(
        case_id=qa_world.case_id,
        question="审计拒绝后能否安全修复？",
        top_k=5,
        idempotency_key="unit-audit-repair",
    )

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert response.answer_status == "ANSWERED"
    assert auditor.calls == 2
    assert agent.feedback[0] == []
    assert {item["code"] for item in agent.feedback[1]} == {
        "NUMERIC_TOKEN_NOT_IN_CITATION",
        "SUPPORTED_WITHOUT_EVIDENCE",
    }
    assert all(item["scope"] == "CLAIM" for item in agent.feedback[1])
    assert all(item["category"] == "MISSING_MATERIAL" for item in agent.feedback[1])
    assert all(item["supporting_evidence_ids"] for item in agent.feedback[1])
    assert all(item["opposing_evidence_ids"] == [] for item in agent.feedback[1])
    assert all(item["apply_to"] == "MATCHING_CLAIM" for item in agent.feedback[1])
    assert {
        item["repair_hint"] for item in agent.feedback[1]
    } == {
        grounded_qa_repair_hint("NUMERIC_TOKEN_NOT_IN_CITATION"),
        grounded_qa_repair_hint("SUPPORTED_WITHOUT_EVIDENCE"),
    }
    serialized_feedback = json.dumps(agent.feedback[1], ensure_ascii=False)
    assert str(auditor.rejected_claim_id) not in serialized_feedback
    assert "70%" not in serialized_feedback
    model_event_ids = [
        event.payload_redacted["invocation_id"]
        for event in events
        if event.event_type == "MODEL_INVOCATION_COMPLETED"
    ]
    assert len(model_event_ids) == 2
    assert run.model_manifest["model_invocation_ids"] == model_event_ids
    assert run.model_manifest["generation_repairs"] == 0
    assert run.model_manifest["audit_repairs"] == 1
    rejected = next(event for event in events if event.event_type == "ANSWER_AUDIT_REJECTED")
    assert rejected.payload_redacted["violation_codes"] == [
        "NUMERIC_TOKEN_NOT_IN_CITATION",
        "SUPPORTED_WITHOUT_EVIDENCE",
    ]
    assert "70%" not in json.dumps(rejected.payload_redacted)


async def test_non_blocking_audit_review_is_normalized_without_model_repair(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent, _ReviewOnlyAuditor())

    response = await service.ask(
        case_id=qa_world.case_id,
        question="结构覆盖不足时应进入人工复核",
        top_k=5,
        idempotency_key="unit-review-normalization",
    )

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        artifact = await session.scalar(
            select(ArtifactRecord).where(ArtifactRecord.run_id == response.run_id)
        )
        claim = await session.scalar(
            select(ClaimRecord).where(ClaimRecord.run_id == response.run_id)
        )
        completed = await session.scalar(
            select(RunEvent).where(
                RunEvent.run_id == response.run_id,
                RunEvent.event_type == "ANSWER_AUDIT_COMPLETED",
            )
        )

    assert response.answer_status == "NEEDS_REVIEW"
    assert response.answer == ""
    assert agent.calls == 1
    assert run.status == "COMPLETED"
    assert run.model_manifest["audit_repairs"] == 0
    assert run.model_manifest["review_normalized"] is True
    assert artifact.execution_status == "PARTIAL"
    assert artifact.payload["answer_status"] == "NEEDS_REVIEW"
    assert artifact.payload["direct_answer"] is None
    assert claim.review_status == "PENDING"
    assert completed.payload_redacted["violation_codes"] == ["STRUCTURAL_COVERAGE_INSUFFICIENT"]


async def test_terminal_audit_failure_exposes_stable_safe_error_type(qa_world):
    service = _service(
        qa_world,
        _packed(_section()),
        _AnsweredAgent(),
        _AlwaysRejectingAuditor(),
    )

    with pytest.raises(QAServiceError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="持续审计失败不能伪装成业务结果",
            top_k=5,
            idempotency_key="unit-terminal-audit-failure",
        )

    async with qa_world.factory() as session:
        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == captured.value.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()
        artifact_count = await session.scalar(
            select(func.count())
            .select_from(ArtifactRecord)
            .where(ArtifactRecord.run_id == captured.value.run_id)
        )

    assert captured.value.error_type == "ANSWER_AUDIT_FAILED"
    assert events[-1].event_type == "QA_EXECUTION_FAILED"
    assert events[-1].payload_redacted["error_type"] == "ANSWER_AUDIT_FAILED"
    rejected = [event for event in events if event.event_type == "ANSWER_AUDIT_REJECTED"]
    assert rejected[-1].payload_redacted["terminal"] is True
    assert rejected[-1].payload_redacted["violation_codes"] == ["NUMERIC_TOKEN_NOT_IN_CITATION"]
    assert "70%" not in json.dumps([event.payload_redacted for event in events])
    assert artifact_count == 0


async def test_model_output_rejection_repairs_once_with_safe_code(qa_world):
    agent = _RejectingAgent(reject_times=1)
    service = _service(qa_world, _packed(_section()), agent)

    response = await service.ask(
        case_id=qa_world.case_id,
        question="模型输出违规后能否修复？",
        top_k=5,
        idempotency_key="unit-output-repair",
    )

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert response.answer_status == "ANSWERED"
    assert agent.feedback == [[], ["UNKNOWN_EVIDENCE_ID"]]
    assert run.model_manifest["generation_repairs"] == 1
    assert len(run.model_manifest["model_invocation_ids"]) == 2
    assert run.model_manifest["model_invocation_ids"][0] == str(agent.rejected_invocation_ids[0])
    rejected = [event for event in events if event.event_type == "ANSWER_GENERATION_REJECTED"]
    assert [event.payload_redacted for event in rejected] == [
        {
            "repair_attempt": 1,
            "error_code": "UNKNOWN_EVIDENCE_ID",
            "terminal": False,
        }
    ]


async def test_invalid_constructed_model_output_repairs_and_persists_safe_trace(qa_world):
    section = _section()
    packed = _packed(section)
    evidence = evidence_ref_from_packed(section, AS_OF)
    invalid_trace = ModelInvocationTrace(
        provider="fake",
        model="fake-grounded-model",
        prompt_version="grounded_qa_v1",
        prompt_sha256="a" * 64,
        request_sha256="b" * 64,
        response_sha256="c" * 64,
        latency_ms=1.0,
        attempts=1,
        status="SUCCESS",
    )
    repaired_trace = invalid_trace.model_copy(update={"invocation_id": uuid.uuid4()})
    invalid_claim = DraftAnswerClaim.model_construct(
        category="ELIGIBILITY",
        statement=" \t\u3000 ",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
    )
    invalid_draft = GroundedAnswerDraft.model_construct(claims=[invalid_claim])
    repaired_draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="MISSING_MATERIAL",
                statement="现有材料要求提交最近一期经审计财务报表。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[evidence.evidence_id],
            )
        ]
    )
    chat = _SequencedChat(
        [
            (invalid_draft, [invalid_trace]),
            (repaired_draft, [repaired_trace]),
        ]
    )
    agent = GroundedQAAgent(
        chat,
        prompt_path=DEFAULT_PROMPT_PATH,
        prompt_version="grounded_qa_v1",
        max_claims=6,
        max_tokens=1024,
    )
    service = _service(qa_world, packed, agent)

    request = {
        "case_id": qa_world.case_id,
        "question": "材料是否齐全？",
        "top_k": 5,
        "idempotency_key": "unit-invalid-model-output-repair",
    }
    response = await service.ask(**request)
    replay = await service.ask(**request)

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, response.run_id)
        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == response.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()

    assert response.answer_status == "ANSWERED"
    assert replay.idempotent_replay is True
    assert replay.model_invocation_ids == [repaired_trace.invocation_id]
    assert len(chat.calls) == 2
    assert "INVALID_MODEL_OUTPUT" in chat.calls[1]["user"]
    assert run.model_manifest["generation_repairs"] == 1
    assert run.model_manifest["model_invocation_ids"] == [
        str(invalid_trace.invocation_id),
        str(repaired_trace.invocation_id),
    ]
    model_event_ids = [
        event.payload_redacted["invocation_id"]
        for event in events
        if event.event_type == "MODEL_INVOCATION_COMPLETED"
    ]
    assert model_event_ids == run.model_manifest["model_invocation_ids"]
    rejection = next(event for event in events if event.event_type == "ANSWER_GENERATION_REJECTED")
    assert rejection.payload_redacted == {
        "repair_attempt": 1,
        "error_code": "INVALID_MODEL_OUTPUT",
        "terminal": False,
    }
    persisted_events = json.dumps([event.payload_redacted for event in events], ensure_ascii=False)
    assert "statement must not be blank" not in persisted_events


async def test_terminal_model_output_rejection_is_failed_not_abstained(qa_world):
    agent = _RejectingAgent(reject_times=2)
    service = _service(qa_world, _packed(_section()), agent)

    with pytest.raises(QAServiceError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="持续违规的模型输出不能伪装弃答",
            top_k=5,
            idempotency_key="unit-output-terminal",
        )

    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, captured.value.run_id)
        artifact_count = await session.scalar(
            select(func.count()).select_from(ArtifactRecord).where(ArtifactRecord.run_id == run.id)
        )
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.sequence_no)
            )
        ).all()

    assert captured.value.error_type == "GroundedQAOutputRejected"
    assert run.status == "FAILED"
    assert run.completed_at is not None
    assert artifact_count == 0
    assert len(run.model_manifest["model_invocation_ids"]) == 2
    rejection_events = [
        event.payload_redacted
        for event in events
        if event.event_type == "ANSWER_GENERATION_REJECTED"
    ]
    assert rejection_events[-1]["terminal"] is True
    assert rejection_events[-1]["error_code"] == "UNKNOWN_EVIDENCE_ID"
    assert events[-1].event_type == "QA_EXECUTION_FAILED"
    assert "ABSTAINED" not in json.dumps(
        [event.payload_redacted for event in events], ensure_ascii=False
    )


@pytest.mark.parametrize(
    ("update", "expected_error"),
    [
        ({"run_id": uuid.uuid4()}, "QA_ARTIFACT_RUN_ID_MISMATCH"),
        ({"task_id": "untrusted_task"}, "QA_ARTIFACT_TASK_ID_MISMATCH"),
        ({"producer": "untrusted_agent"}, "QA_ARTIFACT_PRODUCER_MISMATCH"),
        ({"prompt_version": "untrusted_prompt"}, "QA_ARTIFACT_PROVENANCE_MISMATCH"),
        ({"model_invocation_ids": []}, "QA_ARTIFACT_PROVENANCE_MISMATCH"),
        (
            {"refusal_reason_code": RefusalReasonCode.INSUFFICIENT_EVIDENCE},
            "QA_ARTIFACT_REFUSAL_REASON_MISMATCH",
        ),
    ],
)
async def test_untrusted_artifact_identity_or_provenance_fails_before_persist(
    qa_world,
    update,
    expected_error,
):
    service = _service(qa_world, _packed(_section()), _TamperedAgent(update))

    with pytest.raises(QAServiceError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="不可信 Artifact 来源",
            top_k=5,
            idempotency_key="unit-untrusted-artifact",
        )

    assert str(captured.value.__cause__) == expected_error
    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, captured.value.run_id)
        artifact_count = await session.scalar(
            select(func.count()).select_from(ArtifactRecord).where(ArtifactRecord.run_id == run.id)
        )
    assert run.status == "FAILED"
    assert run.completed_at is not None
    assert artifact_count == 0


async def test_needs_review_claims_remain_pending_for_human_review(qa_world):
    service = _service(qa_world, _packed(_section()), _NeedsReviewAgent())

    response = await service.ask(
        case_id=qa_world.case_id,
        question="冲突材料应如何处理？",
        top_k=5,
        idempotency_key="unit-needs-review",
    )

    async with qa_world.factory() as session:
        claim = await session.scalar(
            select(ClaimRecord).where(ClaimRecord.run_id == response.run_id)
        )
    assert response.answer_status == "NEEDS_REVIEW"
    assert len(response.model_invocation_ids) == 1
    assert claim.review_status == "PENDING"
    assert claim.payload["answer_status"] == "NEEDS_REVIEW"


async def test_snapshot_parent_binding_is_checked_before_loading_context(qa_world):
    async with qa_world.factory() as session:
        snapshot = await session.get(CaseSnapshot, qa_world.snapshot.snapshot_id)
        snapshot.case_id = qa_world.other_case_id
        await session.commit()
    service = _service(qa_world, _packed(_section()), _AnsweredAgent())

    with pytest.raises(QAServiceError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="错绑快照不能进入检索",
            top_k=5,
            idempotency_key="unit-snapshot-binding",
        )

    assert str(captured.value.__cause__) == "QA_SNAPSHOT_BINDING_MISMATCH"
    assert qa_world.snapshot_load_calls == []
    async with qa_world.factory() as session:
        run = await session.get(ReviewRun, captured.value.run_id)
    assert run.status == "FAILED"
    assert run.completed_at is not None


async def test_completed_idempotent_request_replays_without_second_execution(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "同一问题重试应复用原 Run",
        "top_k": 5,
        "idempotency_key": "unit-idempotent-replay",
    }

    first = await service.ask(**kwargs)
    replay = await service.ask(**kwargs)

    assert replay.run_id == first.run_id
    assert replay.snapshot_id == first.snapshot_id
    assert replay.answer == first.answer
    assert replay.model_invocation_ids == first.model_invocation_ids
    assert len(replay.model_invocation_ids) == 1
    assert replay.idempotent_replay is True
    assert replay.candidates == []
    assert replay.channel_config == {"idempotent_replay": True}
    assert agent.calls == 1
    async with qa_world.factory() as session:
        run_count = await session.scalar(
            select(func.count())
            .select_from(ReviewRun)
            .where(ReviewRun.request_idempotency_key == kwargs["idempotency_key"])
        )
        artifact_count = await session.scalar(
            select(func.count())
            .select_from(ArtifactRecord)
            .where(ArtifactRecord.run_id == first.run_id)
        )
    assert run_count == artifact_count == 1


async def test_replay_rejects_synchronized_answer_and_claim_payload_tampering(qa_world):
    service = _service(qa_world, _packed(_section()), _AnsweredAgent())
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "篡改后的答案不能通过回放",
        "top_k": 5,
        "idempotency_key": "unit-replay-payload-integrity",
    }
    first = await service.ask(**kwargs)
    tampered_text = "同步伪造的答案与 Claim，不得返回给调用方。"

    async with qa_world.factory() as session:
        record = await session.scalar(
            select(ArtifactRecord).where(ArtifactRecord.run_id == first.run_id)
        )
        payload = json.loads(json.dumps(record.payload, ensure_ascii=False))
        payload["direct_answer"] = tampered_text
        payload["claims"][0]["statement"] = tampered_text
        record.payload = payload
        await session.commit()

    with pytest.raises(QAServiceError) as captured:
        await service.ask(**kwargs)

    assert captured.value.error_type == "IDEMPOTENT_REPLAY_INTEGRITY_FAILED"
    assert tampered_text not in str(captured.value)


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    [
        ("task_id", "untrusted_task"),
        ("producer", "untrusted_agent"),
        ("lifecycle_status", "ACCEPTED"),
        ("execution_status", "PARTIAL"),
        ("contract_version", "9.9"),
        ("input_hash", "0" * 64),
    ],
)
async def test_replay_rejects_record_metadata_tampering(
    qa_world,
    field_name,
    tampered_value,
):
    service = _service(qa_world, _packed(_section()), _AnsweredAgent())
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "元数据篡改后不能回放",
        "top_k": 5,
        "idempotency_key": f"unit-replay-metadata-{field_name}",
    }
    first = await service.ask(**kwargs)

    async with qa_world.factory() as session:
        record = await session.scalar(
            select(ArtifactRecord).where(ArtifactRecord.run_id == first.run_id)
        )
        setattr(record, field_name, tampered_value)
        await session.commit()

    with pytest.raises(QAServiceError) as captured:
        await service.ask(**kwargs)

    assert captured.value.error_type == "IDEMPOTENT_REPLAY_INTEGRITY_FAILED"


async def test_idempotency_key_reuse_with_different_payload_is_conflict(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    key = "unit-idempotency-conflict"
    await service.ask(
        case_id=qa_world.case_id,
        question="原始问题",
        top_k=5,
        idempotency_key=key,
    )

    with pytest.raises(IdempotencyConflictError) as captured:
        await service.ask(
            case_id=qa_world.case_id,
            question="内容不同的问题",
            top_k=5,
            idempotency_key=key,
        )

    assert captured.value.error_code == "IDEMPOTENCY_CONFLICT"
    assert agent.calls == 1


async def test_idempotency_hash_prevents_replaying_answer_from_different_model(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "模型版本变化后不得冒充当前实验结果",
        "top_k": 5,
        "idempotency_key": "unit-model-provenance-conflict",
    }
    await service.ask(**kwargs)
    service.settings.llm_model = "different-grounded-model"

    with pytest.raises(IdempotencyConflictError):
        await service.ask(**kwargs)

    assert agent.calls == 1


async def test_idempotency_hash_tracks_extractive_fallback_policy(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "抽取兜底策略变化后不得重放旧答案",
        "top_k": 5,
        "idempotency_key": "unit-fallback-policy-conflict",
    }
    await service.ask(**kwargs)
    service.settings.qa_allow_extractive_fallback = False

    with pytest.raises(IdempotencyConflictError):
        await service.ask(**kwargs)

    assert agent.calls == 1


async def test_idempotency_hash_tracks_runtime_fusion_configuration(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    service.orchestrator.rrf_k = 60
    service.orchestrator.route_weights = {"DENSE": 1.0, "SPARSE": 1.0}
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "融合参数变化后不得重放旧答案",
        "top_k": 5,
        "idempotency_key": "unit-fusion-config-conflict",
    }
    await service.ask(**kwargs)
    service.orchestrator.route_weights = {"DENSE": 2.0, "SPARSE": 1.0}

    with pytest.raises(IdempotencyConflictError):
        await service.ask(**kwargs)

    assert agent.calls == 1


@pytest.mark.parametrize(
    ("version_attribute", "new_version"),
    [
        ("_audit_implementation_version", "structural_evidence_v3"),
        ("_grounded_answer_contract_version", "1.2"),
    ],
)
async def test_idempotency_hash_tracks_audit_and_answer_contract_versions(
    qa_world,
    version_attribute,
    new_version,
):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)
    kwargs = {
        "case_id": qa_world.case_id,
        "question": "审计契约变化后不得命中旧问答结果",
        "top_k": 5,
        "idempotency_key": f"unit-audit-version-{version_attribute}",
    }
    await service.ask(**kwargs)
    setattr(service, version_attribute, new_version)

    with pytest.raises(IdempotencyConflictError):
        await service.ask(**kwargs)

    assert agent.calls == 1


async def test_service_rejects_naive_decision_cutoff_before_creating_run(qa_world):
    agent = _AnsweredAgent()
    service = _service(qa_world, _packed(_section()), agent)

    with pytest.raises(ValueError, match="must include a timezone"):
        await service.ask(
            case_id=qa_world.case_id,
            question="无时区截止时点不得被本地时区隐式解释",
            top_k=5,
            decision_cutoff_at=datetime(2026, 6, 30, 12),
            idempotency_key="unit-naive-cutoff-rejected",
        )

    async with qa_world.factory() as session:
        run_count = await session.scalar(select(func.count()).select_from(ReviewRun))
    assert run_count == 0
    assert agent.calls == 0
