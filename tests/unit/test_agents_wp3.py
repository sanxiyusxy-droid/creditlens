"""WP3 Agent 单测与故障注入（Risk/Challenger/Auditor/Report）。

覆盖：
- Risk：版本化阈值配置、工具异常记录降级（不静默吞掉）、消费 Packed Sections；
- Challenger：五维冲突判断（指标/期间/单位/口径/数值）、冲突 vs 补充材料分流、
  source_claim_id 持久化；
- Auditor：Case/cutoff/Snapshot 复核、真冲突送 HITL 而补充材料不送、
  核心 Agent 失败阻断 vs 非核心降级；
- Report：真实 Evidence locator、source_claim_id 可追踪、
  VERIFIED_DRAFT / APPROVED_DRAFT 状态区分。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from creditlens.agents.auditor import EvidenceAuditor
from creditlens.agents.challenger import Challenger, assess_conflict
from creditlens.agents.contracts import (
    AgentArtifact,
    AgentClaim,
    AgentEvidenceRef,
    AnswerStatus,
    DraftAnswerClaim,
    GroundedAnswerArtifact,
    GroundedAnswerDraft,
    RefusalReasonCode,
)
from creditlens.agents.grounded_qa import GroundedQAAgent
from creditlens.agents.report_agent import ReportAgent
from creditlens.agents.risk_agent import RiskAgent
from creditlens.application.qa_service import (
    _audit_repair_feedback,
    _normalize_non_blocking_review,
)
from creditlens.application.snapshot_service import SnapshotContext
from creditlens.formulas.engine import CalculationArtifact, FormulaRegistry
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    CaseDocument,
    ClaimRecord,
    CreditCase,
    Document,
    DocumentSection,
    DocumentVersion,
    Entity,
    FinancialFact,
    ParseRun,
    ReviewRun,
    Tenant,
)
from creditlens.retrieval.context_packing import PackedContext, PackedSection
from creditlens.retrieval.contracts import RetrievedCandidate, TrustedRequestContext

TENANT = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
CASE = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
ENTITY = uuid.UUID("00000000-0000-0000-0000-0000000000cc")
AS_OF = date(2026, 5, 1)
CUTOFF = datetime(2026, 5, 1, tzinfo=UTC)


class _FakeGateway:
    """可编程工具网关：按工具名返回响应或抛异常。"""

    def __init__(self, responses: dict | None = None, errors: dict | None = None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls: list[str] = []
        self.invocations: list[dict] = []

    async def invoke(self, role, tool, **kwargs):
        self.calls.append(tool)
        self.invocations.append({"role": role, "tool": tool, **kwargs})
        if tool in self.errors:
            raise self.errors[tool]
        if tool not in self.responses:
            raise AssertionError(f"未预期的工具调用: {tool}")
        return self.responses[tool]


class _SearchResult:
    def __init__(self, candidates=None, packing=None):
        self.candidates = candidates or []
        self.packing = packing


def _candidate(text: str, section_id=None) -> RetrievedCandidate:
    return RetrievedCandidate(
        section_id=section_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        parse_run_id=uuid.uuid4(),
        page_start=1,
        page_end=1,
        heading_path=["第五章", "风险提示"],
        text=text,
        text_hash=f"hash-{text[:6]}",
        channel="DENSE",
        rank=1,
        raw_score=0.9,
    )


def _trusted(case_id=CASE, cutoff=CUTOFF) -> TrustedRequestContext:
    return TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=TENANT,
        case_id=case_id,
        as_of_date=AS_OF,
        decision_cutoff_at=cutoff,
    )


def _calc(result: Decimal, status: str = "CALCULATED") -> CalculationArtifact:
    return CalculationArtifact(
        calculation_id=uuid.uuid4(),
        metric_code="debt_ratio",
        formula_version="1.0",
        expression="total_liabilities / total_assets * 100",
        inputs=[],
        parameters={},
        result=result,
        result_unit="%",
        status=status,
        trace_hash=f"trace-{uuid.uuid4().hex[:8]}",
    )


# ====================== Risk Agent ======================


async def test_risk_threshold_versioned_and_recorded():
    """阈值来自版本化配置，版本随 Artifact 可追溯；未知版本显式报错。"""
    gateway = _FakeGateway(
        responses={
            "compute_metric": _calc(Decimal("50.0")),
            "search_risk_evidence": _SearchResult(),
        }
    )
    agent = RiskAgent(gateway)
    run_id = uuid.uuid4()
    artifact = await agent.run(run_id, "risk_review", _trusted())
    assert agent.threshold_version == "risk-thresholds-v1"
    assert {"risk_threshold_version": "risk-thresholds-v1"} in artifact.unresolved_issues
    assert {call["task_id"] for call in gateway.invocations} == {f"{run_id}:risk_review"}
    with pytest.raises(KeyError):
        RiskAgent(gateway, threshold_version="risk-thresholds-v99")


async def test_risk_threshold_triggers_high_claim():
    """资产负债率超过 warn_above 必须产生 HIGH 风险 Claim（阈值来自配置）。"""
    gateway = _FakeGateway(
        responses={
            "compute_metric": _calc(Decimal("75.0")),
            "search_risk_evidence": _SearchResult(),
        }
    )
    artifact = await RiskAgent(gateway).run(uuid.uuid4(), "risk_review", _trusted())
    triggered = [c for c in artifact.claims if c.severity == "HIGH"]
    assert triggered, "75% > 70% 预警线应触发风险 Claim"
    assert triggered[0].category == "FINANCIAL"
    assert triggered[0].calculation_ids, "风险 Claim 必须绑定可重放的计算痕迹"


async def test_risk_tool_failure_records_degradation():
    """WP3：工具异常必须记录降级，不能静默吞掉，也不得让 Run 假成功。"""
    gateway = _FakeGateway(
        errors={
            "compute_metric": RuntimeError("metric service down"),
            "search_risk_evidence": ConnectionError("search service down"),
        }
    )
    artifact = await RiskAgent(gateway).run(uuid.uuid4(), "risk_review", _trusted())
    degraded = [i for i in artifact.unresolved_issues if i.get("degraded")]
    metric_deg = [i for i in degraded if i["tool"] == "compute_metric"]
    search_deg = [i for i in degraded if i["tool"] == "search_risk_evidence"]
    assert len(metric_deg) == 2, "两个指标各自记录一条降级"
    assert len(search_deg) == 4, "四个检索主题各自记录一条降级"
    assert all(i["error"] for i in degraded)
    # 无证据支撑时不得产出 SUPPORTED Claim
    assert all(c.verdict != "SUPPORTED" for c in artifact.claims)


async def test_risk_consumes_packed_sections():
    """WP2：Risk 实际消费 Packed Sections 原文而非仅元数据。"""
    packed = {
        "sections": [
            {
                "section_id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "document_version_id": str(uuid.uuid4()),
                "parse_run_id": str(uuid.uuid4()),
                "heading_path": ["第四章", "客户集中度"],
                "text": "第一大客户收入占比超过 40%，存在集中度风险。",
                "text_hash": "hash-packed",
                "page_start": 4,
                "page_end": 5,
                "rank": 1,
            }
        ]
    }
    gateway = _FakeGateway(
        responses={
            "compute_metric": _calc(Decimal("50.0")),
            "search_risk_evidence": _SearchResult(packing=packed),
        }
    )
    artifact = await RiskAgent(gateway).run(uuid.uuid4(), "risk_review", _trusted())
    section_claims = [c for c in artifact.claims if c.category == "CONCENTRATION"]
    assert section_claims, "Packed Section 应转化为风险 Claim"
    assert artifact.evidence, "证据必须携带原文定位"
    assert any(e.content_hash == "hash-packed" for e in artifact.evidence)


# ====================== Challenger ======================


def test_challenger_five_dimension_conflict_matrix():
    """WP3 五维判断：同期间同单位数值不一致=冲突；期间/单位不同=口径差异。"""
    conflict, _ = assess_conflict("2025年资产负债率为 65%", "2025年资产负债率为 72%")
    assert conflict is True

    conflict, reason = assess_conflict("2024年资产负债率为 65%", "2025年资产负债率为 72%")
    assert conflict is False and "期间" in reason

    conflict, reason = assess_conflict("应收账款 500 万元", "应收账款 8000 元")
    assert conflict is False and "单位" in reason

    conflict, reason = assess_conflict("现金流出现紧张迹象", "现金流为负")
    assert conflict is False and "数值" in reason

    conflict, reason = assess_conflict("资产负债率为 65%", "年报口径仍为 65%")
    assert conflict is False and "一致" in reason


def _source_artifact(run_id, statement, section_id=None) -> tuple[AgentArtifact, AgentClaim]:
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id or uuid.uuid4(),
        content_hash="hash-src",
        section_id=section_id,
        source_available_at=datetime.now(UTC),
    )
    claim = AgentClaim(
        category="FINANCIAL",
        statement=statement,
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
        as_of_date=AS_OF,
    )
    artifact = AgentArtifact(
        run_id=run_id,
        task_id="financial_review",
        producer="financial_analyst",
        claims=[claim],
        evidence=[evidence],
    )
    return artifact, claim


async def test_challenger_true_conflict_keeps_both_sides():
    """真冲突：DATA_CONFLICT + PARTIALLY_SUPPORTED，source_claim_id 回指原 Claim。"""
    run_id = uuid.uuid4()
    source, claim = _source_artifact(run_id, "2025年资产负债率为 65%，处于可控水平")
    gateway = _FakeGateway(
        responses={
            "search_counter_evidence": _SearchResult(
                candidates=[_candidate("审计口径下2025年资产负债率为 72%。")]
            )
        }
    )
    artifact = await Challenger(gateway).run(run_id, "challenge", _trusted(), [source])
    assert len(artifact.claims) == 1
    conflict = artifact.claims[0]
    assert conflict.category == "DATA_CONFLICT"
    assert conflict.verdict == "PARTIALLY_SUPPORTED"
    assert conflict.source_claim_id == claim.claim_id, "source_claim_id 必须持久化可追踪"
    assert conflict.opposing_evidence_ids, "反证必须保留"


async def test_challenger_supplement_material_not_conflict():
    """补充材料（无数值矛盾）：MISSING_MATERIAL，不阻断流程。"""
    run_id = uuid.uuid4()
    source, claim = _source_artifact(run_id, "2025年资产负债率为 65%，处于可控水平")
    gateway = _FakeGateway(
        responses={
            "search_counter_evidence": _SearchResult(
                candidates=[_candidate("另有客户集中度管理的补充说明，未列示具体数字。")]
            )
        }
    )
    artifact = await Challenger(gateway).run(run_id, "challenge", _trusted(), [source])
    assert len(artifact.claims) == 1
    supplement = artifact.claims[0]
    assert supplement.category == "MISSING_MATERIAL"
    assert supplement.verdict == "INSUFFICIENT_EVIDENCE"
    assert supplement.uncertainty_reason, "补充材料必须说明原因（契约要求）"
    assert supplement.source_claim_id == claim.claim_id


async def test_challenger_consumes_packed_sections():
    """WP2：Challenger 消费 Packed Sections。"""
    run_id = uuid.uuid4()
    source, _ = _source_artifact(run_id, "2025年资产负债率为 65%")
    packed = {
        "sections": [
            {
                "section_id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "document_version_id": str(uuid.uuid4()),
                "parse_run_id": str(uuid.uuid4()),
                "heading_path": ["附注"],
                "text": "复核口径下2025年资产负债率为 70%。",
                "text_hash": "hash-packed",
                "page_start": 8,
                "page_end": 9,
                "rank": 1,
            }
        ]
    }
    gateway = _FakeGateway(responses={"search_counter_evidence": _SearchResult(packing=packed)})
    artifact = await Challenger(gateway).run(run_id, "challenge", _trusted(), [source])
    assert artifact.claims and artifact.claims[0].category == "DATA_CONFLICT"


# ====================== Auditor ======================


async def _make_world(session, *, available_at=datetime(2026, 4, 30, tzinfo=UTC)):
    section_id = uuid.uuid4()
    parse_run_id = uuid.uuid4()
    version_id = uuid.uuid4()
    session.add_all(
        [
            Tenant(id=TENANT, name="T"),
            Entity(id=ENTITY, tenant_id=TENANT, entity_type="COMPANY", canonical_name="示例公司"),
            CreditCase(
                id=CASE,
                tenant_id=TENANT,
                case_number="C-001",
                borrower_entity_id=ENTITY,
                product_code="working_capital",
                requested_amount=Decimal("1000000"),
                application_date=AS_OF,
                as_of_date=AS_OF,
                decision_cutoff_at=CUTOFF,
            ),
            Document(
                tenant_id=TENANT,
                logical_key="annual_report_x",
                title="年报",
                document_type="ANNUAL_REPORT",
            ),
        ]
    )
    await session.flush()
    doc = (
        await session.scalars(select(Document).where(Document.logical_key == "annual_report_x"))
    ).one()
    session.add(
        DocumentVersion(
            id=version_id,
            tenant_id=TENANT,
            document_id=doc.id,
            version_label="2025",
            source_available_at=available_at,
            object_uri="obj://x",
            source_filename="x.pdf",
            mime_type="application/pdf",
            file_size=10,
            content_hash="hash-ver",
            active_parse_run_id=parse_run_id,
        )
    )
    session.add(
        ParseRun(
            id=parse_run_id,
            tenant_id=TENANT,
            document_version_id=version_id,
            parser_name="pymupdf",
            parser_version="1.0",
            config_hash="cfg",
            activation_status="ACTIVE",
        )
    )
    session.add(
        DocumentSection(
            id=section_id,
            tenant_id=TENANT,
            document_version_id=version_id,
            parse_run_id=parse_run_id,
            section_type="PARAGRAPH",
            ordinal=1,
            page_start=1,
            page_end=1,
            text="原文内容",
            text_hash="hash-sec",
        )
    )
    session.add(
        CaseDocument(
            case_id=CASE,
            document_version_id=version_id,
            document_role="BORROWER_PROVIDED",
        )
    )
    await session.flush()
    return section_id, parse_run_id, version_id


def _span_artifact(
    run_id,
    section_id,
    version_id,
    parse_run_id,
    text_hash="hash-sec",
    page_number=1,
) -> AgentArtifact:
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        content_hash=text_hash,
        document_version_id=version_id,
        section_id=section_id,
        parse_run_id=parse_run_id,
        page_number=page_number,
        source_available_at=datetime.now(UTC),
    )
    claim = AgentClaim(
        category="ELIGIBILITY",
        statement="材料已齐备，符合审查要求。",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
        as_of_date=AS_OF,
    )
    return AgentArtifact(
        run_id=run_id,
        task_id="policy_review",
        producer="policy_analyst",
        claims=[claim],
        evidence=[evidence],
    )


class _SequentialDraftChat:
    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.calls: list[dict] = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.drafts.pop(0)


async def _packed_world(session, *, text: str) -> tuple[PackedContext, uuid.UUID]:
    section_id, parse_run_id, version_id = await _make_world(session)
    section = await session.get(DocumentSection, section_id)
    version = await session.get(DocumentVersion, version_id)
    section.text = text
    await session.flush()
    packed = PackedContext(
        sections=[
            PackedSection(
                section_id=section_id,
                document_id=version.document_id,
                document_version_id=version_id,
                parse_run_id=parse_run_id,
                heading_path=["审计单测"],
                text=text,
                text_hash=section.text_hash,
                page_start=1,
                page_end=1,
                tokens_est=len(text),
                rank=1,
            )
        ],
        total_tokens_est=len(text),
        budget=2048,
    )
    return packed, parse_run_id


async def test_real_grounded_agent_and_auditor_normalize_review_only_result(session):
    packed, parse_run_id = await _packed_world(
        session,
        text="申请材料应包括最近一期经审计财务报表。",
    )
    ref_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"evidence:{packed.sections[0].section_id}:{packed.sections[0].text_hash}",
    )
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement="该企业已经获得国家级高新技术企业认证。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref_id],
            )
        ]
    )
    generation = await GroundedQAAgent(_SequentialDraftChat(draft)).generate(
        "企业具备什么资质？",
        uuid.uuid4(),
        AS_OF,
        packed,
    )
    audit = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        generation.artifact,
        allowed_evidence_ids={ref_id},
        snapshot=SnapshotContext(
            snapshot_id=uuid.uuid4(),
            allowed_parse_run_ids=[parse_run_id],
        ),
    )

    normalized, changed = _normalize_non_blocking_review(generation.artifact, audit)

    assert audit.ok
    assert audit.review_normalization_required
    assert changed is True
    assert normalized.answer_status == AnswerStatus.NEEDS_REVIEW
    assert normalized.execution_status == "PARTIAL"
    assert normalized.direct_answer is None


async def test_real_grounded_agent_and_auditor_repair_uses_model_visible_locator(session):
    packed, parse_run_id = await _packed_world(
        session,
        text="政策规定资产负债率不得超过70%。",
    )
    ref_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"evidence:{packed.sections[0].section_id}:{packed.sections[0].text_hash}",
    )
    bad_draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement="政策规定资产负债率上限为80%。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref_id],
            )
        ]
    )
    repaired_draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement="政策规定资产负债率上限为70%。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref_id],
            )
        ]
    )
    chat = _SequentialDraftChat(bad_draft, repaired_draft)
    agent = GroundedQAAgent(chat)
    auditor = EvidenceAuditor(FormulaRegistry())
    snapshot = SnapshotContext(
        snapshot_id=uuid.uuid4(),
        allowed_parse_run_ids=[parse_run_id],
    )
    first = await agent.generate("资产负债率上限是多少？", uuid.uuid4(), AS_OF, packed)
    first_audit = await auditor.verify_grounded_answer(
        session,
        _trusted(),
        first.artifact,
        allowed_evidence_ids={ref_id},
        snapshot=snapshot,
    )
    feedback = _audit_repair_feedback(first_audit.violations, first.artifact)
    repaired = await agent.generate(
        "资产负债率上限是多少？",
        uuid.uuid4(),
        AS_OF,
        packed,
        audit_feedback=feedback,
    )
    repaired_audit = await auditor.verify_grounded_answer(
        session,
        _trusted(),
        repaired.artifact,
        allowed_evidence_ids={ref_id},
        snapshot=snapshot,
    )

    claim_id = str(first.artifact.claims[0].claim_id)
    assert not first_audit.ok
    assert not first_audit.review_normalization_required
    assert feedback == [
        {
            "scope": "CLAIM",
            "code": "NUMERIC_TOKEN_NOT_IN_CITATION",
            "repair_hint": feedback[0]["repair_hint"],
            "category": "ELIGIBILITY",
            "supporting_evidence_ids": [str(ref_id)],
            "opposing_evidence_ids": [],
            "apply_to": "MATCHING_CLAIM",
        }
    ]
    assert claim_id not in chat.calls[1]["user"]
    assert str(ref_id) in chat.calls[1]["user"]
    assert '"category": "ELIGIBILITY"' in chat.calls[1]["user"]
    assert "NUMERIC_TOKEN_NOT_IN_CITATION" in chat.calls[1]["user"]
    assert "80%" not in chat.calls[1]["user"]
    assert repaired_audit.ok
    assert repaired_audit.derived_answer_status == "ANSWERED"


async def test_auditor_rejects_when_case_missing(session):
    """WP3：Case 不存在/租户不一致 -> 全部 Claim 拒绝。"""
    await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4())
    result = await auditor.verify(session, _trusted(case_id=uuid.uuid4()), [artifact])
    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    assert "CASE_VALIDATION_FAILED" in result.violations[str(artifact.claims[0].claim_id)]


async def test_auditor_rejects_material_after_cutoff(session):
    """WP3：cutoff 之后才可获得的材料必须拒绝（NOT_AVAILABLE_AT_CUTOFF）。"""
    section_id, parse_run_id, version_id = await _make_world(
        session, available_at=datetime(2026, 6, 1, tzinfo=UTC)
    )
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])
    result = await auditor.verify(session, _trusted(), [artifact], snapshot=snapshot)
    assert result.rejected_claim_ids, "cutoff 后材料不得进入报告"
    violations = [v for vs in result.violations.values() for v in vs]
    assert "NOT_AVAILABLE_AT_CUTOFF" in violations


async def test_auditor_rejects_parse_run_outside_snapshot(session):
    """WP3：证据 Parse Run 不在冻结 Snapshot 集合内必须拒绝。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[uuid.uuid4()])
    result = await auditor.verify(session, _trusted(), [artifact], snapshot=snapshot)
    assert result.rejected_claim_ids
    violations = [v for vs in result.violations.values() for v in vs]
    assert "PARSE_RUN_NOT_IN_SNAPSHOT" in violations


async def test_auditor_passes_valid_evidence_within_snapshot(session):
    """合法证据 + Snapshot 覆盖 -> 接受。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])
    result = await auditor.verify(session, _trusted(), [artifact], snapshot=snapshot)
    assert result.accepted_claim_ids == [artifact.claims[0].claim_id]
    assert not result.needs_human_review_claim_ids


async def test_grounded_answer_audit_accepts_exact_claim_rendering(session):
    section_id, parse_run_id, version_id = await _make_world(session)
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].statement = "原文内容"
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer="原文内容",
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    assert result.ok
    assert result.derived_answer_status == "ANSWERED"
    assert result.verification_scope == "STRUCTURAL_VERIFICATION"
    assert result.semantic_entailment_verified is False


def test_grounded_answer_contract_enforces_strict_execution_status_matrix():
    """Every business state has exactly one permitted technical execution state."""
    claim = AgentClaim(
        category="ELIGIBILITY",
        statement="原文内容",
        verdict="SUPPORTED",
        supporting_evidence_ids=[uuid.uuid4()],
        as_of_date=AS_OF,
    )
    cases = [
        (
            "PARTIAL",
            {
                "answer_status": AnswerStatus.ANSWERED,
                "claims": [claim],
                "direct_answer": claim.statement,
            },
        ),
        (
            "FAILED",
            {
                "answer_status": AnswerStatus.ANSWERED,
                "claims": [claim],
                "direct_answer": claim.statement,
            },
        ),
        (
            "SUCCESS",
            {
                "answer_status": AnswerStatus.NEEDS_REVIEW,
                "claims": [claim],
                "direct_answer": None,
            },
        ),
        (
            "FAILED",
            {
                "answer_status": AnswerStatus.NEEDS_REVIEW,
                "claims": [claim],
                "direct_answer": None,
            },
        ),
        (
            "SUCCESS",
            {
                "answer_status": AnswerStatus.ABSTAINED,
                "claims": [],
                "direct_answer": None,
                "abstention_reason": "现有证据不足。",
                "refusal_reason_code": RefusalReasonCode.INSUFFICIENT_EVIDENCE,
            },
        ),
        (
            "FAILED",
            {
                "answer_status": AnswerStatus.ABSTAINED,
                "claims": [],
                "direct_answer": None,
                "abstention_reason": "现有证据不足。",
                "refusal_reason_code": RefusalReasonCode.INSUFFICIENT_EVIDENCE,
            },
        ),
    ]

    for execution_status, answer_fields in cases:
        with pytest.raises(ValidationError, match="requires execution_status"):
            GroundedAnswerArtifact(
                run_id=uuid.uuid4(),
                task_id="grounded_qa",
                producer="grounded_qa",
                execution_status=execution_status,
                prompt_version="grounded_qa_v1",
                **answer_fields,
            )


def test_grounded_answer_contract_enforces_refusal_reason_status_matrix():
    claim = AgentClaim(
        category="ELIGIBILITY",
        statement="原文内容",
        verdict="SUPPORTED",
        supporting_evidence_ids=[uuid.uuid4()],
        as_of_date=AS_OF,
    )
    with pytest.raises(ValidationError, match="requires refusal_reason_code"):
        GroundedAnswerArtifact(
            run_id=uuid.uuid4(),
            task_id="grounded_qa",
            producer="grounded_qa",
            execution_status="INSUFFICIENT_EVIDENCE",
            answer_status=AnswerStatus.ABSTAINED,
            abstention_reason="现有证据不足。",
            prompt_version="grounded_qa_v1",
        )

    non_refusals = [
        {
            "execution_status": "SUCCESS",
            "answer_status": AnswerStatus.ANSWERED,
            "claims": [claim],
            "direct_answer": claim.statement,
        },
        {
            "execution_status": "PARTIAL",
            "answer_status": AnswerStatus.NEEDS_REVIEW,
            "claims": [claim],
            "direct_answer": None,
        },
    ]
    for fields in non_refusals:
        with pytest.raises(ValidationError, match="non-ABSTAINED artifact"):
            GroundedAnswerArtifact(
                run_id=uuid.uuid4(),
                task_id="grounded_qa",
                producer="grounded_qa",
                refusal_reason_code=RefusalReasonCode.INSUFFICIENT_EVIDENCE,
                prompt_version="grounded_qa_v1",
                **fields,
            )


async def test_grounded_answer_auditor_rejects_bypassed_failed_state_without_derivation(
    session,
):
    """Auditor remains fail-closed when model_copy bypasses Pydantic validation."""
    section_id, parse_run_id, version_id = await _make_world(session)
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].statement = "原文内容"
    answered = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer="原文内容",
        prompt_version="grounded_qa_v1",
    )
    abstained = GroundedAnswerArtifact(
        run_id=uuid.uuid4(),
        task_id="grounded_qa",
        producer="grounded_qa",
        execution_status="INSUFFICIENT_EVIDENCE",
        answer_status=AnswerStatus.ABSTAINED,
        abstention_reason="现有证据不足。",
        refusal_reason_code=RefusalReasonCode.INSUFFICIENT_EVIDENCE,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    for artifact, allowed_ids in (
        (answered, {answered.evidence[0].evidence_id}),
        (abstained, set()),
    ):
        bypassed = artifact.model_copy(update={"execution_status": "FAILED"})
        result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
            session,
            _trusted(),
            bypassed,
            allowed_evidence_ids=allowed_ids,
            snapshot=snapshot,
        )

        assert not result.ok
        assert result.derived_answer_status is None
        assert result.blocking_failures == ["GROUNDING_EXECUTION_FAILED"]
        assert result.violations["grounded_answer"] == ["GROUNDING_EXECUTION_FAILED"]


async def test_grounded_answer_audit_rejects_unpacked_citation_and_numeric_invention(session):
    section_id, parse_run_id, version_id = await _make_world(session)
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].statement = "该政策阈值为 70%"
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer="该政策阈值为 70%",
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids=set(),
        snapshot=snapshot,
    )

    assert not result.ok
    assert not result.review_normalization_required
    violations = [code for codes in result.violations.values() for code in codes]
    assert any(code.startswith("EVIDENCE_NOT_IN_PACKED_CONTEXT") for code in violations)
    assert "CITATION_NOT_IN_PACKED_CONTEXT" in violations
    assert "NUMERIC_TOKEN_NOT_IN_CITATION:70%" in violations


async def test_grounded_financial_number_may_use_verified_document_span(session):
    """Grounded QA's document-number exception opens only after full span checks."""
    section_id, parse_run_id, version_id = await _make_world(session)
    statement = "2025年度营业收入为50000万元。"
    section = await session.get(DocumentSection, section_id)
    section.text = statement
    await session.flush()
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].category = "FINANCIAL"
    base.claims[0].statement = statement
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer=statement,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    assert result.ok
    assert result.accepted_claim_ids == [artifact.claims[0].claim_id]


async def _audit_grounded_pair(
    session,
    *,
    cited_text: str,
    statement: str,
    category: str = "FINANCIAL",
):
    """Build a valid locator so answer-layer guards are isolated in tests."""
    section_id, parse_run_id, version_id = await _make_world(session)
    section = await session.get(DocumentSection, section_id)
    section.text = cited_text
    await session.flush()
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].category = category
    base.claims[0].statement = statement
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer=statement,
        prompt_version="grounded_qa_v1",
    )
    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=SnapshotContext(
            snapshot_id=uuid.uuid4(),
            allowed_parse_run_ids=[parse_run_id],
        ),
    )
    return result, artifact


async def test_grounded_numeric_guard_binds_explicit_sign(session):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text="2025年度营业收入同比增长+10%。",
        statement="2025年度营业收入同比增长-10%。",
    )

    assert not result.ok
    assert not result.review_normalization_required
    assert (
        "NUMERIC_TOKEN_NOT_IN_CITATION:-10%" in result.violations[str(artifact.claims[0].claim_id)]
    )


async def test_grounded_numeric_guard_binds_amount_scale_and_currency(session):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text="2025年度营业收入为50000元。",
        statement="2025年度营业收入为50000万元。",
    )

    assert not result.ok
    assert not result.review_normalization_required
    # The stable external code retains the historical numeric payload, while
    # the internal comparison also binds CNY and the 万 scale.
    assert (
        "NUMERIC_TOKEN_NOT_IN_CITATION:50000" in result.violations[str(artifact.claims[0].claim_id)]
    )


async def test_grounded_numeric_guard_still_runs_structural_coverage(session):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text="2025年度营业收入为50000万元。",
        statement="2025年度重大诉讼金额为50000万元。",
    )

    claim_id = artifact.claims[0].claim_id
    assert result.ok
    assert result.review_normalization_required
    assert claim_id in result.needs_human_review_claim_ids
    assert "STRUCTURAL_COVERAGE_INSUFFICIENT" in result.violations[str(claim_id)]


async def test_grounded_numeric_guard_normalizes_nfkc_spaces_and_thousands(session):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text="2025年度营业收入为50000万元。",
        statement="２０２５ 年度营业收入为 ５０,０００ 万元。",
    )

    assert result.ok, result.violations
    assert result.accepted_claim_ids == [artifact.claims[0].claim_id]


@pytest.mark.parametrize(
    ("cited_text", "statement"),
    [
        ("该企业不存在重大诉讼。", "该企业存在重大诉讼。"),
        ("申请人不满足准入要求。", "申请人满足准入要求。"),
        ("营业收入同比下降。", "营业收入同比上升。"),
    ],
)
async def test_grounded_obvious_polarity_reversal_routes_to_review(
    session,
    cited_text,
    statement,
):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text=cited_text,
        statement=statement,
    )

    claim_id = artifact.claims[0].claim_id
    assert result.ok
    assert result.review_normalization_required
    assert claim_id in result.needs_human_review_claim_ids
    assert claim_id not in result.rejected_claim_ids
    assert "POLARITY_CONFLICT_WITH_CITATION" in result.violations[str(claim_id)]


async def test_grounded_threshold_wording_is_not_a_polarity_false_positive(session):
    result, artifact = await _audit_grounded_pair(
        session,
        cited_text="政策规定资产负债率不得超过70%。",
        statement="政策规定资产负债率上限为70%。",
        category="ELIGIBILITY",
    )

    assert result.ok, result.violations
    assert artifact.claims[0].claim_id in result.accepted_claim_ids


async def test_grounded_financial_document_number_requires_verbatim_coverage(session):
    section_id, parse_run_id, version_id = await _make_world(session)
    section = await session.get(DocumentSection, section_id)
    section.text = "2025年度营业收入尚待披露。"
    await session.flush()
    statement = "2025年度营业收入为50000万元。"
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].category = "FINANCIAL"
    base.claims[0].statement = statement
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer=statement,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    assert not result.ok
    assert artifact.claims[0].claim_id in result.rejected_claim_ids
    assert (
        "NUMERIC_TOKEN_NOT_IN_CITATION:50000" in result.violations[str(artifact.claims[0].claim_id)]
    )


async def test_grounded_financial_document_number_requires_matching_hash(session):
    section_id, parse_run_id, version_id = await _make_world(session)
    statement = "2025年度营业收入为50000万元。"
    section = await session.get(DocumentSection, section_id)
    section.text = statement
    await session.flush()
    base = _span_artifact(
        uuid.uuid4(),
        section_id,
        version_id,
        parse_run_id,
        text_hash="tampered-hash",
    )
    base.producer = "grounded_qa"
    base.claims[0].category = "FINANCIAL"
    base.claims[0].statement = statement
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer=statement,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    assert not result.ok
    assert "CONTENT_HASH_MISMATCH" in result.violations[str(artifact.evidence[0].evidence_id)]
    assert artifact.claims[0].claim_id in result.rejected_claim_ids


async def test_generic_agent_financial_document_number_still_requires_structured_fact(
    session,
):
    """The document-number exception is not available to ordinary Agent artifacts."""
    section_id, parse_run_id, version_id = await _make_world(session)
    statement = "2025年度营业收入为50000万元。"
    section = await session.get(DocumentSection, section_id)
    section.text = statement
    await session.flush()
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    artifact.claims[0].category = "FINANCIAL"
    artifact.claims[0].statement = statement
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session,
        _trusted(),
        [artifact],
        snapshot=snapshot,
    )

    claim = artifact.claims[0]
    assert claim.claim_id in result.rejected_claim_ids
    assert any(
        code.endswith(":NUMERIC_CLAIM_WITHOUT_FACT")
        for code in result.violations[str(claim.claim_id)]
    )


async def test_grounded_nonnumeric_counterexample_is_forced_to_human_review(session):
    """A valid locator cannot turn an unrelated citation into semantic support."""
    section_id, parse_run_id, version_id = await _make_world(session)
    section = await session.get(DocumentSection, section_id)
    section.text = "申请材料应包括最近一期经审计财务报表。"
    await session.flush()
    statement = "该企业已经获得国家级高新技术企业认证。"
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].statement = statement
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.ANSWERED,
        direct_answer=statement,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    claim_id = artifact.claims[0].claim_id
    assert result.ok
    assert result.review_normalization_required
    assert result.derived_answer_status == "NEEDS_REVIEW"
    assert result.requires_human_review
    assert result.needs_human_review_claim_ids == [claim_id]
    assert claim_id not in result.accepted_claim_ids
    assert claim_id not in result.rejected_claim_ids
    assert "STRUCTURAL_COVERAGE_INSUFFICIENT" in result.violations[str(claim_id)]
    assert result.semantic_entailment_verified is False


async def test_grounded_answer_audit_derives_review_status_instead_of_trusting_model(session):
    section_id, parse_run_id, version_id = await _make_world(session)
    base = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    base.producer = "grounded_qa"
    base.claims[0].statement = "原文内容"
    base.execution_status = "PARTIAL"
    artifact = GroundedAnswerArtifact(
        **base.model_dump(),
        answer_status=AnswerStatus.NEEDS_REVIEW,
        direct_answer=None,
        prompt_version="grounded_qa_v1",
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids={artifact.evidence[0].evidence_id},
        snapshot=snapshot,
    )

    assert not result.ok
    assert result.derived_answer_status == "ANSWERED"
    assert "ANSWER_STATUS_MISMATCH:NEEDS_REVIEW:ANSWERED" in result.violations["grounded_answer"]


async def test_grounded_answer_audit_accepts_structured_abstention(session):
    await _make_world(session)
    artifact = GroundedAnswerArtifact(
        run_id=uuid.uuid4(),
        task_id="grounded_qa",
        producer="grounded_qa",
        execution_status="INSUFFICIENT_EVIDENCE",
        answer_status=AnswerStatus.ABSTAINED,
        direct_answer=None,
        abstention_reason="现有证据不足。",
        refusal_reason_code=RefusalReasonCode.INSUFFICIENT_EVIDENCE,
        prompt_version="grounded_qa_v1",
    )

    result = await EvidenceAuditor(FormulaRegistry()).verify_grounded_answer(
        session,
        _trusted(),
        artifact,
        allowed_evidence_ids=set(),
        snapshot=SnapshotContext(snapshot_id=uuid.uuid4()),
    )

    assert result.ok
    assert result.derived_answer_status == "ABSTAINED"


async def test_auditor_rejects_missing_document_locator(session):
    """DOCUMENT_SPAN 缺少版本、解析批次或页码时必须 fail-closed。"""
    section_id, parse_run_id, _ = await _make_world(session)
    artifact = _span_artifact(
        uuid.uuid4(),
        section_id,
        None,
        None,
        page_number=None,
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    evidence_violations = result.violations[str(artifact.evidence[0].evidence_id)]
    assert "MISSING_DOCUMENT_LOCATOR:document_version_id" in evidence_violations
    assert "MISSING_DOCUMENT_LOCATOR:parse_run_id" in evidence_violations
    assert "MISSING_DOCUMENT_LOCATOR:page_number" in evidence_violations


@pytest.mark.parametrize(
    ("wrong_version", "wrong_parse", "expected_violation"),
    [
        (True, False, "DOCUMENT_VERSION_MISMATCH"),
        (False, True, "PARSE_RUN_MISMATCH"),
    ],
)
async def test_auditor_rejects_inconsistent_version_or_parse_locator(
    session, wrong_version, wrong_parse, expected_violation
):
    """Locator 声明必须与 Section 的版本和 ParseRun 精确一致。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    artifact = _span_artifact(
        uuid.uuid4(),
        section_id,
        uuid.uuid4() if wrong_version else version_id,
        uuid.uuid4() if wrong_parse else parse_run_id,
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    assert expected_violation in result.violations[str(artifact.evidence[0].evidence_id)]


async def test_auditor_requires_snapshot_for_document_span(session):
    """即使 locator 合法，没有冻结 Snapshot 也不得接受文档证据。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=None
    )

    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    assert "SNAPSHOT_REQUIRED" in result.violations[str(artifact.evidence[0].evidence_id)]


async def test_auditor_requires_document_bound_to_case(session):
    """存在的文档版本若未绑定当前案件，同样必须 fail-closed。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    binding = await session.get(
        CaseDocument,
        {"case_id": CASE, "document_version_id": version_id},
    )
    await session.delete(binding)
    await session.flush()
    artifact = _span_artifact(uuid.uuid4(), section_id, version_id, parse_run_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    assert "DOCUMENT_NOT_BOUND_TO_CASE" in result.violations[str(artifact.evidence[0].evidence_id)]


async def test_auditor_calculation_evidence_requires_closed_reference(session):
    """计算证据必须完整绑定 Artifact 内计算，并与 trace_hash 一致。"""
    await _make_world(session)
    calculation = _calc(Decimal("50.0"))
    absent_calculation_id = uuid.uuid4()
    cases = [
        (None, calculation.calculation_id, calculation.trace_hash, "MISSING_CALCULATION_ID"),
        (
            calculation.calculation_id,
            uuid.uuid4(),
            calculation.trace_hash,
            "CALCULATION_SOURCE_MISMATCH",
        ),
        (
            absent_calculation_id,
            absent_calculation_id,
            calculation.trace_hash,
            "CALCULATION_NOT_FOUND",
        ),
        (
            calculation.calculation_id,
            calculation.calculation_id,
            "tampered-trace",
            "CALCULATION_CONTENT_HASH_MISMATCH",
        ),
    ]

    for calculation_id, source_id, content_hash, expected in cases:
        evidence = AgentEvidenceRef(
            evidence_id=uuid.uuid4(),
            evidence_type="CALCULATION",
            source_id=source_id,
            content_hash=content_hash,
            calculation_id=calculation_id,
            source_available_at=CUTOFF,
        )
        claim = AgentClaim(
            category="FINANCIAL",
            statement="财务指标已完成计算。",
            verdict="SUPPORTED",
            supporting_evidence_ids=[evidence.evidence_id],
            as_of_date=AS_OF,
        )
        artifact = AgentArtifact(
            run_id=uuid.uuid4(),
            task_id="financial_review",
            producer="financial_analyst",
            evidence=[evidence],
            claims=[claim],
            calculations=[calculation],
        )

        result = await EvidenceAuditor(FormulaRegistry()).verify(session, _trusted(), [artifact])

        assert claim.claim_id in result.rejected_claim_ids
        assert expected in result.violations[str(evidence.evidence_id)]


async def test_auditor_sql_fact_checks_identity_case_and_cutoff(session):
    """SQL_FACT 必须回表验证身份、案件归属与截止时点。"""
    await _make_world(session)
    fact = FinancialFact(
        tenant_id=TENANT,
        case_id=CASE,
        entity_id=ENTITY,
        metric_code="revenue",
        period_end=AS_OF,
        value=Decimal("100"),
        canonical_value=Decimal("100"),
        source_available_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    session.add(fact)
    await session.flush()
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="SQL_FACT",
        source_id=uuid.uuid4(),
        fact_id=fact.id,
        content_hash="fact-row",
        source_available_at=fact.source_available_at,
    )
    claim = AgentClaim(
        category="FINANCIAL",
        statement="财务事实可用于分析。",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
        as_of_date=AS_OF,
    )
    artifact = AgentArtifact(
        run_id=uuid.uuid4(),
        task_id="financial_review",
        producer="financial_analyst",
        evidence=[evidence],
        claims=[claim],
    )
    snapshot = SnapshotContext(
        snapshot_id=uuid.uuid4(),
        allowed_fact_ids=[fact.id],
    )

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert result.rejected_claim_ids == [claim.claim_id]
    violations = result.violations[str(evidence.evidence_id)]
    assert "FACT_SOURCE_MISMATCH" in violations
    assert "FACT_NOT_AVAILABLE_AT_CUTOFF" in violations


async def test_auditor_table_cell_document_locator_is_fail_closed(session):
    """带 Section 的 TABLE_CELL 采用与 DOCUMENT_SPAN 相同的强定位校验。"""
    section_id, parse_run_id, _ = await _make_world(session)
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="TABLE_CELL",
        source_id=section_id,
        section_id=section_id,
        content_hash="hash-sec",
        source_available_at=CUTOFF,
    )
    claim = AgentClaim(
        category="FINANCIAL",
        statement="表格单元格提供财务依据。",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
        as_of_date=AS_OF,
    )
    artifact = AgentArtifact(
        run_id=uuid.uuid4(),
        task_id="financial_review",
        producer="financial_analyst",
        evidence=[evidence],
        claims=[claim],
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert result.rejected_claim_ids == [claim.claim_id]
    violations = result.violations[str(evidence.evidence_id)]
    assert "MISSING_DOCUMENT_LOCATOR:document_version_id" in violations
    assert "MISSING_DOCUMENT_LOCATOR:parse_run_id" in violations
    assert "MISSING_DOCUMENT_LOCATOR:page_number" in violations


def _conflict_and_supplement_artifacts(
    run_id, section_id, version_id, parse_run_id
) -> list[AgentArtifact]:
    dummy = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        content_hash="hash-sec",
        document_version_id=version_id,
        section_id=section_id,
        parse_run_id=parse_run_id,
        page_number=1,
        source_available_at=datetime.now(UTC),
    )
    conflict_claim = AgentClaim(
        category="DATA_CONFLICT",
        statement="正反证据数值不一致，保留双方待裁决。",
        verdict="PARTIALLY_SUPPORTED",
        opposing_evidence_ids=[dummy.evidence_id],
        as_of_date=AS_OF,
        uncertainty_reason="五维冲突判断：数值不一致",
    )
    supplement_claim = AgentClaim(
        category="MISSING_MATERIAL",
        statement="存在补充性材料，建议审查员参阅。",
        verdict="INSUFFICIENT_EVIDENCE",
        opposing_evidence_ids=[dummy.evidence_id],
        as_of_date=AS_OF,
        uncertainty_reason="补充材料，仅供审阅参考",
    )
    return [
        AgentArtifact(
            run_id=run_id,
            task_id="challenge",
            producer="challenger",
            claims=[conflict_claim, supplement_claim],
            evidence=[dummy],
        )
    ]


async def test_auditor_conflict_to_hitl_supplement_not(session):
    """WP3：真冲突送人工复核；补充材料不阻断流程、不送 HITL。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifacts = _conflict_and_supplement_artifacts(
        uuid.uuid4(), section_id, version_id, parse_run_id
    )
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])
    result = await auditor.verify(session, _trusted(), artifacts, snapshot=snapshot)
    conflict_claim, supplement_claim = artifacts[0].claims
    assert result.needs_human_review_claim_ids == [conflict_claim.claim_id]
    assert supplement_claim.claim_id in result.accepted_claim_ids


async def test_auditor_rejects_bad_opposing_evidence_before_hitl(session):
    """坏反证不能绕过审计进入 HITL；引用该反证的 Claim 全部拒绝。"""
    section_id, parse_run_id, version_id = await _make_world(session)
    artifact = _conflict_and_supplement_artifacts(
        uuid.uuid4(), section_id, version_id, parse_run_id
    )[0]
    bad_evidence = artifact.evidence[0].model_copy(update={"parse_run_id": uuid.uuid4()})
    artifact = artifact.model_copy(update={"evidence": [bad_evidence]})
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])

    result = await EvidenceAuditor(FormulaRegistry()).verify(
        session, _trusted(), [artifact], snapshot=snapshot
    )

    assert set(result.rejected_claim_ids) == {claim.claim_id for claim in artifact.claims}
    assert not result.needs_human_review_claim_ids
    assert all(
        "INVALID_OPPOSING_EVIDENCE" in result.violations[str(claim.claim_id)]
        for claim in artifact.claims
    )


async def test_auditor_core_failed_blocks_noncore_degrades(session):
    """WP1/WP3：核心 Agent 失败 -> Claim 全拒；非核心失败 -> DEGRADED 继续。"""
    await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    core = AgentArtifact(
        run_id=uuid.uuid4(),
        task_id="policy_review",
        producer="policy_analyst",
        execution_status="FAILED",
        claims=[
            AgentClaim(
                category="ELIGIBILITY",
                statement="核心结论",
                verdict="SUPPORTED",
                supporting_evidence_ids=[],
                as_of_date=AS_OF,
            )
        ],
    )
    noncore = AgentArtifact(
        run_id=uuid.uuid4(),
        task_id="risk_review",
        producer="risk_analyst",
        execution_status="FAILED",
        claims=[
            AgentClaim(
                category="FINANCIAL",
                statement="风险结论",
                verdict="INSUFFICIENT_EVIDENCE",
                uncertainty_reason="工具不可用",
                as_of_date=AS_OF,
            )
        ],
    )
    result = await auditor.verify(session, _trusted(), [core, noncore])
    assert result.rejected_claim_ids == [core.claims[0].claim_id]
    assert "CORE_AGENT_FAILED:policy_analyst" in result.violations[str(core.claims[0].claim_id)]
    assert result.degraded is True
    assert noncore.claims[0].claim_id not in result.accepted_claim_ids


# ====================== Report Agent ======================


async def _make_report_world(session):
    from creditlens.agents.contracts import AgentEvidenceRef
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )

    await _make_world(session)
    run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
        model_manifest={"degraded": True, "degraded_agents": ["risk", "challenger"]},
    )
    session.add(run)
    await session.flush()

    section_id = uuid.uuid4()
    opposing_section_id = uuid.uuid4()
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        document_version_id=uuid.uuid4(),
        section_id=section_id,
        page_number=3,
        content_hash="hash-loc",
        source_available_at=CUTOFF,
    )
    opposing_evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=opposing_section_id,
        document_version_id=uuid.uuid4(),
        section_id=opposing_section_id,
        page_number=4,
        content_hash="hash-opposing",
        source_available_at=CUTOFF,
    )
    source_claim_id = uuid.uuid4()
    source_claim = AgentClaim(
        category="ELIGIBILITY",
        statement="符合准入条件。",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence.evidence_id],
        opposing_evidence_ids=[opposing_evidence.evidence_id],
        as_of_date=AS_OF,
        source_claim_id=source_claim_id,
    )
    excluded_source_claim = AgentClaim(
        category="FINANCIAL",
        statement="被排除的结论。",
        verdict="SUPPORTED",
        as_of_date=AS_OF,
    )
    source_artifact = AgentArtifact(
        run_id=run.id,
        task_id="policy_review",
        producer="policy_analyst",
        lifecycle_status="VALIDATED",
        claims=[source_claim, excluded_source_claim],
        evidence=[evidence, opposing_evidence],
    )
    persisted_payload = source_artifact.model_dump(mode="json", exclude={"output_hash"})
    artifact = ArtifactRecord(
        id=source_artifact.artifact_id,
        tenant_id=TENANT,
        run_id=run.id,
        task_id=source_artifact.task_id,
        artifact_type=source_artifact.producer,
        producer=source_artifact.producer,
        lifecycle_status=source_artifact.lifecycle_status,
        execution_status=source_artifact.execution_status,
        payload=persisted_payload,
        output_hash=canonical_artifact_payload_hash(persisted_payload),
    )
    session.add(artifact)
    await session.flush()

    claim = ClaimRecord(
        id=source_claim.claim_id,
        tenant_id=TENANT,
        run_id=run.id,
        artifact_id=artifact.id,
        category=source_claim.category,
        statement=source_claim.statement,
        verdict=source_claim.verdict,
        as_of_date=AS_OF,
        review_status="AUDITED",
        payload={
            "supporting_evidence_ids": [str(evidence.evidence_id)],
            "opposing_evidence_ids": [str(opposing_evidence.evidence_id)],
            "calculation_ids": [],
            "source_claim_id": str(source_claim_id),
        },
    )
    session.add(claim)
    # 未通过审计的 Claim：不得进入报告
    session.add(
        ClaimRecord(
            id=excluded_source_claim.claim_id,
            tenant_id=TENANT,
            run_id=run.id,
            artifact_id=artifact.id,
            category=excluded_source_claim.category,
            statement=excluded_source_claim.statement,
            verdict=excluded_source_claim.verdict,
            as_of_date=AS_OF,
            review_status="NEEDS_REWORK",
            payload={
                "supporting_evidence_ids": [],
                "opposing_evidence_ids": [],
                "calculation_ids": [],
                "source_claim_id": None,
            },
        )
    )
    await session.flush()
    return run, claim, section_id, source_claim_id


async def test_report_agent_real_locators_and_source_claim(session):
    """WP3：报告携带真实 Evidence locator（可回原文）与 source_claim_id。"""
    run, claim, section_id, source_claim_id = await _make_report_world(session)
    agent = ReportAgent()
    content = await agent.generate(session, run)

    assert len(content.claims) == 1
    assert content.excluded_claims == 1, "NEEDS_REWORK Claim 不得进入报告"
    entry = content.claims[0]
    assert entry.evidence_locators, "必须携带真实 locator"
    assert entry.evidence_locators[0]["section_id"] == str(section_id)
    assert entry.evidence_locators[0]["page_number"] == 3
    assert entry.opposing_evidence_refs
    assert entry.opposing_evidence_locators[0]["page_number"] == 4
    assert entry.source_claim_id == str(source_claim_id)
    assert content.references and all(
        ref["claim_id"] == str(claim.id) for ref in content.references
    )
    assert {ref["polarity"] for ref in content.references} == {"SUPPORTING", "OPPOSING"}
    assert content.degraded is True
    assert content.degraded_agents == ["risk", "challenger"]
    markdown = agent.render_markdown(content)
    assert "DEGRADED（部分审查覆盖缺失）" in markdown
    assert "risk, challenger" in markdown


async def test_report_agent_rejects_tampered_claim_projection(session):
    """Report generation fails closed if mutable storage diverges from the Artifact."""
    from creditlens.infrastructure.postgres.artifact_integrity import ArtifactIntegrityError

    run, claim, *_ = await _make_report_world(session)
    claim.verdict = "CONTRADICTED"
    await session.flush()

    with pytest.raises(ArtifactIntegrityError, match="PERSISTED_ARTIFACT_INTEGRITY_FAILED"):
        await ReportAgent().generate(session, run)


async def test_report_agent_status_verified_then_approved(session):
    """WP3：自动链路 VERIFIED_DRAFT；人工批准后显式 APPROVED_DRAFT。"""
    run, *_ = await _make_report_world(session)
    agent = ReportAgent()
    content = await agent.generate(session, run)

    draft = await agent.persist(session, run, content)
    assert draft.status == "VERIFIED_DRAFT"
    assert draft.version_no == 1

    approved = await agent.persist(session, run, content, status="APPROVED_DRAFT")
    assert approved.status == "APPROVED_DRAFT"
    assert approved.version_no == 2
    assert approved.content_hash == draft.content_hash, "内容不变则哈希不变"


# ====================== P1：Evidence 闭环 ======================


async def test_evidence_record_persisted_with_parse_run_locator(session):
    """P1：Agent 引用的证据独立落库为 EvidenceRecord，locator 含 parse_run_id。

    报告可从 evidence 表直接追溯到冻结 Snapshot 中的原始段落，
    不必解析 Artifact payload。
    """
    from creditlens.agents.contracts import AgentEvidenceRef
    from creditlens.agents.supervisor import Supervisor
    from creditlens.infrastructure.postgres.models import EvidenceRecord

    await _make_world(session)
    run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    session.add(run)
    await session.flush()

    section_id = uuid.uuid4()
    version_id = uuid.uuid4()
    parse_run_id = uuid.uuid4()
    ref = AgentEvidenceRef(
        evidence_id=uuid.uuid5(uuid.NAMESPACE_URL, f"evidence:{section_id}:hash-x"),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        content_hash="hash-x",
        document_version_id=version_id,
        section_id=section_id,
        parse_run_id=parse_run_id,
        page_number=7,
        source_available_at=CUTOFF,
    )
    artifact = AgentArtifact(
        run_id=run.id,
        task_id="policy_review",
        producer="policy_analyst",
        evidence=[ref],
        claims=[
            AgentClaim(
                category="ELIGIBILITY",
                statement="符合准入条件。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
                as_of_date=AS_OF,
            )
        ],
    )
    # 同一证据被两个 Artifact 引用：EvidenceRecord 只应存在一条
    await Supervisor._persist_evidence(session, run, artifact)
    await Supervisor._persist_evidence(session, run, artifact)

    rows = (
        await session.scalars(select(EvidenceRecord).where(EvidenceRecord.run_id == run.id))
    ).all()
    assert len(rows) == 1, "同一 evidence_id 在同 Run 内不得重复落库"
    record = rows[0]
    assert record.evidence_key == ref.evidence_id
    assert record.section_id == section_id
    assert record.page_number == 7
    assert record.locator["parse_run_id"] == str(parse_run_id), "locator 必须含 parse_run_id"
    assert record.locator["section_id"] == str(section_id)
    assert record.content_hash == "hash-x"


async def test_evidence_key_is_idempotent_per_run_not_global(session):
    """稳定 evidence_key 可跨 Run 复用，但每个 Run 内只落一行。"""
    from creditlens.agents.supervisor import Supervisor
    from creditlens.infrastructure.postgres.models import EvidenceRecord

    await _make_world(session)
    first_run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    second_run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    session.add_all([first_run, second_run])
    await session.flush()

    section_id = uuid.uuid4()
    stable_key = uuid.uuid5(uuid.NAMESPACE_URL, f"evidence:{section_id}:hash-shared")
    ref = AgentEvidenceRef(
        evidence_id=stable_key,
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        content_hash="hash-shared",
        document_version_id=uuid.uuid4(),
        section_id=section_id,
        parse_run_id=uuid.uuid4(),
        page_number=1,
        source_available_at=CUTOFF,
    )

    def artifact_for(run_id):
        return AgentArtifact(
            run_id=run_id,
            task_id="policy_review",
            producer="policy_analyst",
            evidence=[ref],
        )

    await Supervisor._persist_evidence(session, first_run, artifact_for(first_run.id))
    await Supervisor._persist_evidence(session, first_run, artifact_for(first_run.id))
    await Supervisor._persist_evidence(session, second_run, artifact_for(second_run.id))

    rows = (
        await session.scalars(
            select(EvidenceRecord).where(EvidenceRecord.evidence_key == stable_key)
        )
    ).all()
    assert len(rows) == 2
    assert {row.run_id for row in rows} == {first_run.id, second_run.id}
    assert len({row.id for row in rows}) == 2
    assert {row.evidence_key for row in rows} == {stable_key}


async def test_report_locator_carries_parse_run_id_from_evidence_table(session):
    """P1：报告 Locator 带 parse_run_id（来源为落库的 EvidenceRecord）。"""
    from creditlens.infrastructure.postgres.models import EvidenceRecord

    run, claim, section_id, _ = await _make_report_world(session)
    evidence_id = uuid.UUID(claim.payload["supporting_evidence_ids"][0])
    parse_run_id = uuid.uuid4()
    session.add(
        EvidenceRecord(
            evidence_key=evidence_id,
            tenant_id=TENANT,
            run_id=run.id,
            evidence_type="DOCUMENT_SPAN",
            source_id=section_id,
            section_id=section_id,
            page_number=3,
            locator={"section_id": str(section_id), "parse_run_id": str(parse_run_id)},
            content_hash="hash-loc",
            source_available_at=CUTOFF,
        )
    )
    await session.flush()

    content = await ReportAgent().generate(session, run)
    locator = content.claims[0].evidence_locators[0]
    assert locator["parse_run_id"] == str(parse_run_id)
    assert locator["section_id"] == str(section_id)
