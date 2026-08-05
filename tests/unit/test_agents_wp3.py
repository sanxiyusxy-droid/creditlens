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
from sqlalchemy import select

from creditlens.agents.auditor import EvidenceAuditor
from creditlens.agents.challenger import Challenger, assess_conflict
from creditlens.agents.contracts import AgentArtifact, AgentClaim, AgentEvidenceRef
from creditlens.agents.report_agent import ReportAgent
from creditlens.agents.risk_agent import RiskAgent
from creditlens.application.snapshot_service import SnapshotContext
from creditlens.formulas.engine import CalculationArtifact, FormulaRegistry
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    ClaimRecord,
    CreditCase,
    Document,
    DocumentSection,
    DocumentVersion,
    Entity,
    ParseRun,
    ReviewRun,
    Tenant,
)
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

    async def invoke(self, role, tool, **kwargs):
        self.calls.append(tool)
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
    artifact = await agent.run(uuid.uuid4(), "risk_review", _trusted())
    assert agent.threshold_version == "risk-thresholds-v1"
    assert {"risk_threshold_version": "risk-thresholds-v1"} in artifact.unresolved_issues
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
                "heading_path": ["第四章", "客户集中度"],
                "text": "第一大客户收入占比超过 40%，存在集中度风险。",
                "text_hash": "hash-packed",
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
                "heading_path": ["附注"],
                "text": "复核口径下2025年资产负债率为 70%。",
                "text_hash": "hash-packed",
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
    await session.flush()
    return section_id, parse_run_id, version_id


def _span_artifact(run_id, section_id, text_hash="hash-sec") -> AgentArtifact:
    evidence = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=section_id,
        content_hash=text_hash,
        section_id=section_id,
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


async def test_auditor_rejects_when_case_missing(session):
    """WP3：Case 不存在/租户不一致 -> 全部 Claim 拒绝。"""
    await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), uuid.uuid4())
    result = await auditor.verify(session, _trusted(case_id=uuid.uuid4()), [artifact])
    assert result.rejected_claim_ids == [artifact.claims[0].claim_id]
    assert "CASE_VALIDATION_FAILED" in result.violations[str(artifact.claims[0].claim_id)]


async def test_auditor_rejects_material_after_cutoff(session):
    """WP3：cutoff 之后才可获得的材料必须拒绝（NOT_AVAILABLE_AT_CUTOFF）。"""
    section_id, _, _ = await _make_world(session, available_at=datetime(2026, 6, 1, tzinfo=UTC))
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id)
    result = await auditor.verify(session, _trusted(), [artifact])
    assert result.rejected_claim_ids, "cutoff 后材料不得进入报告"
    violations = [v for vs in result.violations.values() for v in vs]
    assert "NOT_AVAILABLE_AT_CUTOFF" in violations


async def test_auditor_rejects_parse_run_outside_snapshot(session):
    """WP3：证据 Parse Run 不在冻结 Snapshot 集合内必须拒绝。"""
    section_id, _, _ = await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[uuid.uuid4()])
    result = await auditor.verify(session, _trusted(), [artifact], snapshot=snapshot)
    assert result.rejected_claim_ids
    violations = [v for vs in result.violations.values() for v in vs]
    assert "PARSE_RUN_NOT_IN_SNAPSHOT" in violations


async def test_auditor_passes_valid_evidence_within_snapshot(session):
    """合法证据 + Snapshot 覆盖 -> 接受。"""
    section_id, parse_run_id, _ = await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifact = _span_artifact(uuid.uuid4(), section_id)
    snapshot = SnapshotContext(snapshot_id=uuid.uuid4(), allowed_parse_run_ids=[parse_run_id])
    result = await auditor.verify(session, _trusted(), [artifact], snapshot=snapshot)
    assert result.accepted_claim_ids == [artifact.claims[0].claim_id]
    assert not result.needs_human_review_claim_ids


def _conflict_and_supplement_artifacts(run_id) -> list[AgentArtifact]:
    dummy = AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type="DOCUMENT_SPAN",
        source_id=uuid.uuid4(),
        content_hash="hash-none",
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
    await _make_world(session)
    auditor = EvidenceAuditor(FormulaRegistry())
    artifacts = _conflict_and_supplement_artifacts(uuid.uuid4())
    result = await auditor.verify(session, _trusted(), artifacts)
    conflict_claim, supplement_claim = artifacts[0].claims
    assert result.needs_human_review_claim_ids == [conflict_claim.claim_id]
    assert supplement_claim.claim_id in result.accepted_claim_ids


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
    await _make_world(session)
    run = ReviewRun(
        tenant_id=TENANT,
        case_id=CASE,
        as_of_date=AS_OF,
        decision_cutoff_at=CUTOFF,
    )
    session.add(run)
    await session.flush()

    evidence_id = uuid.uuid4()
    section_id = uuid.uuid4()
    artifact = ArtifactRecord(
        tenant_id=TENANT,
        run_id=run.id,
        task_id="policy_review",
        artifact_type="AGENT",
        producer="policy_analyst",
        payload={
            "evidence": [
                {
                    "evidence_id": str(evidence_id),
                    "evidence_type": "DOCUMENT_SPAN",
                    "document_version_id": "ver-1",
                    "section_id": str(section_id),
                    "page_number": 3,
                    "content_hash": "hash-loc",
                }
            ]
        },
    )
    session.add(artifact)
    await session.flush()

    source_claim_id = uuid.uuid4()
    claim = ClaimRecord(
        tenant_id=TENANT,
        run_id=run.id,
        artifact_id=artifact.id,
        category="ELIGIBILITY",
        statement="符合准入条件。",
        verdict="SUPPORTED",
        as_of_date=AS_OF,
        review_status="AUDITED",
        payload={
            "supporting_evidence_ids": [str(evidence_id)],
            "source_claim_id": str(source_claim_id),
        },
    )
    session.add(claim)
    # 未通过审计的 Claim：不得进入报告
    session.add(
        ClaimRecord(
            tenant_id=TENANT,
            run_id=run.id,
            artifact_id=artifact.id,
            category="FINANCIAL",
            statement="被排除的结论。",
            verdict="SUPPORTED",
            as_of_date=AS_OF,
            review_status="NEEDS_REWORK",
            payload={},
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
    assert entry.source_claim_id == str(source_claim_id)
    assert content.references and content.references[0]["claim_id"] == str(claim.id)


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
