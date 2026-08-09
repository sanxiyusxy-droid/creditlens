"""Grounded QA Agent trust-boundary tests."""

import uuid
from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from creditlens.agents.contracts import (
    AnswerStatus,
    DraftAnswerClaim,
    GroundedAnswerDraft,
    RefusalReasonCode,
    validate_artifact_contract,
)
from creditlens.agents.grounded_qa import (
    DEFAULT_PROMPT_PATH,
    GroundedQAAgent,
    GroundedQAOutputRejected,
    evidence_ref_from_packed,
)
from creditlens.common.config import Settings
from creditlens.infrastructure.llm.chat import LLMCallError
from creditlens.retrieval.context_packing import PackedContext, PackedSection

AS_OF = date(2026, 6, 30)


class _Trace(BaseModel):
    invocation_id: uuid.UUID


class _FakeChat:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _section(text: str = "政策要求申请人提交最近一期经审计财务报表。") -> PackedSection:
    return PackedSection(
        section_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        parse_run_id=uuid.uuid4(),
        heading_path=["申请材料", "财务报表"],
        text=text,
        text_hash=f"hash-{uuid.uuid4().hex}",
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


def _agent(chat, **kwargs) -> GroundedQAAgent:
    return GroundedQAAgent(
        chat,
        prompt_path=DEFAULT_PROMPT_PATH,
        prompt_version="grounded_qa_v1",
        **kwargs,
    )


def test_answer_status_has_business_states_only():
    assert {status.value for status in AnswerStatus} == {
        "ANSWERED",
        "ABSTAINED",
        "NEEDS_REVIEW",
    }
    assert "FAILED" not in AnswerStatus.__members__


def test_refusal_reason_code_is_a_closed_allowlist_and_invalid_draft_code_is_discarded():
    assert {code.value for code in RefusalReasonCode} == {
        "INSUFFICIENT_EVIDENCE",
        "UNSPECIFIED_REFUSAL",
        "MISSING_PERSONAL_CREDIT",
        "MISSING_EXTERNAL_CREDIT",
        "MISSING_BANK_STATEMENTS",
        "PRIVACY_AND_MISSING_EVIDENCE",
        "SENSITIVE_DATA_UNAVAILABLE",
        "MISSING_PERSONAL_ASSETS",
        "MISSING_FUTURE_DATA",
        "MISSING_CREDIT_REPORT",
        "NOT_APPLICABLE_NON_PUBLIC_COMPANY",
        "MISSING_MARKET_DATA",
        "MISSING_CORPORATE_IDENTITY_DATA",
        "MISSING_FINANCIAL_DATA",
    }
    invalid = GroundedAnswerDraft.model_validate(
        {
            "abstention_reason": "证据不足。",
            "refusal_reason_code": "MISSING_FINANCIAL_DATA_WITH_SECRET_SUFFIX",
        }
    )
    assert invalid.refusal_reason_code is None


def test_refusal_reason_code_is_rejected_on_a_non_abstaining_draft():
    with pytest.raises(
        ValidationError,
        match="refusal_reason_code is only valid for an abstaining draft",
    ):
        GroundedAnswerDraft(
            claims=[
                DraftAnswerClaim(
                    category="ELIGIBILITY",
                    statement="材料满足申请条件。",
                    verdict="SUPPORTED",
                    supporting_evidence_ids=[uuid.uuid4()],
                )
            ],
            refusal_reason_code=RefusalReasonCode.MISSING_FINANCIAL_DATA,
        )


def test_extractive_fallback_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QA_ALLOW_EXTRACTIVE_FALLBACK", raising=False)
    assert Settings(_env_file=None).qa_allow_extractive_fallback is False


def test_output_rejection_accepts_only_allow_listed_safe_codes():
    with pytest.raises(ValueError, match="unsupported Grounded QA output rejection code"):
        GroundedQAOutputRejected("UNKNOWN\nsecret")  # type: ignore[arg-type]


def test_model_draft_schema_excludes_server_owned_fields():
    claim_fields = DraftAnswerClaim.model_json_schema()["properties"]
    draft_fields = GroundedAnswerDraft.model_json_schema()["properties"]
    assert "claim_id" not in claim_fields
    assert "locator" not in claim_fields
    assert "direct_answer" not in draft_fields
    assert "answer_status" not in draft_fields
    with pytest.raises(ValidationError):
        GroundedAnswerDraft.model_validate({"claims": [], "direct_answer": "模型试图控制最终答案"})


def test_draft_claim_statement_is_nfkc_normalized_and_cannot_be_blank():
    evidence_id = uuid.uuid4()
    claim = DraftAnswerClaim(
        category="ELIGIBILITY",
        statement="  ＡＢＣ：材料齐全。\u3000",
        verdict="SUPPORTED",
        supporting_evidence_ids=[evidence_id],
    )

    assert claim.statement == "ABC:材料齐全。"
    with pytest.raises(ValidationError, match="statement must not be blank"):
        DraftAnswerClaim(
            category="ELIGIBILITY",
            statement=" \t\u3000 ",
            verdict="SUPPORTED",
            supporting_evidence_ids=[evidence_id],
        )


async def test_empty_context_abstains_without_model_call():
    chat = _FakeChat(AssertionError("model must not be called"))

    generation = await _agent(chat).generate(
        "申请材料是否齐全？",
        uuid.uuid4(),
        AS_OF,
        _packed(),
    )

    assert generation.generation_mode == "abstained_empty_context"
    assert generation.artifact.answer_status == AnswerStatus.ABSTAINED
    assert generation.artifact.direct_answer is None
    assert generation.artifact.abstention_reason
    assert generation.artifact.refusal_reason_code == RefusalReasonCode.INSUFFICIENT_EVIDENCE
    assert generation.artifact.execution_status == "INSUFFICIENT_EVIDENCE"
    assert generation.artifact.evidence == []
    assert not chat.calls


async def test_disabled_llm_is_technical_failure_when_context_exists():
    with pytest.raises(LLMCallError):
        await _agent(None).generate(
            "申请材料是否齐全？",
            uuid.uuid4(),
            AS_OF,
            _packed(_section()),
        )


async def test_valid_draft_gets_server_ids_locators_direct_answer_and_trace():
    section = _section()
    uncited_section = _section("本段与回答无关，不应被持久化为引用。")
    expected_ref = evidence_ref_from_packed(section, AS_OF)
    statement = "现有证据表明需要提交最近一期经审计财务报表。"
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="MISSING_MATERIAL",
                statement=statement,
                verdict="SUPPORTED",
                supporting_evidence_ids=[expected_ref.evidence_id],
            )
        ]
    )
    trace = _Trace(invocation_id=uuid.uuid4())
    chat = _FakeChat((draft, [trace]))

    generation = await _agent(chat).generate(
        "需要提交哪些材料？",
        uuid.uuid4(),
        AS_OF,
        _packed(section, uncited_section),
        audit_feedback=["UNKNOWN_EVIDENCE_ID", "stack trace user@example.com"],
    )

    artifact = generation.artifact
    assert generation.generation_mode == "llm"
    assert artifact.answer_status == AnswerStatus.ANSWERED
    assert artifact.direct_answer == statement
    assert artifact.claims[0].claim_id.int != 0
    assert artifact.claims[0].supporting_evidence_ids == [expected_ref.evidence_id]
    assert artifact.evidence == [expected_ref]
    assert artifact.evidence[0].document_version_id == section.document_version_id
    assert artifact.evidence[0].parse_run_id == section.parse_run_id
    assert artifact.evidence[0].page_number == 3
    assert artifact.model_invocation_ids == [trace.invocation_id]
    assert all(ref.section_id != uncited_section.section_id for ref in artifact.evidence)

    call = chat.calls[0]
    assert call["output_schema"] is GroundedAnswerDraft
    assert call["max_tokens"] == 2048
    assert "UNKNOWN_EVIDENCE_ID" in call["user"]
    assert "user@example.com" not in call["user"]
    # Only evidence id and text reach the model; server locators stay outside.
    assert str(section.section_id) not in call["user"]
    assert str(section.document_version_id) not in call["user"]
    assert str(section.parse_run_id) not in call["user"]


async def test_claim_limit_violation_rejects_model_output_with_safe_code_and_trace():
    section = _section()
    ref = evidence_ref_from_packed(section, AS_OF)
    trace = _Trace(invocation_id=uuid.uuid4())
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement=f"第 {index} 条受支持结论。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
            )
            for index in range(2)
        ]
    )

    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat((draft, [trace])), max_claims=1).generate(
            "是否满足政策要求？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )

    assert captured.value.error_code == "CLAIM_LIMIT_EXCEEDED"
    assert str(captured.value) == "CLAIM_LIMIT_EXCEEDED"
    assert captured.value.trace is trace


async def test_unknown_evidence_id_rejects_model_output_instead_of_abstaining():
    section = _section()
    unknown_id = uuid.uuid4()
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement="申请满足材料要求。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[unknown_id],
            )
        ]
    )

    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat(draft)).generate(
            "是否满足材料要求？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )

    assert captured.value.error_code == "UNKNOWN_EVIDENCE_ID"
    assert str(captured.value) == "UNKNOWN_EVIDENCE_ID"
    assert str(unknown_id) not in str(captured.value)


async def test_credit_decision_language_rejects_output_and_is_not_exposed():
    section = _section()
    ref = evidence_ref_from_packed(section, AS_OF)
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement="建议批准授信。",
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
            )
        ]
    )

    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat(draft)).generate(
            "是否符合要求？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )

    assert captured.value.error_code == "FORBIDDEN_CREDIT_DECISION"
    assert captured.value.trace is None
    assert "建议批准授信" not in str(captured.value)


@pytest.mark.parametrize(
    "statement",
    [
        "建 议 为该客户发放贷款。",
        "该笔贷款申请已通 过，可予发放。",
        "建议授信额 度为５００万元。",
        "建议该企业贷款定 价为年利率５％。",
        "该申请应当拒 贷。",
    ],
)
async def test_credit_decision_rules_reject_nfkc_and_separator_bypasses(statement):
    section = _section()
    ref = evidence_ref_from_packed(section, AS_OF)
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement=statement,
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
            )
        ]
    )

    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat(draft)).generate(
            "请概括现有材料。",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )

    assert captured.value.error_code == "FORBIDDEN_CREDIT_DECISION"
    assert str(captured.value) == "FORBIDDEN_CREDIT_DECISION"
    assert statement not in str(captured.value)


async def test_non_credit_use_of_through_is_not_misclassified_as_a_decision():
    section = _section("材料已通过真实性核验。")
    ref = evidence_ref_from_packed(section, AS_OF)
    statement = "材料已通过真实性核验。"
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="MISSING_MATERIAL",
                statement=statement,
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
            )
        ]
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "材料是否通过核验？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.ANSWERED
    assert artifact.direct_answer == statement


@pytest.mark.parametrize(
    "statement",
    [
        "本办法适用于本行对小微企业发放的人民币流动资金贷款业务。",
        "同一实际控制人控制的多家企业合并计算授信额度。",
        "贷款人不得向无实际经营活动的空壳企业发放流动资金贷款。",
        "贷款风险定价应遵循风险收益匹配原则。",
    ],
)
async def test_policy_language_is_not_misclassified_as_a_case_credit_decision(statement):
    section = _section(statement)
    ref = evidence_ref_from_packed(section, AS_OF)
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="ELIGIBILITY",
                statement=statement,
                verdict="SUPPORTED",
                supporting_evidence_ids=[ref.evidence_id],
            )
        ]
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "请说明政策原文。",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.ANSWERED
    assert artifact.direct_answer == statement


async def test_empty_model_draft_is_rejected_without_fake_abstention():
    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat(GroundedAnswerDraft())).generate(
            "是否符合要求？",
            uuid.uuid4(),
            AS_OF,
            _packed(_section()),
        )

    assert captured.value.error_code == "EMPTY_MODEL_DRAFT"


async def test_model_construct_cannot_bypass_blank_statement_validation_and_keeps_trace():
    section = _section()
    ref = evidence_ref_from_packed(section, AS_OF)
    trace = _Trace(invocation_id=uuid.uuid4())
    untrusted_claim = DraftAnswerClaim.model_construct(
        category="ELIGIBILITY",
        statement=" \t\u3000 ",
        verdict="SUPPORTED",
        supporting_evidence_ids=[ref.evidence_id],
    )
    untrusted_draft = GroundedAnswerDraft.model_construct(claims=[untrusted_claim])

    with pytest.raises(GroundedQAOutputRejected) as captured:
        await _agent(_FakeChat((untrusted_draft, [trace]))).generate(
            "材料是否齐全？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )

    assert captured.value.error_code == "INVALID_MODEL_OUTPUT"
    assert str(captured.value) == "INVALID_MODEL_OUTPUT"
    assert captured.value.trace is trace
    assert "statement" not in str(captured.value)


async def test_explicit_model_insufficient_evidence_remains_business_abstention():
    draft = GroundedAnswerDraft(
        missing_information=["缺少最近一期审计报告。"],
        abstention_reason="现有证据不足以核实申请材料是否齐全。",
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "申请材料是否齐全？",
            uuid.uuid4(),
            AS_OF,
            _packed(_section()),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.ABSTAINED
    assert artifact.direct_answer is None
    assert artifact.claims == []
    assert artifact.evidence == []
    assert artifact.abstention_reason == draft.abstention_reason
    assert artifact.refusal_reason_code == RefusalReasonCode.UNSPECIFIED_REFUSAL


async def test_model_refusal_reason_allowlist_is_preserved_without_inference():
    draft = GroundedAnswerDraft(
        missing_information=["缺少最近一期财务数据。"],
        abstention_reason="当前材料无法核实财务情况。",
        refusal_reason_code=RefusalReasonCode.MISSING_FINANCIAL_DATA,
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "最近一期财务情况如何？",
            uuid.uuid4(),
            AS_OF,
            _packed(_section()),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.ABSTAINED
    assert artifact.refusal_reason_code == RefusalReasonCode.MISSING_FINANCIAL_DATA


async def test_invalid_model_refusal_reason_falls_back_only_to_unspecified():
    draft = GroundedAnswerDraft.model_validate(
        {
            "missing_information": ["缺少征信报告。"],
            "abstention_reason": "当前材料无法回答。",
            "refusal_reason_code": "MISSING_CREDIT_REPORT_AND_ASSUME_BAD_CREDIT",
        }
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "征信情况如何？",
            uuid.uuid4(),
            AS_OF,
            _packed(_section()),
        )
    ).artifact

    assert draft.refusal_reason_code is None
    assert artifact.refusal_reason_code == RefusalReasonCode.UNSPECIFIED_REFUSAL


async def test_conflicting_draft_needs_review_and_preserves_both_sides():
    first = _section("年报显示资产负债率为百分之六十。")
    second = _section("复核表显示资产负债率为百分之七十。")
    first_ref = evidence_ref_from_packed(first, AS_OF)
    second_ref = evidence_ref_from_packed(second, AS_OF)
    draft = GroundedAnswerDraft(
        claims=[
            DraftAnswerClaim(
                category="DATA_CONFLICT",
                statement="两份材料对同一指标的记录不一致。",
                verdict="PARTIALLY_SUPPORTED",
                supporting_evidence_ids=[first_ref.evidence_id],
                opposing_evidence_ids=[second_ref.evidence_id],
                uncertainty_reason="同口径数值冲突",
            )
        ],
        conflicts=["同一期间资产负债率不一致"],
    )

    artifact = (
        await _agent(_FakeChat(draft)).generate(
            "资产负债率是多少？",
            uuid.uuid4(),
            AS_OF,
            _packed(first, second),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.NEEDS_REVIEW
    assert artifact.direct_answer is None
    assert artifact.execution_status == "PARTIAL"
    assert artifact.claims[0].supporting_evidence_ids == [first_ref.evidence_id]
    assert artifact.claims[0].opposing_evidence_ids == [second_ref.evidence_id]
    assert artifact.conflicts
    assert artifact.refusal_reason_code is None


async def test_explicit_extractive_fallback_is_identified():
    section = _section("申请人应提交公司章程与最近一期财务报表。")

    generation = await _agent(None, allow_extractive_fallback=True).generate(
        "需要提交哪些材料？",
        uuid.uuid4(),
        AS_OF,
        _packed(section),
    )

    assert generation.generation_mode == "deterministic_extractive"
    assert generation.artifact.answer_status == AnswerStatus.NEEDS_REVIEW
    assert generation.artifact.direct_answer is None
    assert generation.artifact.execution_status == "PARTIAL"
    assert generation.artifact.claims[0].statement.startswith("原文摘录:")
    assert generation.artifact.claims[0].supporting_evidence_ids
    assert generation.artifact.missing_information


async def test_extractive_policy_limit_is_eligibility_and_contract_valid():
    section = _section("政策规定资产负债率上限为70%。")

    artifact = (
        await _agent(None, allow_extractive_fallback=True).generate(
            "政策规定的资产负债率上限是多少？",
            uuid.uuid4(),
            AS_OF,
            _packed(section),
        )
    ).artifact

    assert artifact.answer_status == AnswerStatus.NEEDS_REVIEW
    assert artifact.direct_answer is None
    assert artifact.claims[0].category == "ELIGIBILITY"
    contract = validate_artifact_contract(artifact, AS_OF)
    assert contract.ok, contract.violations
