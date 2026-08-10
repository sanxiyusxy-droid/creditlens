"""Grounded QA Agent: turn audited packed evidence into a bounded answer artifact.

The language model may only draft claims and evidence ids.  This module owns the
security boundary around that draft: evidence ids are whitelisted, locators and
claim ids are reconstructed from server input, answer state is deterministic,
and decision-making language is blocked before a direct answer is exposed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from creditlens.agents.contracts import (
    AgentClaim,
    AgentClaimCategory,
    AgentEvidenceRef,
    AnswerStatus,
    DraftAnswerClaim,
    GroundedAnswerArtifact,
    GroundedAnswerDraft,
    RefusalReasonCode,
)
from creditlens.infrastructure.llm.chat import LLMCallError, ModelInvocationTrace
from creditlens.retrieval.context_packing import PackedContext, PackedSection

AGENT_ROLE = "grounded_qa"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "prompts" / "grounded_qa_v1.yaml"
)

_GENERATION_MODES = Literal["llm", "deterministic_extractive", "abstained_empty_context"]
_SAFE_VIOLATION_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
GroundedQAOutputErrorCode = Literal[
    "CLAIM_LIMIT_EXCEEDED",
    "EMPTY_MODEL_DRAFT",
    "FORBIDDEN_CREDIT_DECISION",
    "INVALID_MODEL_OUTPUT",
    "UNKNOWN_EVIDENCE_ID",
]
_OUTPUT_REJECTION_CODES = frozenset(
    {
        "CLAIM_LIMIT_EXCEEDED",
        "EMPTY_MODEL_DRAFT",
        "FORBIDDEN_CREDIT_DECISION",
        "INVALID_MODEL_OUTPUT",
        "UNKNOWN_EVIDENCE_ID",
    }
)
_DEFAULT_REPAIR_HINT = "仅依据允许的 evidence 重新生成符合结构契约的 Claim。"
_REPAIR_HINTS = {
    "ARTIFACT_EVIDENCE_NOT_EXACTLY_CITED": "只保留被 Claim 明确引用的 allowed evidence。",
    "CITATION_NOT_IN_PACKED_CONTEXT": "移除该引用，或改用 allowed evidence 中存在的 evidence_id。",
    "CLAIM_LIMIT_EXCEEDED": "减少 Claim 数量，不得超过服务端给出的上限。",
    "DIRECT_ANSWER_NOT_EXACT_CLAIM_RENDERING": "只修复 Claim；direct_answer 由服务端生成。",
    "EMPTY_MODEL_DRAFT": "生成至少一条有证据的 Claim，或明确填写证据不足信息。",
    "EVIDENCE_NOT_IN_PACKED_CONTEXT": "移除该引用，或改用 allowed evidence 中存在的 evidence_id。",
    "FORBIDDEN_CREDIT_DECISION": "删除授信审批、额度或定价决定，只陈述证据支持的事实。",
    "INVALID_MODEL_OUTPUT": "严格按照输出 Schema 重新生成，不增加未声明字段。",
    "NUMERIC_TOKEN_NOT_IN_CITATION": "数字或日期只能来自所选 supporting evidence；否则删除或改为证据不足。",
    "SUPPORTED_WITHOUT_EVIDENCE": "为 SUPPORTED Claim 选择 allowed supporting evidence，或修改 verdict。",
    "UNKNOWN_EVIDENCE": "移除未知引用，或改用 allowed evidence 中存在的 evidence_id。",
    "UNKNOWN_EVIDENCE_ID": "移除未知引用，或改用 allowed evidence 中存在的 evidence_id。",
}
_FORBIDDEN_DECISION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        # Recommendation intent followed by a credit decision or pricing term.
        r"建议.{0,24}(?:批准|核准|通过|发放|授信|贷款|额度|拒贷|拒绝|定价|利率)",
        # Case/application language is required for a decision outcome.  Plain
        # policy text such as “向企业发放流动资金贷款” must remain answerable.
        r"(?:该|本次|本笔|此笔)?(?:贷款|授信|融资)?申请.{0,12}(?:通过|批准|核准|拒绝|拒贷|不予|可予发放|予以发放)",
        r"(?:该|本次|本笔|此笔)?(?:申请|客户|企业).{0,12}(?:应当|建议|决定|拟|予以|不予|可予).{0,12}(?:批准|核准|通过|发放|授信|贷款|额度|拒贷|拒绝|定价|利率)",
        r"(?:批准|同意|核准|拒绝|拒贷|不予|予以发放)(?:该|本次|本笔|此笔)?(?:授信|贷款|融资|申请)",
        # A concrete case-level limit assignment is prohibited; references to
        # policy concepts such as “合并计算授信额度” are not decisions.
        r"授信额度(?:建议|核定|确定|设置|设定|调整|维持|上调|下调|为|是|[:：]|人民币|\d)",
        r"(?:给予|核定|确定|设置|设定|调整|维持|上调|下调).{0,8}(?:额度|利率|定价)",
        r"(?:贷款|授信|客户|企业).{0,12}(?:定价|利率).{0,8}(?:为|建议|确定|设定|调整)",
        r"(?:该|本次|本笔|此笔)?申请.{0,8}(?:应当|建议|决定|拟|予以|不予).{0,8}(?:拒贷|批贷|放款|最终审批)",
    )
)


class GroundedQAOutputRejected(RuntimeError):
    """Reject unsafe/unusable model output without exposing model-authored text."""

    def __init__(
        self,
        error_code: GroundedQAOutputErrorCode,
        trace: Any | None = None,
    ) -> None:
        if error_code not in _OUTPUT_REJECTION_CODES:
            raise ValueError("unsupported Grounded QA output rejection code")
        self.error_code = error_code
        self.trace = trace
        super().__init__(error_code)


def grounded_qa_repair_hint(code: str) -> str:
    """Return one fixed server-owned hint for a stable violation code."""
    if not _SAFE_VIOLATION_CODE.fullmatch(code):
        raise ValueError("invalid Grounded QA repair code")
    return _REPAIR_HINTS.get(code, _DEFAULT_REPAIR_HINT)


class GroundedQAAuditFeedback(BaseModel):
    """Prompt-safe repair locator built only from model-visible fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: Literal["ANSWER", "CLAIM", "EVIDENCE"]
    code: str
    repair_hint: str
    category: AgentClaimCategory | None = None
    supporting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    opposing_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    evidence_id: uuid.UUID | None = None
    apply_to: Literal["MATCHING_CLAIM", "ALL_MATCHING_CLAIMS"] | None = None

    @model_validator(mode="after")
    def validate_safe_shape(self) -> GroundedQAAuditFeedback:
        if not _SAFE_VIOLATION_CODE.fullmatch(self.code):
            raise ValueError("invalid Grounded QA repair code")
        if self.repair_hint != grounded_qa_repair_hint(self.code):
            raise ValueError("repair_hint must be server-owned")
        if self.scope == "CLAIM":
            if self.category is None or self.apply_to is None or self.evidence_id is not None:
                raise ValueError("CLAIM feedback requires a safe claim locator")
        elif self.scope == "EVIDENCE":
            if (
                self.evidence_id is None
                or self.category is not None
                or self.supporting_evidence_ids
                or self.opposing_evidence_ids
                or self.apply_to is not None
            ):
                raise ValueError("EVIDENCE feedback requires only evidence_id")
        elif (
            self.category is not None
            or self.supporting_evidence_ids
            or self.opposing_evidence_ids
            or self.evidence_id is not None
            or self.apply_to is not None
        ):
            raise ValueError("ANSWER feedback must not carry a subject locator")
        return self


class GroundedQAGeneration(BaseModel):
    """Agent return envelope ready for the service-level trace/audit loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifact: GroundedAnswerArtifact
    model_traces: list[Any] = Field(default_factory=list)
    generation_mode: _GENERATION_MODES


def evidence_ref_from_packed(section: PackedSection, as_of_date: date) -> AgentEvidenceRef:
    """Build the immutable locator without trusting or involving the model."""
    evidence_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"evidence:{section.section_id}:{section.text_hash}",
    )
    return AgentEvidenceRef(
        evidence_id=evidence_id,
        evidence_type="DOCUMENT_SPAN",
        source_id=section.section_id,
        content_hash=section.text_hash,
        document_version_id=section.document_version_id,
        section_id=section.section_id,
        parse_run_id=section.parse_run_id,
        page_number=section.page_start,
        # PackedContext has already passed cutoff checks but does not carry the
        # source timestamp.  The database remains authoritative during audit.
        source_available_at=datetime.combine(as_of_date, time.min, tzinfo=UTC),
    )


class GroundedQAAgent:
    """Generate grounded claims while keeping all trust decisions server-side."""

    def __init__(
        self,
        chat,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        prompt_version: str = "grounded_qa_v1",
        max_claims: int = 6,
        max_tokens: int = 2048,
        allow_extractive_fallback: bool = False,
    ):
        if max_claims < 1:
            raise ValueError("max_claims must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        self._chat = chat
        self._prompt_path = Path(prompt_path)
        self.prompt_version = prompt_version
        self.max_claims = max_claims
        self.max_tokens = max_tokens
        self.allow_extractive_fallback = allow_extractive_fallback
        self._system_prompt, self._user_template = self._load_prompt()

    def _load_prompt(self) -> tuple[str, str]:
        try:
            payload = yaml.safe_load(self._prompt_path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ValueError(f"unable to load Grounded QA prompt: {self._prompt_path}") from exc

        version = str(payload.get("version", ""))
        if version != self.prompt_version:
            raise ValueError(
                f"prompt version mismatch: expected {self.prompt_version}, found {version or 'missing'}"
            )
        system = payload.get("system")
        user_template = payload.get("user_template")
        if not isinstance(system, str) or not system.strip():
            raise ValueError("Grounded QA prompt is missing system text")
        if not isinstance(user_template, str) or not user_template.strip():
            raise ValueError("Grounded QA prompt is missing user_template")
        for placeholder in ("{question}", "{evidence_block}", "{audit_feedback}", "{max_claims}"):
            if placeholder not in user_template:
                raise ValueError(f"Grounded QA user_template is missing {placeholder}")
        return system.strip(), user_template

    async def generate(
        self,
        question: str,
        run_id: uuid.UUID,
        as_of_date: date,
        packed: PackedContext,
        audit_feedback: list[Any] | None = None,
    ) -> GroundedQAGeneration:
        """Generate once; the service may call once more with sanitized audit codes."""
        if not question.strip():
            raise ValueError("question must not be empty")

        refs = self._evidence_refs(packed, as_of_date)
        if not refs:
            artifact = self._abstained_artifact(
                run_id,
                reason="未提供可用于回答的经审计证据。",
                refusal_reason_code=RefusalReasonCode.INSUFFICIENT_EVIDENCE,
                missing_information=["缺少与问题相关的经审计证据"],
            )
            return GroundedQAGeneration(
                artifact=artifact,
                generation_mode="abstained_empty_context",
            )

        if self._chat is None:
            if not self.allow_extractive_fallback:
                raise LLMCallError(
                    "Grounded QA requires an enabled chat provider",
                    self._disabled_provider_trace(question, refs),
                )
            draft = self._extractive_draft(question, packed, refs)
            return self._materialize(
                run_id,
                as_of_date,
                refs,
                draft,
                model_traces=[],
                generation_mode="deterministic_extractive",
            )

        user_prompt = self._build_user_prompt(question, packed, refs, audit_feedback)
        kwargs = {
            "system": self._system_prompt,
            "user": user_prompt,
            "output_schema": GroundedAnswerDraft,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt_version,
        }
        if hasattr(self._chat, "generate_structured_traced"):
            raw = await self._chat.generate_structured_traced(**kwargs)
        else:
            raw = await self._chat.generate_structured(**kwargs)
        draft, traces = self._unpack_model_result(raw)
        return self._materialize(
            run_id,
            as_of_date,
            refs,
            draft,
            model_traces=traces,
            generation_mode="llm",
        )

    @staticmethod
    def _evidence_refs(packed: PackedContext, as_of_date: date) -> list[AgentEvidenceRef]:
        unique: dict[uuid.UUID, AgentEvidenceRef] = {}
        for section in packed.sections:
            ref = evidence_ref_from_packed(section, as_of_date)
            unique.setdefault(ref.evidence_id, ref)
        return list(unique.values())

    def _build_user_prompt(
        self,
        question: str,
        packed: PackedContext,
        refs: list[AgentEvidenceRef],
        audit_feedback: list[Any] | None,
    ) -> str:
        ref_by_section = {ref.section_id: ref for ref in refs}
        evidence_payload = []
        for section in packed.sections:
            ref = ref_by_section.get(section.section_id)
            if ref is None:
                continue
            evidence_payload.append(
                {
                    "evidence_id": str(ref.evidence_id),
                    "heading_path": section.heading_path,
                    "text": section.text,
                }
            )
        safe_codes = self._safe_audit_codes(audit_feedback)
        return self._user_template.format(
            question=self._safe_json(question),
            evidence_block=self._safe_json(evidence_payload),
            audit_feedback=self._safe_json(safe_codes),
            max_claims=self.max_claims,
        )

    @staticmethod
    def _safe_json(value: Any) -> str:
        # Escaping angle brackets prevents untrusted text from closing the
        # delimiters used in the prompt while retaining exact semantic content.
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )

    @staticmethod
    def _safe_audit_codes(feedback: list[Any] | None) -> list[dict[str, Any]]:
        """Validate prompt feedback while dropping text, values and unknown fields."""
        safe: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in feedback or []:
            try:
                if isinstance(raw, GroundedQAAuditFeedback):
                    item = raw
                elif isinstance(raw, dict):
                    item = GroundedQAAuditFeedback.model_validate(raw)
                else:
                    code = str(raw).strip()
                    item = GroundedQAAuditFeedback(
                        scope="ANSWER",
                        code=code,
                        repair_hint=grounded_qa_repair_hint(code),
                    )
            except (ValidationError, ValueError):
                continue
            payload = item.model_dump(mode="json", exclude_none=True)
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if canonical in seen:
                continue
            safe.append(payload)
            seen.add(canonical)
            if len(safe) >= 20:
                break
        return safe

    @staticmethod
    def _unpack_model_result(raw: Any) -> tuple[GroundedAnswerDraft, list[Any]]:
        traces: list[Any] = []
        draft_raw = raw
        if hasattr(raw, "output") and hasattr(raw, "trace"):
            draft_raw = raw.output
            traces = [raw.trace]
        elif isinstance(raw, tuple) and len(raw) == 2:
            draft_raw, raw_traces = raw
            if raw_traces is not None:
                traces = list(raw_traces) if isinstance(raw_traces, list | tuple) else [raw_traces]
        try:
            # Revalidate even an apparent GroundedAnswerDraft.  A provider or
            # test double can construct one with model_construct(), bypassing
            # every schema validator at the model trust boundary.
            payload = (
                draft_raw.model_dump(mode="python")
                if isinstance(draft_raw, GroundedAnswerDraft)
                else draft_raw
            )
            draft = GroundedAnswerDraft.model_validate(payload)
        except ValidationError:
            raise GroundedQAOutputRejected(
                "INVALID_MODEL_OUTPUT",
                trace=GroundedQAAgent._rejection_trace(traces),
            ) from None
        return draft, traces

    def _materialize(
        self,
        run_id: uuid.UUID,
        as_of_date: date,
        refs: list[AgentEvidenceRef],
        draft: GroundedAnswerDraft,
        *,
        model_traces: list[Any],
        generation_mode: _GENERATION_MODES,
    ) -> GroundedQAGeneration:
        try:
            # Defense in depth for direct/internal callers that bypass
            # _unpack_model_result with a model_construct() instance.
            draft = GroundedAnswerDraft.model_validate(draft.model_dump(mode="python"))
        except ValidationError:
            raise GroundedQAOutputRejected(
                "INVALID_MODEL_OUTPUT",
                trace=self._rejection_trace(model_traces),
            ) from None

        if len(draft.claims) > self.max_claims:
            raise GroundedQAOutputRejected(
                "CLAIM_LIMIT_EXCEEDED",
                trace=self._rejection_trace(model_traces),
            )

        allowed_ids = {ref.evidence_id for ref in refs}
        cited_ids = {
            evidence_id
            for claim in draft.claims
            for evidence_id in (claim.supporting_evidence_ids + claim.opposing_evidence_ids)
        }
        if not cited_ids.issubset(allowed_ids):
            raise GroundedQAOutputRejected(
                "UNKNOWN_EVIDENCE_ID",
                trace=self._rejection_trace(model_traces),
            )

        if self._contains_forbidden_decision_language(draft):
            raise GroundedQAOutputRejected(
                "FORBIDDEN_CREDIT_DECISION",
                trace=self._rejection_trace(model_traces),
            )

        explicitly_insufficient = bool(
            draft.abstention_reason
            or draft.missing_information
            or any(claim.verdict == "INSUFFICIENT_EVIDENCE" for claim in draft.claims)
        )
        if not draft.claims and not explicitly_insufficient:
            raise GroundedQAOutputRejected(
                "EMPTY_MODEL_DRAFT",
                trace=self._rejection_trace(model_traces),
            )

        claims = [self._materialize_claim(claim, as_of_date) for claim in draft.claims]
        supported = [claim for claim in claims if claim.verdict == "SUPPORTED"]
        insufficient_only = bool(claims) and all(
            claim.verdict == "INSUFFICIENT_EVIDENCE" for claim in claims
        )
        has_conflict_signal = bool(
            draft.conflicts
            or any(claim.verdict != "SUPPORTED" or claim.opposing_evidence_ids for claim in claims)
        )

        if draft.abstention_reason:
            answer_status = AnswerStatus.ABSTAINED
            direct_answer = None
            abstention_reason = draft.abstention_reason
            execution_status = "INSUFFICIENT_EVIDENCE"
        elif insufficient_only:
            answer_status = AnswerStatus.ABSTAINED
            direct_answer = None
            abstention_reason = "模型明确指出现有证据不足，无法形成受支持结论。"
            execution_status = "INSUFFICIENT_EVIDENCE"
        elif has_conflict_signal and claims:
            answer_status = AnswerStatus.NEEDS_REVIEW
            direct_answer = None
            abstention_reason = None
            execution_status = "PARTIAL"
        elif supported and not draft.missing_information:
            answer_status = AnswerStatus.ANSWERED
            direct_answer = " ".join(claim.statement.strip() for claim in supported)
            abstention_reason = None
            execution_status = "SUCCESS"
        elif supported and draft.missing_information:
            answer_status = AnswerStatus.NEEDS_REVIEW
            direct_answer = None
            abstention_reason = None
            execution_status = "PARTIAL"
        else:
            answer_status = AnswerStatus.ABSTAINED
            direct_answer = None
            abstention_reason = "现有证据不足以形成可直接回答的受支持结论。"
            execution_status = "INSUFFICIENT_EVIDENCE"

        # Packed context is a model input, not an implicit citation list.  The
        # artifact persists only the locators referenced by retained claims.
        cited_refs = [ref for ref in refs if ref.evidence_id in cited_ids]
        if answer_status == AnswerStatus.ABSTAINED:
            claims = []
            cited_refs = []
            refusal_reason_code = draft.refusal_reason_code or RefusalReasonCode.UNSPECIFIED_REFUSAL
        else:
            # A model cannot attach a refusal label to an answer or review state.
            refusal_reason_code = None

        artifact = GroundedAnswerArtifact(
            run_id=run_id,
            task_id="grounded_qa",
            producer=AGENT_ROLE,
            execution_status=execution_status,
            claims=claims,
            evidence=cited_refs,
            answer_status=answer_status,
            direct_answer=direct_answer,
            missing_information=draft.missing_information,
            conflicts=draft.conflicts,
            abstention_reason=abstention_reason,
            refusal_reason_code=refusal_reason_code,
            prompt_version=self.prompt_version,
            model_invocation_ids=self._trace_ids(model_traces),
        )
        return GroundedQAGeneration(
            artifact=artifact,
            model_traces=model_traces,
            generation_mode=generation_mode,
        )

    @staticmethod
    def _materialize_claim(draft: DraftAnswerClaim, as_of_date: date) -> AgentClaim:
        # claim_id is deliberately absent from DraftAnswerClaim and generated by
        # AgentClaim here, after the model output has crossed the trust boundary.
        return AgentClaim(
            category=draft.category,
            statement=draft.statement.strip(),
            verdict=draft.verdict,
            supporting_evidence_ids=list(dict.fromkeys(draft.supporting_evidence_ids)),
            opposing_evidence_ids=list(dict.fromkeys(draft.opposing_evidence_ids)),
            as_of_date=as_of_date,
            uncertainty_reason=draft.uncertainty_reason,
        )

    def _abstained_artifact(
        self,
        run_id: uuid.UUID,
        *,
        reason: str,
        refusal_reason_code: RefusalReasonCode,
        missing_information: list[str] | None = None,
        model_traces: list[Any] | None = None,
    ) -> GroundedAnswerArtifact:
        return GroundedAnswerArtifact(
            run_id=run_id,
            task_id="grounded_qa",
            producer=AGENT_ROLE,
            execution_status="INSUFFICIENT_EVIDENCE",
            claims=[],
            evidence=[],
            answer_status=AnswerStatus.ABSTAINED,
            direct_answer=None,
            missing_information=missing_information or [],
            conflicts=[],
            abstention_reason=reason,
            refusal_reason_code=refusal_reason_code,
            prompt_version=self.prompt_version,
            model_invocation_ids=self._trace_ids(model_traces or []),
        )

    def _disabled_provider_trace(
        self,
        question: str,
        refs: list[AgentEvidenceRef],
    ) -> ModelInvocationTrace:
        prompt_hash = hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest()
        request = self._safe_json(
            {
                "question": question,
                "evidence_ids": [str(ref.evidence_id) for ref in refs],
            }
        )
        return ModelInvocationTrace(
            provider="disabled",
            model="disabled",
            prompt_version=self.prompt_version,
            prompt_sha256=prompt_hash,
            request_sha256=hashlib.sha256(request.encode("utf-8")).hexdigest(),
            response_sha256=None,
            latency_ms=0,
            attempts=1,
            status="FAILED",
            error_type="LLM_DISABLED",
        )

    @staticmethod
    def _contains_forbidden_decision_language(draft: GroundedAnswerDraft) -> bool:
        texts = [claim.statement for claim in draft.claims]
        texts.extend(claim.uncertainty_reason or "" for claim in draft.claims)
        texts.extend(draft.missing_information)
        texts.extend(draft.conflicts)
        texts.append(draft.abstention_reason or "")
        for text in texts:
            # Normalize full-width forms and remove separator/punctuation tricks;
            # only the safe, allowlisted rejection code crosses the boundary.
            normalized = unicodedata.normalize("NFKC", text).casefold()
            compact = re.sub(r"[\W_]+", "", normalized)
            if any(pattern.search(compact) for pattern in _FORBIDDEN_DECISION_PATTERNS):
                return True
        return False

    def _extractive_draft(
        self,
        question: str,
        packed: PackedContext,
        refs: list[AgentEvidenceRef],
    ) -> GroundedAnswerDraft:
        ref_by_section = {ref.section_id: ref for ref in refs}
        section = next((item for item in packed.sections if item.text.strip()), None)
        if section is None:
            return GroundedAnswerDraft(
                missing_information=["上下文段落没有可用文本"],
                abstention_reason="经审计上下文为空文本，无法生成摘录。",
            )
        ref = ref_by_section[section.section_id]
        statement = "原文摘录：" + " ".join(section.text.split())[:800]
        return GroundedAnswerDraft(
            claims=[
                DraftAnswerClaim(
                    category=self._infer_category(question),
                    statement=statement,
                    verdict="SUPPORTED",
                    supporting_evidence_ids=[ref.evidence_id],
                )
            ],
            missing_information=["确定性原文摘录未经生成模型综合，必须由人工复核。"],
        )

    @staticmethod
    def _infer_category(question: str) -> str:
        if any(
            token in question for token in ("政策", "要求", "条件", "上限", "准入", "规定", "不得")
        ):
            return "ELIGIBILITY"
        if any(token in question for token in ("现金流", "回款", "经营现金")):
            return "CASH_FLOW"
        if any(token in question for token in ("集中", "客户占比", "供应商占比")):
            return "CONCENTRATION"
        if any(token in question for token in ("关联方", "关联交易")):
            return "RELATED_PARTY"
        if any(token in question for token in ("例外", "豁免", "担保")):
            return "EXCEPTION"
        if any(token in question for token in ("财务", "利润", "负债", "资产", "比率")):
            return "FINANCIAL"
        if any(token in question for token in ("材料", "缺少", "提交")):
            return "MISSING_MATERIAL"
        return "ELIGIBILITY"

    @staticmethod
    def _trace_ids(traces: list[Any]) -> list[uuid.UUID]:
        ids: list[uuid.UUID] = []
        for trace in traces:
            value = None
            if isinstance(trace, dict):
                value = trace.get("invocation_id") or trace.get("id")
            else:
                value = getattr(trace, "invocation_id", None) or getattr(trace, "id", None)
            try:
                parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
            except (TypeError, ValueError, AttributeError):
                continue
            if parsed not in ids:
                ids.append(parsed)
        return ids

    @staticmethod
    def _rejection_trace(traces: list[Any]) -> Any | None:
        return traces[-1] if traces else None
