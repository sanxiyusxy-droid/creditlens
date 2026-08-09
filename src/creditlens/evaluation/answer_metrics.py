"""Deterministic, offline metrics for grounded answer evaluation.

The evaluator intentionally consumes a structured prediction file instead of
calling an LLM judge.  This keeps the frozen score reproducible and makes
technical failures distinguishable from model refusals.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _non_blank(value: str, *, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


class ExpectedKeyPoint(_StrictModel):
    """A semantic point scored by deterministic acceptable-phrase matching."""

    key: str
    acceptable_phrases: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return _non_blank(value, field_name="key")

    @field_validator("acceptable_phrases")
    @classmethod
    def validate_phrases(cls, values: list[str]) -> list[str]:
        cleaned = [_non_blank(value, field_name="acceptable_phrases") for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("acceptable_phrases must be unique")
        return cleaned


class ExpectedNumericFact(_StrictModel):
    """A numeric fact compared against structured model output using Decimal."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    value: Decimal
    unit: str
    tolerance: Decimal = Field(default=Decimal("0"), ge=0)

    @field_validator("name", "unit")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        cleaned = [_non_blank(value, field_name="aliases") for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("aliases must be unique")
        return cleaned


class AnswerExpectation(_StrictModel):
    key_points: list[ExpectedKeyPoint] = Field(default_factory=list)
    numeric_facts: list[ExpectedNumericFact] = Field(default_factory=list)
    forbidden_assertions: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    acceptable_refusal_reason_codes: list[str] = Field(default_factory=list)

    @field_validator("forbidden_assertions", "citation_refs", "acceptable_refusal_reason_codes")
    @classmethod
    def validate_string_lists(cls, values: list[str], info) -> list[str]:
        cleaned = [_non_blank(value, field_name=info.field_name) for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"{info.field_name} must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_unique_fact_keys(self) -> Self:
        key_point_keys = [item.key for item in self.key_points]
        if len(set(key_point_keys)) != len(key_point_keys):
            raise ValueError("key point keys must be unique")
        numeric_names = [item.name for item in self.numeric_facts]
        if len(set(numeric_names)) != len(numeric_names):
            raise ValueError("numeric fact names must be unique")
        return self


class AnswerEvalQuestion(_StrictModel):
    question_id: str
    case_key: str
    question: str
    intent: str
    as_of_date: date
    decision_cutoff_at: datetime
    answerable: bool
    expected: AnswerExpectation
    expected_refusal_reason: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("question_id", "case_key", "question", "intent")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("decision_cutoff_at")
    @classmethod
    def validate_aware_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_cutoff_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> Self:
        if self.answerable:
            if not self.expected.key_points:
                raise ValueError("answerable questions require at least one expected key point")
            if not self.expected.citation_refs:
                raise ValueError("answerable questions require at least one citation ref")
            if self.expected.acceptable_refusal_reason_codes:
                raise ValueError("answerable questions must not define refusal reason codes")
            if self.expected_refusal_reason is not None:
                raise ValueError("answerable questions must not define expected_refusal_reason")
        else:
            if (
                self.expected.key_points
                or self.expected.numeric_facts
                or self.expected.citation_refs
            ):
                raise ValueError("unanswerable questions must not define answer evidence")
            if not self.expected.acceptable_refusal_reason_codes:
                raise ValueError("unanswerable questions require an acceptable refusal reason code")
            if not self.expected_refusal_reason or not self.expected_refusal_reason.strip():
                raise ValueError("unanswerable questions require expected_refusal_reason")
        return self


class AnswerEvalDataset(_StrictModel):
    dataset_id: str
    dataset_version: str
    source_dataset_id: str
    source_dataset_version: str
    source_split: str
    frozen: bool = True
    questions: list[AnswerEvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_question_ids(self) -> Self:
        question_ids = [item.question_id for item in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        return self


class PredictionStatus(StrEnum):
    ANSWERED = "ANSWERED"
    REFUSED = "REFUSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class PredictedNumericFact(_StrictModel):
    name: str
    value: Decimal
    unit: str

    @field_validator("name", "unit")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)


class AnswerPredictionProvenance(_StrictModel):
    """Safe run identity copied verbatim from a Grounded QA response.

    The deliberately small, strict schema prevents prompts, retrieved text,
    provider payloads, or other potentially sensitive runtime data from being
    smuggled into an evaluation artifact.
    """

    run_id: uuid.UUID
    snapshot_id: uuid.UUID
    generation_mode: Literal[
        "llm",
        "abstained_empty_context",
        "deterministic_extractive",
    ]
    model_invocation_ids: list[uuid.UUID]
    idempotent_replay: bool

    @field_validator("model_invocation_ids")
    @classmethod
    def validate_unique_model_invocations(cls, values: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(values)) != len(values):
            raise ValueError("model_invocation_ids must be unique")
        return values


class TechnicalFailureProvenance(_StrictModel):
    """The only safe run attribution retained for a technical failure."""

    run_id: uuid.UUID


class AnswerPrediction(_StrictModel):
    question_id: str
    status: PredictionStatus
    answer: str | None = None
    numeric_facts: list[PredictedNumericFact] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    refusal_reason_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    provenance: AnswerPredictionProvenance | TechnicalFailureProvenance | None = None

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        return _non_blank(value, field_name="question_id")

    @field_validator("citation_refs")
    @classmethod
    def validate_citation_refs(cls, values: list[str]) -> list[str]:
        cleaned = [_non_blank(value, field_name="citation_refs") for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("citation_refs must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status is PredictionStatus.ANSWERED:
            if not self.answer or not self.answer.strip():
                raise ValueError("ANSWERED predictions require answer text")
            if self.refusal_reason_code or self.error_type or self.error_message:
                raise ValueError("ANSWERED predictions must not contain refusal or error fields")
        elif self.status is PredictionStatus.REFUSED:
            if not self.refusal_reason_code or not self.refusal_reason_code.strip():
                raise ValueError("REFUSED predictions require refusal_reason_code")
            if self.answer or self.numeric_facts or self.citation_refs:
                raise ValueError("REFUSED predictions must not contain answer evidence")
            if self.error_type or self.error_message:
                raise ValueError("REFUSED predictions must not contain error fields")
        elif self.status is PredictionStatus.TECHNICAL_FAILURE:
            if not self.error_type or not self.error_type.strip():
                raise ValueError("TECHNICAL_FAILURE predictions require error_type")
            if self.answer or self.numeric_facts or self.citation_refs or self.refusal_reason_code:
                raise ValueError(
                    "TECHNICAL_FAILURE predictions must not contain answer/refusal data"
                )
        elif (
            self.answer is not None
            or self.numeric_facts
            or self.citation_refs
            or self.refusal_reason_code is not None
            or self.error_type is not None
            or self.error_message is not None
        ):
            raise ValueError("NEEDS_REVIEW predictions must not contain answer/refusal/error data")

        if self.status in {
            PredictionStatus.ANSWERED,
            PredictionStatus.REFUSED,
            PredictionStatus.NEEDS_REVIEW,
        }:
            if not isinstance(self.provenance, AnswerPredictionProvenance):
                raise ValueError(
                    "business outcome predictions require complete Grounded QA provenance"
                )
        elif self.provenance is not None and not isinstance(
            self.provenance, TechnicalFailureProvenance
        ):
            raise ValueError(
                "TECHNICAL_FAILURE provenance may contain only the safe failure run_id"
            )
        return self


class AnswerPredictionSet(_StrictModel):
    prediction_set_id: str
    dataset_id: str
    dataset_version: str
    predictions: list[AnswerPrediction]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prediction_ids(self) -> Self:
        question_ids = [item.question_id for item in self.predictions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("prediction question IDs must be unique")
        return self


class AnswerQuestionScore(_StrictModel):
    question_id: str
    case_key: str
    answerable: bool
    prediction_status: PredictionStatus
    deterministic_outcome_pass: bool
    lexical_correctness: bool
    lexically_correct_citation_exact: bool
    correct_refusal: bool
    false_refusal: bool
    technical_failure: bool
    needs_review: bool
    technical_failure_type: str | None
    missing_prediction: bool
    key_points_matched: list[str]
    key_points_missed: list[str]
    key_point_score: float | None
    numeric_facts_matched: list[str]
    numeric_facts_missed: list[str]
    numeric_accuracy: float | None
    expected_citation_refs: list[str]
    predicted_citation_refs: list[str]
    citation_tp: int
    citation_fp: int
    citation_fn: int
    forbidden_assertions_found: list[str]
    refusal_reason_code: str | None


class AnswerEvaluationSummary(_StrictModel):
    total_questions: int
    answerable_questions: int
    unanswerable_questions: int
    lexically_correct_answers: int
    lexical_correctness: float
    lexically_correct_citation_exact_answers: int
    lexically_correct_citation_exact: float
    matched_key_point_weight: float
    total_key_point_weight: float
    key_point_recall: float
    matched_numeric_facts: int
    total_numeric_facts: int
    numeric_accuracy: float
    citation_tp: int
    citation_fp: int
    citation_fn: int
    citation_precision: float
    citation_recall: float
    citation_f1: float
    correct_refusals: int
    refusal_accuracy: float
    false_refusals: int
    false_refusal_rate: float
    technical_failures: int
    technical_failure_rate: float
    technical_failures_answerable: int
    technical_failures_unanswerable: int
    needs_review_count: int
    needs_review_rate: float
    missing_predictions: int
    forbidden_assertion_violations: int
    forbidden_violation_questions: int
    deterministic_outcome_pass_count: int
    deterministic_outcome_pass_rate: float


class AnswerEvaluationReport(_StrictModel):
    report_type: str = "deterministic_offline_answer_evaluation"
    evaluation_scope: Literal["DETERMINISTIC_LEXICAL_AND_CITATION_SET"] = (
        "DETERMINISTIC_LEXICAL_AND_CITATION_SET"
    )
    semantic_entailment_evaluated: Literal[False] = False
    generated_at: datetime
    dataset_id: str
    dataset_version: str
    prediction_set_id: str
    dataset_sha256: str | None = None
    predictions_sha256: str | None = None
    metric_contract_version: str = "2.0.0"
    summary: AnswerEvaluationSummary
    questions: list[AnswerQuestionScore]


_IGNORED_TEXT_CHARS = re.compile(r"[^\w%+./-]+", flags=re.UNICODE)
_NEGATION_MARKERS = (
    "无法确认",
    "不能确认",
    "尚不能",
    "并非",
    "不是",
    "不属于",
    "非",
    "未",
    "无",
    "不",
)
_NEGATION_WINDOW = max(len(marker) for marker in _NEGATION_MARKERS)
_MONEY_UNIT_IN_YUAN = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "亿元": Decimal("100000000"),
}
_EXPLICIT_NUMERIC_NAME_EQUIVALENTS = (
    ("同比增长率", "同比增幅"),
    ("同比增长", "同比增幅"),
)


def normalize_text(value: str) -> str:
    """Normalize Chinese/Latin text for auditable substring matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _IGNORED_TEXT_CHARS.sub("", normalized)


def _contains_non_negated_span(normalized_text: str, normalized_span: str) -> bool:
    """Conservatively reject a lexical hit immediately scoped by negation."""

    start = normalized_text.find(normalized_span)
    while start >= 0:
        prefix = normalized_text[max(0, start - _NEGATION_WINDOW) : start]
        if not any(prefix.endswith(marker) for marker in _NEGATION_MARKERS):
            return True
        start = normalized_text.find(normalized_span, start + 1)
    return False


def _contains_phrase(answer: str, phrase: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase) and _contains_non_negated_span(
        normalized_answer, normalized_phrase
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _normalize_numeric_name(value: str) -> str:
    """Apply a small, explicit synonym table; never use fuzzy similarity."""

    normalized = normalize_text(value)
    for source, target in _EXPLICIT_NUMERIC_NAME_EQUIVALENTS:
        normalized = normalized.replace(source, target)
    return normalized


def _numeric_name_matches(expected: ExpectedNumericFact, predicted: PredictedNumericFact) -> bool:
    expected_names = {
        _normalize_numeric_name(expected.name),
        *(_normalize_numeric_name(alias) for alias in expected.aliases),
    }
    predicted_name = _normalize_numeric_name(predicted.name)
    # 运行器从答案中截取数字前的局部语境作为事实名，例如
    # “2025年营业收入为”而不是人工对齐后的“营业收入”。只允许完整的
    # expected/alias 出现在该局部语境中，避免用模糊相似度把不同指标误配。
    return any(name and _contains_non_negated_span(predicted_name, name) for name in expected_names)


def _numeric_fact_matches(expected: ExpectedNumericFact, predicted: PredictedNumericFact) -> bool:
    if not _numeric_name_matches(expected, predicted):
        return False

    expected_unit = normalize_text(expected.unit)
    predicted_unit = normalize_text(predicted.unit)
    expected_scale = _MONEY_UNIT_IN_YUAN.get(expected_unit)
    predicted_scale = _MONEY_UNIT_IN_YUAN.get(predicted_unit)
    if expected_scale is not None and predicted_scale is not None:
        delta_in_yuan = abs(expected.value * expected_scale - predicted.value * predicted_scale)
        return delta_in_yuan <= expected.tolerance * expected_scale
    return (
        expected_unit == predicted_unit
        and abs(expected.value - predicted.value) <= expected.tolerance
    )


def _score_numeric_facts(
    expected: list[ExpectedNumericFact], predicted: list[PredictedNumericFact]
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missed: list[str] = []
    used_prediction_indexes: set[int] = set()
    for expected_fact in expected:
        match_index = next(
            (
                index
                for index, predicted_fact in enumerate(predicted)
                if index not in used_prediction_indexes
                and _numeric_fact_matches(expected_fact, predicted_fact)
            ),
            None,
        )
        if match_index is None:
            missed.append(expected_fact.name)
        else:
            matched.append(expected_fact.name)
            used_prediction_indexes.add(match_index)
    return matched, missed


def _missing_prediction(question_id: str) -> AnswerPrediction:
    return AnswerPrediction(
        question_id=question_id,
        status=PredictionStatus.TECHNICAL_FAILURE,
        error_type="MISSING_PREDICTION",
        error_message="No prediction was supplied for this frozen question.",
    )


def evaluate_answers(
    dataset: AnswerEvalDataset,
    prediction_set: AnswerPredictionSet,
    *,
    dataset_sha256: str | None = None,
    predictions_sha256: str | None = None,
) -> AnswerEvaluationReport:
    """Score a structured prediction set without network or model calls."""

    if prediction_set.dataset_id != dataset.dataset_id:
        raise ValueError(
            f"prediction dataset_id {prediction_set.dataset_id!r} does not match {dataset.dataset_id!r}"
        )
    if prediction_set.dataset_version != dataset.dataset_version:
        raise ValueError(
            "prediction dataset_version "
            f"{prediction_set.dataset_version!r} does not match {dataset.dataset_version!r}"
        )

    dataset_ids = {question.question_id for question in dataset.questions}
    prediction_by_id = {
        prediction.question_id: prediction for prediction in prediction_set.predictions
    }
    unknown_ids = sorted(set(prediction_by_id) - dataset_ids)
    if unknown_ids:
        raise ValueError(f"predictions contain unknown question IDs: {unknown_ids}")

    question_scores: list[AnswerQuestionScore] = []
    matched_key_point_weight = 0.0
    total_key_point_weight = 0.0
    matched_numeric_facts = 0
    total_numeric_facts = 0
    citation_tp = citation_fp = citation_fn = 0

    for question in dataset.questions:
        supplied_prediction = prediction_by_id.get(question.question_id)
        prediction = supplied_prediction or _missing_prediction(question.question_id)
        technical_failure = prediction.status is PredictionStatus.TECHNICAL_FAILURE
        needs_review = prediction.status is PredictionStatus.NEEDS_REVIEW
        missing_prediction = supplied_prediction is None
        answered = prediction.status is PredictionStatus.ANSWERED
        answer_text = prediction.answer or ""

        key_points_matched: list[str] = []
        key_points_missed: list[str] = []
        question_matched_weight = 0.0
        question_total_weight = sum(item.weight for item in question.expected.key_points)
        if question.answerable:
            total_key_point_weight += question_total_weight
            for key_point in question.expected.key_points:
                if answered and any(
                    _contains_phrase(answer_text, phrase) for phrase in key_point.acceptable_phrases
                ):
                    key_points_matched.append(key_point.key)
                    question_matched_weight += key_point.weight
                else:
                    key_points_missed.append(key_point.key)
            matched_key_point_weight += question_matched_weight

        numeric_facts_matched: list[str] = []
        numeric_facts_missed: list[str] = []
        if question.answerable:
            total_numeric_facts += len(question.expected.numeric_facts)
            numeric_facts_matched, numeric_facts_missed = _score_numeric_facts(
                question.expected.numeric_facts,
                prediction.numeric_facts if answered else [],
            )
            matched_numeric_facts += len(numeric_facts_matched)

        expected_citations = set(question.expected.citation_refs)
        predicted_citations = set(prediction.citation_refs if answered else [])
        question_citation_tp = len(expected_citations & predicted_citations)
        question_citation_fp = len(predicted_citations - expected_citations)
        question_citation_fn = len(expected_citations - predicted_citations)
        citation_tp += question_citation_tp
        citation_fp += question_citation_fp
        citation_fn += question_citation_fn

        forbidden_assertions_found = (
            [
                phrase
                for phrase in question.expected.forbidden_assertions
                if answered and _contains_phrase(answer_text, phrase)
            ]
            if answered
            else []
        )
        all_key_points_matched = not key_points_missed
        all_numeric_facts_matched = not numeric_facts_missed
        lexical_correctness = bool(
            question.answerable
            and answered
            and all_key_points_matched
            and all_numeric_facts_matched
            and not forbidden_assertions_found
        )
        citations_exact = question_citation_fp == 0 and question_citation_fn == 0
        lexically_correct_citation_exact = lexical_correctness and citations_exact
        correct_refusal = bool(
            not question.answerable
            and prediction.status is PredictionStatus.REFUSED
            and prediction.refusal_reason_code in question.expected.acceptable_refusal_reason_codes
        )
        false_refusal = question.answerable and prediction.status is PredictionStatus.REFUSED

        question_scores.append(
            AnswerQuestionScore(
                question_id=question.question_id,
                case_key=question.case_key,
                answerable=question.answerable,
                prediction_status=prediction.status,
                deterministic_outcome_pass=lexical_correctness or correct_refusal,
                lexical_correctness=lexical_correctness,
                lexically_correct_citation_exact=lexically_correct_citation_exact,
                correct_refusal=correct_refusal,
                false_refusal=false_refusal,
                technical_failure=technical_failure,
                needs_review=needs_review,
                technical_failure_type=prediction.error_type if technical_failure else None,
                missing_prediction=missing_prediction,
                key_points_matched=key_points_matched,
                key_points_missed=key_points_missed,
                key_point_score=(
                    _safe_ratio(question_matched_weight, question_total_weight)
                    if question.answerable
                    else None
                ),
                numeric_facts_matched=numeric_facts_matched,
                numeric_facts_missed=numeric_facts_missed,
                numeric_accuracy=(
                    _safe_ratio(len(numeric_facts_matched), len(question.expected.numeric_facts))
                    if question.answerable and question.expected.numeric_facts
                    else None
                ),
                expected_citation_refs=sorted(expected_citations),
                predicted_citation_refs=sorted(predicted_citations),
                citation_tp=question_citation_tp,
                citation_fp=question_citation_fp,
                citation_fn=question_citation_fn,
                forbidden_assertions_found=forbidden_assertions_found,
                refusal_reason_code=prediction.refusal_reason_code,
            )
        )

    answerable_count = sum(question.answerable for question in dataset.questions)
    unanswerable_count = len(dataset.questions) - answerable_count
    lexically_correct_answers = sum(score.lexical_correctness for score in question_scores)
    lexically_correct_citation_exact_answers = sum(
        score.lexically_correct_citation_exact for score in question_scores
    )
    correct_refusals = sum(score.correct_refusal for score in question_scores)
    false_refusals = sum(score.false_refusal for score in question_scores)
    technical_failures = sum(score.technical_failure for score in question_scores)
    needs_review_count = sum(score.needs_review for score in question_scores)
    precision = _safe_ratio(citation_tp, citation_tp + citation_fp)
    recall = _safe_ratio(citation_tp, citation_tp + citation_fn)
    deterministic_outcome_pass_count = lexically_correct_answers + correct_refusals

    summary = AnswerEvaluationSummary(
        total_questions=len(dataset.questions),
        answerable_questions=answerable_count,
        unanswerable_questions=unanswerable_count,
        lexically_correct_answers=lexically_correct_answers,
        lexical_correctness=_safe_ratio(lexically_correct_answers, answerable_count),
        lexically_correct_citation_exact_answers=(lexically_correct_citation_exact_answers),
        lexically_correct_citation_exact=_safe_ratio(
            lexically_correct_citation_exact_answers, answerable_count
        ),
        matched_key_point_weight=round(matched_key_point_weight, 6),
        total_key_point_weight=round(total_key_point_weight, 6),
        key_point_recall=_safe_ratio(matched_key_point_weight, total_key_point_weight),
        matched_numeric_facts=matched_numeric_facts,
        total_numeric_facts=total_numeric_facts,
        numeric_accuracy=_safe_ratio(matched_numeric_facts, total_numeric_facts),
        citation_tp=citation_tp,
        citation_fp=citation_fp,
        citation_fn=citation_fn,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=_f1(precision, recall),
        correct_refusals=correct_refusals,
        refusal_accuracy=_safe_ratio(correct_refusals, unanswerable_count),
        false_refusals=false_refusals,
        false_refusal_rate=_safe_ratio(false_refusals, answerable_count),
        technical_failures=technical_failures,
        technical_failure_rate=_safe_ratio(technical_failures, len(dataset.questions)),
        technical_failures_answerable=sum(
            score.technical_failure and score.answerable for score in question_scores
        ),
        technical_failures_unanswerable=sum(
            score.technical_failure and not score.answerable for score in question_scores
        ),
        needs_review_count=needs_review_count,
        needs_review_rate=_safe_ratio(needs_review_count, len(dataset.questions)),
        missing_predictions=sum(score.missing_prediction for score in question_scores),
        forbidden_assertion_violations=sum(
            len(score.forbidden_assertions_found) for score in question_scores
        ),
        forbidden_violation_questions=sum(
            bool(score.forbidden_assertions_found) for score in question_scores
        ),
        deterministic_outcome_pass_count=deterministic_outcome_pass_count,
        deterministic_outcome_pass_rate=_safe_ratio(
            deterministic_outcome_pass_count, len(dataset.questions)
        ),
    )
    return AnswerEvaluationReport(
        generated_at=datetime.now(UTC),
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        prediction_set_id=prediction_set.prediction_set_id,
        dataset_sha256=dataset_sha256,
        predictions_sha256=predictions_sha256,
        summary=summary,
        questions=question_scores,
    )
