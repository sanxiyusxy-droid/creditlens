from __future__ import annotations

import json
import subprocess
import sys
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from creditlens.evaluation.answer_metrics import (
    LEGACY_PREDICTION_ADAPTER_VERSION,
    AnswerEvalDataset,
    AnswerEvaluationReport,
    AnswerPrediction,
    AnswerPredictionProvenance,
    AnswerPredictionSet,
    PredictedNumericFact,
    PredictionStatus,
    TechnicalFailureProvenance,
    evaluate_answers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "answer_eval_v1.json"
SOURCE_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "frozen_v2.json"

FROZEN_TEST_UNANSWERABLE_IDS = {
    "q069",
    "q070",
    "q071",
    "q072",
    "q133",
    "q134",
    "q135",
    "q136",
    "q137",
    "q168",
    "q169",
}


@pytest.fixture(scope="module")
def dataset() -> AnswerEvalDataset:
    return AnswerEvalDataset.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))


def _provenance(
    question_id: str,
    *,
    generation_mode: str = "llm",
    idempotent_replay: bool = False,
) -> AnswerPredictionProvenance:
    return AnswerPredictionProvenance(
        run_id=uuid.uuid5(uuid.NAMESPACE_URL, f"creditlens:run:{question_id}"),
        snapshot_id=uuid.uuid5(uuid.NAMESPACE_URL, f"creditlens:snapshot:{question_id}"),
        generation_mode=generation_mode,
        model_invocation_ids=[uuid.uuid5(uuid.NAMESPACE_URL, f"creditlens:model:{question_id}")],
        idempotent_replay=idempotent_replay,
    )


def _perfect_prediction_set(dataset: AnswerEvalDataset) -> AnswerPredictionSet:
    predictions: list[AnswerPrediction] = []
    for question in dataset.questions:
        if question.answerable:
            predictions.append(
                AnswerPrediction(
                    question_id=question.question_id,
                    status=PredictionStatus.ANSWERED,
                    answer="；".join(
                        key_point.acceptable_phrases[0]
                        for key_point in question.expected.key_points
                    ),
                    numeric_facts=[
                        PredictedNumericFact(
                            name=fact.name,
                            value=fact.value,
                            unit=fact.unit,
                        )
                        for fact in question.expected.numeric_facts
                    ],
                    citation_refs=question.expected.citation_refs,
                    provenance=_provenance(question.question_id),
                )
            )
        else:
            predictions.append(
                AnswerPrediction(
                    question_id=question.question_id,
                    status=PredictionStatus.REFUSED,
                    refusal_reason_code=question.expected.acceptable_refusal_reason_codes[0],
                    provenance=_provenance(
                        question.question_id,
                        generation_mode="abstained_empty_context",
                    ),
                )
            )
    return AnswerPredictionSet(
        prediction_set_id="perfect-fixture",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        prediction_adapter_version="fixture-adapter-v1",
        predictions=predictions,
    )


def _replace_prediction(
    prediction_set: AnswerPredictionSet,
    question_id: str,
    replacement: AnswerPrediction,
) -> AnswerPredictionSet:
    return prediction_set.model_copy(
        update={
            "predictions": [
                replacement if item.question_id == question_id else item
                for item in prediction_set.predictions
            ]
        },
        deep=True,
    )


def _question_score(report, question_id: str):
    return next(item for item in report.questions if item.question_id == question_id)


def test_answer_dataset_is_frozen_41_question_test_subset(
    dataset: AnswerEvalDataset,
) -> None:
    source = json.loads(SOURCE_DATASET_PATH.read_text(encoding="utf-8"))
    source_by_id = {item["question_id"]: item for item in source["questions"]}
    source_unanswerable = {
        item["question_id"]
        for item in source["questions"]
        if item.get("split") == "test" and item.get("answerable", True) is False
    }

    assert dataset.frozen is True
    assert dataset.source_dataset_id == source["dataset_id"] == "frozen_v2"
    assert dataset.source_dataset_version == source["dataset_version"] == "2.1.0"
    assert len(dataset.questions) == 41
    assert sum(question.answerable for question in dataset.questions) == 30
    assert sum(not question.answerable for question in dataset.questions) == 11
    assert source_unanswerable == FROZEN_TEST_UNANSWERABLE_IDS
    assert {
        question.question_id for question in dataset.questions if not question.answerable
    } == FROZEN_TEST_UNANSWERABLE_IDS

    for question in dataset.questions:
        source_question = source_by_id[question.question_id]
        assert source_question["split"] == "test"
        assert question.question == source_question["question"]
        assert question.case_key == source_question["case_key"]
        if question.answerable:
            assert question.expected.citation_refs in source_question["required_evidence_sets"]
        else:
            assert question.expected_refusal_reason == source_question["expected_refusal_reason"]


def test_question_contract_rejects_naive_decision_cutoff(
    dataset: AnswerEvalDataset,
) -> None:
    payload = dataset.questions[0].model_dump(mode="json")
    payload["decision_cutoff_at"] = "2026-06-30T15:59:59"

    with pytest.raises(ValueError, match="must include a timezone"):
        type(dataset.questions[0]).model_validate(payload)


def test_prediction_provenance_is_strict_and_survives_json_roundtrip() -> None:
    provenance = _provenance("q-roundtrip", idempotent_replay=True)
    prediction_set = AnswerPredictionSet(
        prediction_set_id="provenance-roundtrip",
        dataset_id="answer_eval_v1",
        dataset_version="1.0.0",
        predictions=[
            AnswerPrediction(
                question_id="q-roundtrip",
                status=PredictionStatus.NEEDS_REVIEW,
                provenance=provenance,
            )
        ],
    )

    restored = AnswerPredictionSet.model_validate_json(prediction_set.model_dump_json())

    assert restored == prediction_set
    assert restored.predictions[0].provenance == provenance
    assert set(json.loads(prediction_set.model_dump_json())["predictions"][0]["provenance"]) == {
        "run_id",
        "snapshot_id",
        "generation_mode",
        "model_invocation_ids",
        "idempotent_replay",
    }

    unsafe = provenance.model_dump(mode="json") | {"prompt": "must never persist"}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AnswerPrediction(
            question_id="q-unsafe",
            status=PredictionStatus.NEEDS_REVIEW,
            provenance=unsafe,
        )
    with pytest.raises(ValueError, match="require complete Grounded QA provenance"):
        AnswerPrediction(
            question_id="q-missing",
            status=PredictionStatus.NEEDS_REVIEW,
        )

    failure_run_id = uuid.uuid4()
    technical = AnswerPrediction(
        question_id="q-failure",
        status=PredictionStatus.TECHNICAL_FAILURE,
        error_type="UPSTREAM_TIMEOUT",
        provenance=TechnicalFailureProvenance(run_id=failure_run_id),
    )
    assert technical.model_dump(mode="json")["provenance"] == {"run_id": str(failure_run_id)}
    with pytest.raises(ValueError, match="only the safe failure run_id"):
        AnswerPrediction(
            question_id="q-overattributed-failure",
            status=PredictionStatus.TECHNICAL_FAILURE,
            error_type="UPSTREAM_TIMEOUT",
            provenance=provenance,
        )


def test_perfect_predictions_score_every_metric_without_failures(
    dataset: AnswerEvalDataset,
) -> None:
    report = evaluate_answers(dataset, _perfect_prediction_set(dataset))
    summary = report.summary

    assert summary.total_questions == 41
    assert summary.lexical_correctness == 1.0
    assert summary.key_point_recall == 1.0
    assert summary.numeric_accuracy == 1.0
    assert summary.citation_precision == 1.0
    assert summary.citation_recall == 1.0
    assert summary.citation_f1 == 1.0
    assert summary.refusal_accuracy == 1.0
    assert summary.false_refusal_rate == 0.0
    assert summary.technical_failures == 0
    assert summary.forbidden_assertion_violations == 0
    assert summary.lexically_correct_citation_exact == 1.0
    assert summary.deterministic_outcome_pass_rate == 1.0


def test_report_contract_names_scope_and_semantic_limit(
    dataset: AnswerEvalDataset,
) -> None:
    report = evaluate_answers(dataset, _perfect_prediction_set(dataset))
    payload = report.model_dump(mode="json")
    summary = payload["summary"]
    question = payload["questions"][0]

    assert payload["evaluation_scope"] == "DETERMINISTIC_LEXICAL_AND_CITATION_SET"
    assert payload["semantic_entailment_evaluated"] is False
    assert payload["prediction_adapter_version"] == "fixture-adapter-v1"
    assert payload["metric_contract_version"] == "2.1.0"
    assert "lexical_correctness" in summary
    assert "lexically_correct_citation_exact" in summary
    assert "deterministic_outcome_pass_rate" in summary
    assert "answer_correctness" not in summary
    assert "fully_grounded_answer_accuracy" not in summary
    assert "overall_accuracy" not in summary
    assert "lexical_correctness" in question
    assert "lexically_correct_citation_exact" in question
    assert "deterministic_outcome_pass" in question
    assert "answer_correct" not in question
    assert "fully_grounded_answer" not in question
    assert "outcome_correct" not in question


def test_metric_2_report_without_adapter_version_is_labeled_legacy(
    dataset: AnswerEvalDataset,
) -> None:
    payload = evaluate_answers(dataset, _perfect_prediction_set(dataset)).model_dump(mode="json")
    payload.pop("prediction_adapter_version")
    payload["metric_contract_version"] = "2.0.0"

    restored = AnswerEvaluationReport.model_validate(payload)

    assert restored.metric_contract_version == "2.0.0"
    assert restored.prediction_adapter_version == LEGACY_PREDICTION_ADAPTER_VERSION


@pytest.mark.parametrize(
    ("question_id", "answer", "facts", "expected_key_points"),
    [
        (
            "q001",
            "借款人资产负债率不得高于百分之七十。",
            [("借款人资产负债率上限", "70", "%")],
            ["debt_ratio_cap"],
        ),
        (
            "q003",
            "单一小微企业借款人的流动资金贷款余额上限为人民币一千万元。",
            [("单一小微企业借款人的流动资金贷款余额上限", "1000", "万元")],
            ["single_borrower_limit"],
        ),
        (
            "q081",
            "资产负债率不得高于百分之七十五。",
            [("资产负债率上限", "75", "%")],
            ["debt_ratio_cap"],
        ),
        (
            "q101",
            "星辰微电子2025年度营业收入为人民币1.85亿元 2025年度营业收入同比增长23.3%",
            [
                ("星辰微电子2025年度营业收入", "1.85", "亿元"),
                ("2025年度营业收入同比增长", "23.3", "%"),
            ],
            ["revenue", "revenue_growth"],
        ),
        (
            "q086",
            "研发投入占营业收入比例不得低于百分之三。",
            [("研发投入占营业收入比例下限", "3", "%")],
            ["rd_ratio_floor"],
        ),
        (
            "q153",
            "一般风险准备金余额不低于融资余额的百分之一。",
            [("一般风险准备金余额不低于融资余额的", "1", "%")],
            ["general_reserve_floor"],
        ),
    ],
)
def test_first_run_chinese_number_answers_match_deterministically(
    dataset: AnswerEvalDataset,
    question_id: str,
    answer: str,
    facts: list[tuple[str, str, str]],
    expected_key_points: list[str],
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == question_id)
    replacement = original.model_copy(
        update={
            "answer": answer,
            "numeric_facts": [
                PredictedNumericFact(name=name, value=Decimal(value), unit=unit)
                for name, value, unit in facts
            ],
        },
        deep=True,
    )

    score = _question_score(
        evaluate_answers(dataset, _replace_prediction(predictions, question_id, replacement)),
        question_id,
    )

    assert score.key_points_matched == expected_key_points
    assert score.key_points_missed == []
    assert score.numeric_facts_missed == []
    assert score.lexical_correctness is True


@pytest.mark.parametrize(
    ("question_id", "answer", "fact_name", "missed_key", "missed_fact"),
    [
        (
            "q086",
            "市场投入占营业收入比例不得低于3%。",
            "市场投入占营业收入比例下限",
            "rd_ratio_floor",
            "研发投入占比下限",
        ),
        (
            "q086",
            "研发投入占营业收入比例不得高于3%。",
            "研发投入占营业收入比例上限",
            "rd_ratio_floor",
            "研发投入占比下限",
        ),
        (
            "q153",
            "专项风险准备金余额不低于融资余额的1%。",
            "专项风险准备金余额不低于融资余额的",
            "general_reserve_floor",
            "一般风险准备金比例下限",
        ),
        (
            "q153",
            "一般风险准备金余额不高于融资余额的1%。",
            "一般风险准备金余额上限",
            "general_reserve_floor",
            "一般风险准备金比例下限",
        ),
    ],
)
def test_bounded_threshold_equivalents_do_not_match_other_metric_or_direction(
    dataset: AnswerEvalDataset,
    question_id: str,
    answer: str,
    fact_name: str,
    missed_key: str,
    missed_fact: str,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == question_id)
    replacement = original.model_copy(
        update={
            "answer": answer,
            "numeric_facts": [
                PredictedNumericFact(
                    name=fact_name, value=Decimal("3" if question_id == "q086" else "1"), unit="%"
                )
            ],
        },
        deep=True,
    )

    score = _question_score(
        evaluate_answers(dataset, _replace_prediction(predictions, question_id, replacement)),
        question_id,
    )

    assert score.key_points_matched == []
    assert score.key_points_missed == [missed_key]
    assert score.numeric_facts_matched == []
    assert score.numeric_facts_missed == [missed_fact]
    assert score.lexical_correctness is False


def test_non_revenue_does_not_match_revenue_metric_or_phrase(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "answer": "非营业收入为人民币1.85亿元；同比增长23.3%",
            "numeric_facts": [
                PredictedNumericFact(name="非营业收入", value=Decimal("1.85"), unit="亿元"),
                PredictedNumericFact(name="同比增长", value=Decimal("23.3"), unit="%"),
            ],
        },
        deep=True,
    )

    score = _question_score(
        evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement)),
        "q101",
    )

    assert score.key_points_matched == ["revenue_growth"]
    assert score.key_points_missed == ["revenue"]
    assert score.numeric_facts_matched == ["营业收入同比增幅"]
    assert score.numeric_facts_missed == ["营业收入"]
    assert score.lexical_correctness is False


def test_chinese_number_phrase_still_respects_negation_window(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q001")
    replacement = original.model_copy(
        update={
            "answer": "无法确认资产负债率不得高于百分之七十。",
            "numeric_facts": [
                PredictedNumericFact(name="资产负债率", value=Decimal("70"), unit="%")
            ],
        },
        deep=True,
    )

    score = _question_score(
        evaluate_answers(dataset, _replace_prediction(predictions, "q001", replacement)),
        "q001",
    )

    assert score.key_points_matched == []
    assert score.key_points_missed == ["debt_ratio_cap"]
    assert score.numeric_accuracy == 1.0
    assert score.lexical_correctness is False


@pytest.mark.parametrize("negation", ["不存在", "没有证据表明", "无法据此确认"])
def test_high_confidence_negation_rejects_phrase_and_numeric_name_at_report_level(
    dataset: AnswerEvalDataset,
    negation: str,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "answer": (f"{negation}营业收入1.85亿元，{negation}同比增长23.3%。"),
            "numeric_facts": [
                PredictedNumericFact(
                    name=f"{negation}营业收入",
                    value=Decimal("1.85"),
                    unit="亿元",
                ),
                PredictedNumericFact(
                    name=f"{negation}同比增长",
                    value=Decimal("23.3"),
                    unit="%",
                ),
            ],
        },
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement))
    score = _question_score(report, "q101")

    assert score.key_points_matched == []
    assert score.key_points_missed == ["revenue", "revenue_growth"]
    assert score.numeric_facts_matched == []
    assert score.numeric_facts_missed == ["营业收入", "营业收入同比增幅"]
    assert score.citation_fp == 0
    assert score.citation_fn == 0
    assert score.lexical_correctness is False
    assert score.lexically_correct_citation_exact is False
    assert report.summary.lexically_correct_answers == 29


def test_negation_of_prior_clause_does_not_suppress_positive_revenue_fact(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "answer": "无重大异常但营业收入1.85亿元，同比增长23.3%。",
            "numeric_facts": [
                PredictedNumericFact(name="营业收入", value=Decimal("1.85"), unit="亿元"),
                PredictedNumericFact(name="同比增长", value=Decimal("23.3"), unit="%"),
            ],
        },
        deep=True,
    )

    score = _question_score(
        evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement)),
        "q101",
    )

    assert score.key_points_missed == []
    assert score.numeric_facts_missed == []
    assert score.lexical_correctness is True
    assert score.lexically_correct_citation_exact is True


def test_numeric_alias_nfkc_unit_and_tolerance_are_supported(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q095")
    replacement = original.model_copy(
        update={
            "numeric_facts": [
                PredictedNumericFact(
                    name="离职比例阈值",
                    value=Decimal("33.34"),
                    unit="％",
                )
            ]
        },
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q095", replacement))
    score = _question_score(report, "q095")

    assert score.numeric_facts_matched == ["核心技术人员离职预警比例"]
    assert score.numeric_accuracy == 1.0
    assert score.lexical_correctness is True


@pytest.mark.parametrize(
    ("revenue_value", "revenue_unit"),
    [
        (Decimal("1.85"), "亿元"),
        (Decimal("18500"), "万元"),
        (Decimal("185000000"), "元"),
    ],
)
def test_q101_money_conversion_and_explicit_growth_synonym_are_lexically_correct(
    dataset: AnswerEvalDataset,
    revenue_value: Decimal,
    revenue_unit: str,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "answer": "营业收入1.85亿元；同比增长23.3%",
            "numeric_facts": [
                PredictedNumericFact(
                    name="营业收入",
                    value=revenue_value,
                    unit=revenue_unit,
                ),
                PredictedNumericFact(
                    name="同比增长",
                    value=Decimal("23.3"),
                    unit="%",
                ),
            ],
        },
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement))
    score = _question_score(report, "q101")

    assert score.key_points_missed == []
    assert score.numeric_facts_matched == ["营业收入", "营业收入同比增幅"]
    assert score.numeric_accuracy == 1.0
    assert score.lexical_correctness is True


def test_q101_negated_phrases_do_not_count_as_lexical_support(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "answer": "无法确认：营业收入1.85亿元；不能确认同比增长23.3%",
            "numeric_facts": [
                PredictedNumericFact(
                    name="营业收入",
                    value=Decimal("1.85"),
                    unit="亿元",
                ),
                PredictedNumericFact(
                    name="同比增长",
                    value=Decimal("23.3"),
                    unit="%",
                ),
            ],
        },
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement))
    score = _question_score(report, "q101")

    assert score.key_points_matched == []
    assert score.key_points_missed == ["revenue", "revenue_growth"]
    assert score.numeric_accuracy == 1.0
    assert score.lexical_correctness is False
    assert score.deterministic_outcome_pass is False


def test_q101_unlisted_numeric_name_synonym_is_not_fuzzily_matched(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q101")
    replacement = original.model_copy(
        update={
            "numeric_facts": [
                PredictedNumericFact(
                    name="营业收入",
                    value=Decimal("1.85"),
                    unit="亿元",
                ),
                PredictedNumericFact(
                    name="同比上升",
                    value=Decimal("23.3"),
                    unit="%",
                ),
            ]
        },
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q101", replacement))
    score = _question_score(report, "q101")

    assert score.numeric_facts_matched == ["营业收入"]
    assert score.numeric_facts_missed == ["营业收入同比增幅"]
    assert score.lexical_correctness is False


def test_citation_micro_precision_recall_and_f1_count_fp_and_fn(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q001")
    replacement = original.model_copy(update={"citation_refs": ["madeup:ref"]}, deep=True)
    total_expected = sum(len(question.expected.citation_refs) for question in dataset.questions)

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q001", replacement))
    summary = report.summary

    assert summary.citation_tp == total_expected - 1
    assert summary.citation_fp == 1
    assert summary.citation_fn == 1
    assert summary.citation_precision == pytest.approx((total_expected - 1) / total_expected)
    assert summary.citation_recall == pytest.approx((total_expected - 1) / total_expected)
    assert summary.lexically_correct_citation_exact_answers == 29


def test_false_refusal_is_not_a_correct_answer(dataset: AnswerEvalDataset) -> None:
    predictions = _perfect_prediction_set(dataset)
    replacement = AnswerPrediction(
        question_id="q001",
        status=PredictionStatus.REFUSED,
        refusal_reason_code="INSUFFICIENT_EVIDENCE",
        provenance=_provenance("q001", generation_mode="abstained_empty_context"),
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q001", replacement))
    score = _question_score(report, "q001")

    assert score.false_refusal is True
    assert score.deterministic_outcome_pass is False
    assert report.summary.false_refusals == 1
    assert report.summary.false_refusal_rate == pytest.approx(1 / 30, abs=1e-6)
    assert report.summary.lexically_correct_answers == 29


def test_technical_failure_on_unanswerable_is_never_a_correct_refusal(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    replacement = AnswerPrediction(
        question_id="q069",
        status=PredictionStatus.TECHNICAL_FAILURE,
        error_type="UPSTREAM_TIMEOUT",
        error_message="model gateway timed out",
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q069", replacement))
    score = _question_score(report, "q069")

    assert score.technical_failure is True
    assert score.correct_refusal is False
    assert score.deterministic_outcome_pass is False
    assert report.summary.technical_failures == 1
    assert report.summary.technical_failures_unanswerable == 1
    assert report.summary.correct_refusals == 10
    assert report.summary.refusal_accuracy == pytest.approx(10 / 11)


def test_needs_review_is_separate_from_refusal_and_technical_failure(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    predictions = _replace_prediction(
        predictions,
        "q001",
        AnswerPrediction(
            question_id="q001",
            status=PredictionStatus.NEEDS_REVIEW,
            provenance=_provenance("q001"),
        ),
    )
    predictions = _replace_prediction(
        predictions,
        "q069",
        AnswerPrediction(
            question_id="q069",
            status=PredictionStatus.NEEDS_REVIEW,
            provenance=_provenance("q069"),
        ),
    )

    report = evaluate_answers(dataset, predictions)

    for question_id in ("q001", "q069"):
        score = _question_score(report, question_id)
        assert score.needs_review is True
        assert score.technical_failure is False
        assert score.correct_refusal is False
        assert score.false_refusal is False
        assert score.deterministic_outcome_pass is False
    assert report.summary.needs_review_count == 2
    assert report.summary.needs_review_rate == pytest.approx(2 / 41, abs=1e-6)
    assert report.summary.technical_failures == 0
    assert report.summary.false_refusals == 0
    assert report.summary.correct_refusals == 10


def test_missing_prediction_is_materialized_as_technical_failure(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    predictions = predictions.model_copy(
        update={
            "predictions": [item for item in predictions.predictions if item.question_id != "q169"]
        },
        deep=True,
    )

    report = evaluate_answers(dataset, predictions)
    score = _question_score(report, "q169")

    assert score.prediction_status is PredictionStatus.TECHNICAL_FAILURE
    assert score.technical_failure_type == "MISSING_PREDICTION"
    assert score.missing_prediction is True
    assert score.correct_refusal is False
    assert report.summary.missing_predictions == 1
    assert report.summary.technical_failures == 1


def test_wrong_refusal_reason_is_incorrect_but_not_technical_failure(
    dataset: AnswerEvalDataset,
) -> None:
    predictions = _perfect_prediction_set(dataset)
    replacement = AnswerPrediction(
        question_id="q069",
        status=PredictionStatus.REFUSED,
        refusal_reason_code="POLICY_PROHIBITS_ANSWER",
        provenance=_provenance("q069", generation_mode="abstained_empty_context"),
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q069", replacement))
    score = _question_score(report, "q069")

    assert score.correct_refusal is False
    assert score.technical_failure is False
    assert report.summary.correct_refusals == 10
    assert report.summary.technical_failures == 0


def test_forbidden_assertion_blocks_correctness(dataset: AnswerEvalDataset) -> None:
    predictions = _perfect_prediction_set(dataset)
    original = next(item for item in predictions.predictions if item.question_id == "q061")
    replacement = original.model_copy(
        update={"answer": f"{original.answer}；不满足政策要求"},
        deep=True,
    )

    report = evaluate_answers(dataset, _replace_prediction(predictions, "q061", replacement))
    score = _question_score(report, "q061")

    assert score.key_point_score == 1.0
    assert score.forbidden_assertions_found == ["不满足政策要求"]
    assert score.lexical_correctness is False
    assert report.summary.forbidden_assertion_violations == 1


def test_cli_scores_prediction_json_and_records_hashes(
    dataset: AnswerEvalDataset,
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    report_path = tmp_path / "report.json"
    predictions_path.write_text(
        _perfect_prediction_set(dataset).model_dump_json(indent=2),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_answer_evaluation.py"),
            "--dataset",
            str(DATASET_PATH),
            "--predictions",
            str(predictions_path),
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_type"] == "deterministic_offline_answer_evaluation"
    assert len(report["dataset_sha256"]) == 64
    assert len(report["predictions_sha256"]) == 64
    assert report["summary"]["deterministic_outcome_pass_rate"] == 1.0
