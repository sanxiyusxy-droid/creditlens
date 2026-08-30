"""Committed JSON Schema files must stay aligned with v1.6 Pydantic contracts."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from creditlens.evaluation.agent_ablation import (
    AblationHarnessArtifact,
    AgentAblationDataset,
    AgentAblationObservationSet,
    AgentAblationReport,
)
from creditlens.evaluation.answer_metrics import AnswerEvaluationSummary
from creditlens.evaluation.answer_suite import (
    AnswerReevaluationManifest,
    AnswerSuiteRunRecord,
)
from creditlens.evaluation.failure_cases import FailureCaseDataset, FailureCaseReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = PROJECT_ROOT / "evaluation" / "schemas"


def _answer_summary(*, total_questions: int, technical_failures: int = 0) -> dict:
    summary = {field: 0 for field in AnswerEvaluationSummary.model_fields}
    summary.update(
        total_questions=total_questions,
        technical_failures=technical_failures,
    )
    return summary


def _completed_stage(*, name: str = "smoke") -> dict:
    expected_count = 3 if name == "smoke" else 41
    digest = "a" * 64
    return {
        "name": name,
        "status": "COMPLETED",
        "expected_question_count": expected_count,
        "execution_nonce": "nonce-12345678",
        "prediction_count": expected_count,
        "idempotent_replay_count": 0,
        "generation_return_code": 0,
        "scoring_return_code": 0,
        "predictions_path": "evaluation/reports/predictions.json",
        "scoring_dataset_path": "evaluation/reports/scoring-dataset.json",
        "report_path": "evaluation/reports/report.json",
        "query_dataset_sha256": digest,
        "answer_dataset_sha256": digest,
        "source_gold_sha256": digest,
        "predictions_sha256": digest,
        "scoring_dataset_sha256": digest,
        "report_sha256": digest,
        "runtime_profile_sha256": digest,
        "git_commit": "b" * 40,
        "git_dirty": True,
        "source_state_sha256": digest,
        "source_state_algorithm": "sha256-canonical-file-manifest-v1",
        "source_state_scope": "creditlens-runtime-evidence-v1",
        "source_state_file_count": 1,
        "evidence_maturity": "DEVELOPMENT_SOURCE_BOUND",
        "summary": _answer_summary(total_questions=expected_count),
        "failure_phase": None,
    }


def _answer_run(stage: dict) -> dict:
    return {
        "protocol_id": "creditlens_answer_reevaluation_v1",
        "protocol_version": "1.0.0",
        "run_id": "schema-contract-test",
        "generated_at": "2026-08-29T00:00:00Z",
        "execution_requested": True,
        "gold_boundary": "ONLINE_QUERY_ONLY_THEN_OFFLINE_GOLD_MAPPING",
        "manifest_sha256": "c" * 64,
        "stages": [stage],
    }


def _answer_run_validator() -> Draft202012Validator:
    schema = AnswerSuiteRunRecord.model_json_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("answer_reevaluation_manifest_v1.schema.json", AnswerReevaluationManifest),
        ("answer_reevaluation_run_v1.schema.json", AnswerSuiteRunRecord),
        ("agent_ablation_dataset_v1.schema.json", AgentAblationDataset),
        ("agent_ablation_observations_v1.schema.json", AgentAblationObservationSet),
        ("agent_ablation_harness_artifact_v1.schema.json", AblationHarnessArtifact),
        ("agent_ablation_report_v1.schema.json", AgentAblationReport),
        ("fail_closed_cases_v1.schema.json", FailureCaseDataset),
        ("fail_closed_report_v1.schema.json", FailureCaseReport),
    ],
)
def test_committed_schema_matches_model(filename: str, model) -> None:
    committed = json.loads((SCHEMA_ROOT / filename).read_text(encoding="utf-8"))

    assert committed == model.model_json_schema()


def test_evidence_schemas_publish_non_vacuous_collection_constraints() -> None:
    failure_results = FailureCaseReport.model_json_schema()["properties"]["results"]
    assert failure_results["minItems"] == 2
    assert failure_results["maxItems"] == 2

    ablation_metrics = AgentAblationReport.model_json_schema()["properties"]["metrics"]
    assert ablation_metrics["minItems"] == 4
    assert ablation_metrics["maxItems"] == 4


def test_answer_run_schema_publishes_completed_stage_requirements() -> None:
    schema = AnswerSuiteRunRecord.model_json_schema()
    stage_schema = schema["$defs"]["AnswerSuiteStageRecord"]
    completed_rule = next(
        item
        for item in stage_schema["allOf"]
        if item["if"]["properties"].get("status", {}).get("const") == "COMPLETED"
    )

    assert {
        "predictions_sha256",
        "scoring_dataset_sha256",
        "report_sha256",
        "summary",
    } <= set(completed_rule["then"]["required"])
    assert completed_rule["then"]["properties"]["generation_return_code"] == {
        "const": 0,
        "type": "integer",
    }


def test_answer_run_schema_accepts_non_vacuous_completed_stages() -> None:
    validator = _answer_run_validator()

    validator.validate(_answer_run(_completed_stage(name="smoke")))
    validator.validate(_answer_run(_completed_stage(name="full")))


def test_answer_run_schema_rejects_vacuous_completed_stage() -> None:
    stage = {
        "name": "smoke",
        "status": "COMPLETED",
        "expected_question_count": 3,
        "execution_nonce": "nonce-12345678",
    }

    with pytest.raises(ValidationError):
        _answer_run_validator().validate(_answer_run(stage))


def test_answer_run_schema_rejects_null_completed_evidence_fields() -> None:
    validator = _answer_run_validator()
    required_evidence_fields = {
        "prediction_count",
        "idempotent_replay_count",
        "generation_return_code",
        "scoring_return_code",
        "predictions_path",
        "scoring_dataset_path",
        "report_path",
        "query_dataset_sha256",
        "answer_dataset_sha256",
        "source_gold_sha256",
        "predictions_sha256",
        "scoring_dataset_sha256",
        "report_sha256",
        "runtime_profile_sha256",
        "git_commit",
        "git_dirty",
        "source_state_sha256",
        "source_state_algorithm",
        "source_state_scope",
        "source_state_file_count",
        "evidence_maturity",
        "summary",
    }

    for field in required_evidence_fields:
        payload = _answer_run(_completed_stage())
        payload["stages"][0][field] = None
        with pytest.raises(ValidationError, match=field.replace("_", " ") + "|None"):
            validator.validate(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("predictions_path", ""),
        ("query_dataset_sha256", "A" * 64),
        ("report_sha256", "a" * 63),
        ("runtime_profile_sha256", 123),
        ("git_commit", "not-a-commit"),
        ("source_state_file_count", 0),
    ],
)
def test_answer_run_schema_rejects_invalid_completed_evidence_types_or_values(
    field: str,
    invalid_value,
) -> None:
    payload = _answer_run(_completed_stage())
    payload["stages"][0][field] = invalid_value

    with pytest.raises(ValidationError):
        _answer_run_validator().validate(payload)


@pytest.mark.parametrize(
    ("name", "field", "invalid_value"),
    [
        ("smoke", "expected_question_count", 41),
        ("smoke", "prediction_count", 2),
        ("smoke", "summary.total_questions", 2),
        ("smoke", "summary.missing_predictions", 1),
        ("smoke", "summary.technical_failures", 1),
        ("full", "expected_question_count", 3),
        ("full", "prediction_count", 40),
        ("full", "summary.total_questions", 40),
        ("full", "summary.missing_predictions", 1),
    ],
)
def test_answer_run_schema_enforces_frozen_completed_stage_counts(
    name: str,
    field: str,
    invalid_value: int,
) -> None:
    payload = _answer_run(_completed_stage(name=name))
    if field.startswith("summary."):
        payload["stages"][0]["summary"][field.removeprefix("summary.")] = invalid_value
    else:
        payload["stages"][0][field] = invalid_value

    with pytest.raises(ValidationError):
        _answer_run_validator().validate(payload)


def test_answer_run_schema_enforces_non_completed_state_contracts() -> None:
    validator = _answer_run_validator()
    base_stage = {
        "name": "smoke",
        "expected_question_count": 3,
        "execution_nonce": "nonce-12345678",
    }

    planned = {**base_stage, "status": "PLANNED"}
    validator.validate(_answer_run(planned))
    with pytest.raises(ValidationError):
        validator.validate(_answer_run({**planned, "predictions_path": "unexpected.json"}))

    running = {**base_stage, "status": "RUNNING"}
    validator.validate(_answer_run(running))
    with pytest.raises(ValidationError):
        validator.validate(_answer_run({**running, "failure_phase": "INTERRUPTED"}))

    validation_failure = {**base_stage, "status": "FAILED", "failure_phase": "VALIDATION"}
    validator.validate(_answer_run(validation_failure))
    with pytest.raises(ValidationError):
        validator.validate(_answer_run({**base_stage, "status": "FAILED"}))

    generation_failure = {
        **base_stage,
        "status": "FAILED",
        "failure_phase": "GENERATION",
        "generation_return_code": -1,
    }
    validator.validate(_answer_run(generation_failure))
    for invalid_return_code in (None, 0):
        with pytest.raises(ValidationError):
            validator.validate(
                _answer_run({**generation_failure, "generation_return_code": invalid_return_code})
            )

    scoring_failure = {
        **base_stage,
        "status": "FAILED",
        "failure_phase": "SCORING",
        "scoring_return_code": -1,
    }
    validator.validate(_answer_run(scoring_failure))
    for invalid_return_code in (None, 0):
        with pytest.raises(ValidationError):
            validator.validate(
                _answer_run({**scoring_failure, "scoring_return_code": invalid_return_code})
            )
