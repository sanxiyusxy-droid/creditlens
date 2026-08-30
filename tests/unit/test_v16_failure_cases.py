"""v1.6 reproducible fail-closed and safe-display tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.run_fail_closed_cases import main as failure_cli_main

from creditlens.evaluation.failure_cases import (
    FailureCaseDataset,
    FailureCaseReport,
    FailureProofScope,
    evaluate_failure_cases,
)
from creditlens.evaluation.source_state import (
    EvidenceMaturity,
    classify_evidence_maturity,
    verify_source_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "fail_closed_cases_v1.json"


def _evaluate():
    payload = DATASET_PATH.read_bytes()
    dataset = FailureCaseDataset.model_validate_json(payload)
    return dataset, evaluate_failure_cases(
        dataset,
        dataset_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_both_frozen_injections_fail_closed_through_production_validator() -> None:
    _dataset, report = _evaluate()

    assert report.all_passed is True
    assert report.proof_scope is FailureProofScope.CONTRACT_VALIDATOR_PRECHECK
    assert report.system_execution_performed is False
    assert report.http_endpoint_called is False
    assert report.evidence_maturity is classify_evidence_maturity(
        git_commit=report.git_commit,
        git_dirty=report.git_dirty,
    )
    verify_source_state(
        PROJECT_ROOT,
        expected_sha256=report.source_state_sha256,
        expected_file_count=report.source_state_file_count,
    )
    by_case = {item.case_id: item for item in report.results}
    assert by_case["numeric_evidence_misbinding"].observed_violation_codes == [
        "NUMERIC_CLAIM_WITHOUT_FACT",
        "UNKNOWN_EVIDENCE",
    ]
    assert by_case["prohibited_credit_determination"].observed_violation_codes == [
        "FORBIDDEN_DETERMINATION"
    ]
    assert all(item.safe_display.report_allowed is False for item in report.results)
    assert all(item.safe_display.workflow_status == "HUMAN_REVIEW" for item in report.results)
    assert all(
        item.safe_display.verification_scope == "STATIC_PUBLIC_CONTRACT_NOT_HTTP"
        for item in report.results
    )
    assert all(item.system_evidence is None for item in report.results)


def test_public_failure_report_does_not_expose_injected_claim_text() -> None:
    dataset, report = _evaluate()
    rendered = report.model_dump_json()

    for case in dataset.cases:
        assert case.synthetic_claim not in rendered
    assert "自动放款" not in rendered
    assert all(item.safe_display.unsafe_content_exposed is False for item in report.results)


def test_missing_expected_violation_marks_regression_without_opening_report_gate() -> None:
    dataset, _report = _evaluate()
    first = dataset.cases[0].model_copy(
        update={"expected_violation_codes": ["IMPOSSIBLE_EXPECTATION"]}
    )
    changed = dataset.model_copy(update={"cases": [first, dataset.cases[1]]}, deep=True)

    report = evaluate_failure_cases(changed, dataset_sha256="f" * 64)

    assert report.all_passed is False
    result = report.results[0]
    assert result.passed is False
    assert result.missing_expected_violation_codes == ["IMPOSSIBLE_EXPECTATION"]
    assert result.safe_display.report_allowed is False
    assert result.safe_display.action == "BLOCK"


@pytest.mark.parametrize("reported_all_passed", [False, True])
def test_report_rejects_all_passed_that_disagrees_with_results(
    reported_all_passed: bool,
) -> None:
    dataset, passing_report = _evaluate()
    if reported_all_passed:
        first = dataset.cases[0].model_copy(
            update={"expected_violation_codes": ["IMPOSSIBLE_EXPECTATION"]}
        )
        changed = dataset.model_copy(update={"cases": [first, dataset.cases[1]]}, deep=True)
        source_report = evaluate_failure_cases(changed, dataset_sha256="f" * 64)
    else:
        source_report = passing_report

    payload = source_report.model_dump()
    payload["all_passed"] = reported_all_passed

    with pytest.raises(
        ValidationError,
        match=r"all_passed must equal the conjunction of result\.passed values",
    ):
        FailureCaseReport.model_validate(payload)


def test_report_rejects_release_maturity_for_dirty_source_state() -> None:
    _dataset, report = _evaluate()
    payload = report.model_dump(mode="json")
    payload["git_dirty"] = True
    payload["evidence_maturity"] = EvidenceMaturity.RELEASE_CANDIDATE.value

    with pytest.raises(ValidationError, match="does not match git_commit/git_dirty"):
        FailureCaseReport.model_validate(payload)


@pytest.mark.parametrize(
    "dataset_sha256",
    [
        "f" * 63,
        "f" * 65,
        "g" * 64,
        "F" * 64,
        ("f" * 63) + " ",
    ],
)
def test_report_rejects_noncanonical_dataset_sha256(dataset_sha256: str) -> None:
    _dataset, report = _evaluate()
    payload = report.model_dump()
    payload["dataset_sha256"] = dataset_sha256

    with pytest.raises(
        ValidationError,
        match="dataset_sha256 must be a lowercase SHA-256 digest",
    ):
        FailureCaseReport.model_validate(payload)


def test_report_rejects_vacuous_or_duplicate_failure_evidence() -> None:
    _dataset, report = _evaluate()
    empty_payload = report.model_dump(mode="json")
    empty_payload["results"] = []

    with pytest.raises(ValidationError):
        FailureCaseReport.model_validate(empty_payload)

    duplicate_payload = report.model_dump(mode="json")
    duplicate_payload["results"] = [duplicate_payload["results"][0]] * 2

    with pytest.raises(
        ValidationError,
        match="failure report case IDs must be unique",
    ):
        FailureCaseReport.model_validate(duplicate_payload)


def test_cli_default_is_explicitly_contract_only(tmp_path, capsys) -> None:
    output = tmp_path / "precheck.json"

    exit_code = failure_cli_main(["--dataset", str(DATASET_PATH), "--output", str(output)])

    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["proof_scope"] == "CONTRACT_VALIDATOR_PRECHECK"
    assert stdout_payload["system_execution_performed"] is False
    assert stdout_payload["http_endpoint_called"] is False
    assert all(item["system_evidence"] is None for item in stdout_payload["results"])
