"""v1.6 frozen answer-suite protocol tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_v16_answer_reevaluation as suite_runner

from creditlens.evaluation.answer_metrics import (
    AnswerEvalDataset,
    AnswerPrediction,
    AnswerPredictionSet,
    PredictionStatus,
    evaluate_answers,
)
from creditlens.evaluation.answer_suite import (
    AnswerReevaluationManifest,
    AnswerSuiteStageName,
    AnswerSuiteStageRecord,
    build_scoring_dataset,
)
from creditlens.evaluation.source_state import capture_source_state_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "answer_reevaluation_manifest_v1.json"
ANSWER_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "answer_eval_v1.json"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "evaluation" / "predictions" / "answer_eval_v1_c4289a1_20260810T125041Z.json"
)
FAKE_RUNTIME_PROFILE_JSON = '{"profile":"v16-suite-unit-test"}'
FAKE_RUNTIME_PROFILE_SHA256 = hashlib.sha256(FAKE_RUNTIME_PROFILE_JSON.encode("utf-8")).hexdigest()


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _install_fake_answer_pipeline(
    monkeypatch,
    *,
    technical_failure: bool = False,
    malformed_report: bool = False,
    mismatched_report: bool = False,
    wrong_query_prefix: bool = False,
    runtime_exception_script: str | None = None,
) -> list[str]:
    calls: list[str] = []

    def fake_run(command: list[str], **_kwargs):
        script_name = Path(command[1]).name
        calls.append(script_name)
        if script_name == runtime_exception_script:
            raise RuntimeError("synthetic subprocess orchestration failure")
        if script_name == "generate_answer_predictions.py":
            base = AnswerPredictionSet.model_validate_json(PREDICTIONS_PATH.read_bytes())
            limit = int(_argument(command, "--limit"))
            start = 1 if wrong_query_prefix else 0
            selected = []
            for prediction in base.predictions[start : start + limit]:
                provenance = prediction.provenance
                if hasattr(provenance, "idempotent_replay"):
                    provenance = provenance.model_copy(update={"idempotent_replay": False})
                selected.append(prediction.model_copy(update={"provenance": provenance}))
            if technical_failure:
                selected[0] = AnswerPrediction(
                    question_id=selected[0].question_id,
                    status=PredictionStatus.TECHNICAL_FAILURE,
                    error_type="SYNTHETIC_PROVIDER_FAILURE",
                    error_message="safe synthetic test failure",
                )

            manifest = AnswerReevaluationManifest.model_validate_json(MANIFEST_PATH.read_bytes())
            query_path = PROJECT_ROOT / manifest.query_dataset
            answer_path = PROJECT_ROOT / manifest.answer_dataset
            gold_path = PROJECT_ROOT / manifest.source_gold_dataset
            source_state = capture_source_state_evidence(PROJECT_ROOT, strict_git=True)
            if limit == 41:
                assert _argument(command, "--expected-runtime-profile-sha256") == (
                    FAKE_RUNTIME_PROFILE_SHA256
                )
                assert _argument(command, "--expected-source-state-sha256") == (
                    source_state.source_state_sha256
                )
                assert int(_argument(command, "--expected-source-state-file-count")) == (
                    source_state.source_state_file_count
                )
                assert _argument(command, "--expected-git-commit") == source_state.git_commit
                assert (
                    _argument(command, "--expected-git-dirty")
                    == str(source_state.git_dirty).lower()
                )
            prediction_set = base.model_copy(
                update={
                    "prediction_set_id": "v16-suite-unit-test",
                    "predictions": selected,
                    "metadata": {
                        **source_state.as_metadata(),
                        "execution_nonce": _argument(command, "--execution-nonce"),
                        "query_dataset_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
                        "answer_eval_dataset_sha256": hashlib.sha256(
                            answer_path.read_bytes()
                        ).hexdigest(),
                        "source_gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
                        "runtime_profile_sha256": FAKE_RUNTIME_PROFILE_SHA256,
                        "runtime_profile_json": FAKE_RUNTIME_PROFILE_JSON,
                    },
                },
                deep=True,
            )
            output_path = Path(_argument(command, "--output"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                prediction_set.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        elif script_name == "run_answer_evaluation.py":
            dataset_path = Path(_argument(command, "--dataset"))
            predictions_path = Path(_argument(command, "--predictions"))
            report_path = Path(_argument(command, "--output"))
            if malformed_report:
                report_path.write_text(
                    json.dumps({"summary": {"technical_failures": 0}}),
                    encoding="utf-8",
                )
            else:
                dataset_bytes = dataset_path.read_bytes()
                prediction_bytes = predictions_path.read_bytes()
                dataset = AnswerEvalDataset.model_validate_json(dataset_bytes)
                prediction_set = AnswerPredictionSet.model_validate_json(prediction_bytes)
                report = evaluate_answers(
                    dataset,
                    prediction_set,
                    dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
                    predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
                )
                if mismatched_report:
                    payload = report.model_dump(mode="json")
                    payload["prediction_set_id"] = "unrelated-prediction-set"
                    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
                else:
                    rendered = report.model_dump_json(indent=2)
                report_path.write_text(rendered + "\n", encoding="utf-8")
        else:  # pragma: no cover - protects the fake from silently accepting new subprocesses
            raise AssertionError(f"unexpected suite subprocess: {command}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    # Replace only the runner's namespace.  Patching ``subprocess.run`` on the
    # shared module would also intercept source-state Git probes.
    monkeypatch.setattr(suite_runner, "subprocess", SimpleNamespace(run=fake_run))
    return calls


def test_frozen_manifest_defines_exact_three_and_forty_one_question_stages() -> None:
    manifest = AnswerReevaluationManifest.model_validate_json(MANIFEST_PATH.read_bytes())

    assert manifest.gold_boundary == "ONLINE_QUERY_ONLY_THEN_OFFLINE_GOLD_MAPPING"
    assert [(item.name, item.question_limit) for item in manifest.stages] == [
        (AnswerSuiteStageName.SMOKE, 3),
        (AnswerSuiteStageName.FULL, 41),
    ]


def test_manifest_rejects_full_before_smoke() -> None:
    manifest = AnswerReevaluationManifest.model_validate_json(MANIFEST_PATH.read_bytes())
    payload = manifest.model_dump(mode="json")
    payload["stages"] = list(reversed(payload["stages"]))

    with pytest.raises(ValueError, match="smoke then full"):
        AnswerReevaluationManifest.model_validate(payload)


def test_scoring_projection_contains_only_completed_prediction_ids() -> None:
    dataset = AnswerEvalDataset.model_validate_json(ANSWER_DATASET_PATH.read_bytes())
    all_predictions = AnswerPredictionSet.model_validate_json(PREDICTIONS_PATH.read_bytes())
    predictions = all_predictions.model_copy(
        update={"predictions": all_predictions.predictions[:3]},
        deep=True,
    )

    projected = build_scoring_dataset(dataset, predictions, expected_count=3)

    assert len(projected.questions) == 3
    assert {item.question_id for item in projected.questions} == {
        item.question_id for item in predictions.predictions
    }


def test_scoring_projection_rejects_partial_checkpoint_count() -> None:
    dataset = AnswerEvalDataset.model_validate_json(ANSWER_DATASET_PATH.read_bytes())
    predictions = AnswerPredictionSet.model_validate_json(PREDICTIONS_PATH.read_bytes())

    with pytest.raises(ValueError, match="expected 3"):
        build_scoring_dataset(dataset, predictions, expected_count=3)


def test_answer_suite_stage_can_record_an_operator_interruption() -> None:
    record = AnswerSuiteStageRecord(
        name=AnswerSuiteStageName.FULL,
        status="FAILED",
        expected_question_count=41,
        execution_nonce="interrupted-full-run",
        failure_phase="INTERRUPTED",
    )

    assert record.failure_phase == "INTERRUPTED"


def test_stage_record_is_immutable_after_validation() -> None:
    record = AnswerSuiteStageRecord(
        name=AnswerSuiteStageName.SMOKE,
        status="PLANNED",
        expected_question_count=3,
        execution_nonce="immutable-smoke",
    )

    with pytest.raises(ValueError, match="frozen"):
        record.status = "COMPLETED"


def test_completed_stage_rejects_vacuous_success() -> None:
    with pytest.raises(ValueError, match="COMPLETED stages require artifacts and hashes"):
        AnswerSuiteStageRecord(
            name=AnswerSuiteStageName.FULL,
            status="COMPLETED",
            expected_question_count=41,
            execution_nonce="vacuous-completion",
        )


def test_answer_suite_defaults_to_plan_and_produces_no_metrics(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_v16_answer_reevaluation.py"),
            "--stage",
            "all",
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    public_plan = json.loads(completed.stdout)
    assert "no model call" in public_plan["warning"]
    record = json.loads(Path(public_plan["run_manifest"]).read_text(encoding="utf-8"))
    assert record["execution_requested"] is False
    assert [item["status"] for item in record["stages"]] == ["PLANNED", "PLANNED"]
    assert len({item["execution_nonce"] for item in record["stages"]}) == 2
    assert all(item["idempotent_replay_count"] is None for item in record["stages"])
    assert all(item["summary"] is None for item in record["stages"])
    assert not list(tmp_path.rglob("predictions.json"))


def test_smoke_technical_failure_fails_gate_before_paid_scoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch, technical_failure=True)

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert calls == ["generate_answer_predictions.py"]
    record_path = next(tmp_path.rglob("run_manifest.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["stages"][0]["status"] == "FAILED"
    assert record["stages"][0]["failure_phase"] == "VALIDATION"
    assert record["stages"][0]["scoring_return_code"] is None


def test_smoke_rejects_predictions_outside_the_frozen_query_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch, wrong_query_prefix=True)

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert calls == ["generate_answer_predictions.py"]
    stage = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))[
        "stages"
    ][0]
    assert stage["status"] == "FAILED"
    assert stage["failure_phase"] == "VALIDATION"


def test_full_only_execution_requires_an_intact_passed_smoke_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("full generation must not start before the smoke gate")

    monkeypatch.setattr(suite_runner.subprocess, "run", unexpected_subprocess)

    with pytest.raises(SystemExit) as exc_info:
        suite_runner.main(["--stage", "full", "--execute", "--output-root", str(tmp_path)])

    assert exc_info.value.code == 2
    assert not list(tmp_path.rglob("run_manifest.json"))


def test_malformed_scorer_output_fails_validation_instead_of_completing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch, malformed_report=True)

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert calls == ["generate_answer_predictions.py", "run_answer_evaluation.py"]
    record = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))
    assert record["stages"][0]["status"] == "FAILED"
    assert record["stages"][0]["failure_phase"] == "VALIDATION"
    assert record["stages"][0]["scoring_return_code"] == 0
    assert record["stages"][0]["report_sha256"] is None


def test_valid_but_mismatched_scorer_report_cannot_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch, mismatched_report=True)

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert calls == ["generate_answer_predictions.py", "run_answer_evaluation.py"]
    stage = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))[
        "stages"
    ][0]
    assert stage["status"] == "FAILED"
    assert stage["failure_phase"] == "VALIDATION"
    assert stage["report_sha256"] is None


def test_validated_scorer_output_completes_with_full_hash_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch)

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 0
    assert calls == ["generate_answer_predictions.py", "run_answer_evaluation.py"]
    record = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))
    stage = record["stages"][0]
    assert stage["status"] == "COMPLETED"
    assert stage["prediction_count"] == 3
    assert stage["idempotent_replay_count"] == 0
    assert stage["summary"]["technical_failures"] == 0
    assert stage["summary"]["missing_predictions"] == 0
    assert all(
        len(stage[field]) == 64
        for field in (
            "query_dataset_sha256",
            "answer_dataset_sha256",
            "source_gold_sha256",
            "predictions_sha256",
            "scoring_dataset_sha256",
            "report_sha256",
            "runtime_profile_sha256",
        )
    )


def test_stage_all_completes_smoke_then_full_with_one_source_and_runtime_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_answer_pipeline(monkeypatch)

    exit_code = suite_runner.main(["--stage", "all", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 0
    assert calls == [
        "generate_answer_predictions.py",
        "run_answer_evaluation.py",
        "generate_answer_predictions.py",
        "run_answer_evaluation.py",
    ]
    record = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))
    smoke, full = record["stages"]
    assert [
        (smoke["name"], smoke["prediction_count"], smoke["status"]),
        (full["name"], full["prediction_count"], full["status"]),
    ] == [
        ("smoke", 3, "COMPLETED"),
        ("full", 41, "COMPLETED"),
    ]
    assert smoke["source_state_sha256"] == full["source_state_sha256"]
    assert smoke["git_commit"] == full["git_commit"]
    assert smoke["git_dirty"] == full["git_dirty"]
    assert (
        smoke["runtime_profile_sha256"]
        == full["runtime_profile_sha256"]
        == FAKE_RUNTIME_PROFILE_SHA256
    )


@pytest.mark.parametrize(
    ("runtime_exception_script", "expected_calls", "failure_phase", "return_code_field"),
    [
        (
            "generate_answer_predictions.py",
            ["generate_answer_predictions.py"],
            "GENERATION",
            "generation_return_code",
        ),
        (
            "run_answer_evaluation.py",
            ["generate_answer_predictions.py", "run_answer_evaluation.py"],
            "SCORING",
            "scoring_return_code",
        ),
    ],
)
def test_unknown_subprocess_exception_never_leaves_stage_running(
    tmp_path: Path,
    monkeypatch,
    runtime_exception_script: str,
    expected_calls: list[str],
    failure_phase: str,
    return_code_field: str,
) -> None:
    calls = _install_fake_answer_pipeline(
        monkeypatch,
        runtime_exception_script=runtime_exception_script,
    )

    exit_code = suite_runner.main(["--stage", "smoke", "--execute", "--output-root", str(tmp_path)])

    assert exit_code == 1
    assert calls == expected_calls
    stage = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text(encoding="utf-8"))[
        "stages"
    ][0]
    assert stage["status"] == "FAILED"
    assert stage["failure_phase"] == failure_phase
    assert stage[return_code_field] == -1
    assert all(item["status"] != "RUNNING" for item in [stage])
