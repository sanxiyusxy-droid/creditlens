"""v1.6 deterministic component ablation collection and integrity tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from creditlens.evaluation.agent_ablation import (
    AblationHarnessArtifact,
    AblationVariant,
    AgentAblationDataset,
    AgentAblationObservationSet,
    AgentAblationReport,
    ExecutionSemantics,
    build_observation_template,
    evaluate_agent_ablation,
)
from creditlens.evaluation.agent_ablation_collector import (
    collect_agent_ablation,
    discover_git_state,
)
from creditlens.evaluation.source_state import (
    EvidenceMaturity,
    classify_evidence_maturity,
    verify_source_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "agent_ablation_v1.json"


def _dataset() -> tuple[AgentAblationDataset, str]:
    payload = DATASET_PATH.read_bytes()
    return AgentAblationDataset.model_validate_json(payload), hashlib.sha256(payload).hexdigest()


def _collect(tmp_path: Path):
    dataset, dataset_hash = _dataset()
    output_dir = tmp_path / "ablation"
    git_commit, git_dirty = discover_git_state(PROJECT_ROOT)
    observations_path, report_path, report = collect_agent_ablation(
        dataset,
        dataset_sha256=dataset_hash,
        output_dir=output_dir,
        git_commit=git_commit,
        git_dirty=git_dirty,
        collector_version="test-component-collector-v1",
    )
    observations = AgentAblationObservationSet.model_validate_json(observations_path.read_bytes())
    return dataset, dataset_hash, output_dir, observations_path, report_path, observations, report


def _sidecar(output_dir: Path, observations, scenario_id: str, variant: AblationVariant):
    observation = next(
        item
        for item in observations.observations
        if item.scenario_id == scenario_id and item.variant is variant
    )
    path = output_dir / observation.source_artifact_path
    return observation, path, AblationHarnessArtifact.model_validate_json(path.read_bytes())


def test_collector_executes_complete_component_matrix_with_honest_scope(
    tmp_path: Path,
) -> None:
    _, _, output_dir, observations_path, report_path, observations, report = _collect(tmp_path)

    assert observations_path.is_file()
    assert report_path.is_file()
    assert len(observations.observations) == 24
    assert len({item.source_run_id for item in observations.observations}) == 24
    assert len(list((output_dir / "artifacts").glob("*.json"))) == 24
    assert {item.execution_semantics for item in observations.observations} == {
        ExecutionSemantics.DETERMINISTIC_COMPONENT_HARNESS
    }
    assert all(item.total_tokens is None for item in observations.observations)
    assert all(item.estimated_cost is None for item in observations.observations)
    assert observations.metadata.source_state_sha256 == report.source_state_sha256
    assert observations.metadata.evidence_maturity == report.evidence_maturity
    assert report.evidence_maturity is classify_evidence_maturity(
        git_commit=report.git_commit,
        git_dirty=report.git_dirty,
    )
    verify_source_state(
        PROJECT_ROOT,
        expected_sha256=report.source_state_sha256,
        expected_file_count=report.source_state_file_count,
    )

    by_variant = {item.variant: item for item in report.metrics}
    assert report.comparison_valid is True
    assert report.comparison_scope == "WITHIN_DETERMINISTIC_COMPONENT_HARNESS"
    assert (
        report.artifact_verification
        == "ALL_SIDECARS_SCHEMA_SHA256_AND_PRIMITIVE_RECOMPUTATION_VERIFIED"
    )
    assert any("does not execute the Supervisor" in item for item in report.limitations)
    assert by_variant[AblationVariant.FULL].unsupported_claim_interception_rate == 1.0
    assert by_variant[AblationVariant.NO_AUDITOR].unsupported_claim_interception_rate == 0.0
    assert by_variant[AblationVariant.FULL].counter_evidence_recall == 1.0
    assert by_variant[AblationVariant.NO_CHALLENGER].counter_evidence_recall == 0.0
    assert by_variant[AblationVariant.FULL].hitl_trials == 4
    assert by_variant[AblationVariant.NO_CHALLENGER].hitl_trials == 3
    assert by_variant[AblationVariant.NO_AUDITOR].hitl_trials == 0
    assert all(item.token_coverage == 0.0 for item in report.metrics)
    assert all(item.cost_coverage == 0.0 for item in report.metrics)
    assert all(item.total_tokens is None for item in report.metrics)
    assert all(item.estimated_cost is None for item in report.metrics)


def test_report_rejects_release_maturity_for_dirty_observations(tmp_path: Path) -> None:
    _, _, _, _, _, _, report = _collect(tmp_path)
    payload = report.model_dump(mode="json")
    payload["git_dirty"] = True
    payload["evidence_maturity"] = EvidenceMaturity.RELEASE_CANDIDATE.value

    with pytest.raises(ValidationError, match="does not match git_commit/git_dirty"):
        AgentAblationReport.model_validate(payload)


def test_sidecars_prove_which_production_primitives_actually_executed(tmp_path: Path) -> None:
    _, _, output_dir, _, _, observations, _ = _collect(tmp_path)

    _, _, unsupported_full = _sidecar(
        output_dir,
        observations,
        "unsupported_numeric_without_fact",
        AblationVariant.FULL,
    )
    assert unsupported_full.contract_validation.executed is True
    assert any(
        "NUMERIC_CLAIM_WITHOUT_FACT" in item
        for item in unsupported_full.contract_validation.violations
    )

    _, _, conflict_full = _sidecar(
        output_dir,
        observations,
        "same_period_numeric_conflict",
        AblationVariant.FULL,
    )
    assert conflict_full.conflict_assessments[0].expected_conflict is True
    assert conflict_full.conflict_assessments[0].is_conflict is True
    assert conflict_full.outcome.hitl_triggered is True

    _, _, period_full = _sidecar(
        output_dir,
        observations,
        "period_mismatch_supplement",
        AblationVariant.FULL,
    )
    assert period_full.conflict_assessments[0].expected_conflict is False
    assert period_full.conflict_assessments[0].is_conflict is False
    assert "期间不一致" in period_full.conflict_assessments[0].reason
    assert period_full.outcome.hitl_triggered is False

    _, _, no_auditor = _sidecar(
        output_dir,
        observations,
        "unsupported_numeric_without_fact",
        AblationVariant.NO_AUDITOR,
    )
    assert no_auditor.contract_validation.executed is False
    assert no_auditor.outcome.unsupported_claim_ids_intercepted == []

    _, _, no_challenger = _sidecar(
        output_dir,
        observations,
        "same_period_numeric_conflict",
        AblationVariant.NO_CHALLENGER,
    )
    assert no_challenger.conflict_assessments == []
    assert no_challenger.outcome.counter_evidence_target_ids_found == []


def test_incomplete_matrix_is_rejected_instead_of_scored(tmp_path: Path) -> None:
    dataset, dataset_hash, output_dir, _, _, observations, _ = _collect(tmp_path)
    observations = observations.model_copy(
        update={"observations": observations.observations[:-1]},
        deep=True,
    )

    with pytest.raises(ValueError, match="incomplete ablation matrix"):
        evaluate_agent_ablation(
            dataset,
            observations,
            dataset_sha256=dataset_hash,
            observations_sha256="b" * 64,
            artifact_root=output_dir,
        )


def test_tampered_sidecar_sha_is_rejected_before_scoring(tmp_path: Path) -> None:
    dataset, dataset_hash, output_dir, _, _, observations, _ = _collect(tmp_path)
    target = output_dir / observations.observations[0].source_artifact_path
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_agent_ablation(
            dataset,
            observations,
            dataset_sha256=dataset_hash,
            observations_sha256="c" * 64,
            artifact_root=output_dir,
        )


def test_sidecar_observation_correlation_is_fail_closed(tmp_path: Path) -> None:
    dataset, dataset_hash, output_dir, _, _, observations, _ = _collect(tmp_path)
    first = observations.observations[0].model_copy(update={"hitl_triggered": False})
    changed = observations.model_copy(
        update={"observations": [first, *observations.observations[1:]]},
        deep=True,
    )

    with pytest.raises(ValueError, match="sidecar/observation mismatch"):
        evaluate_agent_ablation(
            dataset,
            changed,
            dataset_sha256=dataset_hash,
            observations_sha256="d" * 64,
            artifact_root=output_dir,
        )


def test_synchronized_sidecar_and_observation_metric_forgery_is_recomputed(
    tmp_path: Path,
) -> None:
    dataset, dataset_hash, output_dir, _, _, observations, _ = _collect(tmp_path)
    target_index = next(
        index
        for index, item in enumerate(observations.observations)
        if item.scenario_id == "unsupported_numeric_without_fact"
        and item.variant is AblationVariant.FULL
    )
    original = observations.observations[target_index]
    sidecar_path = output_dir / original.source_artifact_path
    sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_payload["outcome"]["unsupported_claim_ids_intercepted"] = []
    sidecar_payload["outcome"]["hitl_triggered"] = False
    sidecar_payload["outcome"]["terminal_status"] = "COMPLETED"
    rewritten = (json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n").encode()
    sidecar_path.write_bytes(rewritten)
    changed_observation = original.model_copy(
        update={
            "source_artifact_sha256": hashlib.sha256(rewritten).hexdigest(),
            "unsupported_claim_ids_intercepted": [],
            "hitl_triggered": False,
            "terminal_status": "COMPLETED",
        }
    )
    changed_items = list(observations.observations)
    changed_items[target_index] = changed_observation
    changed = observations.model_copy(update={"observations": changed_items}, deep=True)

    with pytest.raises(ValueError, match="intercepted labels do not match contract recomputation"):
        evaluate_agent_ablation(
            dataset,
            changed,
            dataset_sha256=dataset_hash,
            observations_sha256="e" * 64,
            artifact_root=output_dir,
        )


def test_artifact_path_must_be_relative_normalized_json(tmp_path: Path) -> None:
    _, _, _, _, _, observations, _ = _collect(tmp_path)
    payload = observations.observations[0].model_dump(mode="json")
    payload["source_artifact_path"] = "../outside.json"

    with pytest.raises(ValidationError, match="normalized relative POSIX JSON path"):
        type(observations.observations[0]).model_validate(payload)


def test_template_is_explicitly_not_run_and_cannot_validate_as_results() -> None:
    dataset, _ = _dataset()
    template = build_observation_template(dataset)

    assert template["template_only"] is True
    assert {item["status"] for item in template["observations"]} == {"NOT_RUN"}
    assert all(item["source_run_id"] is None for item in template["observations"])
    assert all(item["source_artifact_path"] is None for item in template["observations"])
    with pytest.raises(ValidationError):
        AgentAblationObservationSet.model_validate(json.loads(json.dumps(template)))


def test_cli_collects_sidecars_observations_and_verified_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-ablation"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_agent_ablation.py"),
            "--collect",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert summary["execution_semantics"] == "DETERMINISTIC_COMPONENT_HARNESS"
    assert report["comparison_valid"] is True
    assert len(report["metrics"]) == 4
    assert len(list((output_dir / "artifacts").glob("*.json"))) == 24


def test_cli_scoring_revalidates_collected_sidecar_hashes(tmp_path: Path) -> None:
    _, _, output_dir, observations_path, _, observations, _ = _collect(tmp_path)
    target = output_dir / observations.observations[0].source_artifact_path
    target.write_bytes(target.read_bytes() + b"tampered")
    report_path = output_dir / "rescored.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_agent_ablation.py"),
            "--observations",
            str(observations_path),
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert not report_path.exists()


def test_report_rejects_vacuous_or_internally_inconsistent_metrics(tmp_path: Path) -> None:
    _, _, _, _, _, _, report = _collect(tmp_path)
    empty_payload = report.model_dump(mode="json")
    empty_payload["metrics"] = []
    with pytest.raises(ValidationError):
        AgentAblationReport.model_validate(empty_payload)

    duplicate_payload = report.model_dump(mode="json")
    duplicate_payload["metrics"][1]["variant"] = duplicate_payload["metrics"][0]["variant"]
    with pytest.raises(ValidationError, match="exactly one metric for every frozen variant"):
        AgentAblationReport.model_validate(duplicate_payload)

    bad_rate_payload = report.model_dump(mode="json")
    bad_rate_payload["metrics"][0]["hitl_rate"] = 0.0
    with pytest.raises(ValidationError, match="HITL rate does not match"):
        AgentAblationReport.model_validate(bad_rate_payload)
