"""Run the frozen 3-question smoke and/or 41-question answer re-evaluation.

Without ``--execute`` this command emits a plan only and never calls a model.
During execution it delegates generation to ``generate_answer_predictions.py``,
which enforces the query-only online phase and opens gold only after its raw QA
checkpoint is complete.  Scoring starts in a separate process afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.answer_metrics import (  # noqa: E402
    AnswerEvalDataset,
    AnswerEvaluationReport,
    AnswerPredictionProvenance,
    AnswerPredictionSet,
    PredictionStatus,
    evaluate_answers,
)
from creditlens.evaluation.answer_suite import (  # noqa: E402
    AnswerReevaluationManifest,
    AnswerSuiteRunRecord,
    AnswerSuiteStageName,
    AnswerSuiteStageRecord,
    build_scoring_dataset,
    resolve_project_path,
    sha256_file,
)
from creditlens.evaluation.source_state import (  # noqa: E402
    SourceStateEvidence,
    source_state_evidence_from_metadata,
    verify_captured_source_state,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "datasets" / "answer_reevaluation_manifest_v1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "reports" / "local"


def _subprocess_env() -> dict[str, str]:
    """Make captured child output deterministic across Windows code pages."""

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if isinstance(payload, BaseModel):
            # Stage records are updated incrementally.  Re-validate the complete
            # object before every checkpoint so assignment cannot bypass the
            # state-dependent Pydantic invariants.
            validated = type(payload).model_validate(payload.model_dump(mode="python"))
            rendered = validated.model_dump_json(indent=2)
        else:
            rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _with_stage_updates(
    run_record: AnswerSuiteRunRecord,
    stage_index: int,
    **updates: object,
) -> tuple[AnswerSuiteRunRecord, AnswerSuiteStageRecord]:
    """Return a fully revalidated immutable stage and enclosing run record."""

    stage_payload = run_record.stages[stage_index].model_dump(mode="python")
    stage_payload.update(updates)
    replacement = AnswerSuiteStageRecord.model_validate(stage_payload)
    stages = list(run_record.stages)
    stages[stage_index] = replacement
    run_payload = run_record.model_dump(mode="python")
    run_payload["stages"] = stages
    updated_run = AnswerSuiteRunRecord.model_validate(run_payload)
    return updated_run, updated_run.stages[stage_index]


def _load_manifest(path: Path) -> tuple[AnswerReevaluationManifest, str]:
    payload = path.read_bytes()
    import hashlib

    return (
        AnswerReevaluationManifest.model_validate_json(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _generation_command(
    manifest: AnswerReevaluationManifest,
    *,
    query_path: Path,
    answer_path: Path,
    gold_path: Path,
    predictions_path: Path,
    limit: int,
    execution_nonce: str,
    allow_disabled_llm: bool,
    expected_baseline: AnswerSuiteStageRecord | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_answer_predictions.py"),
        "--query-dataset",
        str(query_path),
        "--dataset",
        str(answer_path),
        "--gold-dataset",
        str(gold_path),
        "--output",
        str(predictions_path),
        "--top-k",
        str(manifest.top_k),
        "--limit",
        str(limit),
        "--execution-nonce",
        execution_nonce,
    ]
    if allow_disabled_llm:
        command.append("--allow-disabled-llm")
    if expected_baseline is not None:
        if (
            expected_baseline.runtime_profile_sha256 is None
            or expected_baseline.source_state_sha256 is None
            or expected_baseline.source_state_file_count is None
            or expected_baseline.git_commit is None
            or expected_baseline.git_dirty is None
        ):
            raise ValueError("completed smoke baseline is missing generation preflight fields")
        command.extend(
            [
                "--expected-runtime-profile-sha256",
                expected_baseline.runtime_profile_sha256,
                "--expected-source-state-sha256",
                expected_baseline.source_state_sha256,
                "--expected-source-state-file-count",
                str(expected_baseline.source_state_file_count),
                "--expected-git-commit",
                expected_baseline.git_commit,
                "--expected-git-dirty",
                str(expected_baseline.git_dirty).lower(),
            ]
        )
    return command


def _scoring_command(scoring_dataset: Path, predictions: Path, report: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_answer_evaluation.py"),
        "--dataset",
        str(scoring_dataset),
        "--predictions",
        str(predictions),
        "--output",
        str(report),
    ]


def _query_question_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("query dataset must contain a questions list")
    question_ids: list[str] = []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("query dataset questions must be JSON objects")
        question_id = item.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("query dataset question_id values must be non-blank strings")
        question_ids.append(question_id.strip())
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("query dataset question_id values must be unique")
    return question_ids


def _validated_prediction_source_state(
    metadata: dict[str, str | int | float | bool | None],
) -> SourceStateEvidence:
    evidence = source_state_evidence_from_metadata(metadata)
    if evidence.git_commit is None or evidence.git_dirty is None:
        raise ValueError("prediction provenance requires measured Git commit and dirty state")
    verify_captured_source_state(PROJECT_ROOT, evidence, strict_git=True)
    return evidence


def _metadata_sha256(
    metadata: dict[str, str | int | float | bool | None],
    field_name: str,
) -> str:
    value = metadata.get(field_name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"prediction provenance {field_name} must be a lowercase SHA-256")
    return value


def _validated_runtime_profile(
    metadata: dict[str, str | int | float | bool | None],
) -> str:
    """Verify the persisted canonical profile instead of trusting an opaque hash."""

    expected_sha256 = _metadata_sha256(metadata, "runtime_profile_sha256")
    raw_profile = metadata.get("runtime_profile_json")
    if not isinstance(raw_profile, str) or not raw_profile:
        raise ValueError("prediction provenance runtime_profile_json must be non-empty")
    parsed = json.loads(raw_profile)
    if not isinstance(parsed, dict):
        raise ValueError("prediction runtime profile must be a JSON object")
    canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if raw_profile != canonical:
        raise ValueError("prediction runtime profile is not canonical JSON")
    import hashlib

    actual_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("prediction runtime profile hash does not match its canonical JSON")
    return expected_sha256


def _validated_stage_source_state(stage: AnswerSuiteStageRecord) -> SourceStateEvidence:
    evidence = source_state_evidence_from_metadata(stage.model_dump(mode="python"))
    if evidence.git_commit is None or evidence.git_dirty is None:
        raise ValueError("smoke stage requires measured Git commit and dirty state")
    verify_captured_source_state(PROJECT_ROOT, evidence, strict_git=True)
    return evidence


def _validate_smoke_baseline_is_current(
    smoke: AnswerSuiteStageRecord,
    *,
    query_path: Path,
    answer_path: Path,
    gold_path: Path,
) -> SourceStateEvidence:
    source_state = _validated_stage_source_state(smoke)
    current_hashes = {
        "query_dataset_sha256": sha256_file(query_path),
        "answer_dataset_sha256": sha256_file(answer_path),
        "source_gold_sha256": sha256_file(gold_path),
    }
    mismatches = [
        field_name
        for field_name, expected in current_hashes.items()
        if getattr(smoke, field_name) != expected
    ]
    if mismatches:
        raise ValueError(f"smoke baseline dataset hashes changed: {sorted(mismatches)}")
    return source_state


def _validate_scoring_report(
    report_path: Path,
    *,
    scoring_dataset: AnswerEvalDataset,
    prediction_set: AnswerPredictionSet,
    expected_question_count: int,
    scoring_dataset_sha256: str,
    predictions_sha256: str,
) -> AnswerEvaluationReport:
    """Parse, bind and independently recompute an offline scorer report."""

    report = AnswerEvaluationReport.model_validate_json(report_path.read_bytes())
    identity_checks = {
        "dataset_id": (report.dataset_id, scoring_dataset.dataset_id),
        "dataset_version": (report.dataset_version, scoring_dataset.dataset_version),
        "prediction_set_id": (report.prediction_set_id, prediction_set.prediction_set_id),
        "prediction_adapter_version": (
            report.prediction_adapter_version,
            prediction_set.prediction_adapter_version,
        ),
        "dataset_sha256": (report.dataset_sha256, scoring_dataset_sha256),
        "predictions_sha256": (report.predictions_sha256, predictions_sha256),
        "total_questions": (report.summary.total_questions, expected_question_count),
        "question_score_count": (len(report.questions), expected_question_count),
    }
    mismatches = [
        field_name
        for field_name, (actual, expected) in identity_checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"scoring report identity/hash mismatch: {sorted(mismatches)}")

    expected_report = evaluate_answers(
        scoring_dataset,
        prediction_set,
        dataset_sha256=scoring_dataset_sha256,
        predictions_sha256=predictions_sha256,
    )
    actual_payload = report.model_dump(mode="json", exclude={"generated_at"})
    expected_payload = expected_report.model_dump(mode="json", exclude={"generated_at"})
    if actual_payload != expected_payload:
        raise ValueError("scoring report does not match independent deterministic recomputation")
    return report


def _selected_stages(
    manifest: AnswerReevaluationManifest,
    requested: str,
) -> list:
    if requested == "all":
        return manifest.stages
    target = AnswerSuiteStageName(requested)
    return [item for item in manifest.stages if item.name is target]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the frozen CreditLens v1.6 answer re-evaluation suite."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["smoke", "full", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the configured model service; omission is a no-network plan.",
    )
    parser.add_argument(
        "--allow-disabled-llm",
        action="store_true",
        help="Record explicit technical failures when LLM_PROVIDER=disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    smoke_baseline: AnswerSuiteStageRecord | None = None
    try:
        manifest_path = args.manifest.resolve()
        manifest, manifest_sha256 = _load_manifest(manifest_path)
        query_path = resolve_project_path(PROJECT_ROOT, manifest.query_dataset)
        query_question_ids = _query_question_ids(query_path)
        if len(query_question_ids) != 41:
            raise ValueError("the frozen query projection must contain exactly 41 questions")
        answer_path = resolve_project_path(PROJECT_ROOT, manifest.answer_dataset)
        gold_path = resolve_project_path(PROJECT_ROOT, manifest.source_gold_dataset)
        stages = _selected_stages(manifest, args.stage)
        if args.execute and args.stage == "full":
            raise ValueError(
                "full-only execution is disabled because it cannot prove the current provider "
                "profile passed smoke; run --stage all --execute"
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    started = datetime.now(UTC)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_root = args.output_root.resolve() / f"answer-reevaluation-{run_id}"
    records = [
        AnswerSuiteStageRecord(
            name=stage.name,
            status="PLANNED",
            expected_question_count=stage.expected_question_count,
            execution_nonce=f"{run_id}-{stage.name.value}",
        )
        for stage in stages
    ]
    run_record = AnswerSuiteRunRecord(
        protocol_id=manifest.protocol_id,
        protocol_version=manifest.protocol_version,
        run_id=run_id,
        generated_at=started.isoformat(),
        execution_requested=args.execute,
        gold_boundary=manifest.gold_boundary,
        manifest_sha256=manifest_sha256,
        stages=records,
    )
    record_path = run_root / "run_manifest.json"
    _atomic_write_json(record_path, run_record)

    if not args.execute:
        plan = {
            "warning": "PLAN ONLY: no model call was made and no metrics were produced.",
            "run_manifest": str(record_path),
            "stages": [item.model_dump(mode="json") for item in records],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    for stage_index, stage in enumerate(stages):
        baseline_source: SourceStateEvidence | None = None
        if stage.name is AnswerSuiteStageName.FULL:
            if smoke_baseline is None:
                run_record, _record = _with_stage_updates(
                    run_record,
                    stage_index,
                    status="FAILED",
                    failure_phase="VALIDATION",
                )
                _atomic_write_json(record_path, run_record)
                print("full stage reached without a verified smoke baseline", file=sys.stderr)
                return 1
            try:
                baseline_source = _validate_smoke_baseline_is_current(
                    smoke_baseline,
                    query_path=query_path,
                    answer_path=answer_path,
                    gold_path=gold_path,
                )
            except Exception as exc:
                run_record, _record = _with_stage_updates(
                    run_record,
                    stage_index,
                    status="FAILED",
                    failure_phase="VALIDATION",
                )
                _atomic_write_json(record_path, run_record)
                print(f"full-stage smoke baseline validation failed: {exc}", file=sys.stderr)
                return 1
        stage_root = run_root / stage.name.value
        predictions_path = stage_root / "predictions.json"
        scoring_dataset_path = stage_root / "scoring_dataset.json"
        report_path = stage_root / "report.json"
        run_record, record = _with_stage_updates(
            run_record,
            stage_index,
            status="RUNNING",
            predictions_path=str(predictions_path),
            scoring_dataset_path=str(scoring_dataset_path),
            report_path=str(report_path),
        )
        _atomic_write_json(record_path, run_record)

        try:
            generation = subprocess.run(
                _generation_command(
                    manifest,
                    query_path=query_path,
                    answer_path=answer_path,
                    gold_path=gold_path,
                    predictions_path=predictions_path,
                    limit=stage.question_limit,
                    execution_nonce=record.execution_nonce,
                    allow_disabled_llm=args.allow_disabled_llm,
                    expected_baseline=smoke_baseline
                    if stage.name is AnswerSuiteStageName.FULL
                    else None,
                ),
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=_subprocess_env(),
            )
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                generation_return_code=-1,
                status="FAILED",
                failure_phase="GENERATION",
            )
            _atomic_write_json(record_path, run_record)
            print(
                "answer re-evaluation generation process raised "
                f"{type(exc).__name__}; checkpoint retained",
                file=sys.stderr,
            )
            return 1
        if generation.returncode != 0:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                generation_return_code=generation.returncode,
                status="FAILED",
                failure_phase="GENERATION",
            )
            _atomic_write_json(record_path, run_record)
            print(
                "answer re-evaluation generation failed; inspect the safe run manifest",
                file=sys.stderr,
            )
            return generation.returncode or 1
        try:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                generation_return_code=0,
            )
            _atomic_write_json(record_path, run_record)
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="VALIDATION",
            )
            _atomic_write_json(record_path, run_record)
            print(f"generation checkpoint validation failed: {exc}", file=sys.stderr)
            return 1

        try:
            # This is the first suite-level read of answer labels/source gold.
            # The generator has already completed and checkpointed the online phase.
            prediction_set = AnswerPredictionSet.model_validate_json(predictions_path.read_bytes())
            answer_dataset = AnswerEvalDataset.model_validate_json(answer_path.read_bytes())
            scoring_dataset = build_scoring_dataset(
                answer_dataset,
                prediction_set,
                expected_count=stage.expected_question_count,
            )
            prediction_count = len(prediction_set.predictions)
            predicted_question_ids = [
                prediction.question_id for prediction in prediction_set.predictions
            ]
            expected_question_ids = query_question_ids[: stage.question_limit]
            if predicted_question_ids != expected_question_ids:
                raise ValueError(
                    "prediction question IDs/order do not match the frozen query stage prefix"
                )
            query_dataset_sha256 = sha256_file(query_path)
            answer_dataset_sha256 = sha256_file(answer_path)
            source_gold_sha256 = sha256_file(gold_path)
            predictions_sha256 = sha256_file(predictions_path)
            metadata = prediction_set.metadata
            source_state = _validated_prediction_source_state(metadata)
            runtime_profile_sha256 = _validated_runtime_profile(metadata)
            if baseline_source is not None and source_state != baseline_source:
                raise ValueError("full prediction source state differs from its smoke baseline")
            if (
                smoke_baseline is not None
                and stage.name is AnswerSuiteStageName.FULL
                and runtime_profile_sha256 != smoke_baseline.runtime_profile_sha256
            ):
                raise ValueError("full runtime profile differs from its smoke baseline")
            replay_count = sum(
                1
                for prediction in prediction_set.predictions
                if isinstance(prediction.provenance, AnswerPredictionProvenance)
                and prediction.provenance.idempotent_replay
            )
            if metadata.get("execution_nonce") != record.execution_nonce:
                raise ValueError("prediction execution nonce does not match this suite stage")
            if replay_count:
                raise ValueError(
                    "freshness gate failed: answer re-evaluation reused "
                    f"{replay_count} persisted QA run(s)"
                )
            technical_failures = sum(
                prediction.status is PredictionStatus.TECHNICAL_FAILURE
                for prediction in prediction_set.predictions
            )
            if stage.name is AnswerSuiteStageName.SMOKE and technical_failures:
                raise ValueError(
                    "smoke gate failed: expected zero technical failures, "
                    f"observed {technical_failures}"
                )
            expected_hashes = {
                "query_dataset_sha256": query_dataset_sha256,
                "answer_eval_dataset_sha256": answer_dataset_sha256,
                "source_gold_sha256": source_gold_sha256,
            }
            for key, expected in expected_hashes.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"prediction provenance hash mismatch: {key}")
            _atomic_write_json(scoring_dataset_path, scoring_dataset)
            scoring_dataset_sha256 = sha256_file(scoring_dataset_path)
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                prediction_count=prediction_count,
                idempotent_replay_count=replay_count,
                query_dataset_sha256=query_dataset_sha256,
                answer_dataset_sha256=answer_dataset_sha256,
                source_gold_sha256=source_gold_sha256,
                predictions_sha256=predictions_sha256,
                scoring_dataset_sha256=scoring_dataset_sha256,
                runtime_profile_sha256=runtime_profile_sha256,
                git_commit=source_state.git_commit,
                git_dirty=source_state.git_dirty,
                source_state_sha256=source_state.source_state_sha256,
                source_state_algorithm=source_state.source_state_algorithm,
                source_state_scope=source_state.source_state_scope,
                source_state_file_count=source_state.source_state_file_count,
                evidence_maturity=source_state.evidence_maturity,
            )
            _atomic_write_json(record_path, run_record)
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="VALIDATION",
            )
            _atomic_write_json(record_path, run_record)
            print(f"post-generation validation failed: {exc}", file=sys.stderr)
            return 1

        try:
            scoring = subprocess.run(
                _scoring_command(scoring_dataset_path, predictions_path, report_path),
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=_subprocess_env(),
            )
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                scoring_return_code=-1,
                status="FAILED",
                failure_phase="SCORING",
            )
            _atomic_write_json(record_path, run_record)
            print(
                "answer re-evaluation scoring process raised "
                f"{type(exc).__name__}; checkpoint retained",
                file=sys.stderr,
            )
            return 1
        if scoring.returncode != 0:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                scoring_return_code=scoring.returncode,
                status="FAILED",
                failure_phase="SCORING",
            )
            _atomic_write_json(record_path, run_record)
            print(
                "answer re-evaluation scoring failed; inspect the safe run manifest",
                file=sys.stderr,
            )
            return scoring.returncode or 1
        try:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                scoring_return_code=0,
            )
            _atomic_write_json(record_path, run_record)
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="VALIDATION",
            )
            _atomic_write_json(record_path, run_record)
            print(f"scoring checkpoint validation failed: {exc}", file=sys.stderr)
            return 1
        try:
            # Scoring is a separate process and can be long-running.  Re-bind
            # every frozen artifact and the source tree immediately before the
            # only transition that may claim this stage COMPLETED.
            verify_captured_source_state(PROJECT_ROOT, source_state, strict_git=True)
            final_artifact_hashes = {
                "predictions_sha256": sha256_file(predictions_path),
                "scoring_dataset_sha256": sha256_file(scoring_dataset_path),
            }
            changed_artifacts = [
                field_name
                for field_name, current_hash in final_artifact_hashes.items()
                if getattr(record, field_name) != current_hash
            ]
            if changed_artifacts:
                raise ValueError(
                    f"evaluation artifacts changed during scoring: {sorted(changed_artifacts)}"
                )
            if _validated_runtime_profile(prediction_set.metadata) != record.runtime_profile_sha256:
                raise ValueError("prediction runtime profile changed during scoring")
            report = _validate_scoring_report(
                report_path,
                scoring_dataset=scoring_dataset,
                prediction_set=prediction_set,
                expected_question_count=stage.expected_question_count,
                scoring_dataset_sha256=record.scoring_dataset_sha256 or "",
                predictions_sha256=record.predictions_sha256 or "",
            )
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                report_sha256=sha256_file(report_path),
                summary=report.summary,
                status="COMPLETED",
            )
            _atomic_write_json(record_path, run_record)
            if stage.name is AnswerSuiteStageName.SMOKE:
                smoke_baseline = record
        except KeyboardInterrupt:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="INTERRUPTED",
            )
            _atomic_write_json(record_path, run_record)
            print("answer re-evaluation interrupted; checkpoint retained", file=sys.stderr)
            return 130
        except Exception as exc:
            run_record, record = _with_stage_updates(
                run_record,
                stage_index,
                status="FAILED",
                failure_phase="VALIDATION",
            )
            _atomic_write_json(record_path, run_record)
            print(f"post-scoring validation failed: {exc}", file=sys.stderr)
            return 1

    print(run_record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
