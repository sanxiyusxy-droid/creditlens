"""Offline collector for the honest v1.6 deterministic component ablation.

This module executes production *deterministic primitives*, not the online
Supervisor graph.  Its sidecars make that boundary inspectable for every cell.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from creditlens.agents.challenger import assess_conflict
from creditlens.agents.contracts import (
    AgentArtifact,
    AgentClaim,
    AgentEvidenceRef,
    validate_artifact_contract,
)
from creditlens.evaluation.agent_ablation import (
    HARNESS_LIMITATIONS,
    AblationHarnessArtifact,
    AblationHarnessOutcome,
    AblationObservationMetadata,
    AblationScenario,
    AblationTrialObservation,
    AblationVariant,
    AgentAblationDataset,
    AgentAblationObservationSet,
    AgentAblationReport,
    ConflictAssessmentTrace,
    ContractValidationTrace,
    ExecutionSemantics,
    HarnessEvidenceMode,
    evaluate_agent_ablation,
)
from creditlens.evaluation.source_state import (
    build_source_state_evidence,
    verify_captured_source_state,
)
from creditlens.evaluation.source_state import discover_git_state as measure_git_state

COLLECTOR_VERSION = "1.0.0"


def discover_git_state(project_root: Path) -> tuple[str, bool]:
    """Return the measured source revision and dirty state; never invent either."""

    commit, dirty = measure_git_state(project_root, strict=True)
    if commit is None or dirty is None:  # pragma: no cover - strict mode guarantees both.
        raise ValueError("unable to measure git revision/dirty state")
    return commit, dirty


def _stable_uuid(run_id: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(run_id, name)


def _build_input_artifact(
    run_id: uuid.UUID,
    scenario: AblationScenario,
) -> AgentArtifact:
    fixture = scenario.component_harness
    evidence_id = _stable_uuid(run_id, "primary-evidence")
    evidence: list[AgentEvidenceRef] = []
    if fixture.evidence_mode != HarnessEvidenceMode.UNKNOWN_REFERENCE:
        evidence_kwargs: dict = {}
        if fixture.evidence_mode == HarnessEvidenceMode.DOCUMENT_SPAN:
            evidence_kwargs.update(
                document_version_id=_stable_uuid(run_id, "document-version"),
                section_id=_stable_uuid(run_id, "section"),
                parse_run_id=_stable_uuid(run_id, "parse-run"),
                page_number=1,
            )
        elif fixture.evidence_mode == HarnessEvidenceMode.SQL_FACT:
            evidence_kwargs["fact_id"] = _stable_uuid(run_id, "fact")
        elif fixture.evidence_mode == HarnessEvidenceMode.POLICY_RULE:
            evidence_kwargs["valid_from"] = fixture.as_of_date
        evidence.append(
            AgentEvidenceRef(
                evidence_id=evidence_id,
                evidence_type=fixture.evidence_mode.value,
                source_id=_stable_uuid(run_id, "source"),
                content_hash=hashlib.sha256(fixture.claim_statement.encode("utf-8")).hexdigest(),
                source_available_at=datetime.combine(
                    fixture.as_of_date,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                **evidence_kwargs,
            )
        )
    claim = AgentClaim(
        claim_id=_stable_uuid(run_id, "claim"),
        category=fixture.claim_category,
        statement=fixture.claim_statement,
        verdict="SUPPORTED",
        severity="HIGH",
        supporting_evidence_ids=[evidence_id],
        as_of_date=fixture.as_of_date,
    )
    return AgentArtifact(
        artifact_id=_stable_uuid(run_id, "input-artifact"),
        run_id=run_id,
        task_id=f"ablation:{scenario.scenario_id}",
        producer="deterministic_component_harness",
        input_hash=hashlib.sha256(
            scenario.component_harness.model_dump_json().encode("utf-8")
        ).hexdigest(),
        claims=[claim],
        evidence=evidence,
    )


def _execute_cell(
    scenario: AblationScenario,
    *,
    variant: AblationVariant,
    disabled_components: list[str],
    collector_version: str,
) -> tuple[AblationHarnessArtifact, AblationTrialObservation]:
    run_id = uuid.uuid4()
    started_ns = time.perf_counter_ns()
    artifact = _build_input_artifact(run_id, scenario)

    auditor_enabled = "auditor" not in disabled_components
    challenger_enabled = "challenger" not in disabled_components
    if auditor_enabled:
        validation = validate_artifact_contract(
            artifact,
            scenario.component_harness.as_of_date,
        )
        contract_trace = ContractValidationTrace(
            component_enabled=True,
            executed=True,
            enforced=True,
            ok=validation.ok,
            violations=validation.violations,
        )
    else:
        contract_trace = ContractValidationTrace(
            component_enabled=False,
            executed=False,
            enforced=False,
        )

    conflict_traces: list[ConflictAssessmentTrace] = []
    if challenger_enabled:
        for counter in scenario.component_harness.counter_evidence:
            is_conflict, reason = assess_conflict(
                scenario.component_harness.claim_statement,
                counter.text,
            )
            conflict_traces.append(
                ConflictAssessmentTrace(
                    target_id=counter.target_id,
                    claim_text=scenario.component_harness.claim_statement,
                    counter_text=counter.text,
                    expected_conflict=counter.expected_conflict,
                    is_conflict=is_conflict,
                    reason=reason,
                )
            )

    emitted = list(scenario.unsupported_claim_ids)
    intercepted = []
    if contract_trace.executed:
        intercepted = [
            label
            for label, expected_fragment in (
                scenario.component_harness.expected_contract_violation_substrings.items()
            )
            if any(expected_fragment in violation for violation in contract_trace.violations)
        ]
    counter_found = [item.target_id for item in conflict_traces]
    direct_conflict_found = any(item.is_conflict for item in conflict_traces)
    hitl_triggered = auditor_enabled and bool(intercepted or direct_conflict_found)
    terminal_status = "HUMAN_REVIEW" if hitl_triggered else "COMPLETED"
    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    recorded_at = datetime.now(UTC)
    outcome = AblationHarnessOutcome(
        unsupported_claim_ids_emitted=emitted,
        unsupported_claim_ids_intercepted=intercepted,
        counter_evidence_target_ids_found=counter_found,
        hitl_triggered=hitl_triggered,
        terminal_status=terminal_status,
    )
    sidecar = AblationHarnessArtifact(
        artifact_type="creditlens_agent_ablation_component_harness_v1",
        protocol_id="creditlens_multi_agent_ablation_v1",
        protocol_version="1.0.0",
        collector_version=collector_version,
        scenario_id=scenario.scenario_id,
        variant=variant,
        execution_semantics=ExecutionSemantics.DETERMINISTIC_COMPONENT_HARNESS,
        source_run_id=run_id,
        recorded_at=recorded_at,
        disabled_components=list(disabled_components),
        challenger_enabled=challenger_enabled,
        auditor_enabled=auditor_enabled,
        input_artifact=artifact,
        contract_validation=contract_trace,
        conflict_assessments=conflict_traces,
        outcome=outcome,
        measured_latency_ms=latency_ms,
        limitations=list(HARNESS_LIMITATIONS),
    )
    observation = AblationTrialObservation(
        scenario_id=scenario.scenario_id,
        variant=variant,
        execution_semantics=ExecutionSemantics.DETERMINISTIC_COMPONENT_HARNESS,
        source_run_id=run_id,
        source_artifact_path="pending.json",
        source_artifact_sha256="0" * 64,
        recorded_at=recorded_at,
        unsupported_claim_ids_emitted=emitted,
        unsupported_claim_ids_intercepted=intercepted,
        counter_evidence_target_ids_found=counter_found,
        hitl_triggered=hitl_triggered,
        latency_ms=latency_ms,
        terminal_status=terminal_status,
    )
    return sidecar, observation


def _write_json(path: Path, model) -> str:
    payload = (model.model_dump_json(indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def collect_agent_ablation(
    dataset: AgentAblationDataset,
    *,
    dataset_sha256: str,
    output_dir: Path,
    git_commit: str,
    git_dirty: bool,
    collector_version: str = COLLECTOR_VERSION,
    project_root: Path | None = None,
) -> tuple[Path, Path, AgentAblationReport]:
    """Execute all 24 cells, persist sidecars, verify them, then score."""

    resolved_project_root = project_root or Path(__file__).resolve().parents[3]
    source_state = build_source_state_evidence(
        resolved_project_root,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--collect output directory must be empty")
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    definitions = {definition.variant: definition for definition in dataset.variants}
    observations: list[AblationTrialObservation] = []
    for scenario in dataset.scenarios:
        for variant in AblationVariant:
            definition = definitions[variant]
            sidecar, pending = _execute_cell(
                scenario,
                variant=variant,
                disabled_components=definition.disabled_components,
                collector_version=collector_version,
            )
            filename = (
                f"{scenario.scenario_id}--{variant.value.lower()}--{pending.source_run_id}.json"
            )
            relative_path = (Path("artifacts") / filename).as_posix()
            artifact_hash = _write_json(output_dir / relative_path, sidecar)
            observations.append(
                pending.model_copy(
                    update={
                        "source_artifact_path": relative_path,
                        "source_artifact_sha256": artifact_hash,
                    }
                )
            )

    observation_set = AgentAblationObservationSet(
        observation_set_id=f"component-harness-{uuid.uuid4()}",
        protocol_id=dataset.protocol_id,
        protocol_version=dataset.protocol_version,
        observations=observations,
        metadata=AblationObservationMetadata(
            collector_version=collector_version,
            git_commit=source_state.git_commit,
            git_dirty=source_state.git_dirty,
            source_state_sha256=source_state.source_state_sha256,
            source_state_algorithm=source_state.source_state_algorithm,
            source_state_scope=source_state.source_state_scope,
            source_state_file_count=source_state.source_state_file_count,
            evidence_maturity=source_state.evidence_maturity,
            dataset_sha256=dataset_sha256,
        ),
    )
    verify_captured_source_state(resolved_project_root, source_state, strict_git=True)
    observations_path = output_dir / "observations.json"
    observations_hash = _write_json(observations_path, observation_set)
    report = evaluate_agent_ablation(
        dataset,
        observation_set,
        dataset_sha256=dataset_sha256,
        observations_sha256=observations_hash,
        artifact_root=output_dir,
    )
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    return observations_path, report_path, report
