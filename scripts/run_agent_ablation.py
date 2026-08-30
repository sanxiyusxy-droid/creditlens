"""Collect or score the offline deterministic component ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.agent_ablation import (  # noqa: E402
    AgentAblationDataset,
    AgentAblationObservationSet,
    build_observation_template,
    evaluate_agent_ablation,
)
from creditlens.evaluation.agent_ablation_collector import (  # noqa: E402
    collect_agent_ablation,
    discover_git_state,
)
from creditlens.evaluation.source_state import verify_source_state  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "agent_ablation_v1.json"


def _load(path: Path, model_type):
    payload = path.read_bytes()
    return model_type.model_validate_json(payload), hashlib.sha256(payload).hexdigest()


def _write(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.rstrip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a complete, evidence-backed Multi-Agent ablation matrix."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observations", type=Path, help="Completed observation set to score.")
    mode.add_argument(
        "--write-template",
        type=Path,
        help="Write NOT_RUN cells only. The template is deliberately not scoreable.",
    )
    mode.add_argument(
        "--collect",
        type=Path,
        metavar="OUTPUT_DIR",
        help=(
            "Execute 6 frozen scenarios x 4 variants in the offline deterministic "
            "component harness, then write sidecars, observations.json and report.json."
        ),
    )
    parser.add_argument("--output", type=Path, help="Required with --observations.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Sidecar root for --observations (default: observations file directory).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.observations is not None and args.output is None:
        parser.error("--output is required with --observations")
    if args.observations is None and args.output is not None:
        parser.error("--output is only valid with --observations")
    if args.observations is None and args.artifact_root is not None:
        parser.error("--artifact-root is only valid with --observations")
    try:
        dataset, dataset_hash = _load(args.dataset, AgentAblationDataset)
        if args.write_template is not None:
            template = build_observation_template(dataset, dataset_sha256=dataset_hash)
            _write(args.write_template, json.dumps(template, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {
                        "template": str(args.write_template.resolve()),
                        "scoreable": False,
                        "warning": template["warning"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.collect is not None:
            git_commit, git_dirty = discover_git_state(PROJECT_ROOT)
            observations_path, report_path, report = collect_agent_ablation(
                dataset,
                dataset_sha256=dataset_hash,
                output_dir=args.collect,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
            print(
                json.dumps(
                    {
                        "execution_semantics": "DETERMINISTIC_COMPONENT_HARNESS",
                        "observations": str(observations_path.resolve()),
                        "report": str(report_path.resolve()),
                        "comparison_scope": report.comparison_scope,
                        "limitations": report.limitations,
                        "metrics": [item.model_dump(mode="json") for item in report.metrics],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        observations, observations_hash = _load(args.observations, AgentAblationObservationSet)
        verify_source_state(
            PROJECT_ROOT,
            expected_sha256=observations.metadata.source_state_sha256,
            expected_file_count=observations.metadata.source_state_file_count,
        )
        report = evaluate_agent_ablation(
            dataset,
            observations,
            dataset_sha256=dataset_hash,
            observations_sha256=observations_hash,
            artifact_root=args.artifact_root or args.observations.parent,
        )
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        parser.error(str(exc))
    _write(args.output, report.model_dump_json(indent=2))
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
