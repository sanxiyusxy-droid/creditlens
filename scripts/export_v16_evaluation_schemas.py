"""Export the committed v1.6 evaluation contracts as JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.agent_ablation import (  # noqa: E402
    AblationHarnessArtifact,
    AgentAblationDataset,
    AgentAblationObservationSet,
    AgentAblationReport,
)
from creditlens.evaluation.answer_suite import (  # noqa: E402
    AnswerReevaluationManifest,
    AnswerSuiteRunRecord,
)
from creditlens.evaluation.failure_cases import (  # noqa: E402
    FailureCaseDataset,
    FailureCaseReport,
)

SCHEMAS = {
    "answer_reevaluation_manifest_v1.schema.json": AnswerReevaluationManifest,
    "answer_reevaluation_run_v1.schema.json": AnswerSuiteRunRecord,
    "agent_ablation_dataset_v1.schema.json": AgentAblationDataset,
    "agent_ablation_observations_v1.schema.json": AgentAblationObservationSet,
    "agent_ablation_harness_artifact_v1.schema.json": AblationHarnessArtifact,
    "agent_ablation_report_v1.schema.json": AgentAblationReport,
    "fail_closed_cases_v1.schema.json": FailureCaseDataset,
    "fail_closed_report_v1.schema.json": FailureCaseReport,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export v1.6 evaluation JSON Schemas.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "schemas",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in SCHEMAS.items():
        path = args.output_dir / filename
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
