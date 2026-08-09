"""Run the deterministic answer-layer evaluation from prediction JSON.

This command never calls a model.  A real model runner can be added separately
as long as it emits the validated ``AnswerPredictionSet`` contract consumed here.

Example:
    uv run python scripts/run_answer_evaluation.py \
        --predictions evaluation/predictions/answer_eval_v1.json \
        --output evaluation/reports/answer_eval_v1_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.answer_metrics import (  # noqa: E402
    AnswerEvalDataset,
    AnswerPredictionSet,
    evaluate_answers,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "answer_eval_v1.json"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> tuple[object, str]:
    content = path.read_bytes()
    return json.loads(content.decode("utf-8")), _sha256(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score deterministic CreditLens answer predictions offline.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Frozen answer-evaluation dataset JSON.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Structured AnswerPredictionSet JSON produced by a system under test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the actual score report; stdout is used when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_payload, dataset_hash = _read_json(args.dataset)
        prediction_payload, prediction_hash = _read_json(args.predictions)
        dataset = AnswerEvalDataset.model_validate(dataset_payload)
        prediction_set = AnswerPredictionSet.model_validate(prediction_payload)
        report = evaluate_answers(
            dataset,
            prediction_set,
            dataset_sha256=dataset_hash,
            predictions_sha256=prediction_hash,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    rendered = report.model_dump_json(indent=2)
    if args.output is None:
        print(rendered)
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "report": str(output_path),
                    "dataset_sha256": dataset_hash,
                    "predictions_sha256": prediction_hash,
                    "summary": report.summary.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
