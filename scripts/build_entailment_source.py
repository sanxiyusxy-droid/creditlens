"""Build a gold-free semantic-entailment source from frozen QA artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from itertools import combinations
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    build_source_from_grounded_qa_artifacts,
    serialize_json_model,
    sha256_bytes,
)


def _distinct_paths(paths: list[Path]) -> list[Path]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "prediction, checkpoint, evidence-checkpoint, and output paths must differ"
        )
    for left, right in combinations(paths, 2):
        if left.exists() and right.exists() and left.samefile(right):
            raise ValueError(
                "prediction, checkpoint, evidence-checkpoint, and output paths "
                "must not refer to the same underlying file"
            )
    return resolved


def _aware_iso8601(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "created-at must be a valid ISO 8601 timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("created-at must include a timezone")
    return parsed


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a gold-free source from a prediction set and raw QA checkpoints.",
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evidence-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Complete gold-free checkpoint used only to supply frozen candidate text.",
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--created-at",
        type=_aware_iso8601,
        help=(
            "Explicit timezone-aware ISO 8601 source timestamp for reproducible output "
            "(for example 2026-08-12T09:30:00+08:00); defaults to the current UTC time."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = _distinct_paths(
            [args.prediction, args.raw_checkpoint, *args.evidence_checkpoint, args.output]
        )
        prediction_path = paths[0]
        raw_path = paths[1]
        evidence_paths = paths[2:-1]
        output = paths[-1]
        prediction_bytes = prediction_path.read_bytes()
        raw_bytes = raw_path.read_bytes()
        evidence = [
            (f"evidence-checkpoint-{index}:{path.name}", path.read_bytes())
            for index, path in enumerate(evidence_paths, start=1)
        ]
        source = build_source_from_grounded_qa_artifacts(
            prediction_bytes=prediction_bytes,
            raw_checkpoint_bytes=raw_bytes,
            evidence_checkpoints=evidence,
            source_id=args.source_id,
            prediction_artifact_id=prediction_path.name,
            raw_checkpoint_artifact_id=raw_path.name,
            created_at=args.created_at,
        )
        source_bytes = serialize_json_model(source)
        _atomic_write(output, source_bytes, overwrite=args.overwrite)
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "source": str(output),
                "source_sha256": sha256_bytes(source_bytes),
                "prediction_sha256": source.prediction_sha256,
                "input_artifacts": [
                    item.model_dump(mode="json") for item in source.input_artifacts
                ],
                "claims": len(source.items),
                "gold_read": False,
                "model_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
