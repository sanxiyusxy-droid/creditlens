"""Aggregate independently completed human semantic-entailment reviews offline.

The command validates raw file hashes and blind mappings before reporting. It
does not call a model, and citation-set equality is not treated as faithfulness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    AdjudicationSet,
    BlindReviewMapping,
    BlindReviewPackage,
    ReviewSubmission,
    SemanticEntailmentSource,
    evaluate_entailment_reviews,
    serialize_json_model,
    sha256_bytes,
)


def _read_model(path: Path, model_type):
    content = path.read_bytes()
    return model_type.model_validate_json(content), sha256_bytes(content)


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two existing or prospective paths address the same file."""

    first_resolved = first.resolve()
    second_resolved = second.resolve()
    if first_resolved == second_resolved:
        return True
    return (
        first_resolved.exists()
        and second_resolved.exists()
        and first_resolved.samefile(second_resolved)
    )


def _validated_paths(input_paths: list[Path], output_path: Path) -> tuple[list[Path], Path]:
    resolved_inputs = [path.resolve() for path in input_paths]
    output = output_path.resolve()
    if any(_paths_alias(path, output) for path in resolved_inputs):
        raise ValueError("output path must differ from every input path")
    return resolved_inputs, output


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
            # Publishing through a hard link makes no-clobber race-safe: the
            # operation fails if another process creates the destination first.
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score manual claim-evidence entailment reviews; no model is called.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--mapping", type=Path, action="append", required=True)
    parser.add_argument("--submission", type=Path, action="append", default=[])
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raw_inputs = [
            args.source,
            *args.package,
            *args.mapping,
            *args.submission,
            *([args.adjudication] if args.adjudication else []),
        ]
        inputs, output = _validated_paths(raw_inputs, args.output)
        input_iterator = iter(inputs)
        source, source_hash = _read_model(next(input_iterator), SemanticEntailmentSource)
        packages = [_read_model(next(input_iterator), BlindReviewPackage) for _path in args.package]
        mappings = [_read_model(next(input_iterator), BlindReviewMapping) for _path in args.mapping]
        submissions = [
            _read_model(next(input_iterator), ReviewSubmission) for _path in args.submission
        ]
        adjudication = (
            _read_model(next(input_iterator), AdjudicationSet) if args.adjudication else None
        )
        report = evaluate_entailment_reviews(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=submissions,
            adjudication=adjudication,
        )
        _atomic_write(output, serialize_json_model(report), overwrite=args.overwrite)
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "report": str(output),
                "summary": report.summary.model_dump(mode="json"),
                "model_called": False,
                "citation_set_used_as_faithfulness": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
