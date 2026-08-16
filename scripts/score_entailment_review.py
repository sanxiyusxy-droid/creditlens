"""Aggregate independently completed human semantic-entailment reviews offline.

The command validates raw file hashes and blind mappings before reporting. It
does not call a model, and citation-set equality is not treated as faithfulness.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    BlindAdjudicationMapping,
    BlindAdjudicationPackage,
    BlindAdjudicationSubmission,
    BlindReviewMapping,
    BlindReviewPackage,
    FormalCompletionStatus,
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


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following its final filesystem entry."""

    return Path(os.path.abspath(path))


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _validated_paths(input_paths: list[Path], output_path: Path) -> tuple[list[Path], Path]:
    resolved_inputs = [path.resolve() for path in input_paths]
    output = _lexical_absolute(output_path)
    if _is_reparse_or_symlink(output):
        raise ValueError("output path must not be a symbolic link or reparse point")
    if any(_paths_alias(path, output) for path in resolved_inputs):
        raise ValueError("output path must differ from every input path")
    return resolved_inputs, output


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_reparse_or_symlink(path):
        raise ValueError("output path must not be a symbolic link or reparse point")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        if _is_reparse_or_symlink(path):
            raise ValueError("output path became a symbolic link or reparse point")
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
    parser.add_argument("--adjudication-package", type=Path)
    parser.add_argument("--adjudication-mapping", type=Path)
    parser.add_argument("--adjudication-submission", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail without publishing output unless the formal completion gate is COMPLETE.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        adjudication_paths = (
            args.adjudication_package,
            args.adjudication_mapping,
            args.adjudication_submission,
        )
        if any(path is not None for path in adjudication_paths) and not all(
            path is not None for path in adjudication_paths
        ):
            raise ValueError(
                "--adjudication-package, --adjudication-mapping, and "
                "--adjudication-submission are all-or-none"
            )
        raw_inputs = [
            args.source,
            *args.package,
            *args.mapping,
            *args.submission,
            *(path for path in adjudication_paths if path is not None),
        ]
        inputs, output = _validated_paths(raw_inputs, args.output)
        input_iterator = iter(inputs)
        source, source_hash = _read_model(next(input_iterator), SemanticEntailmentSource)
        packages = [_read_model(next(input_iterator), BlindReviewPackage) for _path in args.package]
        mappings = [_read_model(next(input_iterator), BlindReviewMapping) for _path in args.mapping]
        submissions = [
            _read_model(next(input_iterator), ReviewSubmission) for _path in args.submission
        ]
        adjudication_package = (
            _read_model(next(input_iterator), BlindAdjudicationPackage)
            if args.adjudication_package
            else None
        )
        adjudication_mapping = (
            _read_model(next(input_iterator), BlindAdjudicationMapping)
            if args.adjudication_mapping
            else None
        )
        adjudication_submission = (
            _read_model(next(input_iterator), BlindAdjudicationSubmission)
            if args.adjudication_submission
            else None
        )
        report = evaluate_entailment_reviews(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=submissions,
            adjudication_package=adjudication_package,
            adjudication_mapping=adjudication_mapping,
            adjudication_submission=adjudication_submission,
        )
        if (
            args.require_complete
            and report.formal_completion.status is not FormalCompletionStatus.COMPLETE
        ):
            codes = ", ".join(blocker.code.value for blocker in report.formal_completion.blockers)
            raise ValueError(f"formal semantic review is INCOMPLETE: {codes}")
        _atomic_write(output, serialize_json_model(report), overwrite=args.overwrite)
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "report": str(output),
                "summary": report.summary.model_dump(mode="json"),
                "formal_completion": report.formal_completion.model_dump(mode="json"),
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
