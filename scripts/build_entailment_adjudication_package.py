"""Build a randomized dispute-only package and a private identity mapping."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from datetime import datetime
from itertools import combinations
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    BlindReviewMapping,
    BlindReviewPackage,
    ReviewSubmission,
    SemanticEntailmentSource,
    build_blind_adjudication_package,
    serialize_json_model,
    sha256_bytes,
)


def _aware_iso8601(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "generated-at must be a valid ISO 8601 timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("generated-at must include a timezone")
    return parsed


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def _validated_paths(
    inputs: list[Path], output: Path, mapping_output: Path
) -> tuple[list[Path], Path, Path]:
    resolved_inputs = [path.resolve() for path in inputs]
    outputs = (_lexical_absolute(output), _lexical_absolute(mapping_output))
    if any(_is_reparse_or_symlink(path) for path in outputs):
        raise ValueError("outputs must not be symbolic links or reparse points")
    for left, right in combinations([*resolved_inputs, *outputs], 2):
        if _paths_alias(left, right):
            raise ValueError("all adjudication input and output paths must be distinct files")
    return resolved_inputs, outputs[0], outputs[1]


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
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_pair(
    first_path: Path,
    first_payload: bytes,
    second_path: Path,
    second_payload: bytes,
    *,
    overwrite: bool,
) -> None:
    created: list[Path] = []
    if overwrite:
        _atomic_write(first_path, first_payload, overwrite=True)
        _atomic_write(second_path, second_payload, overwrite=True)
        return
    try:
        _atomic_write(first_path, first_payload, overwrite=False)
        created.append(first_path)
        _atomic_write(second_path, second_payload, overwrite=False)
        created.append(second_path)
    except (OSError, ValueError):
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _read_model(path: Path, model_type):
    content = path.read_bytes()
    return model_type.model_validate_json(content), sha256_bytes(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a blind human adjudication package; no model is called.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--mapping", type=Path, action="append", required=True)
    parser.add_argument("--submission", type=Path, action="append", required=True)
    parser.add_argument(
        "--adjudicator-id",
        required=True,
        help="Assigned 22-128 character URL-safe pseudonym; never a real name.",
    )
    parser.add_argument("--ordering-seed", required=True)
    parser.add_argument("--generated-at", type=_aware_iso8601)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        requested_output = _lexical_absolute(args.output)
        requested_mapping = (
            _lexical_absolute(args.mapping_output)
            if args.mapping_output
            else requested_output.with_suffix(requested_output.suffix + ".mapping.json")
        )
        raw_inputs = [args.source, *args.package, *args.mapping, *args.submission]
        inputs, output, mapping_output = _validated_paths(
            raw_inputs, requested_output, requested_mapping
        )
        iterator = iter(inputs)
        source, source_hash = _read_model(next(iterator), SemanticEntailmentSource)
        packages = [_read_model(next(iterator), BlindReviewPackage) for _ in args.package]
        mappings = [_read_model(next(iterator), BlindReviewMapping) for _ in args.mapping]
        submissions = [_read_model(next(iterator), ReviewSubmission) for _ in args.submission]
        package, mapping = build_blind_adjudication_package(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=submissions,
            adjudicator_id=args.adjudicator_id,
            ordering_seed=args.ordering_seed,
            generated_at=args.generated_at,
        )
        package_bytes = serialize_json_model(package)
        mapping_bytes = serialize_json_model(mapping)
        existing = [path for path in (output, mapping_output) if path.exists()]
        if existing and not args.overwrite:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"output already exists: {joined}; pass --overwrite to replace it"
            )
        _atomic_write_pair(
            output,
            package_bytes,
            mapping_output,
            mapping_bytes,
            overwrite=args.overwrite,
        )
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "package": str(output),
                "package_sha256": sha256_bytes(package_bytes),
                "private_mapping": str(mapping_output),
                "mapping_sha256": sha256_bytes(mapping_bytes),
                "disputes": len(package.items),
                "model_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
