"""Build a reviewer-editable worksheet from exact blind-package bytes.

The command is an offline file transformation. It does not call a model and
does not read a Source, private mapping, gold label, or raw reviewer name. The
reviewer identifier must be a high-entropy pseudonym rather than a real name.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    BlindReviewPackage,
    build_review_worksheet,
    serialize_json_model,
    sha256_bytes,
)


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


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following its final directory entry."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _reject_output_reparse(path: Path) -> None:
    if _is_reparse_point(path):
        raise ValueError(f"output path must not be a symbolic link or reparse point: {path}")


def _validated_paths(package: Path, output: Path) -> tuple[Path, Path]:
    package_resolved = package.resolve()
    output_lexical = _lexical_absolute(output)
    _reject_output_reparse(output_lexical)
    if package_resolved == output_lexical.resolve() or (
        package.exists() and output_lexical.exists() and package.samefile(output_lexical)
    ):
        raise ValueError("output path must differ from the blind-package input path")
    return package_resolved, output_lexical


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    _reject_output_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        _reject_output_reparse(path)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a blank blind-review worksheet; no model is called.",
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--reviewer-id",
        required=True,
        help=(
            "Assigned high-entropy URL-safe pseudonym (22-128 characters); "
            "never provide a real name. Only its SHA-256 is written."
        ),
    )
    parser.add_argument(
        "--created-at",
        type=_aware_iso8601,
        help=(
            "Timezone-aware ISO 8601 worksheet timestamp for reproducible output; "
            "defaults to current UTC time."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package_path, output = _validated_paths(args.package, args.output)
        package_bytes = package_path.read_bytes()
        package_sha256 = sha256_bytes(package_bytes)
        package = BlindReviewPackage.model_validate_json(package_bytes)
        worksheet = build_review_worksheet(
            package,
            package_sha256=package_sha256,
            reviewer_id=args.reviewer_id,
            created_at=args.created_at,
        )
        worksheet_bytes = serialize_json_model(worksheet)
        _atomic_write(output, worksheet_bytes, overwrite=args.overwrite)
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "worksheet": str(output),
                "worksheet_sha256": sha256_bytes(worksheet_bytes),
                "package_id": worksheet.package_id,
                "package_sha256": worksheet.package_sha256,
                "reviewer_id_sha256": worksheet.reviewer_id_sha256,
                "items": len(worksheet.items),
                "model_called": False,
                "source_or_mapping_read": False,
                "gold_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
