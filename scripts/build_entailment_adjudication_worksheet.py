"""Build an adjudicator-editable worksheet from public dispute-package bytes."""

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
    BlindAdjudicationPackage,
    build_adjudication_worksheet,
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


def _validated_paths(package: Path, output: Path) -> tuple[Path, Path]:
    package = package.resolve()
    output = _lexical_absolute(output)
    if _is_reparse_or_symlink(output):
        raise ValueError("output path must not be a symbolic link or reparse point")
    if _paths_alias(package, output):
        raise ValueError("output path must differ from the dispute-package input")
    return package, output


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a blank blind adjudication worksheet; no model is called.",
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--adjudicator-id",
        required=True,
        help="Assigned 22-128 character URL-safe pseudonym; never a real name.",
    )
    parser.add_argument("--created-at", type=_aware_iso8601)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package_path, output = _validated_paths(args.package, args.output)
        package_bytes = package_path.read_bytes()
        package_hash = sha256_bytes(package_bytes)
        package = BlindAdjudicationPackage.model_validate_json(package_bytes)
        worksheet = build_adjudication_worksheet(
            package,
            package_sha256=package_hash,
            adjudicator_id=args.adjudicator_id,
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
                "disputes": len(worksheet.items),
                "model_called": False,
                "source_or_mapping_read": False,
                "reviewer_identity_read": False,
                "gold_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
