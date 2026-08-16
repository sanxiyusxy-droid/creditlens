"""Compile a filled blind worksheet into a human adjudication submission."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    AdjudicationWorksheet,
    AdjudicatorAttestationV1,
    BlindAdjudicationPackage,
    compile_adjudication_submission,
    serialize_json_model,
    sha256_bytes,
)


def _aware_iso8601(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "submitted-at must be a valid ISO 8601 timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("submitted-at must include a timezone")
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


def _validated_paths(package: Path, worksheet: Path, output: Path) -> tuple[Path, Path, Path]:
    inputs = [package.resolve(), worksheet.resolve()]
    output = _lexical_absolute(output)
    if _is_reparse_or_symlink(output):
        raise ValueError("output path must not be a symbolic link or reparse point")
    for left, right in combinations([*inputs, output], 2):
        if _paths_alias(left, right):
            raise ValueError("package, worksheet, and output must be distinct files")
    return inputs[0], inputs[1], output


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
        description=(
            "Compile a complete blind adjudication worksheet; no model is called. "
            "All isolation attestations are self-declarations, not technical proof."
        )
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument(
        "--adjudicator-id",
        required=True,
        help="Assigned 22-128 character URL-safe pseudonym; never a real name.",
    )
    parser.add_argument("--submission-id")
    parser.add_argument("--submitted-at", type=_aware_iso8601)
    parser.add_argument("--attest-human", action="store_true")
    parser.add_argument("--attest-no-model-assistance", action="store_true")
    parser.add_argument("--attest-no-repository-access", action="store_true")
    parser.add_argument("--attest-no-gold-access", action="store_true")
    parser.add_argument("--attest-no-source-or-private-mapping-access", action="store_true")
    parser.add_argument(
        "--attest-no-raw-submission-or-reviewer-identity-access", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _require_attestations(args: argparse.Namespace) -> None:
    required = {
        "--attest-human": args.attest_human,
        "--attest-no-model-assistance": args.attest_no_model_assistance,
        "--attest-no-repository-access": args.attest_no_repository_access,
        "--attest-no-gold-access": args.attest_no_gold_access,
        "--attest-no-source-or-private-mapping-access": (
            args.attest_no_source_or_private_mapping_access
        ),
        "--attest-no-raw-submission-or-reviewer-identity-access": (
            args.attest_no_raw_submission_or_reviewer_identity_access
        ),
    }
    missing = [flag for flag, supplied in required.items() if not supplied]
    if missing:
        raise ValueError(
            "all human isolation attestations are required (self-declarations only); "
            f"missing: {', '.join(missing)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package_path, worksheet_path, output = _validated_paths(
            args.package, args.worksheet, args.output
        )
        _require_attestations(args)
        package_bytes = package_path.read_bytes()
        package_hash = sha256_bytes(package_bytes)
        package = BlindAdjudicationPackage.model_validate_json(package_bytes)
        worksheet = AdjudicationWorksheet.model_validate_json(worksheet_path.read_bytes())
        submitted_at = args.submitted_at or datetime.now(UTC)
        attestation = AdjudicatorAttestationV1(
            actor_type="HUMAN",
            model_assistance_used=False,
            accessed_repository=False,
            accessed_gold=False,
            accessed_source_or_private_mapping=False,
            accessed_raw_submissions_or_reviewer_identity=False,
            attested_at=submitted_at,
        )
        submission = compile_adjudication_submission(
            package,
            worksheet,
            package_sha256=package_hash,
            adjudicator_id=args.adjudicator_id,
            adjudicator_attestation=attestation,
            submitted_at=submitted_at,
            submission_id=args.submission_id,
        )
        submission_bytes = serialize_json_model(submission)
        _atomic_write(output, submission_bytes, overwrite=args.overwrite)
    except (OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "submission": str(output),
                "submission_id": submission.submission_id,
                "submission_sha256": sha256_bytes(submission_bytes),
                "package_id": submission.package_id,
                "decisions": len(submission.decisions),
                "model_called": False,
                "source_or_mapping_read": False,
                "reviewer_identity_read": False,
                "gold_read": False,
                "attestation_is_self_declaration_not_technical_proof": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
