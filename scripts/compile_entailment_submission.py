"""Compile a completed worksheet into a strict human ReviewSubmission.

The attestation flags are reviewer self-declarations, not technical proof. The
command performs no model call and never reads Source, mapping, or gold files.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    BlindReviewPackage,
    ReviewerAttestationV1,
    ReviewWorksheet,
    compile_review_submission,
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


def _validated_paths(package: Path, worksheet: Path, output: Path) -> tuple[Path, Path, Path]:
    inputs = (package, worksheet)
    resolved_inputs = tuple(path.resolve() for path in inputs)
    output_lexical = _lexical_absolute(output)
    _reject_output_reparse(output_lexical)
    if resolved_inputs[0] == resolved_inputs[1] or (
        package.exists() and worksheet.exists() and package.samefile(worksheet)
    ):
        raise ValueError("package, worksheet, and output paths must refer to distinct files")
    output_resolved = output_lexical.resolve()
    for input_path, resolved_input in zip(inputs, resolved_inputs, strict=True):
        if resolved_input == output_resolved or (
            input_path.exists() and output_lexical.exists() and input_path.samefile(output_lexical)
        ):
            raise ValueError("package, worksheet, and output paths must refer to distinct files")
    return (*resolved_inputs, output_lexical)


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
        description=(
            "Compile a complete blind worksheet into ReviewSubmission; no model is called. "
            "Attestation flags are self-declarations, not technical proof."
        ),
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument(
        "--reviewer-id",
        required=True,
        help=(
            "Assigned high-entropy URL-safe pseudonym (22-128 characters); "
            "never provide a real name."
        ),
    )
    parser.add_argument("--submission-id")
    parser.add_argument(
        "--submitted-at",
        type=_aware_iso8601,
        help=(
            "Timezone-aware ISO 8601 submission and attestation timestamp for "
            "reproducible output; defaults to current UTC time."
        ),
    )
    parser.add_argument("--attest-human", action="store_true")
    parser.add_argument("--attest-no-model-assistance", action="store_true")
    parser.add_argument("--attest-no-repo-or-gold-access", action="store_true")
    parser.add_argument("--attest-no-source-or-mapping-access", action="store_true")
    parser.add_argument("--attest-no-reviewer-coordination", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _require_attestations(args: argparse.Namespace) -> None:
    required = {
        "--attest-human": args.attest_human,
        "--attest-no-model-assistance": args.attest_no_model_assistance,
        "--attest-no-repo-or-gold-access": args.attest_no_repo_or_gold_access,
        "--attest-no-source-or-mapping-access": args.attest_no_source_or_mapping_access,
        "--attest-no-reviewer-coordination": args.attest_no_reviewer_coordination,
    }
    missing = [flag for flag, supplied in required.items() if not supplied]
    if missing:
        raise ValueError(
            "all human attestation self-declarations are required (not technical proof); "
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
        package_sha256 = sha256_bytes(package_bytes)
        package = BlindReviewPackage.model_validate_json(package_bytes)
        worksheet = ReviewWorksheet.model_validate_json(worksheet_path.read_bytes())
        submitted_at = args.submitted_at or datetime.now(UTC)
        attestation = ReviewerAttestationV1(
            actor_type="HUMAN",
            model_assistance_used=False,
            accessed_repo_or_gold=False,
            accessed_source_or_mapping=False,
            coordinated_with_other_reviewers=False,
            attested_at=submitted_at,
        )
        submission = compile_review_submission(
            package,
            worksheet,
            package_sha256=package_sha256,
            reviewer_id=args.reviewer_id,
            reviewer_attestation=attestation,
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
                "reviews": len(submission.reviews),
                "model_called": False,
                "source_or_mapping_read": False,
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
