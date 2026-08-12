"""Build a reviewer-specific, gold-free semantic-entailment review package.

This is an offline file transformation. It never calls an LLM or any service.
Keep the generated ``.mapping.json`` file away from reviewers until scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from itertools import combinations
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.semantic_entailment import (  # noqa: E402
    SemanticEntailmentSource,
    build_blind_review_package,
    serialize_json_model,
    sha256_bytes,
)


def _distinct_paths(source: Path, output: Path, mapping_output: Path) -> tuple[Path, Path, Path]:
    paths = source, output, mapping_output
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("source, output, and mapping-output paths must differ")
    for left, right in combinations(paths, 2):
        if left.exists() and right.exists() and left.samefile(right):
            raise ValueError(
                "source, output, and mapping-output paths must not refer to the "
                "same underlying file"
            )
    return resolved


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
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a randomized human blind-review package; no model is called.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--ordering-seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping-output",
        type=Path,
        help="Private identity map (default: <output>.mapping.json).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        requested_output = args.output.resolve()
        requested_mapping = (
            args.mapping_output.resolve()
            if args.mapping_output
            else requested_output.with_suffix(requested_output.suffix + ".mapping.json")
        )
        source_path, output, mapping_output = _distinct_paths(
            args.source, requested_output, requested_mapping
        )
        source_bytes = source_path.read_bytes()
        source = SemanticEntailmentSource.model_validate_json(source_bytes)
        package, mapping = build_blind_review_package(
            source,
            source_sha256=sha256_bytes(source_bytes),
            reviewer_id=args.reviewer_id,
            ordering_seed=args.ordering_seed,
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
                "items": len(package.items),
                "model_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
