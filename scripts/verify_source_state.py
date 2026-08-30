"""Recompute or verify the deterministic source-state binding on an artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.evaluation.source_state import (  # noqa: E402
    SOURCE_STATE_ALGORITHM,
    SOURCE_STATE_SCOPE,
    compute_source_state,
    source_state_evidence_from_metadata,
    verify_source_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute CreditLens runtime-evidence source-state SHA-256."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--artifact",
        type=Path,
        help="Prediction, ablation, or fail-closed JSON carrying source-state metadata.",
    )
    mode.add_argument("--expected-sha256", help="Expected source-state SHA-256.")
    parser.add_argument(
        "--expected-file-count",
        type=int,
        help="Optional expected covered-file count (or read automatically from --artifact).",
    )
    return parser


def _artifact_binding(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("artifact root must be a JSON object")
    candidate = payload.get("metadata", payload)
    if not isinstance(candidate, Mapping):
        raise ValueError("artifact metadata must be a JSON object")
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.artifact is not None:
            binding = _artifact_binding(args.artifact)
            evidence = source_state_evidence_from_metadata(binding)
            snapshot = verify_source_state(
                args.project_root,
                expected_sha256=evidence.source_state_sha256,
                expected_file_count=evidence.source_state_file_count,
            )
            verified = True
        elif args.expected_sha256 is not None:
            snapshot = verify_source_state(
                args.project_root,
                expected_sha256=args.expected_sha256,
                expected_file_count=args.expected_file_count,
            )
            verified = True
        else:
            snapshot = compute_source_state(args.project_root)
            verified = None
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "verified": verified,
                "source_state_sha256": snapshot.sha256,
                "source_state_algorithm": SOURCE_STATE_ALGORITHM,
                "source_state_scope": SOURCE_STATE_SCOPE,
                "source_state_file_count": snapshot.file_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
