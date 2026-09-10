"""Run the synthetic demo's complete TCP HTTP acceptance path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.demo_acceptance import (  # noqa: E402
    AcceptanceFailure,
    load_frozen_hitl_allowlist,
    run_http_acceptance,
)

DEFAULT_HITL_ALLOWLIST = PROJECT_ROOT / "evaluation" / "datasets" / "http_acceptance_hitl_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CreditLens 合成案件 HTTP 全链路验收")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional machine-readable acceptance report.")
    parser.add_argument(
        "--hitl-allowlist",
        type=Path,
        default=DEFAULT_HITL_ALLOWLIST,
        help="Frozen public fingerprints of the only synthetic blocking claims safe to approve.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Explicitly permit a non-local base URL (disabled by default).",
    )
    parser.add_argument(
        "--profile",
        choices=("deterministic-offline", "configured-models"),
        default="deterministic-offline",
        help="Runtime profile that the frozen HITL fingerprints were reviewed against.",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> tuple[dict, int]:
    try:
        frozen_hitl = load_frozen_hitl_allowlist(
            args.hitl_allowlist,
            expected_profile=args.profile,
        )
        report = await run_http_acceptance(
            args.base_url,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            expected_hitl_claim_fingerprints=(
                frozen_hitl.blocking_claim_fingerprints if frozen_hitl is not None else None
            ),
            allow_non_loopback=args.allow_non_loopback,
        )
        return report.to_dict(), 0
    except AcceptanceFailure as exc:
        return {
            "schema_version": "creditlens.http-acceptance.v1",
            "passed": False,
            "error_code": exc.code,
        }, 1
    except Exception as exc:
        # Never emit response bodies, exception messages, DSNs or provider details.
        return {
            "schema_version": "creditlens.http-acceptance.v1",
            "passed": False,
            "error_code": "HTTP_ACCEPTANCE_FAILED",
            "error_type": type(exc).__name__,
        }, 1


if __name__ == "__main__":
    parsed = _parse_args()
    result, exit_code = asyncio.run(_main(parsed))
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = parsed.output.with_name(f".{parsed.output.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            temporary.replace(parsed.output)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(rendered)
    raise SystemExit(exit_code)
