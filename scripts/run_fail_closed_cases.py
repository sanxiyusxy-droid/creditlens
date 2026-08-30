"""Execute the two frozen v1.6 fail-closed cases.

The default mode is a contract-validator precheck only.  Pass
``--execute-system`` to run the real Supervisor/Auditor/database report gate.
Neither mode is an HTTP acceptance test.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditlens.common.config import get_settings  # noqa: E402
from creditlens.demo_acceptance import DEMO_CASE_ID  # noqa: E402
from creditlens.demo_bootstrap import (  # noqa: E402
    DEMO_TENANT_ID,
    DEMO_USER_ID,
    validate_demo_settings,
)
from creditlens.evaluation.failure_cases import (  # noqa: E402
    FailureCaseDataset,
    evaluate_failure_cases,
    execute_failure_cases_system,
)
from creditlens.infrastructure.postgres.session import (  # noqa: E402
    create_engine,
    create_session_factory,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "fail_closed_cases_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic fail-closed cases as a contract-only precheck, "
            "or explicitly through the Supervisor/Auditor/database gate."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute-system",
        action="store_true",
        help=(
            "Use the configured database and seeded demo identity/case to execute "
            "the real workflow. This still does not call an HTTP endpoint."
        ),
    )
    return parser


def _failure_payload(
    *,
    execute_system: bool,
    system_execution_attempted: bool,
    code: str,
    error_type: str,
) -> dict:
    return {
        "report_type": "fail_closed_regression_error",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_id": str(uuid.uuid4()),
        "requested_proof_scope": (
            "SUPERVISOR_AUDITOR_DATABASE_GATE" if execute_system else "CONTRACT_VALIDATOR_PRECHECK"
        ),
        "system_execution_requested": execute_system,
        "system_execution_attempted": system_execution_attempted,
        "system_proof_completed": False,
        "http_endpoint_called": False,
        "all_passed": False,
        "error_code": code,
        "error_type": error_type,
    }


async def _run(args: argparse.Namespace) -> tuple[dict, int]:
    try:
        payload = args.dataset.read_bytes()
        dataset = FailureCaseDataset.model_validate_json(payload)
        dataset_sha256 = hashlib.sha256(payload).hexdigest()
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        return (
            _failure_payload(
                execute_system=args.execute_system,
                system_execution_attempted=False,
                code="FAILURE_CASE_DATASET_INVALID",
                error_type=type(exc).__name__,
            ),
            2,
        )

    engine = None
    try:
        if args.execute_system:
            settings = get_settings()
            # The injection is intentionally limited to the local synthetic
            # demo profile.  Configured model providers are harmless here: this
            # runner constructs no model, embedding, reranker or HTTP client.
            validate_demo_settings(settings, allow_configured_models=True)
            engine = create_engine(settings.database_url)
            report = await execute_failure_cases_system(
                dataset,
                dataset_sha256=dataset_sha256,
                session_factory=create_session_factory(engine),
                tenant_id=DEMO_TENANT_ID,
                user_id=DEMO_USER_ID,
                case_id=DEMO_CASE_ID,
            )
        else:
            report = evaluate_failure_cases(
                dataset,
                dataset_sha256=dataset_sha256,
            )
        return report.model_dump(mode="json"), 0 if report.all_passed else 1
    except Exception as exc:
        # Exception messages may contain SQL, DSNs or injected content.  Emit
        # only a stable code and class name at this command boundary.
        return (
            _failure_payload(
                execute_system=args.execute_system,
                system_execution_attempted=args.execute_system,
                code="FAILURE_CASE_SYSTEM_EXECUTION_FAILED"
                if args.execute_system
                else "FAILURE_CASE_PRECHECK_FAILED",
                error_type=type(exc).__name__,
            ),
            2,
        )
    finally:
        if engine is not None:
            await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, exit_code = asyncio.run(_run(args))
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            temporary.replace(args.output)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
