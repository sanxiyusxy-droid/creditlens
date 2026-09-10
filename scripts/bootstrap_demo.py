"""Prepare and verify the local synthetic CreditLens demo.

Privileged schema/RLS setup is intentionally performed by ``start_demo.ps1``.
This command then runs through the ordinary RLS-constrained application role.
Its stdout is exactly one JSON document and never includes URLs or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from creditlens.common.config import get_settings  # noqa: E402
from creditlens.demo_bootstrap import (  # noqa: E402
    BootstrapInvariantError,
    ensure_demo_financial_facts,
    run_preflight,
    validate_demo_settings,
    verify_runtime_database_gate,
)
from creditlens.infrastructure.objectstore import build_object_store  # noqa: E402
from creditlens.infrastructure.postgres.session import (  # noqa: E402
    create_engine,
    create_session_factory,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client  # noqa: E402
from seed_synthetic_data import seed_environment  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="幂等准备并验证 CreditLens 合成演示环境")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只读检查，不补 Seed、Collection 或 FinancialFact",
    )
    parser.add_argument(
        "--allow-configured-models",
        action="store_true",
        help="显式允许把合成演示材料发往当前已配置的模型服务",
    )
    return parser.parse_args()


def _failure(*, mode: str, code: str, error_type: str | None = None) -> dict:
    details = {"error_type": error_type} if error_type else {}
    return {
        "schema_version": "creditlens.demo-preflight.v1",
        "ready": False,
        "profile": "unknown",
        "mode": mode,
        "components": [
            {
                "name": "bootstrap",
                "status": "FAIL",
                "code": code,
                "details": details,
            }
        ],
    }


async def _main(args: argparse.Namespace) -> tuple[dict, int]:
    mode = "check" if args.check_only else "prepare"
    settings = get_settings()
    engine = None
    qdrant = None
    try:
        # This guard runs before any provider is constructed or any synthetic data is read.
        validate_demo_settings(
            settings,
            allow_configured_models=args.allow_configured_models,
        )
        engine = create_engine(settings.database_url)
        session_factory = create_session_factory(engine)
        # No object store/vector provider is constructed and no seed write is
        # attempted until the runtime role, database identity, migration head,
        # complete RLS policy set and exact grant matrix pass read-only checks.
        await verify_runtime_database_gate(session_factory, PROJECT_ROOT, settings)
        qdrant = build_qdrant_client(settings)

        inserted_facts = 0
        if not args.check_only:
            store = build_object_store(settings)
            # The legacy seed reports human progress to stdout. Suppress it so this
            # command keeps a strict one-document JSON protocol.
            with contextlib.redirect_stdout(io.StringIO()):
                await seed_environment(session_factory, store, qdrant, settings)
            inserted_facts = await ensure_demo_financial_facts(session_factory)

        report = await run_preflight(
            settings=settings,
            session_factory=session_factory,
            qdrant=qdrant,
            project_root=PROJECT_ROOT,
            allow_configured_models=args.allow_configured_models,
        )
        payload = report.to_dict()
        payload["mode"] = mode
        payload["changes"] = {
            "financial_facts_inserted": inserted_facts,
            "destructive_reset_performed": False,
        }
        return payload, 0 if report.ready else 1
    except BootstrapInvariantError as exc:
        return _failure(mode=mode, code=exc.code), 1
    except Exception as exc:
        # Exception messages may contain DSNs, provider URLs or SQL. Only the
        # class is safe for machine-readable diagnostics.
        return _failure(
            mode=mode,
            code="BOOTSTRAP_FAILED",
            error_type=type(exc).__name__,
        ), 1
    finally:
        if qdrant is not None:
            with contextlib.suppress(Exception):
                qdrant.close()
        if engine is not None:
            with contextlib.suppress(Exception):
                await engine.dispose()


if __name__ == "__main__":
    parsed = _parse_args()
    result, exit_code = asyncio.run(_main(parsed))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(exit_code)
