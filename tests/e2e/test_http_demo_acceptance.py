"""Real services exercised through an explicitly in-process ASGI transport.

This test is evidence for PostgreSQL/Qdrant/RLS and the public FastAPI contract.
It is deliberately not socket/TCP black-box evidence; the standalone acceptance
script owns that proof boundary.
"""

import asyncio
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

from creditlens.demo_acceptance import (
    DEMO_CASE_ID,
    load_frozen_hitl_allowlist,
    run_http_acceptance,
)
from creditlens.demo_bootstrap import ensure_demo_financial_facts
from creditlens.infrastructure.postgres.session import create_session_factory
from creditlens.observability.outbox_worker import (
    LocalDirectoryTelemetryExporter,
    TelemetryOutboxWorker,
)
from tests.conftest import requires_integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HITL_ALLOWLIST = PROJECT_ROOT / "evaluation" / "datasets" / "http_acceptance_hitl_v1.json"

pytestmark = [
    pytest.mark.integration,
    requires_integration,
    pytest.mark.asyncio(loop_scope="session"),
]


async def test_real_services_via_in_process_asgi_question_review_report_trace(
    pg_engine,
    monkeypatch,
):
    """Use real services while reporting the transport as IN_PROCESS_TEST_TRANSPORT."""

    frozen_hitl = load_frozen_hitl_allowlist(
        HITL_ALLOWLIST,
        expected_case_id=DEMO_CASE_ID,
        expected_profile="deterministic-offline",
    )
    if frozen_hitl is None:
        pytest.skip(
            "frozen deterministic HITL allowlist is not available; "
            "unknown claims must never be auto-approved"
        )
    assert frozen_hitl.case_id == str(DEMO_CASE_ID)
    assert frozen_hitl.profile == "deterministic-offline"
    assert frozen_hitl.blocking_claim_fingerprints

    from apps.api import main as api_main

    # A deterministic allowlist cannot authorize claims from a configured-model
    # profile. Keep the evidence boundary explicit instead of guessing equivalence.
    if not (
        api_main.settings.llm_provider == "disabled"
        and api_main.settings.embedding_provider == "hash_fallback"
        and api_main.settings.rerank_provider == "lexical_fallback"
    ):
        pytest.skip("integration runtime is not the deterministic-offline profile")

    # ASGITransport does not own this application's lifespan. Start the real
    # lease-based worker explicitly with the local durable exporter; do not enter
    # api_main.lifespan because its shutdown closes process-global API resources.
    api_session_factory = create_session_factory(pg_engine)
    monkeypatch.setattr(api_main, "session_factory", api_session_factory)
    monkeypatch.setattr(api_main.settings, "qa_allow_extractive_fallback", True)
    monkeypatch.setattr(api_main.settings, "telemetry_export_poll_seconds", 0.01)
    monkeypatch.setattr(api_main.settings, "telemetry_export_batch_size", 128)
    await ensure_demo_financial_facts(api_session_factory)

    telemetry_directory = (
        PROJECT_ROOT
        / "evaluation"
        / "reports"
        / "local"
        / f"integration_http_delivery_{uuid.uuid4().hex}"
    )
    exporter = LocalDirectoryTelemetryExporter(telemetry_directory)
    telemetry_worker = TelemetryOutboxWorker(
        api_session_factory,
        exporter,
        max_attempts=api_main.settings.telemetry_export_max_attempts,
        lease_seconds=api_main.settings.telemetry_export_lease_seconds,
        base_backoff_seconds=api_main.settings.telemetry_export_base_backoff_seconds,
        max_backoff_seconds=api_main.settings.telemetry_export_max_backoff_seconds,
        tenant_id=api_main.DEFAULT_TENANT_ID,
        user_id=api_main.DEMO_USER_ID,
    )
    telemetry_stop = asyncio.Event()
    telemetry_task = asyncio.create_task(
        api_main._run_telemetry_worker(telemetry_worker, telemetry_stop),
        name="test-creditlens-telemetry-outbox",
    )
    await asyncio.sleep(0)
    assert not telemetry_task.done()
    monkeypatch.setattr(api_main.app.state, "runtime_started", True)
    monkeypatch.setattr(api_main.app.state, "telemetry_worker_enabled", True)
    monkeypatch.setattr(api_main.app.state, "telemetry_worker", telemetry_worker)
    monkeypatch.setattr(api_main.app.state, "telemetry_task", telemetry_task)

    try:
        report = await run_http_acceptance(
            "http://creditlens.test",
            timeout_seconds=120,
            poll_seconds=0.05,
            transport=httpx.ASGITransport(app=api_main.app),
            expected_hitl_claim_fingerprints=frozen_hitl.blocking_claim_fingerprints,
        )
        assert report.passed is True
        assert report.transport_scope == "IN_PROCESS_TEST_TRANSPORT"
        assert report.qa_candidate_count > 0
        assert report.review_final_status == "COMPLETED"
        assert report.report_status == (
            "APPROVED_DRAFT" if report.human_review_exercised else "VERIFIED_DRAFT"
        )
        assert report.trace_integrity_status == "VALID"
        assert report.trace_delivery_status == "COMPLETE"
        assert report.trace_event_count > 0
        assert report.trace_invocation_count > 0
        delivered_files = list(exporter.directory.glob("*.json"))
        assert len(delivered_files) >= report.trace_invocation_count
    finally:
        await api_main._stop_telemetry_worker(telemetry_task, telemetry_stop)
        shutil.rmtree(telemetry_directory, ignore_errors=True)
