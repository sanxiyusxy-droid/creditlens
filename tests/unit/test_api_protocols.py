"""API 协议边界：SSE 续传/停止语义与错误码映射。"""

import asyncio
import time
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import MethodType, SimpleNamespace

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from creditlens.common.errors import IdempotencyConflictError
from creditlens.infrastructure.postgres.models import (
    AppUser,
    ArtifactRecord,
    Base,
    CaseMembership,
    ClaimRecord,
    CreditCase,
    Entity,
    EvidenceRecord,
    InvocationRecord,
    ReviewRun,
    RunEvent,
    TelemetryOutbox,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory
from creditlens.observability.invocation import (
    InvocationEnvelope,
    InvocationKind,
    InvocationStatus,
)
from creditlens.observability.writer import InvocationWriter


def test_fixed_demo_identity_is_fail_closed_outside_local_environment():
    from apps.api import main as api_main

    from creditlens.common.config import Settings

    api_main._assert_demo_identity_is_safe(
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        )
    )
    assert Settings.model_fields["api_identity_mode"].default == "required"
    assert Settings.model_fields["allow_insecure_demo_identity"].default is False
    with pytest.raises(RuntimeError, match="INSECURE_DEMO_IDENTITY_NOT_ALLOWED"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="demo",
                allow_insecure_demo_identity=False,
                app_env="test",
            )
        )
    with pytest.raises(RuntimeError, match="DEMO_IDENTITY_FORBIDDEN"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="demo",
                allow_insecure_demo_identity=True,
                app_env="production",
            )
        )
    with pytest.raises(RuntimeError, match="IDENTITY_PROVIDER_NOT_CONFIGURED"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="required",
                allow_insecure_demo_identity=True,
                app_env="test",
            )
        )


@pytest.mark.parametrize(
    "unsafe_settings",
    [
        SimpleNamespace(
            api_identity_mode="required",
            allow_insecure_demo_identity=False,
            app_env="local",
        ),
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=False,
            app_env="test",
        ),
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="production",
        ),
    ],
    ids=["required-provider-missing", "demo-not-opted-in", "demo-in-production"],
)
async def test_request_identity_gate_fails_closed_without_lifespan(monkeypatch, unsafe_settings):
    """ASGITransport skips lifespan, so the request gate must independently reject access."""
    from apps.api import main as api_main

    monkeypatch.setattr(api_main, "settings", unsafe_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/cases/{uuid.uuid4()}/questions",
            json={"question": "is the case complete?", "idempotency_key": "identity-gate-001"},
        )
        live_response = await client.get("/health/live")

    assert response.status_code == 503
    assert response.json() == {"detail": {"error_code": "API_IDENTITY_UNAVAILABLE"}}
    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}


async def test_explicit_safe_demo_identity_works_without_lifespan(monkeypatch):
    """The local/test demo opt-in remains usable and supplies identity to the endpoint."""
    from apps.api import main as api_main

    captured: dict = {}

    class Result:
        @staticmethod
        def model_dump(*, mode: str):
            assert mode == "json"
            return {"accepted": True}

    class QAService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def ask(self, **_kwargs):
            return Result()

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        ),
    )
    monkeypatch.setattr(api_main, "QAService", QAService)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/cases/{uuid.uuid4()}/questions",
            json={"question": "is the case complete?", "idempotency_key": "identity-gate-002"},
        )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert captured["tenant_id"] == api_main.DEFAULT_TENANT_ID
    assert captured["user_id"] == api_main.DEMO_USER_ID


async def test_lifespan_does_not_infer_stale_runs_at_startup(monkeypatch):
    """没有 lease/heartbeat 时，启动只建表，不扫描或改写旧的非终态 Run。"""
    from apps.api import main as api_main

    calls: list[str] = []

    class Connection:
        async def run_sync(self, _operation):
            calls.append("create_all")

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    class Engine:
        def begin(self):
            return BeginContext()

        async def dispose(self):
            calls.append("engine_disposed")

    def fail_if_session_is_opened(*_args, **_kwargs):
        raise AssertionError("lifespan must not scan ReviewRun without a lease protocol")

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        ),
    )
    monkeypatch.setattr(api_main, "engine", Engine())
    monkeypatch.setattr(api_main, "session_scope", fail_if_session_is_opened)
    monkeypatch.setattr(api_main, "chat_provider", None)
    monkeypatch.setattr(api_main, "embedder", object())
    monkeypatch.setattr(api_main, "reranker", None)
    monkeypatch.setattr(api_main, "qdrant", object())

    async with api_main.lifespan(SimpleNamespace(state=SimpleNamespace())):
        calls.append("serving")

    assert calls == ["create_all", "serving", "engine_disposed"]


async def test_telemetry_worker_loop_is_explicit_and_stops_cleanly(monkeypatch):
    from apps.api import main as api_main

    stop = asyncio.Event()
    batches: list[int] = []

    class Worker:
        async def process_batch(self, *, batch_size: int):
            batches.append(batch_size)
            stop.set()

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            telemetry_export_poll_seconds=0.01,
            telemetry_export_batch_size=7,
        ),
    )
    await api_main._run_telemetry_worker(Worker(), stop)

    assert batches == [7]


@pytest.mark.parametrize(
    ("poll_seconds", "batch_size", "error_code"),
    [
        (0, 32, "TELEMETRY_EXPORT_POLL_SECONDS_INVALID"),
        (1, 0, "TELEMETRY_EXPORT_BATCH_SIZE_INVALID"),
        (1, 1001, "TELEMETRY_EXPORT_BATCH_SIZE_INVALID"),
    ],
)
def test_telemetry_worker_config_is_rejected_before_background_start(
    monkeypatch,
    poll_seconds,
    batch_size,
    error_code,
):
    from apps.api import main as api_main

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            telemetry_outbox_worker_enabled=True,
            telemetry_exporter_backend="noop",
            app_env="test",
            telemetry_export_poll_seconds=poll_seconds,
            telemetry_export_batch_size=batch_size,
            telemetry_export_max_attempts=3,
            telemetry_export_lease_seconds=10,
            telemetry_export_base_backoff_seconds=1,
            telemetry_export_max_backoff_seconds=10,
        ),
    )

    with pytest.raises(RuntimeError, match=error_code):
        api_main._build_api_telemetry_worker()


async def test_immediate_telemetry_start_failure_still_closes_every_resource(monkeypatch):
    from apps.api import main as api_main

    closed: list[str] = []

    class Connection:
        async def run_sync(self, _operation):
            return None

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    class Engine:
        def begin(self):
            return BeginContext()

        async def dispose(self):
            closed.append("engine")

    class AsyncResource:
        def __init__(self, name: str):
            self.name = name

        async def aclose(self):
            closed.append(self.name)

    class SyncResource:
        def close(self):
            closed.append("qdrant")

    async def fail_immediately(_worker, _stop):
        raise RuntimeError("private exporter startup detail")

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        ),
    )
    monkeypatch.setattr(api_main, "engine", Engine())
    monkeypatch.setattr(api_main, "chat_provider", AsyncResource("chat"))
    monkeypatch.setattr(api_main, "embedder", AsyncResource("embedding"))
    monkeypatch.setattr(api_main, "reranker", None)
    monkeypatch.setattr(api_main, "qdrant", SyncResource())
    monkeypatch.setattr(api_main, "_build_api_telemetry_worker", lambda: object())
    monkeypatch.setattr(api_main, "_run_telemetry_worker", fail_immediately)

    with pytest.raises(RuntimeError, match="TELEMETRY_WORKER_START_FAILED"):
        async with api_main.lifespan(SimpleNamespace(state=SimpleNamespace())):
            raise AssertionError("failed worker must not reach serving state")

    assert closed == ["chat", "embedding", "qdrant", "engine"]


async def test_shutdown_closes_every_resource_when_one_close_fails():
    from apps.api import main as api_main

    closed: list[str] = []

    class AsyncResource:
        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail

        async def aclose(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    class SyncResource:
        def close(self):
            closed.append("qdrant")

    class Engine:
        async def dispose(self):
            closed.append("engine")

    failures = await api_main._close_resources_best_effort(
        [
            ("chat", AsyncResource("chat", fail=True), ("aclose", "close")),
            ("embedding", AsyncResource("embedding"), ("aclose", "close")),
            ("reranker", AsyncResource("reranker"), ("aclose", "close")),
            ("qdrant", SyncResource(), ("aclose", "close")),
            ("engine", Engine(), ("dispose",)),
        ]
    )

    assert failures == ["chat"]
    assert closed == ["chat", "embedding", "reranker", "qdrant", "engine"]


async def test_http_embedding_and_reranker_expose_connection_pool_close():
    from creditlens.infrastructure.llm.embedding import OpenAICompatEmbedding
    from creditlens.retrieval.rerank import HttpCrossEncoderReranker

    closed: list[str] = []

    class Client:
        def __init__(self, name: str):
            self.name = name

        async def aclose(self):
            closed.append(self.name)

    embedding = object.__new__(OpenAICompatEmbedding)
    embedding._client = Client("embedding")
    reranker = object.__new__(HttpCrossEncoderReranker)
    reranker._client = Client("reranker")

    await embedding.aclose()
    await reranker.aclose()

    assert closed == ["embedding", "reranker"]


@pytest.fixture
async def protocol_client(tmp_path, monkeypatch):
    """只构造 SSE 所需的最小案件/Membership 世界。"""
    from apps.api import main as api_main

    tenant_id = api_main.DEFAULT_TENANT_ID
    user_id = api_main.DEMO_USER_ID
    entity_id = uuid.uuid4()
    case_id = uuid.uuid4()
    run_id = uuid.uuid4()
    now = datetime(2026, 5, 1, tzinfo=UTC)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api_protocols.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    monkeypatch.setattr(api_main, "session_factory", factory)

    async with factory() as session:
        session.add_all(
            [
                Tenant(id=tenant_id, name="API protocol"),
                AppUser(
                    id=user_id,
                    tenant_id=tenant_id,
                    external_subject="api-protocol-user",
                    display_name="API protocol user",
                ),
                Entity(
                    id=entity_id,
                    tenant_id=tenant_id,
                    entity_type="COMPANY",
                    canonical_name="协议测试企业",
                ),
                CreditCase(
                    id=case_id,
                    tenant_id=tenant_id,
                    case_number="API-PROTOCOL",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1"),
                    application_date=date(2026, 5, 1),
                    as_of_date=date(2026, 5, 1),
                    decision_cutoff_at=now,
                ),
                CaseMembership(case_id=case_id, user_id=user_id, case_role="REVIEWER"),
                ReviewRun(
                    id=run_id,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    status="REWORK",
                    as_of_date=date(2026, 5, 1),
                    decision_cutoff_at=now,
                ),
                RunEvent(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    sequence_no=1,
                    event_type="STATE_CHANGED",
                    payload_redacted={"to": "REWORK"},
                ),
            ]
        )
        await session.commit()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        yield client, factory, run_id, case_id
    await engine.dispose()


async def test_sse_last_event_id_header_overrides_legacy_query(protocol_client):
    """标准 Last-Event-ID 头优先于兼容用 query 参数。"""
    client, _, run_id, _ = protocol_client
    response = await client.get(
        f"/api/v1/runs/{run_id}/events?last_event_id=0",
        headers={"Last-Event-ID": "1"},
    )

    assert response.status_code == 200
    assert "id: 1" not in response.text
    assert 'event: DONE\ndata: {"status": "REWORK"}' in response.text


async def test_sse_rework_stops_stream_and_query_cursor_remains_supported(protocol_client):
    """REWORK 不会让前端长连接悬挂，旧 query 续传仍会返回遗漏事件。"""
    client, _, run_id, _ = protocol_client
    response = await client.get(f"/api/v1/runs/{run_id}/events?last_event_id=0")

    assert response.status_code == 200
    assert "id: 1" in response.text
    assert 'event: DONE\ndata: {"status": "REWORK"}' in response.text


async def test_sse_rejects_negative_resume_cursor(protocol_client):
    """负游标不是合法的 RunEvent sequence，query/header 均应在入口返回 422。"""
    client, _, run_id, _ = protocol_client
    query_response = await client.get(f"/api/v1/runs/{run_id}/events?last_event_id=-1")
    header_response = await client.get(
        f"/api/v1/runs/{run_id}/events", headers={"Last-Event-ID": "-1"}
    )

    assert query_response.status_code == 422
    assert header_response.status_code == 422


async def test_cancelled_full_review_persists_fresh_invocation_and_terminates_run(
    protocol_client,
    monkeypatch,
):
    from apps.api import main as api_main

    from creditlens.agents import wiring as wiring_module
    from creditlens.agents.supervisor import Supervisor
    from creditlens.application import snapshot_service, trusted_context
    from creditlens.tools.gateway import ToolGateway

    _, factory, _, case_id = protocol_client
    async with factory() as session:
        run = ReviewRun(
            tenant_id=api_main.DEFAULT_TENANT_ID,
            case_id=case_id,
            run_type="FULL_REVIEW",
            status="RECEIVED",
            as_of_date=date(2026, 5, 1),
            decision_cutoff_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    gateway = ToolGateway()
    cancellation_started = asyncio.Event()
    never_release = asyncio.Event()

    async def successful_tool():
        return "ok"

    async def cancelled_tool():
        cancellation_started.set()
        await never_release.wait()

    gateway.register("successful_tool", successful_tool)
    gateway.register("cancelled_tool", cancelled_tool)
    gateway.grant("analyst", ["successful_tool", "cancelled_tool"])

    async def execute_stub(self, session, _trusted, _snapshot, run, events):
        for status in ("AUTHORIZED", "VALIDATING_CASE", "PLANNING", "EXECUTING"):
            await self._transition(session, run, events, status)
        await gateway.invoke(
            "analyst",
            "successful_tool",
            task_id=f"{run.id}:successful-task",
        )
        await gateway.invoke(
            "analyst",
            "cancelled_tool",
            task_id=f"{run.id}:cancelled-task",
        )

    def build_stub(
        _session,
        _qdrant,
        _embedder,
        _snapshot,
        *,
        invocation_session_factory,
        **_kwargs,
    ):
        assert invocation_session_factory is factory
        supervisor = Supervisor(
            policy_agent=object(),
            financial_agent=object(),
            challenger=object(),
            auditor=object(),
            tool_gateway=gateway,
            invocation_session_factory=invocation_session_factory,
        )
        supervisor._execute_with_event_writer = MethodType(execute_stub, supervisor)
        return supervisor, gateway

    trusted = SimpleNamespace(
        tenant_id=api_main.DEFAULT_TENANT_ID,
        user_id=api_main.DEMO_USER_ID,
        case_id=case_id,
    )

    async def trusted_stub(*_args, **_kwargs):
        return trusted

    async def snapshot_stub(*_args, **_kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(wiring_module, "build_supervisor", build_stub)
    monkeypatch.setattr(trusted_context, "build_trusted_context", trusted_stub)
    monkeypatch.setattr(snapshot_service, "load_snapshot_context", snapshot_stub)

    identity = api_main.APIIdentity(
        tenant_id=api_main.DEFAULT_TENANT_ID,
        user_id=api_main.DEMO_USER_ID,
    )
    background = asyncio.create_task(api_main._execute_review_background(run_id, case_id, identity))
    await asyncio.wait_for(cancellation_started.wait(), timeout=2)
    started_at = time.perf_counter()
    background.cancel()
    with pytest.raises(asyncio.CancelledError):
        await background
    elapsed = time.perf_counter() - started_at

    async with factory() as session:
        persisted_run = await session.get(ReviewRun, run_id)
        invocations = (
            await session.scalars(select(InvocationRecord).where(InvocationRecord.run_id == run_id))
        ).all()
        deliveries = (
            await session.scalars(select(TelemetryOutbox).where(TelemetryOutbox.run_id == run_id))
        ).all()
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.sequence_no)
            )
        ).all()

    assert persisted_run.status == "FAILED"
    assert persisted_run.completed_at is not None
    assert elapsed < 5
    assert {(record.kind, record.name, record.status) for record in invocations} == {
        ("TOOL", "successful_tool", "SUCCESS"),
        ("TOOL", "cancelled_tool", "CANCELLED"),
    }
    assert [delivery.status for delivery in deliveries] == ["PENDING", "PENDING"]
    assert events[-1].event_type == "BACKGROUND_CANCELLED"
    assert events[-1].payload_redacted == {"error_type": "FULL_REVIEW_CANCELLED"}


async def test_background_cleanup_failure_does_not_replace_original_cancellation():
    from apps.api import main as api_main

    async def failing_cleanup():
        raise RuntimeError("private cleanup failure")

    async def cancellation_boundary():
        try:
            raise asyncio.CancelledError("original background cancellation")
        except asyncio.CancelledError:
            await api_main._run_bounded_cancellation_cleanup(
                failing_cleanup(),
                timeout_seconds=0.1,
            )
            raise

    with pytest.raises(asyncio.CancelledError) as captured:
        await cancellation_boundary()
    assert captured.value.args == ("original background cancellation",)


async def test_background_cleanup_timeout_is_bounded_and_preserves_cancellation():
    from apps.api import main as api_main

    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def blocked_cleanup():
        cleanup_started.set()
        try:
            await cleanup_release.wait()
        finally:
            cleanup_finished.set()

    async def cancellation_boundary():
        try:
            raise asyncio.CancelledError("original background timeout cancellation")
        except asyncio.CancelledError:
            await api_main._run_bounded_cancellation_cleanup(
                blocked_cleanup(),
                timeout_seconds=0.05,
            )
            raise

    caller = asyncio.create_task(cancellation_boundary())
    await asyncio.wait_for(cleanup_started.wait(), timeout=2)
    started_at = time.perf_counter()
    with pytest.raises(asyncio.CancelledError) as captured:
        await asyncio.wait_for(caller, timeout=1)
    elapsed = time.perf_counter() - started_at

    assert captured.value.args == ("original background timeout cancellation",)
    assert elapsed < 0.5
    cleanup_release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=2)


async def test_trace_marks_legacy_runs_without_invocations_as_unavailable(protocol_client):
    client, _, run_id, _ = protocol_client

    response = await client.get(f"/api/v1/runs/{run_id}/trace")

    assert response.status_code == 200
    body = response.json()
    assert body["invocations"] == []
    assert body["delivery"] == {
        "contract_version": None,
        "status": "LEGACY_UNAVAILABLE",
        "complete": None,
        "total": 0,
        "counts": {
            "PENDING": 0,
            "PROCESSING": 0,
            "DELIVERED": 0,
            "DEAD": 0,
            "MISSING": 0,
            "INVALID": 0,
        },
    }


async def test_trace_marks_v2_run_without_invocations_as_empty_and_incomplete(protocol_client):
    client, factory, run_id, _ = protocol_client
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        run.model_manifest = {"invocation_contract_version": "invocation_v2"}
        await session.commit()

    response = await client.get(f"/api/v1/runs/{run_id}/trace")

    assert response.status_code == 200
    body = response.json()
    assert body["invocations"] == []
    assert body["integrity"] == {"status": "EMPTY", "valid": False, "invalid_count": 0}
    assert body["delivery"]["contract_version"] == "invocation_v2"
    assert body["delivery"]["status"] == "EMPTY"
    assert body["delivery"]["complete"] is False
    assert body["delivery"]["total"] == 0


async def test_trace_exposes_v2_invocations_and_delivery_lifecycle(protocol_client):
    client, factory, run_id, _ = protocol_client
    observed_at = datetime(2026, 5, 1, 1, tzinfo=UTC)
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        run.model_manifest = {"invocation_contract_version": "invocation_v2"}
        writer = InvocationWriter(
            session,
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
        )
        result = await writer.record(
            InvocationEnvelope(
                kind=InvocationKind.MODEL,
                name="grounded_qa_generation",
                provider="openai_compatible",
                model="interview-model",
                actor_role="grounded_qa",
                task_id="answer_generation",
                started_at=observed_at,
                ended_at=observed_at,
                latency_ms=0,
                status=InvocationStatus.SUCCESS,
                request_sha256="a" * 64,
                response_sha256="b" * 64,
            )
        )
        invocation_id = result.record.invocation_id
        await session.commit()

    pending = await client.get(f"/api/v1/runs/{run_id}/trace")
    assert pending.status_code == 200
    body = pending.json()
    assert body["delivery"]["status"] == "PENDING"
    assert body["delivery"]["complete"] is False
    assert body["delivery"]["counts"]["PENDING"] == 1
    assert body["invocations"][0]["invocation_id"] == str(invocation_id)
    assert body["invocations"][0]["envelope"]["kind"] == "MODEL"
    assert body["invocations"][0]["integrity"]["valid"] is True
    assert body["invocations"][0]["delivery"]["status"] == "PENDING"
    assert body["integrity"] == {"status": "VALID", "valid": True, "invalid_count": 0}

    async with factory() as session:
        delivery = await session.scalar(
            select(TelemetryOutbox).where(TelemetryOutbox.invocation_id == invocation_id)
        )
        delivery.status = "DELIVERED"
        delivery.attempts = 1
        delivery.delivered_at = observed_at
        await session.commit()

    delivered = await client.get(f"/api/v1/runs/{run_id}/trace")
    assert delivered.status_code == 200
    assert delivered.json()["delivery"]["status"] == "COMPLETE"
    assert delivered.json()["delivery"]["complete"] is True

    async with factory() as session:
        record = await session.get(InvocationRecord, invocation_id)
        original_payload = dict(record.payload_redacted)
        tampered_payload = dict(original_payload)
        tampered_payload["name"] = "forged private administrator payload"
        record.payload_redacted = tampered_payload
        await session.commit()

    tampered = await client.get(f"/api/v1/runs/{run_id}/trace")
    assert tampered.status_code == 200
    tampered_body = tampered.json()
    assert tampered_body["delivery"]["status"] == "DEGRADED"
    assert tampered_body["delivery"]["complete"] is False
    assert tampered_body["integrity"] == {
        "status": "DEGRADED",
        "valid": False,
        "invalid_count": 1,
    }
    assert tampered_body["invocations"][0]["envelope"] is None
    assert tampered_body["invocations"][0]["integrity"]["error_code"] == (
        "INVOCATION_INTEGRITY_FAILED"
    )
    assert "forged private administrator payload" not in tampered.text

    async with factory() as session:
        record = await session.get(InvocationRecord, invocation_id)
        record.payload_redacted = original_payload
        await session.commit()

    async with factory() as session:
        delivery = await session.scalar(
            select(TelemetryOutbox).where(TelemetryOutbox.invocation_id == invocation_id)
        )
        delivery.status = "DEAD"
        delivery.attempts = 2
        delivery.delivered_at = None
        delivery.dead_at = observed_at
        delivery.last_error_code = "EXPORT_FAILED"
        await session.commit()

    dead = await client.get(f"/api/v1/runs/{run_id}/trace")
    assert dead.status_code == 200
    assert dead.json()["delivery"]["status"] == "DEGRADED"
    assert dead.json()["delivery"]["complete"] is False


async def test_trace_classifies_valid_record_without_outbox_as_missing(protocol_client):
    client, factory, run_id, _ = protocol_client
    observed_at = datetime(2026, 5, 1, 2, tzinfo=UTC)
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        run.model_manifest = {"invocation_contract_version": "invocation_v2"}
        result = await InvocationWriter(
            session,
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
        ).record(
            InvocationEnvelope(
                kind=InvocationKind.TOOL,
                name="missing_delivery_tool",
                actor_role="analyst",
                task_id="missing-delivery",
                started_at=observed_at,
                ended_at=observed_at,
                latency_ms=0,
                status=InvocationStatus.SUCCESS,
                request_sha256="c" * 64,
                response_sha256="d" * 64,
            )
        )
        invocation_id = result.record.invocation_id
        await session.delete(result.outbox)
        await session.commit()

    response = await client.get(f"/api/v1/runs/{run_id}/trace")

    assert response.status_code == 200
    body = response.json()
    assert body["integrity"] == {"status": "VALID", "valid": True, "invalid_count": 0}
    assert body["delivery"]["status"] == "DEGRADED"
    assert body["delivery"]["complete"] is False
    assert body["delivery"]["counts"]["MISSING"] == 1
    assert body["delivery"]["counts"]["INVALID"] == 0
    assert body["invocations"][0]["invocation_id"] == str(invocation_id)
    assert body["invocations"][0]["envelope"]["name"] == "missing_delivery_tool"
    assert body["invocations"][0]["integrity"]["valid"] is True
    assert body["invocations"][0]["delivery"]["status"] == "MISSING"


async def test_idempotency_conflict_maps_to_409():
    from apps.api import main as api_main

    response = await api_main.creditlens_error_handler(
        None, IdempotencyConflictError("同一幂等键的请求内容不一致")
    )

    assert response.status_code == 409


async def test_failure_transition_appends_minimal_trace_and_bumps_version(protocol_client):
    """后台失败路径不应留下无 Trace 的 FAILED Run。"""
    from apps.api import main as api_main

    _, factory, _, case_id = protocol_client
    async with factory() as session:
        run = ReviewRun(
            tenant_id=api_main.DEFAULT_TENANT_ID,
            case_id=case_id,
            status="RUNNING",
            as_of_date=date(2026, 5, 1),
            decision_cutoff_at=datetime(2026, 5, 1, tzinfo=UTC),
            state_version=7,
        )
        session.add(run)
        await session.flush()
        session.add(
            RunEvent(
                run_id=run.id,
                tenant_id=run.tenant_id,
                case_id=run.case_id,
                sequence_no=4,
                event_type="STAGE_STARTED",
                payload_redacted={},
            )
        )
        await session.flush()

        changed = await api_main._mark_run_failed_with_trace(
            session, run.id, "BACKGROUND_FAILED", "RuntimeError"
        )
        assert changed is True
        await session.commit()

    async with factory() as session:
        persisted = await session.get(ReviewRun, run.id)
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.sequence_no)
            )
        ).all()
    assert persisted.status == "FAILED"
    assert persisted.state_version == 8
    assert events[-1].sequence_no == 5
    assert events[-1].event_type == "BACKGROUND_FAILED"
    assert events[-1].payload_redacted == {"error_type": "RuntimeError"}


async def test_get_run_exposes_supporting_and_opposing_locators_by_evidence_key(protocol_client):
    """HITL 返回稳定 evidence_key 对应的本 Run 文档定位，不能泄露其他 Run。"""
    from creditlens.agents.contracts import (
        AgentClaim,
        AnswerStatus,
        GroundedAnswerArtifact,
        RefusalReasonCode,
    )
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )

    client, factory, run_id, _ = protocol_client
    supporting_key = uuid.uuid4()
    opposing_key = uuid.uuid4()
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        run.run_type = "SIMPLE_QA"
        run.model_manifest = {
            "workflow": "grounded_qa_v1",
            "prompt_version": "grounded_qa_v1",
            "model_invocation_ids": [],
        }
        source_claim = AgentClaim(
            category="DATA_CONFLICT",
            statement="conflicting evidence requires review",
            verdict="PARTIALLY_SUPPORTED",
            supporting_evidence_ids=[supporting_key],
            opposing_evidence_ids=[opposing_key],
            as_of_date=run.as_of_date,
        )
        grounded = GroundedAnswerArtifact(
            run_id=run.id,
            task_id="grounded_qa",
            producer="grounded_qa",
            lifecycle_status="VERIFIED",
            execution_status="INSUFFICIENT_EVIDENCE",
            claims=[source_claim],
            answer_status=AnswerStatus.ABSTAINED,
            abstention_reason="Conflicting evidence requires review.",
            refusal_reason_code=RefusalReasonCode.MISSING_BANK_STATEMENTS,
            prompt_version="grounded_qa_v1",
        )
        persisted_payload = grounded.model_dump(mode="json", exclude={"output_hash"})
        persisted_payload["generation_mode"] = "llm"
        artifact = ArtifactRecord(
            id=grounded.artifact_id,
            tenant_id=run.tenant_id,
            run_id=run.id,
            task_id="grounded_qa",
            artifact_type="GROUNDED_ANSWER",
            contract_version=grounded.contract_version,
            producer="grounded_qa",
            lifecycle_status="VERIFIED",
            execution_status=grounded.execution_status,
            payload=persisted_payload,
            output_hash=canonical_artifact_payload_hash(persisted_payload),
        )
        session.add(artifact)
        await session.flush()
        session.add_all(
            [
                RunEvent(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    case_id=run.case_id,
                    sequence_no=2,
                    event_type="EVIDENCE_READY",
                    payload_redacted={},
                ),
                EvidenceRecord(
                    evidence_key=supporting_key,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type="DOCUMENT_SPAN",
                    source_id=uuid.uuid4(),
                    document_version_id=uuid.uuid4(),
                    section_id=uuid.uuid4(),
                    page_number=3,
                    locator={"parse_run_id": str(uuid.uuid4())},
                    content_hash="a" * 64,
                ),
                EvidenceRecord(
                    evidence_key=opposing_key,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type="DOCUMENT_SPAN",
                    source_id=uuid.uuid4(),
                    document_version_id=uuid.uuid4(),
                    section_id=uuid.uuid4(),
                    page_number=4,
                    locator={"parse_run_id": str(uuid.uuid4())},
                    content_hash="b" * 64,
                ),
            ]
        )
        session.add(
            ClaimRecord(
                id=source_claim.claim_id,
                tenant_id=run.tenant_id,
                run_id=run.id,
                artifact_id=artifact.id,
                category=source_claim.category,
                statement=source_claim.statement,
                verdict=source_claim.verdict,
                as_of_date=run.as_of_date,
                review_status="PENDING",
                payload={
                    "supporting_evidence_ids": [str(supporting_key)],
                    "opposing_evidence_ids": [str(opposing_key)],
                    "calculation_ids": [],
                    "answer_status": "ABSTAINED",
                },
            )
        )
        await session.commit()

    response = await client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    evidence = response.json()["claims"][0]["evidence"]
    assert evidence["supporting_locators"][0]["page_number"] == 3
    assert evidence["opposing_locators"][0]["page_number"] == 4
    assert evidence["supporting_locators"][0]["parse_run_id"]
    assert response.json()["grounded_answer"]["refusal_reason_code"] == "MISSING_BANK_STATEMENTS"

    # Simulate storage corruption outside the column-restricted app role.
    async with factory() as session:
        claim = await session.get(ClaimRecord, source_claim.claim_id)
        claim.statement = "tampered claim"
        await session.commit()
    corrupted = await client.get(f"/api/v1/runs/{run_id}")
    assert corrupted.status_code == 409
    assert corrupted.json() == {"detail": "ARTIFACT_INTEGRITY_FAILED"}
