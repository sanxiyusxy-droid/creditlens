"""Runtime boundaries route terminal invocations to the durable v2 sink."""

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from creditlens.agents.policy_agent import PolicyAgent
from creditlens.agents.supervisor import Supervisor, _fail_closed_model_trace_sink
from creditlens.infrastructure.llm.chat import ModelInvocationTrace
from creditlens.infrastructure.postgres.models import InvocationRecord, TelemetryOutbox
from creditlens.infrastructure.postgres.session import create_session_factory
from creditlens.observability.invocation import (
    InvocationEnvelope,
    InvocationKind,
    InvocationStatus,
)
from creditlens.observability.writer import InvocationAuditPersistError
from creditlens.retrieval.contracts import RetrievedCandidate
from creditlens.tools.gateway import ToolCallDeniedError, ToolGateway

_HASH = "a" * 64


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [
        ("success", InvocationStatus.SUCCESS),
        ("failed", InvocationStatus.FAILED),
        ("denied", InvocationStatus.DENIED),
        ("cancelled", InvocationStatus.CANCELLED),
    ],
)
async def test_gateway_v2_sink_owns_every_terminal_status_without_legacy_dual_write(
    mode,
    expected_status,
):
    gateway = ToolGateway()
    invocations = []
    legacy_events = []

    async def tool():
        if mode == "failed":
            raise RuntimeError("private tool failure")
        if mode == "cancelled":
            raise asyncio.CancelledError
        return "ok"

    gateway.register("lookup", tool)
    if mode != "denied":
        gateway.grant("analyst", ["lookup"])
    legacy_token = gateway.bind_event_sink(
        lambda event_type, payload: legacy_events.append((event_type, payload))
    )
    invocation_token = gateway.bind_invocation_sink(invocations.append, fail_closed=True)
    try:
        with contextlib.suppress(RuntimeError, ToolCallDeniedError, asyncio.CancelledError):
            await gateway.invoke("analyst", "lookup", task_id="run:task")
    finally:
        gateway.reset_invocation_sink(invocation_token)
        gateway.reset_event_sink(legacy_token)

    assert [envelope.status for envelope in invocations] == [expected_status]
    assert legacy_events == []


async def test_gateway_fail_closed_preserves_success_fact_and_exposes_only_safe_code():
    gateway = ToolGateway()
    side_effects = []

    async def mutate():
        side_effects.append("committed")
        return "ok"

    async def unavailable_writer(_envelope):
        raise RuntimeError("database password and SQL must not escape")

    gateway.register("mutate", mutate)
    gateway.grant("analyst", ["mutate"])
    token = gateway.bind_invocation_sink(unavailable_writer, fail_closed=True)
    try:
        with pytest.raises(InvocationAuditPersistError) as captured:
            await gateway.invoke("analyst", "mutate", task_id="run:task")
    finally:
        gateway.reset_invocation_sink(token)

    assert captured.value.error_code == "INVOCATION_AUDIT_PERSIST_FAILED"
    assert side_effects == ["committed"]
    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.invocation is not None
    assert record.invocation.status == InvocationStatus.SUCCESS
    assert record.observability_error_codes == ("INVOCATION_AUDIT_PERSIST_FAILED",)


async def test_gateway_cancellation_remains_control_flow_when_v2_writer_fails():
    gateway = ToolGateway()

    async def cancelled():
        raise asyncio.CancelledError

    async def unavailable_writer(_envelope):
        raise RuntimeError("private persistence detail")

    gateway.register("cancelled", cancelled)
    gateway.grant("analyst", ["cancelled"])
    token = gateway.bind_invocation_sink(unavailable_writer, fail_closed=True)
    try:
        with pytest.raises(asyncio.CancelledError):
            await gateway.invoke("analyst", "cancelled", task_id="run:task")
    finally:
        gateway.reset_invocation_sink(token)

    record = gateway.calls[-1]
    assert record.status == "CANCELLED"
    assert record.invocation is not None
    assert record.invocation.status == InvocationStatus.CANCELLED
    assert record.observability_error_codes == ("INVOCATION_AUDIT_PERSIST_FAILED",)


@pytest.mark.parametrize(
    "terminal_status",
    [
        InvocationStatus.SUCCESS,
        InvocationStatus.FAILED,
        InvocationStatus.DENIED,
        InvocationStatus.CANCELLED,
    ],
)
async def test_supervisor_tool_terminal_commit_drains_before_outer_cancellation(
    engine,
    monkeypatch,
    terminal_status,
):
    from creditlens.agents import supervisor as supervisor_module

    factory = create_session_factory(engine)
    started = asyncio.Event()
    release = asyncio.Event()
    real_writer = supervisor_module.InvocationWriter

    class GatedWriter:
        def __init__(self, *args, **kwargs):
            self._delegate = real_writer(*args, **kwargs)

        async def record(self, envelope):
            started.set()
            await release.wait()
            return await self._delegate.record(envelope)

    monkeypatch.setattr(supervisor_module, "InvocationWriter", GatedWriter)
    supervisor = Supervisor(
        policy_agent=object(),
        financial_agent=object(),
        challenger=object(),
        auditor=object(),
        invocation_session_factory=factory,
        invocation_cancel_persist_timeout_seconds=1,
    )
    tenant_id = uuid.uuid4()
    case_id = uuid.uuid4()
    run_id = uuid.uuid4()
    observed_at = datetime(2026, 8, 21, tzinfo=UTC)
    envelope = InvocationEnvelope(
        kind=InvocationKind.TOOL,
        name="gated_tool",
        actor_role="analyst",
        task_id=f"{run_id}:gated-tool",
        started_at=observed_at,
        ended_at=observed_at,
        latency_ms=0,
        status=terminal_status,
        error_code=None if terminal_status == InvocationStatus.SUCCESS else "TOOL_TERMINATED",
        request_sha256="a" * 64,
        response_sha256="b" * 64 if terminal_status == InvocationStatus.SUCCESS else None,
    )
    sink = supervisor._tool_invocation_sink(
        None,
        tenant_id=tenant_id,
        case_id=case_id,
        run_id=run_id,
        user_id=None,
    )
    caller = asyncio.create_task(sink(envelope))
    await asyncio.wait_for(started.wait(), timeout=2)

    caller.cancel(f"outer cancellation after {terminal_status.value}")
    await asyncio.sleep(0)
    assert not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as captured:
        await caller

    assert captured.value.args == (f"outer cancellation after {terminal_status.value}",)
    async with factory() as session:
        record = await session.scalar(
            select(InvocationRecord).where(InvocationRecord.invocation_id == envelope.invocation_id)
        )
        outbox = await session.scalar(
            select(TelemetryOutbox).where(TelemetryOutbox.invocation_id == envelope.invocation_id)
        )
    assert record.status == terminal_status.value
    assert outbox.status == "PENDING"
    assert outbox.invocation_id == record.invocation_id


async def test_gateway_fail_closed_when_terminal_envelope_cannot_be_constructed():
    gateway = ToolGateway(_fingerprint_key_version="unsafe key")

    async def lookup():
        return "ok"

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    token = gateway.bind_invocation_sink(lambda _envelope: None, fail_closed=True)
    try:
        with pytest.raises(InvocationAuditPersistError):
            await gateway.invoke("analyst", "lookup", task_id="run:task")
    finally:
        gateway.reset_invocation_sink(token)

    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.invocation is None
    assert "INVOCATION_AUDIT_PERSIST_FAILED" in record.observability_error_codes


def _trace(*, status: str = "SUCCESS") -> ModelInvocationTrace:
    return ModelInvocationTrace(
        provider="openai_compatible",
        model="policy-model",
        prompt_version="policy_statement_v1",
        prompt_sha256=_HASH,
        request_sha256=_HASH,
        response_sha256=_HASH if status == "SUCCESS" else None,
        latency_ms=2,
        attempts=1,
        status=status,
        error_type="HTTPStatusError" if status == "FAILED" else None,
    )


async def test_supervisor_policy_model_commit_drains_before_outer_cancellation(
    engine,
    monkeypatch,
):
    from creditlens.agents import supervisor as supervisor_module

    factory = create_session_factory(engine)
    started = asyncio.Event()
    release = asyncio.Event()
    real_writer = supervisor_module.InvocationWriter

    class GatedWriter:
        def __init__(self, *args, **kwargs):
            self._delegate = real_writer(*args, **kwargs)

        async def record_model_trace(self, trace, **kwargs):
            started.set()
            await release.wait()
            return await self._delegate.record_model_trace(trace, **kwargs)

    monkeypatch.setattr(supervisor_module, "InvocationWriter", GatedWriter)
    supervisor = Supervisor(
        policy_agent=object(),
        financial_agent=object(),
        challenger=object(),
        auditor=object(),
        invocation_session_factory=factory,
        invocation_cancel_persist_timeout_seconds=1,
    )
    tenant_id = uuid.uuid4()
    case_id = uuid.uuid4()
    run_id = uuid.uuid4()
    trace = _trace()
    sink = supervisor._model_invocation_sink(
        None,
        tenant_id=tenant_id,
        case_id=case_id,
        run_id=run_id,
        user_id=None,
    )
    caller = asyncio.create_task(
        sink(
            trace,
            name="policy_statement",
            actor_role="policy_analyst",
            task_id=f"{run_id}:policy-review",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)

    caller.cancel("outer cancellation after policy model success")
    await asyncio.sleep(0)
    assert not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as captured:
        await caller

    assert captured.value.args == ("outer cancellation after policy model success",)
    async with factory() as session:
        record = await session.scalar(
            select(InvocationRecord).where(InvocationRecord.invocation_id == trace.invocation_id)
        )
        outbox = await session.scalar(
            select(TelemetryOutbox).where(TelemetryOutbox.invocation_id == trace.invocation_id)
        )
    assert record.status == "SUCCESS"
    assert outbox.status == "PENDING"
    assert outbox.invocation_id == record.invocation_id


def _candidate() -> RetrievedCandidate:
    return RetrievedCandidate(
        section_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        parse_run_id=uuid.uuid4(),
        page_start=1,
        page_end=1,
        heading_path=["政策", "准入条件"],
        text="企业应满足政策原文中列明的准入条件。",
        text_hash="b" * 64,
        channel="DENSE",
        rank=1,
        raw_score=1.0,
    )


async def test_policy_traced_success_is_sent_with_stable_runtime_context():
    trace = _trace()

    class Chat:
        async def generate_structured_traced(self, **kwargs):
            assert kwargs["prompt_version"] == "policy_statement_v1"
            return SimpleNamespace(
                output=SimpleNamespace(statement="该企业应满足政策原文列明的准入条件。"),
                trace=trace,
            )

    recorded = []
    agent = PolicyAgent(ToolGateway(), chat=Chat())
    token = agent.bind_model_trace_sink(
        lambda model_trace, **kwargs: recorded.append((model_trace, kwargs))
    )
    try:
        statement = await agent._make_statement(
            "准入条件",
            [_candidate()],
            invocation_task_id="run-id:policy-review",
        )
    finally:
        agent.reset_model_trace_sink(token)

    assert "该企业应满足" in statement
    assert recorded == [
        (
            trace,
            {
                "name": "policy_statement",
                "actor_role": "policy_analyst",
                "task_id": "run-id:policy-review",
            },
        )
    ]


async def test_policy_provider_failure_records_failed_trace_before_safe_fallback():
    trace = _trace(status="FAILED")

    class ProviderFailure(Exception):
        def __init__(self):
            super().__init__("private provider detail")
            self.trace = trace

    class Chat:
        async def generate_structured_traced(self, **_kwargs):
            raise ProviderFailure

    recorded = []
    agent = PolicyAgent(ToolGateway(), chat=Chat())
    token = agent.bind_model_trace_sink(
        lambda model_trace, **kwargs: recorded.append((model_trace, kwargs))
    )
    try:
        statement = await agent._make_statement(
            "准入条件",
            [_candidate()],
            invocation_task_id="run-id:policy-review",
        )
    finally:
        agent.reset_model_trace_sink(token)

    assert statement.startswith("审查日适用政策")
    assert recorded[0][0].status == "FAILED"


async def test_policy_audit_write_failure_is_not_hidden_by_provider_fallback():
    trace = _trace(status="FAILED")

    class ProviderFailure(Exception):
        def __init__(self):
            super().__init__("private provider detail")
            self.trace = trace

    class Chat:
        async def generate_structured_traced(self, **_kwargs):
            raise ProviderFailure

    class Writer:
        async def record_model_trace(self, _trace, **_kwargs):
            raise RuntimeError("private database detail")

    agent = PolicyAgent(ToolGateway(), chat=Chat())
    token = agent.bind_model_trace_sink(_fail_closed_model_trace_sink(Writer()))
    try:
        with pytest.raises(InvocationAuditPersistError) as captured:
            await agent._make_statement(
                "准入条件",
                [_candidate()],
                invocation_task_id="run-id:policy-review",
            )
    finally:
        agent.reset_model_trace_sink(token)

    assert captured.value.error_code == "INVOCATION_AUDIT_PERSIST_FAILED"
