import asyncio
import contextlib
import dataclasses
import types
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum

import pytest

from creditlens.infrastructure.llm.chat import ModelInvocationTrace
from creditlens.observability.invocation import (
    InvocationKind,
    InvocationStatus,
    InvocationTimeQuality,
    ModelPrice,
    PayloadCanonicalizationError,
    PricingCatalog,
    TokenUsage,
    adapt_model_invocation_trace,
    aggregate_invocations,
    best_effort_hmac_fingerprint,
    estimate_model_cost,
    hash_invocation_payload,
    invocation_event_type,
    invocation_run_event_payload,
)
from creditlens.tools.gateway import (
    ToolBudgetExceededError,
    ToolCallDeniedError,
    ToolGateway,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _pricing() -> PricingCatalog:
    return PricingCatalog(
        version="interview-demo-2026-08-12",
        entries=(
            ModelPrice(
                provider="openai_compatible",
                model="priced-model",
                input_per_million_usd=Decimal("1.5"),
                output_per_million_usd=Decimal("2"),
            ),
        ),
    )


def test_adapts_legacy_model_trace_and_estimates_cost_from_versioned_table():
    trace = ModelInvocationTrace(
        provider="openai_compatible",
        model="priced-model",
        prompt_version="grounded-qa-v1",
        prompt_sha256=_HASH_A,
        request_sha256=_HASH_B,
        response_sha256=_HASH_C,
        input_tokens=1_000_000,
        output_tokens=500_000,
        total_tokens=1_500_000,
        latency_ms=125,
        attempts=2,
        status="SUCCESS",
    )
    envelope = adapt_model_invocation_trace(
        trace, ended_at=datetime(2026, 8, 12, tzinfo=UTC), pricing=_pricing()
    )
    assert envelope.invocation_id == trace.invocation_id
    assert envelope.kind == InvocationKind.MODEL
    assert envelope.version == "grounded-qa-v1"
    assert envelope.token_usage == TokenUsage(
        input_tokens=1_000_000, output_tokens=500_000, total_tokens=1_500_000
    )
    assert envelope.cost is not None
    assert envelope.cost.amount_usd == Decimal("2.5")
    assert envelope.cost.estimated is True
    assert envelope.cost.pricing_version == "interview-demo-2026-08-12"
    assert invocation_event_type(envelope) == "MODEL_INVOCATION_SUCCESS"
    assert envelope.time_quality == InvocationTimeQuality.ESTIMATED


def test_unknown_price_or_incomplete_usage_never_guesses_cost():
    assert (
        estimate_model_cost(
            provider="openai_compatible",
            model="unknown-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            pricing=_pricing(),
        )
        is None
    )


def test_aggregate_invocations_sums_only_complete_comparable_costs_and_tokens():
    ended_at = datetime(2026, 8, 12, tzinfo=UTC)
    traces = [
        ModelInvocationTrace(
            provider="openai_compatible",
            model="priced-model",
            prompt_version="v1",
            prompt_sha256=_HASH_A,
            request_sha256=_HASH_B,
            response_sha256=_HASH_C,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=1,
            attempts=1,
            status="SUCCESS",
        ),
        ModelInvocationTrace(
            provider="openai_compatible",
            model="priced-model",
            prompt_version="v1",
            prompt_sha256=_HASH_A,
            request_sha256=_HASH_B,
            response_sha256=_HASH_C,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            latency_ms=2,
            attempts=1,
            status="SUCCESS",
        ),
    ]
    envelopes = [
        adapt_model_invocation_trace(trace, ended_at=ended_at, pricing=_pricing())
        for trace in traces
    ]
    aggregate = aggregate_invocations(envelopes)
    assert aggregate.model_invocation_count == 2
    assert aggregate.input_tokens == 30
    assert aggregate.output_tokens == 15
    assert aggregate.total_tokens == 45
    assert aggregate.token_usage_complete is True
    assert aggregate.estimated_cost_usd == Decimal("0.000075")
    assert aggregate.cost_complete is True
    assert aggregate.pricing_version == "interview-demo-2026-08-12"

    incomplete = aggregate_invocations(
        [envelopes[0], envelopes[1].model_copy(update={"cost": None, "token_usage": None})]
    )
    assert incomplete.input_tokens is None
    assert incomplete.token_usage_complete is False
    assert incomplete.estimated_cost_usd is None
    assert incomplete.cost_complete is False
    assert incomplete.pricing_version is None
    assert (
        estimate_model_cost(
            provider="openai_compatible",
            model="priced-model",
            usage=TokenUsage(input_tokens=10, output_tokens=None, total_tokens=None),
            pricing=_pricing(),
        )
        is None
    )


def test_hashing_and_run_event_payload_do_not_expose_raw_content():
    secret = "customer-id-110101-secret-prompt"
    digest = hash_invocation_payload({"prompt": secret})
    assert digest == hash_invocation_payload({"prompt": secret})
    assert secret not in digest
    trace = ModelInvocationTrace(
        provider="openai_compatible",
        model="unknown-model",
        prompt_version="v1",
        prompt_sha256=_HASH_A,
        request_sha256=digest,
        response_sha256=None,
        latency_ms=1,
        attempts=1,
        status="FAILED",
        error_type="ValidationError",
    )
    payload = invocation_run_event_payload(adapt_model_invocation_trace(trace))
    assert secret not in str(payload)
    assert "prompt" not in payload
    assert payload["error_code"] == "VALIDATION_ERROR"


class _Color(Enum):
    RED = "red"


@dataclasses.dataclass
class _FingerprintFixture:
    as_of: date
    observed_at: datetime
    color: _Color
    amount: Decimal
    blob: bytes


def test_versioned_typed_fingerprint_supports_domain_types_without_collisions():
    fixture = _FingerprintFixture(
        as_of=date(2026, 8, 12),
        observed_at=datetime(2026, 8, 12, 1, 2, 3, tzinfo=UTC),
        color=_Color.RED,
        amount=Decimal("1.00"),
        blob=b"abc",
    )
    assert hash_invocation_payload(fixture) == hash_invocation_payload(fixture)
    assert hash_invocation_payload(date(2026, 8, 12)) != hash_invocation_payload(date(2026, 8, 13))
    assert hash_invocation_payload({1: "x", "1": "y"}) != hash_invocation_payload(
        {"1": "x", 1: "y"}
    )
    assert hash_invocation_payload(Decimal("1.0")) != hash_invocation_payload(Decimal("1.00"))


def test_fingerprint_rejects_recursive_and_unsupported_values_without_content():
    recursive: list[object] = []
    recursive.append(recursive)
    with pytest.raises(PayloadCanonicalizationError, match="FINGERPRINT_RECURSIVE_VALUE"):
        hash_invocation_payload(recursive)
    with pytest.raises(PayloadCanonicalizationError, match="FINGERPRINT_UNSUPPORTED_TYPE"):
        hash_invocation_payload(object())


def test_hmac_fingerprint_is_secret_scoped_and_best_effort_failure_is_safe():
    first, available, error = best_effort_hmac_fingerprint(
        {"account": "1234"}, secret=b"a" * 32, domain="request"
    )
    second, _, _ = best_effort_hmac_fingerprint(
        {"account": "1234"}, secret=b"b" * 32, domain="request"
    )
    assert first != second
    assert available is True
    assert error is None

    fallback, available, error = best_effort_hmac_fingerprint(
        object(), secret=b"a" * 32, domain="request"
    )
    assert len(fallback) == 64
    assert available is False
    assert error == "FINGERPRINT_UNSUPPORTED_TYPE"


async def test_tool_gateway_emits_success_envelope_without_raw_arguments_or_result():
    gateway = ToolGateway()

    async def lookup(account: str):
        return {"account": account, "balance": 100}

    gateway.register("lookup", lookup, provider="postgres", version="schema-v3")
    gateway.grant("analyst", ["lookup"])
    secret = "customer-account-123"
    assert await gateway.invoke("analyst", "lookup", account=secret) == {
        "account": secret,
        "balance": 100,
    }
    record = gateway.calls[-1]
    envelope = record.invocation
    assert envelope is not None
    assert record.invocation_id == envelope.invocation_id
    assert envelope.kind == InvocationKind.TOOL
    assert envelope.status == InvocationStatus.SUCCESS
    assert envelope.provider == "postgres"
    assert envelope.version == "schema-v3"
    assert envelope.response_sha256 is not None
    assert secret not in str(invocation_run_event_payload(envelope))


async def test_tool_gateway_emits_stable_safe_codes_for_denied_and_failed_calls():
    gateway = ToolGateway()

    async def broken():
        raise RuntimeError("customer PII must not be recorded")

    gateway.register("broken", broken)
    gateway.grant("analyst", ["broken"])
    with pytest.raises(RuntimeError):
        await gateway.invoke("analyst", "broken")
    failed = gateway.calls[-1]
    assert failed.error_code == "TOOL_EXECUTION_FAILED"
    assert failed.invocation is not None
    assert failed.invocation.status == InvocationStatus.FAILED
    assert "customer PII" not in str(invocation_run_event_payload(failed.invocation))

    with pytest.raises(ToolCallDeniedError):
        await gateway.invoke("analyst", "not_granted", query="secret")
    denied = gateway.calls[-1]
    assert denied.error_code == "TOOL_CALL_DENIED"
    assert denied.invocation is not None
    assert denied.invocation.status == InvocationStatus.DENIED
    assert denied.invocation.response_sha256 is None


async def test_observability_failure_never_changes_successful_tool_result():
    gateway = ToolGateway()
    side_effects: list[str] = []
    unsupported_result = object()

    async def mutate():
        side_effects.append("done")
        return unsupported_result

    gateway.register("mutate", mutate)
    gateway.grant("analyst", ["mutate"])
    assert await gateway.invoke("analyst", "mutate", task_id="task-a") is unsupported_result
    assert side_effects == ["done"]
    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.invocation is not None
    assert record.invocation.status == InvocationStatus.SUCCESS
    assert record.invocation.response_fingerprint_available is False
    assert record.invocation.observability_error_codes == ("FINGERPRINT_UNSUPPORTED_TYPE",)


async def test_invalid_fingerprint_secret_never_blocks_tool_execution():
    gateway = ToolGateway(_fingerprint_secret=b"short")

    async def lookup():
        return "ok"

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    assert await gateway.invoke("analyst", "lookup", task_id="task-a") == "ok"
    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.invocation is not None
    assert record.invocation.request_fingerprint_available is False
    assert record.invocation.response_fingerprint_available is False
    assert record.invocation.observability_error_codes == ("FINGERPRINT_SECRET_INVALID",)


async def test_budget_counts_attempts_per_task_and_uses_matching_error_code():
    gateway = ToolGateway(max_calls_per_task=1)

    async def broken():
        raise RuntimeError("failed attempt")

    gateway.register("broken", broken)
    gateway.grant("analyst", ["broken"])
    with pytest.raises(RuntimeError):
        await gateway.invoke("analyst", "broken", task_id="task-a")
    with pytest.raises(ToolBudgetExceededError) as captured:
        await gateway.invoke("analyst", "broken", task_id="task-a")
    assert captured.value.error_code == "BUDGET_EXCEEDED"
    assert gateway.calls[-1].error_code == "BUDGET_EXCEEDED"
    assert gateway.calls[-1].invocation is not None
    assert gateway.calls[-1].invocation.error_code == "BUDGET_EXCEEDED"

    with pytest.raises(RuntimeError):
        await gateway.invoke("analyst", "broken", task_id="task-b")


async def test_budget_reserves_in_flight_attempt_before_concurrent_tool_can_start():
    gateway = ToolGateway(max_calls_per_task=1)
    tool_started = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def lookup():
        nonlocal tool_started
        tool_started += 1
        first_started.set()
        await release_first.wait()
        return "ok"

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    first = asyncio.create_task(gateway.invoke("analyst", "lookup", task_id="shared-task"))
    await first_started.wait()

    with pytest.raises(ToolBudgetExceededError):
        await gateway.invoke("analyst", "lookup", task_id="shared-task")

    assert tool_started == 1
    assert gateway.calls[0].status == "IN_FLIGHT"
    assert gateway.calls[1].error_code == "BUDGET_EXCEEDED"
    release_first.set()
    assert await first == "ok"
    assert gateway.calls[0].status == "SUCCESS"


async def test_cancelled_tool_call_is_audited_and_reraised():
    gateway = ToolGateway()

    async def cancelled():
        raise asyncio.CancelledError

    gateway.register("cancelled", cancelled)
    gateway.grant("analyst", ["cancelled"])
    with pytest.raises(asyncio.CancelledError):
        await gateway.invoke("analyst", "cancelled", task_id="task-a")
    record = gateway.calls[-1]
    assert record.status == "CANCELLED"
    assert record.error_code == "TOOL_CALL_CANCELLED"
    assert record.invocation is not None
    assert record.invocation.status == InvocationStatus.CANCELLED


async def test_wall_clock_rollback_is_adjusted_without_changing_result(monkeypatch):
    from creditlens.tools import gateway as gateway_module

    initial = datetime(2026, 8, 12, tzinfo=UTC)
    wall_times = iter((initial, initial - timedelta(seconds=1)))
    monkeypatch.setattr(gateway_module, "utc_now", lambda: next(wall_times))
    gateway = ToolGateway()

    async def lookup():
        return "ok"

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    assert await gateway.invoke("analyst", "lookup", task_id="task-a") == "ok"
    envelope = gateway.calls[-1].invocation
    assert envelope is not None
    assert envelope.time_quality == InvocationTimeQuality.CLOCK_ADJUSTED
    assert envelope.ended_at >= envelope.started_at


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [
        ("success", InvocationStatus.SUCCESS),
        ("failed", InvocationStatus.FAILED),
        ("denied", InvocationStatus.DENIED),
        ("cancelled", InvocationStatus.CANCELLED),
    ],
)
async def test_tool_event_sink_receives_every_terminal_status(mode, expected_status):
    gateway = ToolGateway()
    events: list[tuple[str, dict]] = []
    gateway.bind_event_sink(lambda event_type, payload: events.append((event_type, payload)))

    async def tool():
        if mode == "failed":
            raise RuntimeError("boom")
        if mode == "cancelled":
            raise asyncio.CancelledError
        return "ok"

    gateway.register("tool", tool)
    if mode != "denied":
        gateway.grant("analyst", ["tool"])
    with contextlib.suppress(RuntimeError, ToolCallDeniedError, asyncio.CancelledError):
        await gateway.invoke("analyst", "tool", task_id="task-a")
    assert len(events) == 1
    assert events[0][0] == f"TOOL_INVOCATION_{expected_status.value}"
    assert events[0][1]["status"] == expected_status.value


async def test_sync_event_sink_may_return_a_non_awaitable_value():
    gateway = ToolGateway()
    gateway.bind_event_sink(lambda _event_type, _payload: 1)

    async def lookup():
        return "ok"

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    assert await gateway.invoke("analyst", "lookup", task_id="task-a") == "ok"
    assert gateway.calls[-1].observability_error_codes == ()


async def test_tool_event_sink_failure_never_changes_result_and_is_visible_in_memory():
    gateway = ToolGateway()

    async def broken_sink(_event_type, _payload):
        raise RuntimeError("database unavailable")

    async def lookup():
        return "ok"

    gateway.bind_event_sink(broken_sink)
    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    assert await gateway.invoke("analyst", "lookup", task_id="task-a") == "ok"
    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.observability_error_codes == ("INVOCATION_EVENT_SINK_FAILED",)
    assert record.invocation is not None
    assert record.invocation.observability_error_codes == ("INVOCATION_EVENT_SINK_FAILED",)


async def test_event_sink_cancellation_is_never_swallowed_after_successful_tool():
    gateway = ToolGateway()

    async def cancelled_sink(_event_type, _payload):
        raise asyncio.CancelledError

    async def lookup():
        return "completed side effect"

    gateway.bind_event_sink(cancelled_sink)
    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    with pytest.raises(asyncio.CancelledError):
        await gateway.invoke("analyst", "lookup", task_id="task-a")

    # The tool itself completed before persistence was cancelled. Preserve that
    # fact in memory without misclassifying cancellation as a sink failure.
    record = gateway.calls[-1]
    assert record.status == "SUCCESS"
    assert record.observability_error_codes == ()
    assert record.invocation is not None
    assert record.invocation.status == InvocationStatus.SUCCESS


async def test_concurrent_run_scoped_event_sinks_do_not_cross_write_or_clear_each_other():
    gateway = ToolGateway()
    entered = {"run-a": asyncio.Event(), "run-b": asyncio.Event()}
    release = {"run-a": asyncio.Event(), "run-b": asyncio.Event()}
    events: dict[str, list[tuple[str, dict]]] = {"run-a": [], "run-b": []}

    async def lookup(run_name: str):
        entered[run_name].set()
        await release[run_name].wait()
        return run_name

    async def execute_run(run_name: str):
        token = gateway.bind_event_sink(
            lambda event_type, payload: events[run_name].append((event_type, payload))
        )
        try:
            return await gateway.invoke(
                "analyst",
                "lookup",
                task_id=f"{run_name}:task",
                run_name=run_name,
            )
        finally:
            gateway.reset_event_sink(token)

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    run_a = asyncio.create_task(execute_run("run-a"))
    await entered["run-a"].wait()
    run_b = asyncio.create_task(execute_run("run-b"))
    await entered["run-b"].wait()

    release["run-a"].set()
    assert await run_a == "run-a"
    assert events["run-b"] == []

    # Run A has already reset its sink. Run B must retain its own context binding.
    release["run-b"].set()
    assert await run_b == "run-b"
    assert [payload["task_id"] for _event_type, payload in events["run-a"]] == ["run-a:task"]
    assert [payload["task_id"] for _event_type, payload in events["run-b"]] == ["run-b:task"]


async def test_supervisor_concurrent_runs_keep_tool_events_in_their_own_session():
    from creditlens.agents.supervisor import RunOutcome, Supervisor
    from creditlens.infrastructure.postgres.models import ReviewRun, RunEvent

    class RecordingSession:
        def __init__(self):
            self.added: list[object] = []

        async def scalar(self, _statement):
            return 0

        def add(self, value):
            self.added.append(value)

    gateway = ToolGateway()
    entered = {"run-a": asyncio.Event(), "run-b": asyncio.Event()}
    release = {"run-a": asyncio.Event(), "run-b": asyncio.Event()}

    async def lookup(run_name: str):
        entered[run_name].set()
        await release[run_name].wait()
        return run_name

    gateway.register("lookup", lookup)
    gateway.grant("analyst", ["lookup"])
    supervisor = Supervisor(
        policy_agent=object(),
        financial_agent=object(),
        challenger=object(),
        auditor=object(),
        tool_gateway=gateway,
    )

    async def execute_stub(self, _session, trusted, _snapshot, run, _seq):
        await gateway.invoke(
            "analyst",
            "lookup",
            task_id=f"{run.id}:task",
            run_name=trusted.run_name,
        )
        return RunOutcome(run.id, "COMPLETED")

    supervisor._execute_with_event_writer = types.MethodType(execute_stub, supervisor)
    tenant_id = uuid.uuid4()
    case_id = uuid.uuid4()

    def review_run():
        return ReviewRun(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            case_id=case_id,
            run_type="FULL_REVIEW",
            status="RECEIVED",
            as_of_date=date(2026, 8, 12),
        )

    run_a, run_b = review_run(), review_run()
    session_a, session_b = RecordingSession(), RecordingSession()
    task_a = asyncio.create_task(
        supervisor.execute_full_review(
            session_a,
            types.SimpleNamespace(run_name="run-a"),
            run=run_a,
        )
    )
    await entered["run-a"].wait()
    task_b = asyncio.create_task(
        supervisor.execute_full_review(
            session_b,
            types.SimpleNamespace(run_name="run-b"),
            run=run_b,
        )
    )
    await entered["run-b"].wait()

    release["run-a"].set()
    await task_a
    assert [event.run_id for event in session_a.added if isinstance(event, RunEvent)] == [run_a.id]
    assert [event for event in session_b.added if isinstance(event, RunEvent)] == []

    release["run-b"].set()
    await task_b
    assert [event.run_id for event in session_b.added if isinstance(event, RunEvent)] == [run_b.id]
