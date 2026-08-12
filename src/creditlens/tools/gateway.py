"""Tool Gateway（文档 §11）：Agent 只能通过 Gateway 调用注册工具。

- 每个 Agent Role 有工具 Allowlist；越权调用记录 DENIED 并拒绝；
- 所有调用（含被拒绝的）都进入审计记录；
- Agent 不持有数据库/对象存储凭据。
"""

import asyncio
import contextvars
import inspect
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from creditlens.common.clock import utc_now
from creditlens.common.errors import CreditLensError
from creditlens.observability.invocation import (
    CANONICALIZATION_VERSION,
    FingerprintScheme,
    InvocationEnvelope,
    InvocationKind,
    InvocationStatus,
    InvocationTimeQuality,
    best_effort_hmac_fingerprint,
    invocation_event_type,
    invocation_run_event_payload,
    safe_error_code,
)

InvocationEventSink = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class ToolCallDeniedError(CreditLensError):
    error_code = "TOOL_CALL_DENIED"


class ToolBudgetExceededError(ToolCallDeniedError):
    error_code = "BUDGET_EXCEEDED"


@dataclass
class ToolCallRecord:
    agent_role: str
    tool_name: str
    task_id: str
    status: str  # IN_FLIGHT|SUCCESS|DENIED|FAILED|CANCELLED
    error_code: str | None = None
    called_at: str = ""
    invocation: InvocationEnvelope | None = None
    observability_error_codes: tuple[str, ...] = ()

    @property
    def invocation_id(self):
        return self.invocation.invocation_id if self.invocation is not None else None


@dataclass
class ToolGateway:
    _tools: dict[str, Callable[..., Awaitable[Any]]] = field(default_factory=dict)
    _allowlist: dict[str, set[str]] = field(default_factory=dict)
    _tool_providers: dict[str, str | None] = field(default_factory=dict)
    _tool_versions: dict[str, str | None] = field(default_factory=dict)
    _fingerprint_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    _event_sink: contextvars.ContextVar[InvocationEventSink | None] = field(
        default_factory=lambda: contextvars.ContextVar(
            f"creditlens_tool_event_sink_{uuid.uuid4().hex}",
            default=None,
        ),
        repr=False,
    )
    _budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    calls: list[ToolCallRecord] = field(default_factory=list)
    max_calls_per_task: int = 8

    def register(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        *,
        provider: str | None = "internal",
        version: str | None = None,
    ) -> None:
        self._tools[name] = fn
        self._tool_providers[name] = provider
        self._tool_versions[name] = version

    def grant(self, agent_role: str, tool_names: list[str]) -> None:
        self._allowlist.setdefault(agent_role, set()).update(tool_names)

    def bind_event_sink(self, sink: InvocationEventSink | None) -> contextvars.Token:
        """Bind a sink in the current task context and return its reset token."""

        return self._event_sink.set(sink)

    def reset_event_sink(self, token: contextvars.Token) -> None:
        """Restore the sink that preceded ``bind_event_sink`` in this context."""

        self._event_sink.reset(token)

    async def _emit_invocation_event(self, record: ToolCallRecord) -> None:
        sink = self._event_sink.get()
        if sink is None or record.invocation is None:
            return
        try:
            result = sink(
                invocation_event_type(record.invocation),
                invocation_run_event_payload(record.invocation),
            )
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            # Cancellation is control flow, not an observability failure. A
            # completed tool remains recorded SUCCESS, but the caller's task
            # must still observe cancellation while the sink is awaited.
            raise
        except Exception as error:
            code = safe_error_code(error, fallback="INVOCATION_EVENT_SINK_FAILED")
            record.observability_error_codes = tuple(
                dict.fromkeys((*record.observability_error_codes, code))
            )
            record.invocation = record.invocation.model_copy(
                update={"observability_error_codes": record.observability_error_codes}
            )

    async def invoke(
        self,
        agent_role: str,
        tool_name: str,
        /,
        *,
        task_id: str | uuid.UUID | None = None,
        **kwargs,
    ) -> Any:
        """Invoke a tool and record a content-free, best-effort envelope.

        ``task_id`` scopes the attempt budget. Calls that omit it retain the
        legacy per-role scope; new orchestrators should always provide it.
        Content fingerprints use an instance-secret HMAC. They support local
        correlation and integrity checks, but are not anonymization.
        """

        started_at = utc_now()
        started_clock = time.perf_counter()
        resolved_task_id = str(task_id) if task_id is not None else f"legacy-role:{agent_role}"
        request_sha256, request_available, request_error = best_effort_hmac_fingerprint(
            {"agent_role": agent_role, "tool_name": tool_name, "arguments": kwargs},
            secret=self._fingerprint_secret,
            domain="tool-request",
        )
        record = ToolCallRecord(
            agent_role=agent_role,
            tool_name=tool_name,
            task_id=resolved_task_id,
            status="IN_FLIGHT",
            called_at=started_at.isoformat(),
        )

        def finish(
            status: InvocationStatus,
            *,
            error_code: str | None = None,
            response: Any | None = None,
        ) -> None:
            record.status = status.value
            record.error_code = error_code
            observability_errors = [request_error] if request_error is not None else []
            try:
                latency_ms = max(0.0, (time.perf_counter() - started_clock) * 1000)
                observed_end = utc_now()
                time_quality = InvocationTimeQuality.OBSERVED
                if observed_end < started_at:
                    observed_end = started_at + timedelta(milliseconds=latency_ms)
                    time_quality = InvocationTimeQuality.CLOCK_ADJUSTED
                response_sha256: str | None = None
                response_available: bool | None = None
                if status == InvocationStatus.SUCCESS:
                    response_sha256, response_available, response_error = (
                        best_effort_hmac_fingerprint(
                            response,
                            secret=self._fingerprint_secret,
                            domain="tool-response",
                        )
                    )
                    if response_error is not None:
                        observability_errors.append(response_error)
                record.invocation = InvocationEnvelope(
                    kind=InvocationKind.TOOL,
                    name=tool_name,
                    provider=self._tool_providers.get(tool_name),
                    version=self._tool_versions.get(tool_name),
                    actor_role=agent_role,
                    task_id=resolved_task_id,
                    started_at=started_at,
                    ended_at=observed_end,
                    latency_ms=latency_ms,
                    time_quality=time_quality,
                    status=status,
                    error_code=error_code,
                    request_sha256=request_sha256,
                    response_sha256=response_sha256,
                    attempts=1,
                    fingerprint_scheme=FingerprintScheme.HMAC_SHA256_V1,
                    canonicalization_version=CANONICALIZATION_VERSION,
                    request_fingerprint_available=request_available,
                    response_fingerprint_available=response_available,
                    observability_error_codes=tuple(dict.fromkeys(observability_errors)),
                )
            except BaseException as instrumentation_error:
                observability_errors.append(
                    safe_error_code(
                        instrumentation_error,
                        fallback="INVOCATION_ENVELOPE_UNAVAILABLE",
                    )
                )
                record.observability_error_codes = tuple(dict.fromkeys(observability_errors))
            else:
                record.observability_error_codes = tuple(dict.fromkeys(observability_errors))

        # Reserve an attempt before the tool can start. The record itself is the
        # reservation, so in-flight, failed, cancelled and denied calls all count;
        # BUDGET_EXCEEDED observations do not. The lock closes the check/start race
        # for concurrent calls sharing one task budget.
        async with self._budget_lock:
            attempts = sum(
                1
                for call in self.calls
                if call.task_id == resolved_task_id and call.error_code != "BUDGET_EXCEEDED"
            )
            budget_exceeded = attempts >= self.max_calls_per_task
            if budget_exceeded:
                finish(InvocationStatus.DENIED, error_code="BUDGET_EXCEEDED")
            self.calls.append(record)

        if budget_exceeded:
            await self._emit_invocation_event(record)
            raise ToolBudgetExceededError(
                "工具调用预算超限",
                {"agent_role": agent_role, "task_id": resolved_task_id},
            )
        allowed = self._allowlist.get(agent_role, set())
        if tool_name not in allowed or tool_name not in self._tools:
            finish(InvocationStatus.DENIED, error_code="TOOL_CALL_DENIED")
            await self._emit_invocation_event(record)
            raise ToolCallDeniedError(
                f"agent role {agent_role} 无权调用工具 {tool_name}",
                {"agent_role": agent_role, "tool_name": tool_name},
            )
        try:
            result = await self._tools[tool_name](**kwargs)
        except asyncio.CancelledError:
            finish(InvocationStatus.CANCELLED, error_code="TOOL_CALL_CANCELLED")
            await self._emit_invocation_event(record)
            raise
        except Exception as error:
            finish(
                InvocationStatus.FAILED,
                error_code=safe_error_code(error, fallback="TOOL_EXECUTION_FAILED"),
            )
            await self._emit_invocation_event(record)
            raise
        finish(InvocationStatus.SUCCESS, response=result)
        await self._emit_invocation_event(record)
        return result
