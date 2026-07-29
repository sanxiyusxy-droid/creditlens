"""Tool Gateway（文档 §11）：Agent 只能通过 Gateway 调用注册工具。

- 每个 Agent Role 有工具 Allowlist；越权调用记录 DENIED 并拒绝；
- 所有调用（含被拒绝的）都进入审计记录；
- Agent 不持有数据库/对象存储凭据。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from creditlens.common.clock import utc_now
from creditlens.common.errors import CreditLensError


class ToolCallDeniedError(CreditLensError):
    error_code = "TOOL_CALL_DENIED"


@dataclass
class ToolCallRecord:
    agent_role: str
    tool_name: str
    status: str  # SUCCESS|DENIED|FAILED
    error_code: str | None = None
    called_at: str = ""


@dataclass
class ToolGateway:
    _tools: dict[str, Callable[..., Awaitable[Any]]] = field(default_factory=dict)
    _allowlist: dict[str, set[str]] = field(default_factory=dict)
    calls: list[ToolCallRecord] = field(default_factory=list)
    max_calls_per_task: int = 8

    def register(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        self._tools[name] = fn

    def grant(self, agent_role: str, tool_names: list[str]) -> None:
        self._allowlist.setdefault(agent_role, set()).update(tool_names)

    async def invoke(self, agent_role: str, tool_name: str, /, **kwargs) -> Any:
        record = ToolCallRecord(
            agent_role=agent_role,
            tool_name=tool_name,
            status="SUCCESS",
            called_at=utc_now().isoformat(),
        )
        allowed = self._allowlist.get(agent_role, set())
        if tool_name not in allowed or tool_name not in self._tools:
            record.status = "DENIED"
            record.error_code = "TOOL_CALL_DENIED"
            self.calls.append(record)
            raise ToolCallDeniedError(
                f"agent role {agent_role} 无权调用工具 {tool_name}",
                {"agent_role": agent_role, "tool_name": tool_name},
            )
        used = sum(1 for c in self.calls if c.agent_role == agent_role and c.status == "SUCCESS")
        if used >= self.max_calls_per_task:
            record.status = "DENIED"
            record.error_code = "BUDGET_EXCEEDED"
            self.calls.append(record)
            raise ToolCallDeniedError("工具调用预算超限", {"agent_role": agent_role})
        try:
            result = await self._tools[tool_name](**kwargs)
            self.calls.append(record)
            return result
        except Exception:
            record.status = "FAILED"
            self.calls.append(record)
            raise
