"""OpenAI 兼容 Chat Provider（DeepSeek 等，文档 §3.2 LLMProvider）。

- 规划/抽取/校验类调用 temperature=0，强制 JSON 结构化输出并用 Pydantic 校验；
- 校验失败自动重试一次，仍失败抛出（调用方走确定性降级，不得假成功）；
- 不记录/输出 API Key；请求脱敏由调用方负责。
"""

import hashlib
import json
import time
import uuid
from typing import ClassVar, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


class ModelInvocationTrace(BaseModel):
    """Persistence-safe metadata for one structured generation call."""

    model_config = ConfigDict(frozen=True)

    invocation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    request_sha256: str
    response_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=1)
    status: Literal["SUCCESS", "FAILED"]
    error_type: str | None = None


class TracedStructuredResult[TOutput: BaseModel](BaseModel):
    """Validated output paired with its redacted invocation trace."""

    output: TOutput
    trace: ModelInvocationTrace


class LLMCallError(Exception):
    """Structured generation failed; ``trace`` is safe to persist."""

    def __init__(self, message: str, trace: ModelInvocationTrace):
        super().__init__(message)
        self.trace = trace


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _UsageAccumulator:
    """Aggregate usage across retries without reporting partial totals."""

    _MAPPING: ClassVar[dict[str, str]] = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
    }

    def __init__(self) -> None:
        self._totals = {name: 0 for name in self._MAPPING}
        self._complete = {name: True for name in self._MAPPING}
        self._seen = False

    def add(self, response_payload: dict) -> None:
        self._seen = True
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            for name in self._complete:
                self._complete[name] = False
            return
        for name, provider_name in self._MAPPING.items():
            value = usage.get(provider_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                self._complete[name] = False
            elif self._complete[name]:
                self._totals[name] += value

    def values(self) -> dict[str, int | None]:
        return {
            name: self._totals[name] if self._seen and self._complete[name] else None
            for name in self._MAPPING
        }


def _build_trace(
    *,
    model: str,
    prompt_version: str,
    prompt_sha256: str,
    request_sha256: str,
    response_sha256: str | None,
    usage: _UsageAccumulator,
    started_at: float,
    attempts: int,
    status: Literal["SUCCESS", "FAILED"],
    error_type: str | None = None,
) -> ModelInvocationTrace:
    return ModelInvocationTrace(
        provider="openai_compatible",
        model=model,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        attempts=attempts,
        status=status,
        error_type=error_type,
        **usage.values(),
    )


class OpenAICompatChat:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.model = model

    async def generate_text(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024
    ) -> str:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def generate_structured(
        self,
        system: str,
        user: str,
        output_schema: type[TModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        prompt_version: str = "unversioned",
    ) -> TModel:
        """JSON 结构化输出 + Pydantic 校验；失败重试一次。"""
        result = await self.generate_structured_traced(
            system=system,
            user=user,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )
        return result.output

    async def generate_structured_traced(
        self,
        system: str,
        user: str,
        output_schema: type[TModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        prompt_version: str = "unversioned",
    ) -> TracedStructuredResult[TModel]:
        """Generate validated JSON plus a persistence-safe invocation trace.

        Schema validation keeps the historical two-attempt behavior. Network,
        protocol and HTTP failures are not retried here. Failures raise
        ``LLMCallError`` with a trace containing no raw request or response.
        """
        schema_hint = json.dumps(output_schema.model_json_schema(), ensure_ascii=False)
        system_full = (
            f"{system}\n\n你必须只输出一个 JSON 对象，符合以下 JSON Schema，"
            f"不得输出任何其他文本：\n{schema_hint}"
        )
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_full},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        prompt_hash = _sha256_text(system_full)
        request_hash = _sha256_text(_canonical_json(request_payload))
        response_hash: str | None = None
        usage = _UsageAccumulator()
        started_at = time.perf_counter()
        attempts = 0
        last_error: Exception | None = None
        for _attempt in range(2):
            attempts += 1
            try:
                response = await self._client.post("/chat/completions", json=request_payload)
                response_hash = _sha256_bytes(response.content)
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, dict):
                    raise TypeError("provider response must be a JSON object")
                usage.add(response_payload)
                content = response_payload["choices"][0]["message"]["content"]
                output = output_schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                continue
            except Exception as exc:
                trace = _build_trace(
                    model=self.model,
                    prompt_version=prompt_version,
                    prompt_sha256=prompt_hash,
                    request_sha256=request_hash,
                    response_sha256=response_hash,
                    usage=usage,
                    started_at=started_at,
                    attempts=attempts,
                    status="FAILED",
                    error_type=type(exc).__name__,
                )
                raise LLMCallError(f"结构化模型调用失败: {type(exc).__name__}", trace) from None
            trace = _build_trace(
                model=self.model,
                prompt_version=prompt_version,
                prompt_sha256=prompt_hash,
                request_sha256=request_hash,
                response_sha256=response_hash,
                usage=usage,
                started_at=started_at,
                attempts=attempts,
                status="SUCCESS",
            )
            return TracedStructuredResult(output=output, trace=trace)

        trace = _build_trace(
            model=self.model,
            prompt_version=prompt_version,
            prompt_sha256=prompt_hash,
            request_sha256=request_hash,
            response_sha256=response_hash,
            usage=usage,
            started_at=started_at,
            attempts=attempts,
            status="FAILED",
            error_type=type(last_error).__name__ if last_error is not None else "ValidationError",
        )
        # ValidationError may echo the invalid payload in its string form. Do
        # not retain it in the public exception or persistence-safe trace.
        raise LLMCallError("结构化输出两次校验失败", trace) from None

    async def aclose(self) -> None:
        """Close the underlying pooled HTTP client."""
        await self._client.aclose()


def build_chat_provider(settings) -> OpenAICompatChat | None:
    if settings.llm_provider == "disabled":
        return None
    if settings.llm_provider == "openai_compatible":
        if not (settings.llm_api_base and settings.llm_api_key and settings.llm_model):
            raise ValueError("openai_compatible llm 需要 LLM_API_BASE/KEY/MODEL")
        return OpenAICompatChat(
            base_url=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise NotImplementedError(f"llm provider {settings.llm_provider} 未配置")
