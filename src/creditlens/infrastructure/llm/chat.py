"""OpenAI 兼容 Chat Provider（DeepSeek 等，文档 §3.2 LLMProvider）。

- 规划/抽取/校验类调用 temperature=0，强制 JSON 结构化输出并用 Pydantic 校验；
- 校验失败自动重试一次，仍失败抛出（调用方走确定性降级，不得假成功）；
- 不记录/输出 API Key；请求脱敏由调用方负责。
"""

import hashlib
import json
import time
import uuid
from collections import Counter
from typing import ClassVar, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)

SchemaErrorCode = Literal[
    "ENUM_CONSTRAINT",
    "EXTRA_FIELD",
    "INVALID_JSON",
    "LIST_CONSTRAINT",
    "MISSING_FIELD",
    "OBJECT_CONSTRAINT",
    "STRING_CONSTRAINT",
    "TYPE_MISMATCH",
    "VALIDATION_OTHER",
    "VALUE_CONSTRAINT",
]

_SCHEMA_FINGERPRINT_VERSION = "schema_validation_v1"


class ModelInvocationTrace(BaseModel):
    """Persistence-safe metadata for one structured generation call.

    ``request_sha256`` hashes the ordered per-attempt request digest sequence,
    while ``response_sha256`` hashes the last HTTP response body received.
    ``attempts`` counts requests sent and token usage aggregates every parsed
    response. ``prompt_sha256`` remains the identity of the original structured
    prompt; a schema-repair prompt, when needed, is covered by the request hash.
    """

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
    schema_error_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    schema_error_counts: dict[SchemaErrorCode, int] = Field(default_factory=dict)


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


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _attempt_sequence_hash(attempt_digests: list[str]) -> str:
    """Hash an ordered sequence without retaining request/response content."""
    return _sha256_text(_canonical_json({"attempt_sha256": attempt_digests}))


def _schema_property_names(schema: object) -> set[str]:
    """Collect schema-owned field names; model-produced object keys are untrusted."""
    names: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for value in schema.values():
            names.update(_schema_property_names(value))
    elif isinstance(schema, list):
        for value in schema:
            names.update(_schema_property_names(value))
    return names


def _schema_error_code(error_type: str) -> SchemaErrorCode:
    """Map Pydantic's evolving error taxonomy into a stable small allowlist."""
    if error_type == "missing":
        return "MISSING_FIELD"
    if error_type.startswith("json_"):
        return "INVALID_JSON"
    if error_type == "extra_forbidden":
        return "EXTRA_FIELD"
    if error_type in {"literal_error", "enum"}:
        return "ENUM_CONSTRAINT"
    if error_type.startswith("string_") or error_type in {
        "too_short",
        "too_long",
    }:
        return "STRING_CONSTRAINT"
    if error_type.startswith(("list_", "tuple_", "set_")):
        return "LIST_CONSTRAINT"
    if error_type.startswith(("dict_", "mapping_", "model_")):
        return "OBJECT_CONSTRAINT"
    if error_type.endswith("_type"):
        return "TYPE_MISMATCH"
    if error_type.startswith(("value_", "greater_than", "less_than")) or error_type in {
        "assertion_error",
        "finite_number",
        "multiple_of",
    }:
        return "VALUE_CONSTRAINT"
    return "VALIDATION_OTHER"


def _safe_validation_summary(
    error: ValidationError,
    *,
    schema_property_names: set[str],
) -> tuple[str, dict[SchemaErrorCode, int]]:
    """Return bounded locations/types without Pydantic messages or input values."""
    summaries: list[dict[str, str]] = []
    counts: Counter[SchemaErrorCode] = Counter()
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:8]:
        location = "$"
        raw_location = item.get("loc", ())
        if isinstance(raw_location, (list, tuple)):
            for part in raw_location[:6]:
                if isinstance(part, str) and part in schema_property_names:
                    location += f".{part}"
                elif isinstance(part, int) and not isinstance(part, bool):
                    location += "[]"
                else:
                    location += ".<unknown>"

        raw_type = item.get("type")
        error_type = raw_type if isinstance(raw_type, str) else "validation_error"
        if (
            not error_type
            or len(error_type) > 64
            or any(
                not (character.isascii() and (character.isalnum() or character in "._-"))
                for character in error_type
            )
        ):
            error_type = "validation_error"
        summaries.append({"loc": location, "type": error_type})
        counts[_schema_error_code(error_type)] += 1

    summary = _canonical_json({"errors": summaries, "truncated": error.error_count() > 8})
    return summary, dict(sorted(counts.items()))


def _schema_error_fingerprint(summaries: list[str]) -> str | None:
    """Fingerprint only normalized loc/type summaries, never provider content."""
    if not summaries:
        return None
    return _sha256_text(
        _canonical_json(
            {
                "version": _SCHEMA_FINGERPRINT_VERSION,
                "attempt_summaries": summaries,
            }
        )
    )


def _validate_structured_response[TOutput: BaseModel](
    response_payload: dict,
    output_schema: type[TOutput],
    *,
    schema_property_names: set[str],
) -> tuple[TOutput | None, str | None, dict[SchemaErrorCode, int]]:
    """Validate provider content while containing any raw ValidationError input."""
    content = response_payload["choices"][0]["message"]["content"]
    try:
        return output_schema.model_validate_json(content), None, {}
    except ValidationError as error:
        summary, counts = _safe_validation_summary(
            error,
            schema_property_names=schema_property_names,
        )
        return None, summary, counts


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
    schema_error_fingerprint: str | None = None,
    schema_error_counts: dict[SchemaErrorCode, int] | None = None,
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
        schema_error_fingerprint=schema_error_fingerprint,
        schema_error_counts=schema_error_counts or {},
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
        schema_property_names = _schema_property_names(output_schema.model_json_schema())
        system_full = (
            f"{system}\n\n你必须只输出一个 JSON 对象，符合以下 JSON Schema，"
            f"不得输出任何其他文本：\n{schema_hint}"
        )
        prompt_hash = _sha256_text(system_full)
        request_digests: list[str] = []
        response_hash: str | None = None
        usage = _UsageAccumulator()
        started_at = time.perf_counter()
        attempts = 0
        repair_summary: str | None = None
        schema_error_summaries: list[str] = []
        schema_error_counts: Counter[SchemaErrorCode] = Counter()
        for attempt_index in range(2):
            repair_instruction = ""
            if repair_summary is not None:
                repair_instruction = (
                    "\n\n上一次输出未通过 JSON Schema 校验。不要复述或局部修补上一次"
                    "输出；请依据原始任务从头重新生成完整 JSON 对象。以下仅是脱敏后的"
                    f"字段位置/错误类型摘要，不含上一次输出原文：\n{repair_summary}"
                )
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_full + repair_instruction},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            request_digests.append(_sha256_text(_canonical_json(request_payload)))
            attempts += 1
            try:
                response = await self._client.post("/chat/completions", json=request_payload)
                response_hash = _sha256_bytes(response.content)
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, dict):
                    raise TypeError("provider response must be a JSON object")
                usage.add(response_payload)
                output, validation_summary, validation_counts = _validate_structured_response(
                    response_payload,
                    output_schema,
                    schema_property_names=schema_property_names,
                )
                if output is None:
                    # Do not carry provider content into the retry request or
                    # terminal exception frame. Only the safe summary survives.
                    repair_summary = validation_summary
                    if validation_summary is not None:
                        schema_error_summaries.append(validation_summary)
                    schema_error_counts.update(validation_counts)
                    response = None
                    response_payload = None
                    if attempt_index == 0:
                        continue
                    break
            except Exception as exc:
                trace = _build_trace(
                    model=self.model,
                    prompt_version=prompt_version,
                    prompt_sha256=prompt_hash,
                    request_sha256=_attempt_sequence_hash(request_digests),
                    response_sha256=response_hash,
                    usage=usage,
                    started_at=started_at,
                    attempts=attempts,
                    status="FAILED",
                    error_type=type(exc).__name__,
                    schema_error_fingerprint=_schema_error_fingerprint(schema_error_summaries),
                    schema_error_counts=dict(sorted(schema_error_counts.items())),
                )
                raise LLMCallError(f"结构化模型调用失败: {type(exc).__name__}", trace) from None
            trace = _build_trace(
                model=self.model,
                prompt_version=prompt_version,
                prompt_sha256=prompt_hash,
                request_sha256=_attempt_sequence_hash(request_digests),
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
            request_sha256=_attempt_sequence_hash(request_digests),
            response_sha256=response_hash,
            usage=usage,
            started_at=started_at,
            attempts=attempts,
            status="FAILED",
            error_type="ValidationError",
            schema_error_fingerprint=_schema_error_fingerprint(schema_error_summaries),
            schema_error_counts=dict(sorted(schema_error_counts.items())),
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
