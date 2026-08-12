"""Unified, persistence-safe envelope for model and tool invocations.

Only hashes, bounded identifiers, timing, usage and safe error codes belong in
this module. Raw prompts, model output, tool arguments and tool results are
deliberately absent from the public models and RunEvent adapter.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import math
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
CANONICALIZATION_VERSION = "invocation_typed_json_v1"
_MAX_CANONICAL_DEPTH = 64
_MAX_CANONICAL_NODES = 10_000
_MAX_SCALAR_BYTES = 1_000_000


class InvocationKind(StrEnum):
    MODEL = "MODEL"
    TOOL = "TOOL"


class InvocationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"
    CANCELLED = "CANCELLED"


class InvocationTimeQuality(StrEnum):
    OBSERVED = "OBSERVED"
    CLOCK_ADJUSTED = "CLOCK_ADJUSTED"
    ESTIMATED = "ESTIMATED"


class FingerprintScheme(StrEnum):
    LEGACY_SHA256 = "LEGACY_SHA256"
    HMAC_SHA256_V1 = "HMAC_SHA256_V1"


class PayloadCanonicalizationError(ValueError):
    """Safe, content-free failure raised for unsupported fingerprint inputs."""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(error_code)


class TokenUsage(BaseModel):
    """Provider-reported usage; nullable fields mean unknown, never zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelPrice(BaseModel):
    """One exact provider/model price in USD per one million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    input_per_million_usd: Decimal = Field(ge=0)
    output_per_million_usd: Decimal = Field(ge=0)


class PricingCatalog(BaseModel):
    """Explicitly versioned price table; there is intentionally no implicit latest price."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=64)
    entries: tuple[ModelPrice, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_keys(self) -> PricingCatalog:
        keys = [(item.provider.casefold(), item.model.casefold()) for item in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("pricing catalog contains duplicate provider/model entries")
        return self

    def find(self, provider: str | None, model: str | None) -> ModelPrice | None:
        if not provider or not model:
            return None
        key = (provider.casefold(), model.casefold())
        return next(
            (
                entry
                for entry in self.entries
                if (entry.provider.casefold(), entry.model.casefold()) == key
            ),
            None,
        )


class CostEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_usd: Decimal = Field(ge=0)
    pricing_version: str = Field(min_length=1, max_length=64)
    estimated: Literal[True] = True


class InvocationAggregate(BaseModel):
    """Aggregate comparable model invocations without inventing missing usage/cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_count: int = Field(ge=0)
    model_invocation_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_usage_complete: bool
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_complete: bool
    pricing_version: str | None = Field(default=None, max_length=64)


class InvocationEnvelope(BaseModel):
    """Unified redacted record emitted by model/tool adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: InvocationKind
    name: str = Field(min_length=1, max_length=128)
    provider: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=128)
    version: str | None = Field(default=None, max_length=128)
    actor_role: str | None = Field(default=None, max_length=64)
    task_id: str | None = Field(default=None, max_length=128)
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    time_quality: InvocationTimeQuality = InvocationTimeQuality.OBSERVED
    status: InvocationStatus
    error_code: str | None = Field(default=None, max_length=64)
    request_sha256: str
    response_sha256: str | None = None
    token_usage: TokenUsage | None = None
    attempts: int | None = Field(default=None, ge=1)
    prompt_sha256: str | None = None
    cost: CostEstimate | None = None
    fingerprint_scheme: FingerprintScheme = FingerprintScheme.LEGACY_SHA256
    canonicalization_version: str | None = Field(default=None, max_length=64)
    request_fingerprint_available: bool = True
    response_fingerprint_available: bool | None = None
    observability_error_codes: tuple[str, ...] = ()

    @field_validator("started_at", "ended_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("invocation timestamps must be timezone-aware")
        return value

    @field_validator("request_sha256", "response_sha256", "prompt_sha256")
    @classmethod
    def require_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value

    @field_validator("error_code")
    @classmethod
    def require_safe_error_code(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_CODE_RE.fullmatch(value):
            raise ValueError("error_code must be a bounded stable code")
        return value

    @field_validator("observability_error_codes")
    @classmethod
    def require_safe_observability_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _SAFE_CODE_RE.fullmatch(code) for code in value):
            raise ValueError("observability codes must be bounded stable codes")
        return value

    @field_validator("latency_ms")
    @classmethod
    def require_finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency_ms must be finite")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> InvocationEnvelope:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.status == InvocationStatus.SUCCESS and self.error_code is not None:
            raise ValueError("successful invocation cannot have an error_code")
        if self.status != InvocationStatus.SUCCESS and self.error_code is None:
            raise ValueError("non-successful invocation requires an error_code")
        if self.kind != InvocationKind.MODEL and (
            self.model is not None or self.token_usage is not None or self.cost is not None
        ):
            raise ValueError("model, token usage and cost are MODEL-only fields")
        if (
            self.fingerprint_scheme == FingerprintScheme.HMAC_SHA256_V1
            and self.canonicalization_version != CANONICALIZATION_VERSION
        ):
            raise ValueError("HMAC fingerprints require the current canonicalization version")
        if self.response_fingerprint_available is True and self.response_sha256 is None:
            raise ValueError("available response fingerprint requires a digest")
        return self


def estimate_model_cost(
    *,
    provider: str | None,
    model: str | None,
    usage: TokenUsage | None,
    pricing: PricingCatalog,
) -> CostEstimate | None:
    """Estimate cost only when exact versioned prices and both token classes exist."""

    price = pricing.find(provider, model)
    if price is None or usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    amount = (
        Decimal(usage.input_tokens) * price.input_per_million_usd
        + Decimal(usage.output_tokens) * price.output_per_million_usd
    ) / Decimal(1_000_000)
    return CostEstimate(amount_usd=amount, pricing_version=pricing.version)


def aggregate_invocations(envelopes: list[InvocationEnvelope]) -> InvocationAggregate:
    """Sum tokens/cost only when every model envelope provides comparable values."""

    model_envelopes = [item for item in envelopes if item.kind == InvocationKind.MODEL]
    has_models = bool(model_envelopes)
    usage_complete = all(item.token_usage is not None for item in model_envelopes)
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    token_values: dict[str, int | None] = {}
    for field_name in token_fields:
        values = [
            getattr(item.token_usage, field_name) if item.token_usage is not None else None
            for item in model_envelopes
        ]
        if any(value is None for value in values):
            token_values[field_name] = None
            usage_complete = False
        else:
            token_values[field_name] = sum(value for value in values if value is not None)

    costs = [item.cost for item in model_envelopes]
    versions = {cost.pricing_version for cost in costs if cost is not None}
    cost_complete = has_models and all(cost is not None for cost in costs)
    cost_complete = cost_complete and len(versions) == 1
    estimated_cost = (
        sum((cost.amount_usd for cost in costs if cost is not None), start=Decimal(0))
        if cost_complete
        else None
    )
    return InvocationAggregate(
        invocation_count=len(envelopes),
        model_invocation_count=len(model_envelopes),
        **token_values,
        token_usage_complete=usage_complete,
        estimated_cost_usd=estimated_cost,
        cost_complete=cost_complete,
        pricing_version=next(iter(versions)) if cost_complete else None,
    )


@dataclasses.dataclass
class _CanonicalState:
    seen: set[int] = dataclasses.field(default_factory=set)
    nodes: int = 0


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _bounded_text(value: str) -> str:
    if len(value.encode("utf-8")) > _MAX_SCALAR_BYTES:
        raise PayloadCanonicalizationError("FINGERPRINT_SCALAR_TOO_LARGE")
    return value


def _canonical_key(value: Any, *, state: _CanonicalState, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, str, bytes, Decimal, uuid.UUID, date, Enum)):
        return _canonical_value(value, state=state, depth=depth)
    if isinstance(value, float):
        return _canonical_value(value, state=state, depth=depth)
    raise PayloadCanonicalizationError("FINGERPRINT_UNSUPPORTED_KEY_TYPE")


def _canonical_value(value: Any, *, state: _CanonicalState, depth: int = 0) -> Any:
    """Return versioned, typed JSON data without silently collapsing values."""

    state.nodes += 1
    if depth > _MAX_CANONICAL_DEPTH:
        raise PayloadCanonicalizationError("FINGERPRINT_MAX_DEPTH_EXCEEDED")
    if state.nodes > _MAX_CANONICAL_NODES:
        raise PayloadCanonicalizationError("FINGERPRINT_MAX_NODES_EXCEEDED")

    if value is None:
        return ["null"]
    if isinstance(value, Enum):
        return ["enum", _type_name(value), value.name]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", _bounded_text(value)]
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
        return ["float64", encoded]
    if isinstance(value, Decimal):
        decimal_tuple = value.as_tuple()
        return [
            "decimal",
            decimal_tuple.sign,
            "".join(str(digit) for digit in decimal_tuple.digits),
            str(decimal_tuple.exponent),
        ]
    if isinstance(value, uuid.UUID):
        return ["uuid", str(value)]
    if isinstance(value, datetime):
        timezone_kind = (
            "aware" if value.tzinfo is not None and value.utcoffset() is not None else "naive"
        )
        normalized = value.astimezone(UTC) if timezone_kind == "aware" else value
        return ["datetime", timezone_kind, normalized.isoformat(timespec="microseconds")]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, bytes):
        if len(value) > _MAX_SCALAR_BYTES:
            raise PayloadCanonicalizationError("FINGERPRINT_SCALAR_TOO_LARGE")
        return ["bytes", base64.b64encode(value).decode("ascii")]

    identity = id(value)
    if identity in state.seen:
        raise PayloadCanonicalizationError("FINGERPRINT_RECURSIVE_VALUE")
    state.seen.add(identity)
    try:
        if isinstance(value, BaseModel):
            fields = {name: getattr(value, name) for name in type(value).model_fields}
            return [
                "pydantic",
                _type_name(value),
                _canonical_value(fields, state=state, depth=depth + 1),
            ]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
            return [
                "dataclass",
                _type_name(value),
                _canonical_value(fields, state=state, depth=depth + 1),
            ]
        if isinstance(value, Mapping):
            entries: list[list[Any]] = []
            canonical_keys: set[str] = set()
            for key, item in value.items():
                canonical_key = _canonical_key(key, state=state, depth=depth + 1)
                encoded_key = json.dumps(
                    canonical_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                if encoded_key in canonical_keys:
                    raise PayloadCanonicalizationError("FINGERPRINT_KEY_COLLISION")
                canonical_keys.add(encoded_key)
                entries.append(
                    [
                        canonical_key,
                        _canonical_value(item, state=state, depth=depth + 1),
                    ]
                )
            entries.sort(
                key=lambda entry: json.dumps(
                    entry[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            return ["mapping", entries]
        if isinstance(value, (list, tuple)):
            kind = "list" if isinstance(value, list) else "tuple"
            return [
                kind,
                [_canonical_value(item, state=state, depth=depth + 1) for item in value],
            ]
        if isinstance(value, (set, frozenset)):
            items = [_canonical_value(item, state=state, depth=depth + 1) for item in value]
            encoded_items = [
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for item in items
            ]
            if len(encoded_items) != len(set(encoded_items)):
                raise PayloadCanonicalizationError("FINGERPRINT_SET_COLLISION")
            kind = "frozenset" if isinstance(value, frozenset) else "set"
            return [kind, [item for _, item in sorted(zip(encoded_items, items, strict=True))]]
        raise PayloadCanonicalizationError("FINGERPRINT_UNSUPPORTED_TYPE")
    finally:
        state.seen.remove(identity)


def _canonical_payload(value: Any) -> bytes:
    canonical = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "payload": _canonical_value(value, state=_CanonicalState()),
    }
    return json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_invocation_payload(value: Any) -> str:
    """Return a deterministic fingerprint, not anonymization or secret protection.

    Callers must use :func:`hmac_invocation_payload` for sensitive tool content.
    """

    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def hmac_invocation_payload(value: Any, *, secret: bytes, domain: str) -> str:
    """Fingerprint sensitive content with a secret and explicit domain separation."""

    if not isinstance(secret, bytes) or len(secret) < 16:
        raise PayloadCanonicalizationError("FINGERPRINT_SECRET_INVALID")
    message = _canonical_payload({"domain": domain, "payload": value})
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def best_effort_hmac_fingerprint(
    value: Any, *, secret: bytes, domain: str
) -> tuple[str, bool, str | None]:
    """Always return a safe digest; failure metadata never contains payload content."""

    try:
        return hmac_invocation_payload(value, secret=secret, domain=domain), True, None
    except BaseException as error:  # instrumentation must not change tool behavior
        code = safe_error_code(error, fallback="FINGERPRINT_UNAVAILABLE")
        fallback_message = f"{CANONICALIZATION_VERSION}:{domain}:{code}".encode(
            "ascii", errors="ignore"
        )
        try:
            fallback = hmac.new(secret, fallback_message, hashlib.sha256).hexdigest()
        except BaseException:
            fallback = hashlib.sha256(fallback_message).hexdigest()
        return fallback, False, code


def safe_error_code(error: BaseException, *, fallback: str) -> str:
    """Return a stable code without retaining an exception message or traceback."""

    declared = getattr(error, "error_code", None)
    if isinstance(declared, str) and _SAFE_CODE_RE.fullmatch(declared):
        return declared
    return fallback


def _legacy_value(trace: object, field: str) -> Any:
    if isinstance(trace, Mapping):
        return trace.get(field)
    return getattr(trace, field, None)


def _legacy_error_code(value: Any) -> str | None:
    if value is None:
        return None
    raw = _CAMEL_BOUNDARY_RE.sub("_", str(value)).upper()
    normalized = re.sub(r"[^A-Z0-9_]", "_", raw).strip("_")
    if normalized and _SAFE_CODE_RE.fullmatch(normalized):
        return normalized
    return "MODEL_INVOCATION_FAILED"


def adapt_model_invocation_trace(
    trace: object,
    *,
    name: str = "structured_generation",
    ended_at: datetime | None = None,
    pricing: PricingCatalog | None = None,
) -> InvocationEnvelope:
    """Adapt the existing ModelInvocationTrace without importing its module."""

    latency_ms = float(_legacy_value(trace, "latency_ms"))
    ended = ended_at or datetime.now(UTC)
    started = ended - timedelta(milliseconds=latency_ms)
    usage = TokenUsage(
        input_tokens=_legacy_value(trace, "input_tokens"),
        output_tokens=_legacy_value(trace, "output_tokens"),
        total_tokens=_legacy_value(trace, "total_tokens"),
    )
    if all(value is None for value in usage.model_dump().values()):
        usage = None
    provider = _legacy_value(trace, "provider")
    model = _legacy_value(trace, "model")
    cost = (
        estimate_model_cost(provider=provider, model=model, usage=usage, pricing=pricing)
        if pricing is not None
        else None
    )
    status = InvocationStatus(str(_legacy_value(trace, "status")))
    error_code = _legacy_error_code(_legacy_value(trace, "error_type"))
    if status == InvocationStatus.SUCCESS:
        error_code = None
    elif error_code is None:
        error_code = "MODEL_INVOCATION_FAILED"
    return InvocationEnvelope(
        invocation_id=_legacy_value(trace, "invocation_id"),
        kind=InvocationKind.MODEL,
        name=name,
        provider=provider,
        model=model,
        version=_legacy_value(trace, "prompt_version"),
        started_at=started,
        ended_at=ended,
        latency_ms=latency_ms,
        time_quality=InvocationTimeQuality.ESTIMATED,
        status=status,
        error_code=error_code,
        request_sha256=_legacy_value(trace, "request_sha256"),
        response_sha256=_legacy_value(trace, "response_sha256"),
        token_usage=usage,
        attempts=_legacy_value(trace, "attempts"),
        prompt_sha256=_legacy_value(trace, "prompt_sha256"),
        cost=cost,
    )


def invocation_event_type(envelope: InvocationEnvelope) -> str:
    """Map an envelope to the existing RunEvent naming convention."""

    return f"{envelope.kind.value}_INVOCATION_{envelope.status.value}"


def invocation_run_event_payload(envelope: InvocationEnvelope) -> dict[str, Any]:
    """Build a JSON-safe RunEvent/API payload containing metadata only."""

    return envelope.model_dump(mode="json", exclude_none=True)
