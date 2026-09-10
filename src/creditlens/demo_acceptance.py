"""HTTP-contract acceptance flow with an explicit TCP or in-process scope."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from creditlens.agents.report_agent import DISCLAIMER, ReportContent

DEMO_CASE_ID = uuid.UUID("00000000-0000-0000-0000-000000000201")
_FROZEN_DEMO_CASE_IDENTITY = (
    ("case_number", "golden_case_001", "DEMO_CASE_NUMBER_MISMATCH"),
    ("product_code", "working_capital", "DEMO_CASE_PRODUCT_CODE_MISMATCH"),
    ("requested_amount", "5000000.00", "DEMO_CASE_REQUESTED_AMOUNT_MISMATCH"),
    ("currency", "CNY", "DEMO_CASE_CURRENCY_MISMATCH"),
    ("as_of_date", "2026-06-30", "DEMO_CASE_AS_OF_DATE_MISMATCH"),
    ("status", "DRAFT", "DEMO_CASE_STATUS_MISMATCH"),
)
_RUN_TERMINAL = {
    "COMPLETED",
    "FAILED",
    "DENIED",
    "HUMAN_REVIEW",
    "REWORK",
    "NEED_MORE_INFO",
    "DATA_QUALITY_BLOCKED",
    "SUPERSEDED",
}
_QA_STATUSES = {"ANSWERED", "NEEDS_REVIEW", "ABSTAINED"}
_HITL_ALLOWLIST_SCHEMA = "creditlens.http-hitl-allowlist.v1"
_HITL_ALLOWLIST_PROFILES = {"deterministic-offline", "configured-models"}


class AcceptanceFailure(RuntimeError):
    """Stable acceptance failure code without response bodies or provider details."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FrozenHitlAllowlist:
    """Human-reviewed fingerprints bound to one demo case and runtime profile."""

    schema_version: str
    case_id: str
    profile: str
    blocking_claim_fingerprints: frozenset[str]


def load_frozen_hitl_allowlist(
    path: str | Path,
    *,
    expected_case_id: uuid.UUID = DEMO_CASE_ID,
    expected_profile: str = "deterministic-offline",
) -> FrozenHitlAllowlist | None:
    """Load a frozen HITL allowlist without treating absence as approval.

    The file is created only after a human has inspected a real local run.  A
    missing file therefore returns ``None``; malformed or cross-profile files
    fail with stable codes instead of silently broadening the approval scope.
    """

    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceFailure("HITL_ALLOWLIST_READ_INVALID") from exc
    if not isinstance(payload, dict):
        raise AcceptanceFailure("HITL_ALLOWLIST_SHAPE_INVALID")
    if payload.get("schema_version") != _HITL_ALLOWLIST_SCHEMA:
        raise AcceptanceFailure("HITL_ALLOWLIST_SCHEMA_INVALID")

    case_id = _canonical_uuid(payload.get("case_id"), "HITL_ALLOWLIST_CASE_ID_INVALID")
    if case_id != str(expected_case_id):
        raise AcceptanceFailure("HITL_ALLOWLIST_CASE_MISMATCH")
    profile = str(payload.get("profile", ""))
    if profile not in _HITL_ALLOWLIST_PROFILES:
        raise AcceptanceFailure("HITL_ALLOWLIST_PROFILE_INVALID")
    if expected_profile not in _HITL_ALLOWLIST_PROFILES:
        raise AcceptanceFailure("HITL_EXPECTED_PROFILE_INVALID")
    if profile != expected_profile:
        raise AcceptanceFailure("HITL_ALLOWLIST_PROFILE_MISMATCH")

    values = payload.get("blocking_claim_fingerprints")
    if not isinstance(values, list) or not values:
        raise AcceptanceFailure("HITL_ALLOWLIST_EMPTY")
    normalized_values = [
        _canonical_sha256(value, "HITL_ALLOWLIST_FINGERPRINT_INVALID") for value in values
    ]
    if len(set(normalized_values)) != len(normalized_values):
        raise AcceptanceFailure("HITL_ALLOWLIST_FINGERPRINT_DUPLICATE")
    return FrozenHitlAllowlist(
        schema_version=_HITL_ALLOWLIST_SCHEMA,
        case_id=case_id,
        profile=profile,
        blocking_claim_fingerprints=frozenset(normalized_values),
    )


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    schema_version: str
    generated_at: str
    transport_scope: str
    passed: bool
    case_id: str
    qa_run_id: str
    qa_run_status: str
    qa_answer_status: str
    qa_candidate_count: int
    review_run_id: str
    review_initial_terminal_status: str
    human_review_exercised: bool
    approved_hitl_claim_fingerprints: tuple[str, ...]
    review_final_status: str
    report_status: str
    trace_integrity_status: str
    trace_delivery_status: str
    trace_event_count: int
    trace_invocation_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(response: httpx.Response, expected_status: int, error_code: str) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise AcceptanceFailure(error_code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise AcceptanceFailure("HTTP_RESPONSE_NOT_JSON") from exc
    if not isinstance(payload, dict):
        raise AcceptanceFailure("HTTP_RESPONSE_SHAPE_INVALID")
    return payload


def _canonical_uuid(value: Any, error_code: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise AcceptanceFailure(error_code) from None


def _canonical_sha256(value: Any, error_code: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise AcceptanceFailure(error_code)
    return normalized


def _validate_demo_case_identity(
    payload: dict[str, Any],
    *,
    expected_case_id: uuid.UUID,
) -> None:
    """Bind acceptance to the immutable root identity of golden_case_001."""

    if expected_case_id != DEMO_CASE_ID:
        raise AcceptanceFailure("DEMO_CASE_IDENTITY_NOT_FROZEN")
    if payload.get("case_id") != str(expected_case_id):
        raise AcceptanceFailure("DEMO_CASE_ID_MISMATCH")
    for field, expected, error_code in _FROZEN_DEMO_CASE_IDENTITY:
        if payload.get(field) != expected:
            raise AcceptanceFailure(error_code)


def _validate_document_locator(
    value: Any,
    *,
    hash_field: str,
    page_field: str,
    error_code: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceFailure(error_code)
    canonical = {
        "section_id": _canonical_uuid(value.get("section_id"), error_code),
        "document_version_id": _canonical_uuid(value.get("document_version_id"), error_code),
        "parse_run_id": _canonical_uuid(value.get("parse_run_id"), error_code),
        "content_hash": _canonical_sha256(value.get(hash_field), error_code),
    }
    page = value.get(page_field)
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise AcceptanceFailure(error_code)
    canonical["page_number"] = page
    return canonical


def _locator_key(locator: dict[str, Any]) -> tuple[str, str, str, str, int]:
    """Return the fields that bind one immutable document section version."""

    return (
        str(locator["section_id"]),
        str(locator["document_version_id"]),
        str(locator["parse_run_id"]),
        str(locator["content_hash"]),
        int(locator["page_number"]),
    )


def _register_locator(
    identities: dict[str, tuple[str, str, str, str, int]],
    locator: tuple[str, str, str, str, int],
    *,
    error_code: str,
) -> None:
    """Reject one Section ID being rebound to different immutable metadata."""

    existing = identities.get(locator[0])
    if existing is not None and existing != locator:
        raise AcceptanceFailure(error_code)
    identities[locator[0]] = locator


async def _wait_for_terminal(
    client: httpx.AsyncClient,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        payload = _json(
            await client.get(f"/api/v1/runs/{run_id}"),
            200,
            "RUN_STATUS_HTTP_FAILED",
        )
        status = str(payload.get("status", ""))
        if status in _RUN_TERMINAL:
            return payload
        await asyncio.sleep(poll_seconds)
    raise AcceptanceFailure("RUN_TERMINAL_TIMEOUT")


def _validate_grounded_answer(payload: dict[str, Any]) -> None:
    answer_status = str(payload.get("answer_status", ""))
    if answer_status not in _QA_STATUSES:
        raise AcceptanceFailure("QA_ANSWER_STATUS_INVALID")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AcceptanceFailure("QA_RETRIEVAL_EMPTY")
    locator_identities: dict[str, tuple[str, str, str, str, int]] = {}
    candidate_locators: set[tuple[str, str, str, str, int]] = set()
    for candidate in candidates:
        canonical_candidate = _validate_document_locator(
            candidate,
            hash_field="text_hash",
            page_field="page",
            error_code="QA_CANDIDATE_LOCATOR_INVALID",
        )
        candidate_key = _locator_key(canonical_candidate)
        _register_locator(
            locator_identities,
            candidate_key,
            error_code="QA_RETRIEVAL_LOCATOR_CONFLICT",
        )
        if candidate_key in candidate_locators:
            raise AcceptanceFailure("QA_CANDIDATE_LOCATOR_DUPLICATE")
        candidate_locators.add(candidate_key)
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            raise AcceptanceFailure("QA_CANDIDATE_TEXT_EMPTY")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise AcceptanceFailure("QA_CLAIMS_INVALID")
    cited_locators: set[tuple[str, str, str, str, int]] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise AcceptanceFailure("QA_CLAIM_SHAPE_INVALID")
        citations = claim.get("citations", [])
        opposing_citations = claim.get("opposing_citations", [])
        if not isinstance(citations, list) or not isinstance(opposing_citations, list):
            raise AcceptanceFailure("QA_CITATION_LOCATOR_INVALID")
        if claim.get("verdict") == "SUPPORTED" and not citations:
            raise AcceptanceFailure("QA_SUPPORTED_CLAIM_WITHOUT_CITATION")
        claim_citations: set[tuple[str, str, str, str, int]] = set()
        for citation_group in (citations, opposing_citations):
            for citation in citation_group:
                canonical_citation = _validate_document_locator(
                    citation,
                    hash_field="content_hash",
                    page_field="page_number",
                    error_code="QA_CITATION_LOCATOR_INVALID",
                )
                _canonical_uuid(citation.get("evidence_id"), "QA_CITATION_LOCATOR_INVALID")
                if not str(citation.get("evidence_type", "")).strip():
                    raise AcceptanceFailure("QA_CITATION_LOCATOR_INVALID")
                citation_key = _locator_key(canonical_citation)
                if citation_key in claim_citations:
                    raise AcceptanceFailure("QA_CITATION_LOCATOR_DUPLICATE")
                claim_citations.add(citation_key)
                cited_locators.add(citation_key)
    answer = payload.get("answer")
    refusal = payload.get("refusal_reason_code")
    abstention_reason = payload.get("abstention_reason")
    if answer_status == "ANSWERED":
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not claims
            or refusal is not None
            or abstention_reason is not None
        ):
            raise AcceptanceFailure("QA_ANSWER_STATE_INVALID")
    elif answer_status == "NEEDS_REVIEW":
        if (
            answer not in {"", None}
            or not claims
            or refusal is not None
            or abstention_reason is not None
        ):
            raise AcceptanceFailure("QA_REVIEW_STATE_INVALID")
    elif (
        answer not in {"", None}
        or claims
        or not refusal
        or not isinstance(abstention_reason, str)
        or not abstention_reason.strip()
    ):
        raise AcceptanceFailure("QA_ABSTENTION_STATE_INVALID")

    query_spec = payload.get("query_spec")
    if not isinstance(query_spec, dict):
        raise AcceptanceFailure("QA_QUERY_SPEC_MISSING")
    if not all(
        query_spec.get(field)
        for field in ("original_query", "standalone_query", "product_code", "as_of_date")
    ):
        raise AcceptanceFailure("QA_QUERY_SPEC_INVALID")
    variants = query_spec.get("query_variants")
    if not isinstance(variants, list) or not variants:
        raise AcceptanceFailure("QA_QUERY_VARIANTS_EMPTY")
    for item in variants:
        if (
            not isinstance(item, dict)
            or not str(item.get("variant_id", "")).strip()
            or not str(item.get("text", "")).strip()
            or str(item.get("route", "")).lower() not in {"dense", "sparse"}
        ):
            raise AcceptanceFailure("QA_QUERY_VARIANT_INVALID")
    variant_routes = {
        str(item.get("route", "")).upper() for item in variants if isinstance(item, dict)
    }
    if not {"DENSE", "SPARSE"} <= variant_routes:
        raise AcceptanceFailure("QA_ORIGINAL_ROUTES_MISSING")

    retrieval_trace = payload.get("retrieval_trace")
    if not isinstance(retrieval_trace, dict):
        raise AcceptanceFailure("QA_RETRIEVAL_TRACE_MISSING")
    route_rows = retrieval_trace.get("routes")
    if not isinstance(route_rows, list) or not route_rows:
        raise AcceptanceFailure("QA_ROUTE_TRACE_INVALID")
    for item in route_rows:
        if not isinstance(item, dict):
            raise AcceptanceFailure("QA_ROUTE_TRACE_INVALID")
        for field in ("candidates_count", "rejected_count"):
            count = item.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise AcceptanceFailure("QA_ROUTE_TRACE_INVALID")
        if not isinstance(item.get("rejection_reasons", {}), dict):
            raise AcceptanceFailure("QA_ROUTE_TRACE_INVALID")
    executed_routes = {
        str(item.get("route", "")).upper() for item in route_rows if isinstance(item, dict)
    }
    if not {"DENSE", "SPARSE", "SUMMARY", "EXACT"} <= executed_routes:
        raise AcceptanceFailure("QA_ROUTE_TRACE_INCOMPLETE")
    rrf_k = retrieval_trace.get("rrf_k")
    fusion = retrieval_trace.get("fusion")
    if (
        not isinstance(rrf_k, int)
        or isinstance(rrf_k, bool)
        or rrf_k <= 0
        or not isinstance(fusion, dict)
        or fusion.get("rrf_k") != rrf_k
        or not isinstance(fusion.get("input_lists"), list)
        or not fusion["input_lists"]
    ):
        raise AcceptanceFailure("QA_FUSION_TRACE_INVALID")
    input_lists = {str(value).upper() for value in fusion["input_lists"]}
    if (
        not any(value.startswith("DENSE:") for value in input_lists)
        or not any(value.startswith("SPARSE:") for value in input_lists)
        or "SUMMARY" not in input_lists
    ):
        raise AcceptanceFailure("QA_FUSION_INPUTS_INCOMPLETE")
    fused_count = fusion.get("fused_count")
    final_count = retrieval_trace.get("final_count")
    if (
        not isinstance(fused_count, int)
        or isinstance(fused_count, bool)
        or fused_count < len(candidates)
        or not isinstance(final_count, int)
        or isinstance(final_count, bool)
        or final_count != len(candidates)
    ):
        raise AcceptanceFailure("QA_FUSION_COUNTS_INVALID")
    rerank_applied = retrieval_trace.get("rerank_applied")
    rerank_degraded = retrieval_trace.get("rerank_degraded")
    if not isinstance(rerank_applied, bool) or not isinstance(rerank_degraded, bool):
        raise AcceptanceFailure("QA_RERANK_TRACE_INVALID")
    if rerank_applied and rerank_degraded:
        raise AcceptanceFailure("QA_RERANK_STATE_CONFLICT")
    if not rerank_applied or rerank_degraded:
        raise AcceptanceFailure("QA_RERANK_NOT_APPLIED")

    packing = payload.get("packing")
    if (
        not isinstance(packing, dict)
        or not isinstance(packing.get("sections"), list)
        or not packing["sections"]
    ):
        raise AcceptanceFailure("QA_PACKING_TRACE_INVALID")
    packed_locators: set[tuple[str, str, str, str, int]] = set()
    for section in packing["sections"]:
        canonical_section = _validate_document_locator(
            section,
            hash_field="text_hash",
            page_field="page_start",
            error_code="QA_PACKING_SECTION_INVALID",
        )
        packed_key = _locator_key(canonical_section)
        _register_locator(
            locator_identities,
            packed_key,
            error_code="QA_RETRIEVAL_LOCATOR_CONFLICT",
        )
        if packed_key in packed_locators:
            raise AcceptanceFailure("QA_PACKING_SECTION_DUPLICATE")
        packed_locators.add(packed_key)
        tokens = section.get("tokens_est") if isinstance(section, dict) else None
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise AcceptanceFailure("QA_PACKING_SECTION_INVALID")
    total_tokens = packing.get("total_tokens_est")
    budget = packing.get("budget")
    if (
        not isinstance(total_tokens, int)
        or not isinstance(budget, int)
        or total_tokens <= 0
        or budget <= 0
        or total_tokens > budget
    ):
        raise AcceptanceFailure("QA_PACKING_BUDGET_INVALID")
    # Packing is built from the complete fused pool and may contain candidates
    # beyond the final display limit, plus explicitly marked adjacent sections.
    # Therefore neither candidate list is a subset of the other.  The product
    # invariant is instead: shared Section IDs must have exactly the same
    # immutable locator, and every cited locator must be an exact member of the
    # context actually shown to the answer generator.
    if not cited_locators <= packed_locators:
        raise AcceptanceFailure("QA_CITATION_NOT_PACKED")


def _validate_trace(payload: dict[str, Any], *, allow_empty: bool) -> None:
    integrity = payload.get("integrity")
    delivery = payload.get("delivery")
    if not isinstance(integrity, dict) or not isinstance(delivery, dict):
        raise AcceptanceFailure("TRACE_SUMMARY_MISSING")
    integrity_status = integrity.get("status")
    delivery_status = delivery.get("status")
    allowed_integrity = {"VALID", "EMPTY"} if allow_empty else {"VALID"}
    if integrity_status not in allowed_integrity:
        raise AcceptanceFailure("TRACE_INTEGRITY_NOT_ACCEPTABLE")
    if integrity.get("invalid_count", 0) != 0:
        raise AcceptanceFailure("TRACE_INVALID_INVOCATIONS")
    allowed_delivery = {"COMPLETE", "EMPTY"} if allow_empty else {"COMPLETE"}
    if delivery_status not in allowed_delivery:
        raise AcceptanceFailure("TRACE_DELIVERY_NOT_ACCEPTABLE")
    counts = delivery.get("counts", {})
    expected_count_keys = {
        "PENDING",
        "PROCESSING",
        "DELIVERED",
        "DEAD",
        "MISSING",
        "INVALID",
    }
    if not isinstance(counts, dict) or set(counts) != expected_count_keys:
        raise AcceptanceFailure("TRACE_DELIVERY_COUNTS_INVALID")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise AcceptanceFailure("TRACE_DELIVERY_COUNTS_INVALID")
    invocations = payload.get("invocations")
    if not isinstance(invocations, list):
        raise AcceptanceFailure("TRACE_INVOCATIONS_INVALID")
    total = delivery.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total != len(invocations):
        raise AcceptanceFailure("TRACE_DELIVERY_COUNTS_INVALID")
    if sum(counts.values()) != total:
        raise AcceptanceFailure("TRACE_DELIVERY_COUNTS_INVALID")

    if integrity_status == "EMPTY" or delivery_status == "EMPTY":
        if (
            not allow_empty
            or integrity_status != "EMPTY"
            or delivery_status != "EMPTY"
            or integrity.get("valid") is not False
            or delivery.get("complete") is not False
            or invocations
            or any(counts.values())
        ):
            raise AcceptanceFailure("TRACE_EMPTY_STATE_INVALID")
    elif (
        integrity_status != "VALID"
        or delivery_status != "COMPLETE"
        or integrity.get("valid") is not True
        or delivery.get("complete") is not True
        or not invocations
        or counts["DELIVERED"] != len(invocations)
        or any(counts[key] for key in expected_count_keys - {"DELIVERED"})
    ):
        raise AcceptanceFailure("TRACE_COMPLETE_STATE_INVALID")

    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise AcceptanceFailure("TRACE_INVOCATION_SHAPE_INVALID")
        if (invocation.get("integrity") or {}).get("valid") is not True:
            raise AcceptanceFailure("TRACE_INVOCATION_INTEGRITY_INVALID")
        if (invocation.get("delivery") or {}).get("status") != "DELIVERED":
            raise AcceptanceFailure("TRACE_INVOCATION_DELIVERY_INVALID")
    if counts.get("INVALID", 0) or counts.get("MISSING", 0):
        raise AcceptanceFailure("TRACE_DELIVERY_GAP")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise AcceptanceFailure("TRACE_EVENTS_EMPTY")


async def _wait_for_trace_delivery(
    client: httpx.AsyncClient,
    run_id: str,
    *,
    allow_empty: bool,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Wait until every invocation outbox row reaches its terminal delivery state."""

    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        payload = _json(
            await client.get(f"/api/v1/runs/{run_id}/trace"),
            200,
            "TRACE_HTTP_FAILED",
        )
        delivery = payload.get("delivery")
        if not isinstance(delivery, dict):
            raise AcceptanceFailure("TRACE_SUMMARY_MISSING")
        integrity = payload.get("integrity")
        allowed_integrity = {"VALID", "EMPTY"} if allow_empty else {"VALID"}
        if not isinstance(integrity, dict) or integrity.get("status") not in allowed_integrity:
            _validate_trace(payload, allow_empty=allow_empty)
        if delivery.get("status") == "DEGRADED":
            _validate_trace(payload, allow_empty=allow_empty)
        if delivery.get("status") == "PENDING":
            await asyncio.sleep(poll_seconds)
            continue
        _validate_trace(payload, allow_empty=allow_empty)
        return payload
    raise AcceptanceFailure("TRACE_DELIVERY_TIMEOUT")


def _hitl_locator_rows(values: Any, *, binding: str) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise AcceptanceFailure("HITL_EVIDENCE_SHAPE_INVALID")
    rows: list[dict[str, Any]] = []
    for locator in values:
        if not isinstance(locator, dict) or not str(locator.get("evidence_type", "")).strip():
            raise AcceptanceFailure("HITL_EVIDENCE_SHAPE_INVALID")
        evidence_type = str(locator["evidence_type"]).strip()
        locator_ids = tuple(
            str(locator.get(field) or "")
            for field in ("section_id", "document_version_id", "parse_run_id")
        )
        if evidence_type == "DOCUMENT_SPAN" or any(locator_ids):
            if not all(locator_ids):
                raise AcceptanceFailure("HITL_EVIDENCE_SHAPE_INVALID")
            for value in locator_ids:
                _canonical_uuid(value, "HITL_EVIDENCE_SHAPE_INVALID")
        page_number = locator.get("page_number")
        if evidence_type == "DOCUMENT_SPAN" and (
            not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1
        ):
            raise AcceptanceFailure("HITL_EVIDENCE_SHAPE_INVALID")
        if page_number is not None and (
            not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1
        ):
            raise AcceptanceFailure("HITL_EVIDENCE_SHAPE_INVALID")
        content_hash = _canonical_sha256(locator.get("content_hash"), "HITL_EVIDENCE_HASH_INVALID")
        rows.append(
            {
                "binding": binding,
                "evidence_type": evidence_type,
                # Calculation traces bind database-local fact ids for replay.
                # Validate them, but bind cross-database approval through the
                # audited source statement instead of this local identifier hash.
                "content_hash": None if evidence_type == "CALCULATION" else content_hash,
                "page_number": page_number,
            }
        )
    return rows


def blocking_claim_fingerprint(
    claim: dict[str, Any],
    all_claims: list[dict[str, Any]] | None = None,
) -> str:
    """Hash the frozen, public shape of a known synthetic HITL conflict.

    Database-local run, claim, section, version and parse ids are validated but
    deliberately excluded from the hash. Calculation trace hashes are also only
    validated because they bind database-local fact ids; their formula/version/
    period/result remain bound by the audited source statement. Document hashes,
    evidence polarity/type and page remain bound, so changed source material fails.
    """

    if (
        claim.get("category") != "DATA_CONFLICT"
        or claim.get("verdict") != "PARTIALLY_SUPPORTED"
        or claim.get("review_status") != "PENDING"
        or not isinstance(claim.get("statement"), str)
        or not claim["statement"].strip()
    ):
        raise AcceptanceFailure("HITL_CLAIM_NOT_ALLOWLISTABLE")
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict) or not evidence.get("source_claim_id"):
        raise AcceptanceFailure("HITL_CLAIM_LINK_MISSING")
    if all_claims is None:
        raise AcceptanceFailure("HITL_SOURCE_CONTEXT_REQUIRED")
    conflict_claim_id = _canonical_uuid(claim.get("claim_id"), "HITL_CLAIM_ID_INVALID")
    source_claim_id = _canonical_uuid(
        evidence.get("source_claim_id"), "HITL_SOURCE_CLAIM_ID_INVALID"
    )
    if source_claim_id == conflict_claim_id:
        raise AcceptanceFailure("HITL_SOURCE_CLAIM_SELF_REFERENCE")
    sources = [
        item
        for item in all_claims
        if isinstance(item, dict) and str(item.get("claim_id")) == source_claim_id
    ]
    if len(sources) != 1:
        raise AcceptanceFailure("HITL_SOURCE_CLAIM_NOT_UNIQUE")
    source = sources[0]
    source_statement = source.get("statement")
    if not isinstance(source_statement, str) or not source_statement.strip():
        raise AcceptanceFailure("HITL_SOURCE_CLAIM_INVALID")
    source_evidence = source.get("evidence")
    if not isinstance(source_evidence, dict):
        raise AcceptanceFailure("HITL_SOURCE_EVIDENCE_INVALID")

    positive_rows = [
        *_hitl_locator_rows(
            source_evidence.get("supporting_locators"), binding="source_supporting"
        ),
        *_hitl_locator_rows(evidence.get("supporting_locators"), binding="conflict_supporting"),
    ]
    negative_rows = [
        *_hitl_locator_rows(source_evidence.get("opposing_locators"), binding="source_opposing"),
        *_hitl_locator_rows(evidence.get("opposing_locators"), binding="conflict_opposing"),
    ]
    if not positive_rows or not negative_rows:
        raise AcceptanceFailure("HITL_TWO_SIDED_EVIDENCE_REQUIRED")
    locator_rows = [*positive_rows, *negative_rows]
    canonical = {
        "conflict": {
            "category": claim["category"],
            "verdict": claim["verdict"],
            "review_status": claim["review_status"],
            "statement": claim["statement"].strip(),
        },
        "source": {
            "category": str(source.get("category", "")),
            "verdict": str(source.get("verdict", "")),
            "statement": source_statement.strip(),
        },
        "evidence": sorted(
            locator_rows,
            key=lambda item: (
                item["binding"],
                item["evidence_type"],
                item["content_hash"] or "",
                item["page_number"] if item["page_number"] is not None else -1,
            ),
        ),
    }
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _canonical_id_list(values: Any, error_code: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise AcceptanceFailure(error_code)
    normalized = tuple(_canonical_uuid(value, error_code) for value in values)
    if len(set(normalized)) != len(normalized):
        raise AcceptanceFailure(error_code)
    return normalized


def _canonical_json_rows(values: Any, error_code: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise AcceptanceFailure(error_code)
    rows: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            raise AcceptanceFailure(error_code)
        try:
            rows.append(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise AcceptanceFailure(error_code) from exc
    if len(set(rows)) != len(rows):
        raise AcceptanceFailure(error_code)
    return tuple(sorted(rows))


def _validate_report_locators(values: Any) -> tuple[str, ...]:
    """Validate ReportAgent's persisted locator projection without coercion."""

    expected_fields = {
        "evidence_type",
        "document_version_id",
        "section_id",
        "parse_run_id",
        "page_number",
        "content_hash",
    }
    if not isinstance(values, list):
        raise AcceptanceFailure("REPORT_CLAIM_LOCATORS_INVALID")
    for locator in values:
        if not isinstance(locator, dict) or set(locator) != expected_fields:
            raise AcceptanceFailure("REPORT_CLAIM_LOCATORS_INVALID")
        if not str(locator.get("evidence_type", "")).strip():
            raise AcceptanceFailure("REPORT_CLAIM_LOCATORS_INVALID")
        _canonical_sha256(locator.get("content_hash"), "REPORT_CLAIM_LOCATORS_INVALID")
        document_fields = (
            locator.get("document_version_id"),
            locator.get("section_id"),
            locator.get("parse_run_id"),
            locator.get("page_number"),
        )
        if any(value is not None for value in document_fields):
            _validate_document_locator(
                locator,
                hash_field="content_hash",
                page_field="page_number",
                error_code="REPORT_CLAIM_LOCATORS_INVALID",
            )
        elif locator.get("evidence_type") in {"DOCUMENT_SPAN", "TABLE_CELL"}:
            raise AcceptanceFailure("REPORT_CLAIM_LOCATORS_INVALID")
    return _canonical_json_rows(values, "REPORT_CLAIM_LOCATORS_INVALID")


def _validate_report_content(
    content: Any,
    *,
    review: dict[str, Any],
    review_run_id: str,
    case_payload: dict[str, Any],
    human_review_exercised: bool,
) -> ReportContent:
    """Bind the persisted report to its case-scoped Run and audited Claims."""

    if not isinstance(content, dict):
        raise AcceptanceFailure("REPORT_CONTENT_SCHEMA_INVALID")
    try:
        report = ReportContent.model_validate(content, strict=True)
    except ValidationError as exc:
        raise AcceptanceFailure("REPORT_CONTENT_SCHEMA_INVALID") from exc
    # ReportContent supplies defaults for ordinary in-process construction and
    # currently ignores extras.  An HTTP acceptance artifact must be the exact,
    # fully materialized projection; equality catches omitted defaults, ignored
    # extras and any nested coercion that would weaken the proof.
    if report.model_dump(mode="json") != content:
        raise AcceptanceFailure("REPORT_CONTENT_SCHEMA_INVALID")

    if review.get("run_id") != review_run_id or review.get("status") != "COMPLETED":
        raise AcceptanceFailure("REPORT_RUN_BINDING_INVALID")
    if report.run_id != review_run_id:
        raise AcceptanceFailure("REPORT_RUN_BINDING_INVALID")
    if report.as_of_date != case_payload.get("as_of_date"):
        raise AcceptanceFailure("REPORT_CASE_BINDING_INVALID")
    if report.disclaimer != DISCLAIMER:
        raise AcceptanceFailure("REPORT_DISCLAIMER_INVALID")

    raw_review_claims = review.get("claims")
    if not isinstance(raw_review_claims, list) or not raw_review_claims:
        raise AcceptanceFailure("REPORT_REVIEW_CLAIMS_INVALID")
    expected_claims: dict[str, dict[str, Any]] = {}
    expected_references: list[dict[str, Any]] = []
    human_approved_count = 0
    for raw_claim in raw_review_claims:
        if not isinstance(raw_claim, dict):
            raise AcceptanceFailure("REPORT_REVIEW_CLAIMS_INVALID")
        claim_id = _canonical_uuid(
            raw_claim.get("claim_id"),
            "REPORT_REVIEW_CLAIMS_INVALID",
        )
        if raw_claim.get("claim_id") != claim_id or claim_id in expected_claims:
            raise AcceptanceFailure("REPORT_REVIEW_CLAIMS_INVALID")
        review_status = raw_claim.get("review_status")
        if review_status not in {"AUDITED", "HUMAN_APPROVED"}:
            raise AcceptanceFailure("REPORT_REVIEW_STATE_INVALID")
        if review_status == "HUMAN_APPROVED":
            human_approved_count += 1
        for field in ("category", "statement", "verdict"):
            if not isinstance(raw_claim.get(field), str) or not raw_claim[field].strip():
                raise AcceptanceFailure("REPORT_REVIEW_CLAIMS_INVALID")
        evidence = raw_claim.get("evidence")
        if not isinstance(evidence, dict):
            raise AcceptanceFailure("REPORT_REVIEW_CLAIMS_INVALID")
        supporting_ids = _canonical_id_list(
            evidence.get("supporting_evidence_ids"),
            "REPORT_REVIEW_CLAIMS_INVALID",
        )
        opposing_ids = _canonical_id_list(
            evidence.get("opposing_evidence_ids"),
            "REPORT_REVIEW_CLAIMS_INVALID",
        )
        calculation_ids = _canonical_id_list(
            evidence.get("calculation_ids"),
            "REPORT_REVIEW_CLAIMS_INVALID",
        )
        supporting_locators = evidence.get("supporting_locators")
        opposing_locators = evidence.get("opposing_locators")
        supporting_locator_rows = _validate_report_locators(supporting_locators)
        opposing_locator_rows = _validate_report_locators(opposing_locators)
        if len(supporting_ids) != len(supporting_locator_rows) or len(opposing_ids) != len(
            opposing_locator_rows
        ):
            raise AcceptanceFailure("REPORT_CLAIM_EVIDENCE_BINDING_INVALID")
        source_claim_id = evidence.get("source_claim_id")
        if source_claim_id is not None:
            source_claim_id = _canonical_uuid(
                source_claim_id,
                "REPORT_REVIEW_CLAIMS_INVALID",
            )
        expected_claims[claim_id] = {
            "category": raw_claim["category"],
            "statement": raw_claim["statement"],
            "verdict": raw_claim["verdict"],
            "review_status": review_status,
            "evidence_refs": supporting_ids,
            "opposing_evidence_refs": opposing_ids,
            "calculation_ids": calculation_ids,
            "evidence_locators": supporting_locator_rows,
            "opposing_evidence_locators": opposing_locator_rows,
            "source_claim_id": source_claim_id,
        }
        expected_references.extend(
            {"claim_id": claim_id, "polarity": "SUPPORTING", **locator}
            for locator in supporting_locators
        )
        expected_references.extend(
            {"claim_id": claim_id, "polarity": "OPPOSING", **locator}
            for locator in opposing_locators
        )

    if human_review_exercised != bool(human_approved_count):
        raise AcceptanceFailure("REPORT_REVIEW_STATE_INVALID")
    if len(report.claims) != len(expected_claims) or report.excluded_claims != 0:
        raise AcceptanceFailure("REPORT_CLAIM_SET_INVALID")

    observed_claim_ids: set[str] = set()
    for claim in report.claims:
        claim_id = _canonical_uuid(claim.claim_id, "REPORT_CLAIM_BINDING_INVALID")
        if claim.claim_id != claim_id or claim_id in observed_claim_ids:
            raise AcceptanceFailure("REPORT_CLAIM_BINDING_INVALID")
        observed_claim_ids.add(claim_id)
        expected = expected_claims.get(claim_id)
        if expected is None:
            raise AcceptanceFailure("REPORT_CLAIM_SET_INVALID")
        observed_source_claim_id = claim.source_claim_id
        if observed_source_claim_id is not None:
            observed_source_claim_id = _canonical_uuid(
                observed_source_claim_id,
                "REPORT_CLAIM_BINDING_INVALID",
            )
        if (
            claim.category != expected["category"]
            or claim.statement != expected["statement"]
            or claim.verdict != expected["verdict"]
            or claim.review_status != expected["review_status"]
            or _canonical_id_list(claim.evidence_refs, "REPORT_CLAIM_BINDING_INVALID")
            != expected["evidence_refs"]
            or _canonical_id_list(
                claim.opposing_evidence_refs,
                "REPORT_CLAIM_BINDING_INVALID",
            )
            != expected["opposing_evidence_refs"]
            or _canonical_id_list(claim.calculation_ids, "REPORT_CLAIM_BINDING_INVALID")
            != expected["calculation_ids"]
            or _validate_report_locators(claim.evidence_locators) != expected["evidence_locators"]
            or _validate_report_locators(claim.opposing_evidence_locators)
            != expected["opposing_evidence_locators"]
            or observed_source_claim_id != expected["source_claim_id"]
        ):
            raise AcceptanceFailure("REPORT_CLAIM_BINDING_INVALID")
        if observed_source_claim_id is not None and observed_source_claim_id not in expected_claims:
            raise AcceptanceFailure("REPORT_CLAIM_BINDING_INVALID")
    if observed_claim_ids != set(expected_claims):
        raise AcceptanceFailure("REPORT_CLAIM_SET_INVALID")

    if _canonical_json_rows(
        report.references,
        "REPORT_REFERENCE_BINDING_INVALID",
    ) != _canonical_json_rows(
        expected_references,
        "REPORT_REFERENCE_BINDING_INVALID",
    ):
        raise AcceptanceFailure("REPORT_REFERENCE_BINDING_INVALID")

    execution = review.get("execution")
    if not isinstance(execution, dict):
        raise AcceptanceFailure("REPORT_EXECUTION_BINDING_INVALID")
    degraded = execution.get("degraded")
    degraded_agents = execution.get("degraded_agents")
    if (
        not isinstance(degraded, bool)
        or not isinstance(degraded_agents, list)
        or any(not isinstance(value, str) or not value for value in degraded_agents)
        or len(set(degraded_agents)) != len(degraded_agents)
        or report.degraded is not degraded
        or report.degraded_agents != degraded_agents
    ):
        raise AcceptanceFailure("REPORT_EXECUTION_BINDING_INVALID")
    if any(not isinstance(value, str) or not value.strip() for value in report.missing_materials):
        raise AcceptanceFailure("REPORT_CONTENT_SCHEMA_INVALID")
    return report


async def run_http_acceptance(
    base_url: str,
    *,
    case_id: uuid.UUID = DEMO_CASE_ID,
    timeout_seconds: float = 90.0,
    poll_seconds: float = 0.25,
    transport: httpx.AsyncBaseTransport | None = None,
    expected_hitl_claim_fingerprints: frozenset[str] | None = None,
    allow_non_loopback: bool = False,
) -> AcceptanceReport:
    """Exercise QA -> review -> optional HITL -> report -> trace over HTTP only."""

    parsed_base_url = urlparse(base_url)
    if (
        transport is None
        and not allow_non_loopback
        and parsed_base_url.hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise AcceptanceFailure("NON_LOOPBACK_BASE_URL_FORBIDDEN")
    if (
        transport is None
        and allow_non_loopback
        and parsed_base_url.hostname not in {"localhost", "127.0.0.1", "::1"}
        and parsed_base_url.scheme.lower() != "https"
    ):
        raise AcceptanceFailure("NON_LOOPBACK_HTTPS_REQUIRED")
    if timeout_seconds <= 0 or poll_seconds < 0:
        raise AcceptanceFailure("ACCEPTANCE_TIMEOUT_INVALID")

    timeout = httpx.Timeout(max(15.0, timeout_seconds), connect=min(15.0, timeout_seconds))
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        transport=transport,
        trust_env=False,
    ) as client:
        ready = _json(await client.get("/health/ready"), 200, "READINESS_HTTP_FAILED")
        if ready.get("status") != "ready":
            raise AcceptanceFailure("API_NOT_READY")

        case_payload = _json(
            await client.get(f"/api/v1/cases/{case_id}"),
            200,
            "DEMO_CASE_HTTP_FAILED",
        )
        _validate_demo_case_identity(case_payload, expected_case_id=case_id)

        qa_payload = _json(
            await client.post(
                f"/api/v1/cases/{case_id}/questions",
                json={
                    "idempotency_key": f"demo-http-qa-{uuid.uuid4()}",
                    "question": "截至案件审查时点，借款人是否满足流贷第六条的资产负债率要求？",
                    "top_k": 8,
                },
            ),
            200,
            "QA_HTTP_FAILED",
        )
        _validate_grounded_answer(qa_payload)
        qa_run_id = str(qa_payload.get("run_id", ""))
        try:
            uuid.UUID(qa_run_id)
        except ValueError:
            raise AcceptanceFailure("QA_RUN_ID_MISSING") from None
        qa_run = _json(
            await client.get(f"/api/v1/runs/{qa_run_id}"),
            200,
            "QA_RUN_HTTP_FAILED",
        )
        qa_run_status = str(qa_run.get("status", ""))
        if qa_run_status != "COMPLETED":
            raise AcceptanceFailure("QA_RUN_NOT_TERMINAL")
        await _wait_for_trace_delivery(
            client,
            qa_run_id,
            allow_empty=True,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )

        started = _json(
            await client.post(
                f"/api/v1/cases/{case_id}/runs",
                json={"run_type": "FULL_REVIEW"},
            ),
            202,
            "REVIEW_START_HTTP_FAILED",
        )
        review_run_id = str(started.get("run_id", ""))
        try:
            uuid.UUID(review_run_id)
        except ValueError:
            raise AcceptanceFailure("REVIEW_RUN_ID_MISSING") from None
        review = await _wait_for_terminal(
            client,
            review_run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        initial_status = str(review.get("status", ""))
        human_review_exercised = initial_status == "HUMAN_REVIEW"
        approved_hitl_claim_fingerprints: tuple[str, ...] = ()

        if human_review_exercised:
            claims = review.get("claims")
            if not isinstance(claims, list):
                raise AcceptanceFailure("REVIEW_CLAIMS_INVALID")
            blocking_claims = [
                claim
                for claim in claims
                if isinstance(claim, dict)
                and claim.get("review_status") in {"PENDING", "NEEDS_REWORK"}
            ]
            if not blocking_claims:
                raise AcceptanceFailure("HUMAN_REVIEW_WITHOUT_BLOCKING_CLAIMS")
            if expected_hitl_claim_fingerprints is None:
                raise AcceptanceFailure("HITL_EXPECTATION_REQUIRED")
            observed_fingerprints = tuple(
                sorted(blocking_claim_fingerprint(claim, claims) for claim in blocking_claims)
            )
            if (
                len(set(observed_fingerprints)) != len(observed_fingerprints)
                or frozenset(observed_fingerprints) != expected_hitl_claim_fingerprints
            ):
                raise AcceptanceFailure("HITL_CLAIM_SET_MISMATCH")
            blocking_ids = [str(claim.get("claim_id", "")) for claim in blocking_claims]
            if any(not value for value in blocking_ids):
                raise AcceptanceFailure("HITL_CLAIM_ID_MISSING")
            approved_hitl_claim_fingerprints = observed_fingerprints
            decision = _json(
                await client.post(
                    f"/api/v1/runs/{review_run_id}/review-decisions",
                    json={
                        "action": "APPROVE_CLAIM",
                        "target_claim_ids": blocking_ids,
                        "reason_code": "SYNTHETIC_DEMO_VERIFIED",
                        "reason": "仅用于合成演示：已核对支持证据与反证。",
                        "idempotency_key": f"demo-http-review-{uuid.uuid4()}",
                        "expected_state_version": review.get("state_version"),
                    },
                ),
                200,
                "HUMAN_DECISION_HTTP_FAILED",
            )
            if decision.get("status") not in {"HUMAN_REVIEW", "COMPLETED"}:
                raise AcceptanceFailure("HUMAN_DECISION_STATUS_INVALID")
            review = await _wait_for_terminal(
                client,
                review_run_id,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )

        final_status = str(review.get("status", ""))
        if final_status != "COMPLETED":
            raise AcceptanceFailure(f"REVIEW_FAIL_CLOSED_{final_status or 'UNKNOWN'}")

        report_payload = _json(
            await client.get(f"/api/v1/runs/{review_run_id}/report"),
            200,
            "REPORT_HTTP_FAILED",
        )
        content = report_payload.get("content")
        content_hash = str(report_payload.get("content_hash", ""))
        expected_report_status = "APPROVED_DRAFT" if human_review_exercised else "VERIFIED_DRAFT"
        if (
            report_payload.get("run_id") != review_run_id
            or report_payload.get("status") != expected_report_status
            or not isinstance(report_payload.get("version_no"), int)
            or report_payload["version_no"] < 1
            or len(content_hash) != 64
            or not isinstance(content, dict)
        ):
            raise AcceptanceFailure("REPORT_INTEGRITY_INVALID")
        try:
            int(content_hash, 16)
        except ValueError as exc:
            raise AcceptanceFailure("REPORT_INTEGRITY_INVALID") from exc
        expected_content_hash = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(content_hash, expected_content_hash):
            raise AcceptanceFailure("REPORT_INTEGRITY_INVALID")
        _validate_report_content(
            content,
            review=review,
            review_run_id=review_run_id,
            case_payload=case_payload,
            human_review_exercised=human_review_exercised,
        )

        trace = await _wait_for_trace_delivery(
            client,
            review_run_id,
            allow_empty=False,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        invocations = trace.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            raise AcceptanceFailure("REVIEW_INVOCATIONS_EMPTY")
        for invocation in invocations:
            if not isinstance(invocation, dict):
                raise AcceptanceFailure("REVIEW_INVOCATION_SHAPE_INVALID")
            if (invocation.get("integrity") or {}).get("valid") is not True:
                raise AcceptanceFailure("REVIEW_INVOCATION_INTEGRITY_INVALID")
            if (invocation.get("delivery") or {}).get("status") != "DELIVERED":
                raise AcceptanceFailure("REVIEW_INVOCATION_DELIVERY_INVALID")
        integrity = trace["integrity"]
        delivery = trace["delivery"]
        return AcceptanceReport(
            schema_version="creditlens.http-acceptance.v1",
            generated_at=datetime.now(UTC).isoformat(),
            transport_scope="TCP_HTTP" if transport is None else "IN_PROCESS_TEST_TRANSPORT",
            passed=True,
            case_id=str(case_id),
            qa_run_id=qa_run_id,
            qa_run_status=qa_run_status,
            qa_answer_status=str(qa_payload["answer_status"]),
            qa_candidate_count=len(qa_payload["candidates"]),
            review_run_id=review_run_id,
            review_initial_terminal_status=initial_status,
            human_review_exercised=human_review_exercised,
            approved_hitl_claim_fingerprints=approved_hitl_claim_fingerprints,
            review_final_status=final_status,
            report_status=str(report_payload.get("status", "")),
            trace_integrity_status=str(integrity.get("status", "")),
            trace_delivery_status=str(delivery.get("status", "")),
            trace_event_count=len(trace["events"]),
            trace_invocation_count=len(trace.get("invocations", [])),
        )
