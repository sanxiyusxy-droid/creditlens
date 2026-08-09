"""Fail-closed integrity checks for persisted Agent artifacts and claims.

``claims.review_status`` is workflow state and may legitimately change after an
artifact is written. Every other Claim field exposed by the application must
remain an exact projection of the immutable, append-only Artifact payload.
"""

from __future__ import annotations

import hmac
import json
import re
import uuid
from collections.abc import Iterable
from datetime import date
from typing import Any

from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.postgres.models import ArtifactRecord, ClaimRecord, ReviewRun

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GROUNDING_ARTIFACT_TYPE = "GROUNDED_ANSWER"
_REVIEW_STATUSES = {
    "PENDING",
    "AUDITED",
    "NEEDS_REWORK",
    "HUMAN_APPROVED",
    "HUMAN_REJECTED",
}


class ArtifactIntegrityError(RuntimeError):
    """Persisted Artifact/Claim data no longer matches its trusted envelope."""


def canonical_artifact_payload_hash(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 used by the persistence boundary."""
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _fail() -> None:
    # Deliberately exclude model-authored/persisted values from the exception.
    raise ArtifactIntegrityError("PERSISTED_ARTIFACT_INTEGRITY_FAILED")


def _uuid_text(value: Any) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError):
        _fail()


def _date_text(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        _fail()


def _string_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        _fail()
    return [_uuid_text(item) for item in value]


def _validate_record_envelope(
    *,
    run: ReviewRun,
    artifact: ArtifactRecord,
    require_output_hash: bool,
) -> list[dict[str, Any]]:
    payload = artifact.payload
    if not isinstance(payload, dict):
        _fail()

    output_hash = artifact.output_hash
    if require_output_hash or output_hash:
        if not isinstance(output_hash, str) or _SHA256_HEX.fullmatch(output_hash) is None:
            _fail()
        if not hmac.compare_digest(canonical_artifact_payload_hash(payload), output_hash):
            _fail()

    checks = (
        artifact.tenant_id == run.tenant_id,
        artifact.run_id == run.id,
        _uuid_text(payload.get("artifact_id")) == str(artifact.id),
        _uuid_text(payload.get("run_id")) == str(run.id),
        payload.get("task_id") == artifact.task_id,
        payload.get("producer") == artifact.producer,
        payload.get("contract_version") == artifact.contract_version,
        payload.get("lifecycle_status") == artifact.lifecycle_status,
        payload.get("execution_status") == artifact.execution_status,
        payload.get("input_hash", "") == artifact.input_hash,
        "output_hash" not in payload,
    )
    if not all(checks):
        _fail()

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not all(isinstance(item, dict) for item in raw_claims):
        _fail()
    return raw_claims


def _expected_claim_payload(
    artifact: ArtifactRecord,
    artifact_payload: dict[str, Any],
    source_claim: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    expected: dict[str, Any] = {
        "supporting_evidence_ids": _string_ids(source_claim.get("supporting_evidence_ids")),
        "opposing_evidence_ids": _string_ids(source_claim.get("opposing_evidence_ids")),
        "calculation_ids": _string_ids(source_claim.get("calculation_ids")),
    }
    allowed_keys = set(expected)
    if artifact.artifact_type == _GROUNDING_ARTIFACT_TYPE:
        expected["answer_status"] = artifact_payload.get("answer_status")
        allowed_keys.add("answer_status")
    else:
        raw_source_id = source_claim.get("source_claim_id")
        expected["source_claim_id"] = (
            _uuid_text(raw_source_id) if raw_source_id is not None else None
        )
        allowed_keys.add("source_claim_id")
    return expected, allowed_keys


def _validate_claim_projection(
    *,
    run: ReviewRun,
    artifact: ArtifactRecord,
    artifact_payload: dict[str, Any],
    source_claim: dict[str, Any],
    claim: ClaimRecord,
) -> None:
    payload = claim.payload
    if not isinstance(payload, dict):
        _fail()
    expected_payload, allowed_keys = _expected_claim_payload(
        artifact, artifact_payload, source_claim
    )
    if set(payload) - allowed_keys:
        _fail()

    # Missing empty-list/null fields are a semantically equivalent legacy
    # representation. Non-empty values and unknown additions remain strict.
    actual_payload = {
        key: payload.get(key, [] if key.endswith("_ids") else None) for key in allowed_keys
    }
    for key in (
        "supporting_evidence_ids",
        "opposing_evidence_ids",
        "calculation_ids",
    ):
        actual_payload[key] = _string_ids(actual_payload[key])

    checks = (
        claim.tenant_id == run.tenant_id,
        claim.run_id == run.id,
        claim.artifact_id == artifact.id,
        str(claim.id) == _uuid_text(source_claim.get("claim_id")),
        claim.category == source_claim.get("category"),
        claim.statement == source_claim.get("statement"),
        claim.verdict == source_claim.get("verdict"),
        claim.severity == source_claim.get("severity", "INFO"),
        claim.confidence_level == "MEDIUM",
        _date_text(claim.as_of_date) == _date_text(source_claim.get("as_of_date")),
        claim.uncertainty_reason == source_claim.get("uncertainty_reason"),
        claim.review_status in _REVIEW_STATUSES,
        actual_payload == expected_payload,
    )
    if not all(checks):
        _fail()


def validate_claim_records_against_artifacts(
    *,
    run: ReviewRun,
    artifacts: Iterable[ArtifactRecord],
    claims: Iterable[ClaimRecord],
    require_output_hash_for: frozenset[str] = frozenset({_GROUNDING_ARTIFACT_TYPE}),
) -> None:
    """Verify hashes/envelopes and exact Claim projections for one Run.

    Legacy non-grounding artifacts may have an empty ``output_hash``; their
    append-only payload still anchors Claim immutability. New persistence writes
    hashes for every artifact, while Grounded QA always requires one.
    """
    artifact_rows = list(artifacts)
    claim_rows = list(claims)
    artifacts_by_id: dict[uuid.UUID, ArtifactRecord] = {}
    source_claims_by_artifact: dict[uuid.UUID, dict[str, dict[str, Any]]] = {}

    for artifact in artifact_rows:
        if artifact.id in artifacts_by_id:
            _fail()
        source_claims = _validate_record_envelope(
            run=run,
            artifact=artifact,
            require_output_hash=artifact.artifact_type in require_output_hash_for,
        )
        source_by_id: dict[str, dict[str, Any]] = {}
        for item in source_claims:
            claim_id = _uuid_text(item.get("claim_id"))
            if claim_id in source_by_id:
                _fail()
            source_by_id[claim_id] = item
        artifacts_by_id[artifact.id] = artifact
        source_claims_by_artifact[artifact.id] = source_by_id

    persisted_ids_by_artifact: dict[uuid.UUID, set[str]] = {
        artifact_id: set() for artifact_id in artifacts_by_id
    }
    for claim in claim_rows:
        artifact = artifacts_by_id.get(claim.artifact_id)
        if artifact is None:
            _fail()
        claim_id = str(claim.id)
        if claim_id in persisted_ids_by_artifact[artifact.id]:
            _fail()
        source_claim = source_claims_by_artifact[artifact.id].get(claim_id)
        if source_claim is None:
            _fail()
        _validate_claim_projection(
            run=run,
            artifact=artifact,
            artifact_payload=artifact.payload,
            source_claim=source_claim,
            claim=claim,
        )
        persisted_ids_by_artifact[artifact.id].add(claim_id)

    if any(
        persisted_ids_by_artifact[artifact_id] != set(source_claims)
        for artifact_id, source_claims in source_claims_by_artifact.items()
    ):
        _fail()


def validate_grounded_qa_artifact_and_claims(
    *,
    run: ReviewRun,
    artifact: ArtifactRecord,
    claims: Iterable[ClaimRecord],
    expected_prompt_version: str,
) -> dict[str, Any]:
    """Verify the server-owned Grounded QA envelope and its Claim projections."""
    validate_claim_records_against_artifacts(
        run=run,
        artifacts=[artifact],
        claims=claims,
    )
    payload = artifact.payload
    manifest = run.model_manifest if isinstance(run.model_manifest, dict) else {}
    raw_payload_ids = payload.get("model_invocation_ids")
    raw_manifest_ids = manifest.get("model_invocation_ids")
    if not isinstance(raw_payload_ids, list) or not isinstance(raw_manifest_ids, list):
        _fail()
    payload_ids = {_uuid_text(item) for item in raw_payload_ids}
    manifest_ids = {_uuid_text(item) for item in raw_manifest_ids}
    answer_status = payload.get("answer_status")
    refusal_reason_code = payload.get("refusal_reason_code")
    direct_answer = payload.get("direct_answer")
    checks = (
        run.run_type == "SIMPLE_QA",
        artifact.artifact_type == _GROUNDING_ARTIFACT_TYPE,
        artifact.task_id == "grounded_qa",
        artifact.producer == "grounded_qa",
        artifact.lifecycle_status == "VERIFIED",
        payload.get("prompt_version") == expected_prompt_version,
        manifest.get("workflow") == "grounded_qa_v1",
        manifest.get("prompt_version") == expected_prompt_version,
        payload_ids.issubset(manifest_ids),
        payload.get("generation_mode")
        in {"llm", "deterministic_extractive", "abstained_empty_context"},
        answer_status in {"ANSWERED", "ABSTAINED", "NEEDS_REVIEW"},
        (answer_status == "ANSWERED") == (isinstance(direct_answer, str) and bool(direct_answer)),
        (answer_status == "ABSTAINED") == (refusal_reason_code is not None),
    )
    if not all(checks):
        _fail()
    return payload
