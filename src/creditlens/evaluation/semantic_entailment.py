"""Offline human review protocol for claim--evidence semantic entailment.

The protocol is deliberately separate from deterministic citation-set metrics:

* phase 1 builds reviewer-specific blind packages from predictions and frozen
  evidence text;
* phase 2 validates independently submitted human labels and aggregates them;
* no model or model judge is invoked anywhere in this module.

Raw artifact SHA-256 values, stable section identifiers, exact evidence-content
hashes, and the randomized blind ordering are retained so a reported score can
be traced back to the bytes that a reviewer actually saw.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creditlens.evaluation.answer_metrics import (
    AnswerPrediction,
    AnswerPredictionProvenance,
    AnswerPredictionSet,
    PredictionStatus,
)

SEMANTIC_ENTAILMENT_PROTOCOL_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_PSEUDONYM_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22,128}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _non_blank(value: str, *, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_sha256(value: str, *, field_name: str) -> str:
    value = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _validate_reviewer_pseudonym(value: str, *, field_name: str = "reviewer_id") -> str:
    value = _non_blank(value, field_name=field_name)
    if not _REVIEWER_PSEUDONYM_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a high-entropy pseudonym of 22-128 URL-safe "
            "characters; do not use a real name"
        )
    return value


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def serialize_json_model(model: BaseModel) -> bytes:
    """Use one stable UTF-8 representation for artifact hashing and CLI output."""

    return (model.model_dump_json(indent=2) + "\n").encode("utf-8")


class EntailmentLabel(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class ReviewDecision(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    SKIP = "SKIP"
    CONFLICT = "CONFLICT"

    @property
    def semantic_label(self) -> EntailmentLabel | None:
        if self in {
            ReviewDecision.ENTAILED,
            ReviewDecision.CONTRADICTED,
            ReviewDecision.NOT_ENOUGH_INFO,
        }:
            return EntailmentLabel(self.value)
        return None


class AdjudicationDecision(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    EXCLUDE = "EXCLUDE"

    @property
    def semantic_label(self) -> EntailmentLabel | None:
        if self is AdjudicationDecision.EXCLUDE:
            return None
        return EntailmentLabel(self.value)


class SourceEvidence(_StrictModel):
    section_id: str
    content: str
    content_sha256: str

    @field_validator("section_id", "content")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="content_sha256")

    @model_validator(mode="after")
    def validate_exact_content_hash(self) -> Self:
        if _text_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 does not match the exact UTF-8 evidence content")
        return self


class SourceClaim(_StrictModel):
    question_id: str
    claim_id: str
    claim: str
    evidence: list[SourceEvidence] = Field(default_factory=list)

    @field_validator("question_id", "claim_id", "claim")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_unique_sections(self) -> Self:
        section_ids = [item.section_id for item in self.evidence]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("evidence section IDs must be unique within a claim")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.question_id, self.claim_id


class SourceInputArtifact(_StrictModel):
    """Raw, gold-free input whose exact bytes were used to build a source."""

    role: Literal["prediction_set", "raw_checkpoint", "evidence_checkpoint"]
    artifact_id: str
    sha256: str

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return _non_blank(value, field_name="artifact_id")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="sha256")


class SemanticEntailmentSource(_StrictModel):
    """Gold-free post-prediction input for blind human review.

    Only generated claims and the evidence shown for those claims belong here.
    The strict schema intentionally has no expected answer, gold label, or
    citation-gold fields.
    """

    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    source_id: str
    prediction_set_id: str
    prediction_sha256: str
    created_at: datetime
    input_artifacts: list[SourceInputArtifact] = Field(min_length=2)
    items: list[SourceClaim] = Field(min_length=1)

    @field_validator("source_id", "prediction_set_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("prediction_sha256")
    @classmethod
    def validate_prediction_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="prediction_sha256")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="created_at")

    @model_validator(mode="after")
    def validate_unique_claim_keys(self) -> Self:
        keys = [item.key for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("(question_id, claim_id) pairs must be unique")
        artifact_ids = [item.artifact_id for item in self.input_artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("source input artifact IDs must be unique")
        roles = [item.role for item in self.input_artifacts]
        if roles.count("prediction_set") != 1 or roles.count("raw_checkpoint") != 1:
            raise ValueError("source requires exactly one prediction_set and raw_checkpoint input")
        prediction_artifact = next(
            item for item in self.input_artifacts if item.role == "prediction_set"
        )
        if prediction_artifact.sha256 != self.prediction_sha256:
            raise ValueError("prediction_sha256 does not match prediction_set input bytes")
        return self


def _json_object(content: bytes, *, artifact_name: str) -> dict:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact_name} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object")
    return payload


def _required_text(payload: dict, field_name: str, *, context: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{context} requires string {field_name}")
    return _non_blank(value, field_name=f"{context}.{field_name}")


def _required_list(payload: dict, field_name: str, *, context: str) -> list:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{context} requires list {field_name}")
    return value


def _expected_answer_status(status: PredictionStatus) -> str | None:
    return {
        PredictionStatus.ANSWERED: "ANSWERED",
        PredictionStatus.REFUSED: "ABSTAINED",
        PredictionStatus.NEEDS_REVIEW: "NEEDS_REVIEW",
        PredictionStatus.TECHNICAL_FAILURE: None,
    }[status]


def _validate_checkpoint_provenance(
    prediction: AnswerPrediction,
    raw: dict,
    *,
    context: str,
) -> None:
    expected_status = _expected_answer_status(prediction.status)
    if expected_status is None:
        failure = raw.get("failure")
        if not isinstance(failure, dict):
            raise ValueError(f"{context} technical failure is missing its checkpoint failure")
        for field_name in ("status", "error_type"):
            if failure.get(field_name) != getattr(prediction, field_name):
                raise ValueError(f"{context} {field_name} differs from the prediction set")
        prediction_run_id = getattr(prediction.provenance, "run_id", None)
        raw_run_id = (failure.get("provenance") or {}).get("run_id")
        if str(prediction_run_id) != str(raw_run_id):
            raise ValueError(f"{context} failure run_id differs from the prediction set")
        return

    response = raw.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{context} business outcome is missing its checkpoint response")
    if response.get("answer_status") != expected_status:
        raise ValueError(f"{context} answer status differs from the prediction set")
    provenance = prediction.provenance
    if not isinstance(provenance, AnswerPredictionProvenance):
        raise ValueError(f"{context} is missing complete prediction provenance")
    comparisons = {
        "run_id": str(provenance.run_id),
        "snapshot_id": str(provenance.snapshot_id),
        "generation_mode": provenance.generation_mode,
        "model_invocation_ids": [str(item) for item in provenance.model_invocation_ids],
        "idempotent_replay": provenance.idempotent_replay,
    }
    for field_name, expected in comparisons.items():
        if response.get(field_name) != expected:
            raise ValueError(f"{context} {field_name} differs from the prediction set")
    if (
        prediction.status is PredictionStatus.ANSWERED
        and response.get("answer") != prediction.answer
    ):
        raise ValueError(f"{context} answer text differs from the prediction set")
    if (
        prediction.status is PredictionStatus.REFUSED
        and response.get("refusal_reason_code") != prediction.refusal_reason_code
    ):
        raise ValueError(f"{context} refusal reason differs from the prediction set")


def _candidate_catalog(checkpoints: list[tuple[str, bytes]]) -> dict[tuple[str, str], str]:
    catalog: dict[tuple[str, str], str] = {}
    for artifact_id, content in checkpoints:
        checkpoint = _json_object(content, artifact_name=artifact_id)
        if checkpoint.get("checkpoint_type") != "grounded_qa_raw_phase":
            raise ValueError(f"{artifact_id} has an unexpected checkpoint_type")
        if checkpoint.get("qa_phase_complete") is not True:
            raise ValueError(f"{artifact_id} must be a complete evidence checkpoint")
        results = _required_list(checkpoint, "results", context=artifact_id)
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"{artifact_id}.results entries must be objects")
            response = result.get("response")
            if not isinstance(response, dict):
                continue
            for candidate in _required_list(response, "candidates", context=artifact_id):
                if not isinstance(candidate, dict):
                    raise ValueError(f"{artifact_id} candidate entries must be objects")
                section_id = _required_text(candidate, "section_id", context=artifact_id)
                text_hash = _validate_sha256(
                    _required_text(candidate, "text_hash", context=artifact_id),
                    field_name=f"{artifact_id}.text_hash",
                )
                text = _required_text(candidate, "text", context=artifact_id)
                if _text_sha256(text) != text_hash:
                    raise ValueError(f"{artifact_id} candidate text_hash does not match exact text")
                key = section_id, text_hash
                previous = catalog.get(key)
                if previous is not None and previous != text:
                    raise ValueError(
                        f"conflicting candidate text for section/hash {section_id}/{text_hash}"
                    )
                catalog[key] = text
    return catalog


def build_source_from_grounded_qa_artifacts(
    *,
    prediction_bytes: bytes,
    raw_checkpoint_bytes: bytes,
    evidence_checkpoints: list[tuple[str, bytes]] | None = None,
    source_id: str,
    prediction_artifact_id: str = "prediction-set",
    raw_checkpoint_artifact_id: str = "raw-checkpoint",
    created_at: datetime | None = None,
) -> SemanticEntailmentSource:
    """Build a gold-free source from exact prediction/checkpoint bytes.

    Evidence checkpoints contribute candidate text only.  Claims, outcome
    status, and run provenance always come from the primary raw checkpoint and
    are checked one-for-one against the public prediction set.
    """

    source_id = _non_blank(source_id, field_name="source_id")
    prediction_artifact_id = _non_blank(prediction_artifact_id, field_name="prediction_artifact_id")
    raw_checkpoint_artifact_id = _non_blank(
        raw_checkpoint_artifact_id, field_name="raw_checkpoint_artifact_id"
    )
    evidence_checkpoints = evidence_checkpoints or []
    try:
        predictions = AnswerPredictionSet.model_validate_json(prediction_bytes)
    except ValueError as exc:
        raise ValueError("prediction artifact is not a valid AnswerPredictionSet") from exc
    checkpoint = _json_object(raw_checkpoint_bytes, artifact_name=raw_checkpoint_artifact_id)
    if checkpoint.get("checkpoint_type") != "grounded_qa_raw_phase":
        raise ValueError("raw checkpoint has an unexpected checkpoint_type")
    if checkpoint.get("qa_phase_complete") is not True:
        raise ValueError("raw checkpoint must be complete")
    raw_results = _required_list(checkpoint, "results", context=raw_checkpoint_artifact_id)
    if checkpoint.get("completed_questions") != len(raw_results):
        raise ValueError("raw checkpoint completed_questions does not match results")
    if checkpoint.get("selected_questions") != len(raw_results):
        raise ValueError("raw checkpoint selected_questions does not match results")

    raw_by_question: dict[str, dict] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ValueError("raw checkpoint result entries must be objects")
        question_id = _required_text(raw, "question_id", context="raw checkpoint result")
        if question_id in raw_by_question:
            raise ValueError("raw checkpoint question IDs must be unique")
        raw_by_question[question_id] = raw
    prediction_by_question = {item.question_id: item for item in predictions.predictions}
    if set(raw_by_question) != set(prediction_by_question):
        raise ValueError("raw checkpoint and prediction question IDs do not match")

    checkpoint_inputs = [
        (raw_checkpoint_artifact_id, raw_checkpoint_bytes),
        *evidence_checkpoints,
    ]
    catalog = _candidate_catalog(checkpoint_inputs)
    source_claims: list[SourceClaim] = []
    for prediction in predictions.predictions:
        raw = raw_by_question[prediction.question_id]
        context = f"question {prediction.question_id}"
        _validate_checkpoint_provenance(prediction, raw, context=context)
        response = raw.get("response")
        if not isinstance(response, dict):
            continue
        claims = _required_list(response, "claims", context=context)
        if prediction.status in {PredictionStatus.REFUSED, PredictionStatus.TECHNICAL_FAILURE}:
            if claims:
                raise ValueError(f"{context} non-answer outcome must not contain claims")
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(f"{context} claim entries must be objects")
            claim_id = _required_text(claim, "claim_id", context=context)
            statement = _required_text(claim, "statement", context=context)
            citations = _required_list(claim, "citations", context=context)
            evidence: list[SourceEvidence] = []
            seen_evidence_ids: set[str] = set()
            for citation in citations:
                if not isinstance(citation, dict):
                    raise ValueError(f"{context} citation entries must be objects")
                evidence_id = _required_text(citation, "evidence_id", context=context)
                if evidence_id in seen_evidence_ids:
                    raise ValueError(f"{context} duplicate supporting evidence_id")
                seen_evidence_ids.add(evidence_id)
                section_id = _required_text(citation, "section_id", context=context)
                content_hash = _validate_sha256(
                    _required_text(citation, "content_hash", context=context),
                    field_name=f"{context}.content_hash",
                )
                expected_evidence_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"evidence:{section_id}:{content_hash}")
                )
                if evidence_id != expected_evidence_id:
                    raise ValueError(f"{context} evidence_id does not match section/content hash")
                text = catalog.get((section_id, content_hash))
                if text is None:
                    raise ValueError(
                        f"{context} lacks frozen candidate text for {section_id}/{content_hash}"
                    )
                evidence.append(
                    SourceEvidence(
                        section_id=section_id,
                        content=text,
                        content_sha256=content_hash,
                    )
                )
            source_claims.append(
                SourceClaim(
                    question_id=prediction.question_id,
                    claim_id=claim_id,
                    claim=statement,
                    evidence=evidence,
                )
            )
    if not source_claims:
        raise ValueError("grounded QA artifacts contain no reviewable claims")

    created_at = created_at or datetime.now(UTC)
    inputs = [
        SourceInputArtifact(
            role="prediction_set",
            artifact_id=prediction_artifact_id,
            sha256=sha256_bytes(prediction_bytes),
        ),
        SourceInputArtifact(
            role="raw_checkpoint",
            artifact_id=raw_checkpoint_artifact_id,
            sha256=sha256_bytes(raw_checkpoint_bytes),
        ),
        *[
            SourceInputArtifact(
                role="evidence_checkpoint",
                artifact_id=artifact_id,
                sha256=sha256_bytes(content),
            )
            for artifact_id, content in evidence_checkpoints
        ],
    ]
    return SemanticEntailmentSource(
        source_id=source_id,
        prediction_set_id=predictions.prediction_set_id,
        prediction_sha256=sha256_bytes(prediction_bytes),
        created_at=created_at,
        input_artifacts=inputs,
        items=source_claims,
    )


class BlindEvidence(_StrictModel):
    ordinal: int = Field(ge=1)
    section_id: str
    content: str
    content_sha256: str

    @field_validator("section_id", "content")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="content_sha256")

    @model_validator(mode="after")
    def validate_exact_content_hash(self) -> Self:
        if _text_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 does not match the exact UTF-8 evidence content")
        return self


class BlindReviewItem(_StrictModel):
    ordinal: int = Field(ge=1)
    blind_item_id: str
    claim: str
    evidence: list[BlindEvidence] = Field(default_factory=list)

    @field_validator("blind_item_id", "claim")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_evidence_order(self) -> Self:
        if [item.ordinal for item in self.evidence] != list(range(1, len(self.evidence) + 1)):
            raise ValueError("blind evidence ordinals must be contiguous and ordered from 1")
        section_ids = [item.section_id for item in self.evidence]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("blind evidence section IDs must be unique")
        return self


def _blind_ordering_sha256(items: list[BlindReviewItem]) -> str:
    material = "\n".join(f"{item.ordinal}:{item.blind_item_id}" for item in items)
    return _text_sha256(material)


class BlindReviewPackage(_StrictModel):
    document_type: Literal["semantic_entailment_blind_review_package"] = (
        "semantic_entailment_blind_review_package"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    source_sha256: str
    prediction_sha256: str
    reviewer_id_sha256: str
    generated_at: datetime
    ordering_scheme: Literal["sha256-seeded-fisher-yates-v1"] = "sha256-seeded-fisher-yates-v1"
    blind_ordering_sha256: str
    items: list[BlindReviewItem] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator(
        "source_sha256", "prediction_sha256", "reviewer_id_sha256", "blind_ordering_sha256"
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_blind_order(self) -> Self:
        if [item.ordinal for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("blind item ordinals must be contiguous and ordered from 1")
        ids = [item.blind_item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("blind_item_id values must be unique")
        if _blind_ordering_sha256(self.items) != self.blind_ordering_sha256:
            raise ValueError("blind_ordering_sha256 does not match package item order")
        return self


class BlindMapEntry(_StrictModel):
    ordinal: int = Field(ge=1)
    blind_item_id: str
    question_id: str
    claim_id: str

    @field_validator("blind_item_id", "question_id", "claim_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @property
    def key(self) -> tuple[str, str]:
        return self.question_id, self.claim_id


class BlindReviewMapping(_StrictModel):
    document_type: Literal["semantic_entailment_blind_mapping"] = (
        "semantic_entailment_blind_mapping"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    source_sha256: str
    prediction_sha256: str
    reviewer_id_sha256: str
    generated_at: datetime
    blind_ordering_sha256: str
    entries: list[BlindMapEntry] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator(
        "source_sha256", "prediction_sha256", "reviewer_id_sha256", "blind_ordering_sha256"
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_mapping_order(self) -> Self:
        if [entry.ordinal for entry in self.entries] != list(range(1, len(self.entries) + 1)):
            raise ValueError("mapping ordinals must be contiguous and ordered from 1")
        blind_ids = [entry.blind_item_id for entry in self.entries]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("mapping blind_item_id values must be unique")
        keys = [entry.key for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("mapping claim keys must be unique")
        material = "\n".join(f"{entry.ordinal}:{entry.blind_item_id}" for entry in self.entries)
        if _text_sha256(material) != self.blind_ordering_sha256:
            raise ValueError("mapping order does not match blind_ordering_sha256")
        return self


class ReviewWorksheetItem(_StrictModel):
    """One editable blind row; ``decision`` intentionally starts empty."""

    ordinal: int = Field(ge=1)
    blind_item_id: str
    decision: ReviewDecision | None = None
    rationale: str = ""

    @field_validator("blind_item_id")
    @classmethod
    def validate_blind_item_id(cls, value: str) -> str:
        return _non_blank(value, field_name="blind_item_id")

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        return value.strip()


class ReviewWorksheet(_StrictModel):
    """Reviewer-editable artifact bound to exact blind-package bytes.

    It deliberately excludes raw reviewer identity, question/claim identity,
    gold labels, and all private Source/mapping fields.
    """

    document_type: Literal["semantic_entailment_review_worksheet"] = (
        "semantic_entailment_review_worksheet"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    package_sha256: str
    reviewer_id_sha256: str
    created_at: datetime
    items: list[ReviewWorksheetItem] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator("package_sha256", "reviewer_id_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="created_at")

    @model_validator(mode="after")
    def validate_item_order(self) -> Self:
        if [item.ordinal for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("worksheet ordinals must be contiguous and ordered from 1")
        blind_ids = [item.blind_item_id for item in self.items]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("worksheet blind_item_id values must be unique")
        return self


def build_review_worksheet(
    package: BlindReviewPackage,
    *,
    package_sha256: str,
    reviewer_id: str,
    created_at: datetime | None = None,
) -> ReviewWorksheet:
    """Create a blank worksheet without exposing private claim identities."""

    package_sha256 = _validate_sha256(package_sha256, field_name="package_sha256")
    reviewer_id = _validate_reviewer_pseudonym(reviewer_id)
    if _text_sha256(reviewer_id) != package.reviewer_id_sha256:
        raise ValueError("reviewer_id hash does not match the assigned blind package")
    created_at = created_at or datetime.now(UTC)
    _validate_aware(created_at, field_name="created_at")
    if created_at < package.generated_at:
        raise ValueError("review worksheet cannot predate its blind package")
    return ReviewWorksheet(
        package_id=package.package_id,
        package_sha256=package_sha256,
        reviewer_id_sha256=package.reviewer_id_sha256,
        created_at=created_at,
        items=[
            ReviewWorksheetItem(
                ordinal=item.ordinal,
                blind_item_id=item.blind_item_id,
            )
            for item in package.items
        ],
    )


class BlindReview(_StrictModel):
    blind_item_id: str
    decision: ReviewDecision
    rationale: str = ""

    @field_validator("blind_item_id")
    @classmethod
    def validate_blind_item_id(cls, value: str) -> str:
        return _non_blank(value, field_name="blind_item_id")

    @model_validator(mode="after")
    def validate_exception_rationale(self) -> Self:
        self.rationale = self.rationale.strip()
        if self.decision in {ReviewDecision.SKIP, ReviewDecision.CONFLICT} and not self.rationale:
            raise ValueError("SKIP and CONFLICT decisions require a rationale")
        return self


class ReviewerAttestationV1(_StrictModel):
    """Human reviewer self-declaration; this is not technical proof."""

    actor_type: Literal["HUMAN"]
    model_assistance_used: Literal[False]
    accessed_repo_or_gold: Literal[False]
    accessed_source_or_mapping: Literal[False]
    coordinated_with_other_reviewers: Literal[False]
    attested_at: datetime

    @field_validator(
        "model_assistance_used",
        "accessed_repo_or_gold",
        "accessed_source_or_mapping",
        "coordinated_with_other_reviewers",
        mode="before",
    )
    @classmethod
    def validate_explicit_false(cls, value: object, info) -> object:
        if value is not False:
            raise ValueError(f"{info.field_name} must be the boolean false")
        return value

    @field_validator("attested_at")
    @classmethod
    def validate_attested_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="attested_at")


class ReviewSubmission(_StrictModel):
    document_type: Literal["semantic_entailment_review_submission"] = (
        "semantic_entailment_review_submission"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    submission_id: str
    package_id: str
    package_sha256: str
    reviewer_id: str
    submitted_at: datetime
    reviewer_attestation: ReviewerAttestationV1
    reviews: list[BlindReview] = Field(default_factory=list)

    @field_validator("submission_id", "package_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, value: str) -> str:
        return _validate_reviewer_pseudonym(value)

    @field_validator("package_sha256")
    @classmethod
    def validate_package_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="package_sha256")

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="submitted_at")

    @model_validator(mode="after")
    def validate_unique_reviews(self) -> Self:
        ids = [item.blind_item_id for item in self.reviews]
        if len(set(ids)) != len(ids):
            raise ValueError("submission blind_item_id values must be unique")
        if self.reviewer_attestation.attested_at > self.submitted_at:
            raise ValueError("reviewer attestation cannot be later than submitted_at")
        return self


def compile_review_submission(
    package: BlindReviewPackage,
    worksheet: ReviewWorksheet,
    *,
    package_sha256: str,
    reviewer_id: str,
    reviewer_attestation: ReviewerAttestationV1,
    submitted_at: datetime | None = None,
    submission_id: str | None = None,
) -> ReviewSubmission:
    """Validate a filled worksheet and compile the strict submission artifact."""

    package_sha256 = _validate_sha256(package_sha256, field_name="package_sha256")
    reviewer_id = _validate_reviewer_pseudonym(reviewer_id)
    submitted_at = submitted_at or datetime.now(UTC)
    _validate_aware(submitted_at, field_name="submitted_at")

    if worksheet.package_id != package.package_id:
        raise ValueError("worksheet package_id does not match the blind package")
    if worksheet.package_sha256 != package_sha256:
        raise ValueError("worksheet package_sha256 does not match exact package bytes")
    if worksheet.reviewer_id_sha256 != package.reviewer_id_sha256:
        raise ValueError("worksheet reviewer_id hash differs from the blind package")
    if _text_sha256(reviewer_id) != package.reviewer_id_sha256:
        raise ValueError("reviewer_id hash does not match the assigned blind package")
    if worksheet.created_at < package.generated_at:
        raise ValueError("review worksheet cannot predate its blind package")
    if reviewer_attestation.attested_at < worksheet.created_at:
        raise ValueError("reviewer attestation cannot predate the completed worksheet")
    if reviewer_attestation.attested_at > submitted_at:
        raise ValueError("reviewer attestation cannot be later than submitted_at")

    package_by_id = {item.blind_item_id: item for item in package.items}
    worksheet_by_id = {item.blind_item_id: item for item in worksheet.items}
    unknown = set(worksheet_by_id) - set(package_by_id)
    if unknown:
        raise ValueError("worksheet contains an unknown blind_item_id")
    missing = set(package_by_id) - set(worksheet_by_id)
    if missing:
        raise ValueError("worksheet does not cover every package item exactly once")
    if len(worksheet.items) != len(package.items):
        raise ValueError("worksheet must contain every package item exactly once")
    for worksheet_item in worksheet.items:
        package_item = package_by_id[worksheet_item.blind_item_id]
        if worksheet_item.ordinal != package_item.ordinal:
            raise ValueError("worksheet ordinal does not match its blind package item")
        if worksheet_item.decision is None:
            raise ValueError("every worksheet decision must be non-null before compilation")
        if (
            worksheet_item.decision in {ReviewDecision.SKIP, ReviewDecision.CONFLICT}
            and not worksheet_item.rationale
        ):
            raise ValueError("SKIP and CONFLICT decisions require a rationale")

    reviews = [
        BlindReview(
            blind_item_id=item.blind_item_id,
            decision=item.decision,
            rationale=item.rationale,
        )
        for item in worksheet.items
    ]
    if submission_id is None:
        identity_material = json.dumps(
            {
                "package_sha256": package_sha256,
                "reviewer_id_sha256": package.reviewer_id_sha256,
                "submitted_at": submitted_at.astimezone(UTC).isoformat(),
                "reviewer_attestation": reviewer_attestation.model_dump(mode="json"),
                "reviews": [review.model_dump(mode="json") for review in reviews],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        submission_id = f"submission-{_text_sha256(identity_material)[:24]}"
    else:
        submission_id = _non_blank(submission_id, field_name="submission_id")

    return ReviewSubmission(
        submission_id=submission_id,
        package_id=package.package_id,
        package_sha256=package_sha256,
        reviewer_id=reviewer_id,
        submitted_at=submitted_at,
        reviewer_attestation=reviewer_attestation,
        reviews=reviews,
    )


class BlindAdjudicationEvidence(_StrictModel):
    """Evidence shown to an adjudicator without a stable source identifier."""

    ordinal: int = Field(ge=1)
    content: str
    content_sha256: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _non_blank(value, field_name="content")

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="content_sha256")

    @model_validator(mode="after")
    def validate_exact_content_hash(self) -> Self:
        if _text_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 does not match exact adjudication evidence bytes")
        return self


class BlindPriorDecision(_StrictModel):
    """One anonymous, independently shuffled reviewer decision."""

    ordinal: int = Field(ge=1)
    decision: ReviewDecision


class BlindAdjudicationItem(_StrictModel):
    ordinal: int = Field(ge=1)
    blind_dispute_id: str
    claim: str
    evidence: list[BlindAdjudicationEvidence] = Field(default_factory=list)
    prior_decisions: list[BlindPriorDecision] = Field(min_length=2, max_length=2)

    @field_validator("blind_dispute_id", "claim")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_nested_order(self) -> Self:
        if [item.ordinal for item in self.evidence] != list(range(1, len(self.evidence) + 1)):
            raise ValueError("adjudication evidence ordinals must be contiguous from 1")
        if [item.ordinal for item in self.prior_decisions] != [1, 2]:
            raise ValueError("prior decision ordinals must be exactly [1, 2]")
        return self


def _adjudication_ordering_sha256(items: list[BlindAdjudicationItem]) -> str:
    material = "\n".join(f"{item.ordinal}:{item.blind_dispute_id}" for item in items)
    return _text_sha256(material)


class BlindAdjudicationPackage(_StrictModel):
    """Public dispute-only package; deliberately contains no source/reviewer IDs."""

    document_type: Literal["semantic_entailment_blind_adjudication_package"] = (
        "semantic_entailment_blind_adjudication_package"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    adjudicator_id_sha256: str
    generated_at: datetime
    ordering_scheme: Literal["sha256-seeded-fisher-yates-v1"] = "sha256-seeded-fisher-yates-v1"
    blind_ordering_sha256: str
    items: list[BlindAdjudicationItem] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator("adjudicator_id_sha256", "blind_ordering_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_item_order(self) -> Self:
        if [item.ordinal for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("adjudication item ordinals must be contiguous from 1")
        blind_ids = [item.blind_dispute_id for item in self.items]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("blind_dispute_id values must be unique")
        if _adjudication_ordering_sha256(self.items) != self.blind_ordering_sha256:
            raise ValueError("blind_ordering_sha256 does not match adjudication item order")
        return self


class BlindAdjudicationMapEntry(_StrictModel):
    ordinal: int = Field(ge=1)
    blind_dispute_id: str
    question_id: str
    claim_id: str
    evidence_section_ids: list[str] = Field(default_factory=list)

    @field_validator("blind_dispute_id", "question_id", "claim_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("evidence_section_ids")
    @classmethod
    def validate_section_ids(cls, value: list[str]) -> list[str]:
        normalized = [
            _non_blank(section_id, field_name="evidence_section_ids") for section_id in value
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("adjudication mapping evidence section IDs must be unique")
        return normalized

    @property
    def key(self) -> tuple[str, str]:
        return self.question_id, self.claim_id


class BlindAdjudicationMapping(_StrictModel):
    """Private map binding a public dispute package to exact frozen inputs."""

    document_type: Literal["semantic_entailment_blind_adjudication_mapping"] = (
        "semantic_entailment_blind_adjudication_mapping"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    package_sha256: str
    source_sha256: str
    prediction_sha256: str
    adjudicator_id_sha256: str
    review_bundle_sha256: str
    generated_at: datetime
    blind_ordering_sha256: str
    entries: list[BlindAdjudicationMapEntry] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator(
        "package_sha256",
        "source_sha256",
        "prediction_sha256",
        "adjudicator_id_sha256",
        "review_bundle_sha256",
        "blind_ordering_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_mapping_order(self) -> Self:
        if [entry.ordinal for entry in self.entries] != list(range(1, len(self.entries) + 1)):
            raise ValueError("adjudication mapping ordinals must be contiguous from 1")
        blind_ids = [entry.blind_dispute_id for entry in self.entries]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("adjudication mapping blind_dispute_id values must be unique")
        keys = [entry.key for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("adjudication mapping claim keys must be unique")
        material = "\n".join(f"{entry.ordinal}:{entry.blind_dispute_id}" for entry in self.entries)
        if _text_sha256(material) != self.blind_ordering_sha256:
            raise ValueError("adjudication mapping order does not match blind ordering hash")
        return self


class AdjudicationWorksheetItem(_StrictModel):
    ordinal: int = Field(ge=1)
    blind_dispute_id: str
    decision: AdjudicationDecision | None = None
    rationale: str = ""

    @field_validator("blind_dispute_id")
    @classmethod
    def validate_blind_dispute_id(cls, value: str) -> str:
        return _non_blank(value, field_name="blind_dispute_id")

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        return value.strip()


class AdjudicationWorksheet(_StrictModel):
    document_type: Literal["semantic_entailment_adjudication_worksheet"] = (
        "semantic_entailment_adjudication_worksheet"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    package_id: str
    package_sha256: str
    adjudicator_id_sha256: str
    created_at: datetime
    items: list[AdjudicationWorksheetItem] = Field(min_length=1)

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        return _non_blank(value, field_name="package_id")

    @field_validator("package_sha256", "adjudicator_id_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="created_at")

    @model_validator(mode="after")
    def validate_item_order(self) -> Self:
        if [item.ordinal for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("adjudication worksheet ordinals must be contiguous from 1")
        blind_ids = [item.blind_dispute_id for item in self.items]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("adjudication worksheet blind_dispute_id values must be unique")
        return self


class AdjudicatorAttestationV1(_StrictModel):
    """Human adjudicator isolation self-declaration; not technical proof."""

    actor_type: Literal["HUMAN"]
    model_assistance_used: Literal[False]
    accessed_repository: Literal[False]
    accessed_gold: Literal[False]
    accessed_source_or_private_mapping: Literal[False]
    accessed_raw_submissions_or_reviewer_identity: Literal[False]
    attested_at: datetime

    @field_validator(
        "model_assistance_used",
        "accessed_repository",
        "accessed_gold",
        "accessed_source_or_private_mapping",
        "accessed_raw_submissions_or_reviewer_identity",
        mode="before",
    )
    @classmethod
    def validate_explicit_false(cls, value: object, info) -> object:
        if value is not False:
            raise ValueError(f"{info.field_name} must be the boolean false")
        return value

    @field_validator("attested_at")
    @classmethod
    def validate_attested_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="attested_at")


class AdjudicationSubmissionDecision(_StrictModel):
    blind_dispute_id: str
    decision: AdjudicationDecision
    rationale: str

    @field_validator("blind_dispute_id", "rationale")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)


class BlindAdjudicationSubmission(_StrictModel):
    document_type: Literal["semantic_entailment_blind_adjudication_submission"] = (
        "semantic_entailment_blind_adjudication_submission"
    )
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    submission_id: str
    package_id: str
    package_sha256: str
    adjudicator_id: str
    submitted_at: datetime
    adjudicator_attestation: AdjudicatorAttestationV1
    decisions: list[AdjudicationSubmissionDecision] = Field(min_length=1)

    @field_validator("submission_id", "package_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_blank(value, field_name=info.field_name)

    @field_validator("adjudicator_id")
    @classmethod
    def validate_adjudicator_id(cls, value: str) -> str:
        return _validate_reviewer_pseudonym(value, field_name="adjudicator_id")

    @field_validator("package_sha256")
    @classmethod
    def validate_package_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="package_sha256")

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="submitted_at")

    @model_validator(mode="after")
    def validate_submission(self) -> Self:
        blind_ids = [item.blind_dispute_id for item in self.decisions]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("adjudication submission blind_dispute_id values must be unique")
        if self.adjudicator_attestation.attested_at > self.submitted_at:
            raise ValueError("adjudicator attestation cannot be later than submitted_at")
        return self


def build_adjudication_worksheet(
    package: BlindAdjudicationPackage,
    *,
    package_sha256: str,
    adjudicator_id: str,
    created_at: datetime | None = None,
) -> AdjudicationWorksheet:
    """Create an editable worksheet from public dispute-package bytes only."""

    package_sha256 = _validate_sha256(package_sha256, field_name="package_sha256")
    adjudicator_id = _validate_reviewer_pseudonym(adjudicator_id, field_name="adjudicator_id")
    if _text_sha256(adjudicator_id) != package.adjudicator_id_sha256:
        raise ValueError("adjudicator_id hash does not match the assigned dispute package")
    created_at = created_at or datetime.now(UTC)
    _validate_aware(created_at, field_name="created_at")
    if created_at < package.generated_at:
        raise ValueError("adjudication worksheet cannot predate its dispute package")
    return AdjudicationWorksheet(
        package_id=package.package_id,
        package_sha256=package_sha256,
        adjudicator_id_sha256=package.adjudicator_id_sha256,
        created_at=created_at,
        items=[
            AdjudicationWorksheetItem(
                ordinal=item.ordinal,
                blind_dispute_id=item.blind_dispute_id,
            )
            for item in package.items
        ],
    )


def compile_adjudication_submission(
    package: BlindAdjudicationPackage,
    worksheet: AdjudicationWorksheet,
    *,
    package_sha256: str,
    adjudicator_id: str,
    adjudicator_attestation: AdjudicatorAttestationV1,
    submitted_at: datetime | None = None,
    submission_id: str | None = None,
) -> BlindAdjudicationSubmission:
    """Compile a complete blind worksheet into a strict human submission."""

    package_sha256 = _validate_sha256(package_sha256, field_name="package_sha256")
    adjudicator_id = _validate_reviewer_pseudonym(adjudicator_id, field_name="adjudicator_id")
    submitted_at = submitted_at or datetime.now(UTC)
    _validate_aware(submitted_at, field_name="submitted_at")

    if worksheet.package_id != package.package_id:
        raise ValueError("adjudication worksheet package_id does not match package")
    if worksheet.package_sha256 != package_sha256:
        raise ValueError("adjudication worksheet hash does not match exact package bytes")
    if worksheet.adjudicator_id_sha256 != package.adjudicator_id_sha256:
        raise ValueError("adjudication worksheet assignment differs from package")
    if _text_sha256(adjudicator_id) != package.adjudicator_id_sha256:
        raise ValueError("adjudicator_id hash does not match the assigned dispute package")
    if worksheet.created_at < package.generated_at:
        raise ValueError("adjudication worksheet cannot predate its dispute package")
    if worksheet.created_at > submitted_at:
        raise ValueError("adjudication worksheet cannot be later than submitted_at")
    if adjudicator_attestation.attested_at < worksheet.created_at:
        raise ValueError("adjudicator attestation cannot predate the completed worksheet")
    if adjudicator_attestation.attested_at > submitted_at:
        raise ValueError("adjudicator attestation cannot be later than submitted_at")

    package_by_id = {item.blind_dispute_id: item for item in package.items}
    worksheet_by_id = {item.blind_dispute_id: item for item in worksheet.items}
    if set(worksheet_by_id) != set(package_by_id) or len(worksheet.items) != len(package.items):
        raise ValueError("adjudication worksheet must cover every dispute exactly once")
    for item in worksheet.items:
        if item.ordinal != package_by_id[item.blind_dispute_id].ordinal:
            raise ValueError("adjudication worksheet ordinal differs from package")
        if item.decision is None:
            raise ValueError("every adjudication worksheet decision must be non-null")
        if not item.rationale:
            raise ValueError("every adjudication decision requires a rationale")

    decisions = [
        AdjudicationSubmissionDecision(
            blind_dispute_id=item.blind_dispute_id,
            decision=item.decision,
            rationale=item.rationale,
        )
        for item in worksheet.items
    ]
    if submission_id is None:
        identity_material = json.dumps(
            {
                "package_sha256": package_sha256,
                "adjudicator_id_sha256": package.adjudicator_id_sha256,
                "submitted_at": submitted_at.astimezone(UTC).isoformat(),
                "adjudicator_attestation": adjudicator_attestation.model_dump(mode="json"),
                "decisions": [item.model_dump(mode="json") for item in decisions],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        submission_id = f"adjudication-{_text_sha256(identity_material)[:24]}"
    else:
        submission_id = _non_blank(submission_id, field_name="submission_id")

    return BlindAdjudicationSubmission(
        submission_id=submission_id,
        package_id=package.package_id,
        package_sha256=package_sha256,
        adjudicator_id=adjudicator_id,
        submitted_at=submitted_at,
        adjudicator_attestation=adjudicator_attestation,
        decisions=decisions,
    )


class ArtifactHash(_StrictModel):
    artifact_id: str
    sha256: str

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return _non_blank(value, field_name="artifact_id")

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, field_name="sha256")


class ReviewerDecision(_StrictModel):
    reviewer_id: str
    decision: ReviewDecision


class ClaimAgreementStatus(StrEnum):
    NO_SEMANTIC_RATING = "NO_SEMANTIC_RATING"
    SINGLE_RATING = "SINGLE_RATING"
    CONSENSUS = "CONSENSUS"
    DISAGREEMENT = "DISAGREEMENT"
    CONFLICT_FLAGGED = "CONFLICT_FLAGGED"
    ADJUDICATION_REQUIRED = "ADJUDICATION_REQUIRED"
    ADJUDICATED = "ADJUDICATED"
    EXCLUDED = "EXCLUDED"


class ClaimEntailmentScore(_StrictModel):
    question_id: str
    claim_id: str
    assigned_reviewers: int
    received_reviews: int
    semantic_reviews: int
    skipped_reviews: int
    conflict_reviews: int
    entailed_reviews: int
    contradicted_reviews: int
    not_enough_info_reviews: int
    reviewer_decisions: list[ReviewerDecision]
    agreement_status: ClaimAgreementStatus
    consensus_label: EntailmentLabel | None
    final_label: EntailmentLabel | None
    adjudication_needed: bool
    adjudicated: bool
    excluded: bool


class LabelRates(_StrictModel):
    entailed: float
    contradicted: float
    not_enough_info: float


class LabelCounts(_StrictModel):
    entailed: int = Field(ge=0)
    contradicted: int = Field(ge=0)
    not_enough_info: int = Field(ge=0)


class AgreementMetrics(_StrictModel):
    reviewer_count: int
    pairwise_comparable_ratings: int
    pairwise_exact_agreements: int
    pairwise_percent_agreement: float | None
    cohen_kappa: float | None
    cohen_kappa_comparable_items: int
    cohen_kappa_reason: str | None


class EntailmentSummary(_StrictModel):
    total_claims: int
    assigned_ratings: int
    received_ratings: int
    semantic_ratings: int
    skipped_ratings: int
    conflict_ratings: int
    rating_coverage: float
    semantic_rating_coverage: float
    claims_with_semantic_rating: int
    claim_semantic_coverage: float
    micro_label_rates: LabelRates
    macro_label_rates: LabelRates
    final_labeled_claims: int
    final_label_counts: LabelCounts
    final_label_rates: LabelRates
    final_label_coverage: float
    consensus_claims: int
    resolved_claims: int
    unresolved_claims: int
    excluded_claims: int
    adjudication_needed_claims: int
    adjudicated_claims: int
    adjudication_coverage: float
    agreement: AgreementMetrics


class FormalCompletionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class FormalCompletionBlockerCode(StrEnum):
    REVIEW_PACKAGE_COUNT_NOT_2 = "REVIEW_PACKAGE_COUNT_NOT_2"
    REVIEWER_COUNT_NOT_2 = "REVIEWER_COUNT_NOT_2"
    RECEIVED_RATING_COUNT_MISMATCH = "RECEIVED_RATING_COUNT_MISMATCH"
    PER_CLAIM_REVIEW_COVERAGE_INCOMPLETE = "PER_CLAIM_REVIEW_COVERAGE_INCOMPLETE"
    ADJUDICATION_INCOMPLETE = "ADJUDICATION_INCOMPLETE"
    UNRESOLVED_CLAIMS = "UNRESOLVED_CLAIMS"


class FormalCompletionBlocker(_StrictModel):
    code: FormalCompletionBlockerCode
    message: str
    expected: int | None = None
    actual: int | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _non_blank(value, field_name="message")


class FormalCompletion(_StrictModel):
    status: FormalCompletionStatus
    expected_claims: int = Field(ge=1)
    expected_reviewers: Literal[2] = 2
    expected_ratings: int = Field(ge=2)
    actual_review_packages: int = Field(ge=0)
    actual_reviewers: int = Field(ge=0)
    actual_ratings: int = Field(ge=0)
    resolved_or_excluded_claims: int = Field(ge=0)
    required_adjudications: int = Field(ge=0)
    completed_adjudications: int = Field(ge=0)
    blockers: list[FormalCompletionBlocker] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_matches_blockers(self) -> Self:
        codes = [item.code for item in self.blockers]
        if len(set(codes)) != len(codes):
            raise ValueError("formal completion blocker codes must be unique")
        complete = not self.blockers
        if complete != (self.status is FormalCompletionStatus.COMPLETE):
            raise ValueError("formal completion status must agree with blockers")
        return self


class SemanticEntailmentReport(_StrictModel):
    report_type: Literal["manual_claim_evidence_semantic_entailment"] = (
        "manual_claim_evidence_semantic_entailment"
    )
    evaluation_scope: Literal["HUMAN_BLIND_CLAIM_EVIDENCE_SEMANTIC_ENTAILMENT"] = (
        "HUMAN_BLIND_CLAIM_EVIDENCE_SEMANTIC_ENTAILMENT"
    )
    model_judge_used: Literal[False] = False
    citation_set_used_as_faithfulness: Literal[False] = False
    protocol_version: Literal[SEMANTIC_ENTAILMENT_PROTOCOL_VERSION] = (
        SEMANTIC_ENTAILMENT_PROTOCOL_VERSION
    )
    generated_at: datetime
    source_id: str
    prediction_set_id: str
    prediction_sha256: str
    source_sha256: str
    package_artifacts: list[ArtifactHash]
    mapping_artifacts: list[ArtifactHash]
    submission_artifacts: list[ArtifactHash]
    adjudication_package_artifact: ArtifactHash | None
    adjudication_mapping_artifact: ArtifactHash | None
    adjudication_submission_artifact: ArtifactHash | None
    formal_completion: FormalCompletion
    summary: EntailmentSummary
    claims: list[ClaimEntailmentScore]

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_adjudication_artifact_set(self) -> Self:
        artifacts = (
            self.adjudication_package_artifact,
            self.adjudication_mapping_artifact,
            self.adjudication_submission_artifact,
        )
        if any(item is not None for item in artifacts) and not all(
            item is not None for item in artifacts
        ):
            raise ValueError("adjudication package, mapping, and submission hashes are all-or-none")
        return self


def _seeded_random(*parts: str) -> random.Random:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def build_blind_review_package(
    source: SemanticEntailmentSource,
    *,
    source_sha256: str,
    reviewer_id: str,
    ordering_seed: str,
    generated_at: datetime | None = None,
) -> tuple[BlindReviewPackage, BlindReviewMapping]:
    """Create a reviewer-specific package and a separately held identity map."""

    source_sha256 = _validate_sha256(source_sha256, field_name="source_sha256")
    reviewer_id = _validate_reviewer_pseudonym(reviewer_id)
    ordering_seed = _non_blank(ordering_seed, field_name="ordering_seed")
    generated_at = generated_at or datetime.now(UTC)
    _validate_aware(generated_at, field_name="generated_at")

    reviewer_hash = _text_sha256(reviewer_id)
    package_material = "\0".join(
        (source_sha256, source.prediction_sha256, reviewer_hash, ordering_seed)
    )
    package_id = f"sep-{_text_sha256(package_material)[:24]}"
    rng = _seeded_random(package_material)
    source_items = list(source.items)
    rng.shuffle(source_items)

    package_items: list[BlindReviewItem] = []
    map_entries: list[BlindMapEntry] = []
    for ordinal, source_item in enumerate(source_items, start=1):
        blind_id = f"item-{_text_sha256(f'{package_id}\0{source_item.question_id}\0{source_item.claim_id}')[:24]}"
        evidence = list(source_item.evidence)
        rng.shuffle(evidence)
        package_items.append(
            BlindReviewItem(
                ordinal=ordinal,
                blind_item_id=blind_id,
                claim=source_item.claim,
                evidence=[
                    BlindEvidence(
                        ordinal=evidence_ordinal,
                        section_id=item.section_id,
                        content=item.content,
                        content_sha256=item.content_sha256,
                    )
                    for evidence_ordinal, item in enumerate(evidence, start=1)
                ],
            )
        )
        map_entries.append(
            BlindMapEntry(
                ordinal=ordinal,
                blind_item_id=blind_id,
                question_id=source_item.question_id,
                claim_id=source_item.claim_id,
            )
        )

    ordering_hash = _blind_ordering_sha256(package_items)
    shared = {
        "package_id": package_id,
        "source_sha256": source_sha256,
        "prediction_sha256": source.prediction_sha256,
        "reviewer_id_sha256": reviewer_hash,
        "generated_at": generated_at,
        "blind_ordering_sha256": ordering_hash,
    }
    return (
        BlindReviewPackage(**shared, items=package_items),
        BlindReviewMapping(**shared, entries=map_entries),
    )


def _verify_package_mapping(
    source: SemanticEntailmentSource,
    *,
    source_sha256: str,
    package: BlindReviewPackage,
    mapping: BlindReviewMapping,
) -> None:
    if package.package_id != mapping.package_id:
        raise ValueError("package and mapping package_id values differ")
    shared_fields = (
        "source_sha256",
        "prediction_sha256",
        "reviewer_id_sha256",
        "blind_ordering_sha256",
    )
    for field_name in shared_fields:
        if getattr(package, field_name) != getattr(mapping, field_name):
            raise ValueError(f"package and mapping {field_name} values differ")
    if package.source_sha256 != source_sha256:
        raise ValueError("package source_sha256 does not match source bytes")
    if package.prediction_sha256 != source.prediction_sha256:
        raise ValueError("package prediction_sha256 does not match source provenance")

    source_by_key = {item.key: item for item in source.items}
    package_by_id = {item.blind_item_id: item for item in package.items}
    if set(package_by_id) != {entry.blind_item_id for entry in mapping.entries}:
        raise ValueError("package and mapping blind item sets differ")
    if {entry.key for entry in mapping.entries} != set(source_by_key):
        raise ValueError("mapping does not cover the source claims exactly once")

    for entry in mapping.entries:
        source_item = source_by_key.get(entry.key)
        package_item = package_by_id[entry.blind_item_id]
        if source_item is None:
            raise ValueError("mapping references an unknown source claim")
        if package_item.claim != source_item.claim:
            raise ValueError("blind package claim text drifted from its source")
        expected_evidence = sorted(
            (item.section_id, item.content_sha256, item.content) for item in source_item.evidence
        )
        actual_evidence = sorted(
            (item.section_id, item.content_sha256, item.content) for item in package_item.evidence
        )
        if actual_evidence != expected_evidence:
            raise ValueError("blind package evidence drifted from its source")


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _label_rates(counts: dict[EntailmentLabel, int], denominator: int) -> LabelRates:
    return LabelRates(
        entailed=_rate(counts[EntailmentLabel.ENTAILED], denominator),
        contradicted=_rate(counts[EntailmentLabel.CONTRADICTED], denominator),
        not_enough_info=_rate(counts[EntailmentLabel.NOT_ENOUGH_INFO], denominator),
    )


def _agreement_metrics(
    reviewer_labels: dict[str, dict[tuple[str, str], EntailmentLabel]],
) -> AgreementMetrics:
    reviewers = sorted(reviewer_labels)
    pairwise_comparable = 0
    pairwise_agreements = 0
    for first, second in combinations(reviewers, 2):
        comparable = set(reviewer_labels[first]) & set(reviewer_labels[second])
        pairwise_comparable += len(comparable)
        pairwise_agreements += sum(
            reviewer_labels[first][key] == reviewer_labels[second][key] for key in comparable
        )

    kappa: float | None = None
    comparable_count = 0
    reason: str | None = None
    if len(reviewers) != 2:
        reason = "Cohen kappa is reported only when exactly two reviewers submitted labels"
    else:
        first, second = reviewers
        comparable = sorted(set(reviewer_labels[first]) & set(reviewer_labels[second]))
        comparable_count = len(comparable)
        if not comparable:
            reason = "No items received a semantic label from both reviewers"
        else:
            observed = _rate(
                sum(
                    reviewer_labels[first][key] == reviewer_labels[second][key]
                    for key in comparable
                ),
                comparable_count,
            )
            expected = sum(
                _rate(
                    sum(reviewer_labels[first][key] is label for key in comparable),
                    comparable_count,
                )
                * _rate(
                    sum(reviewer_labels[second][key] is label for key in comparable),
                    comparable_count,
                )
                for label in EntailmentLabel
            )
            if expected == 1.0:
                reason = "Cohen kappa is undefined because expected agreement is 1"
            else:
                kappa = (observed - expected) / (1.0 - expected)

    return AgreementMetrics(
        reviewer_count=len(reviewers),
        pairwise_comparable_ratings=pairwise_comparable,
        pairwise_exact_agreements=pairwise_agreements,
        pairwise_percent_agreement=(
            _rate(pairwise_agreements, pairwise_comparable) if pairwise_comparable else None
        ),
        cohen_kappa=kappa,
        cohen_kappa_comparable_items=comparable_count,
        cohen_kappa_reason=reason,
    )


def _strict_two_review_consensus(
    decisions: list[ReviewerDecision],
) -> EntailmentLabel | None:
    """Return a label only for two matching, non-abstaining human reviews."""

    if len(decisions) != 2:
        return None
    labels = [item.decision.semantic_label for item in decisions]
    if labels[0] is None or labels[1] is None or labels[0] is not labels[1]:
        return None
    return labels[0]


def _review_bundle_sha256(
    packages: list[tuple[BlindReviewPackage, str]],
    mappings: list[tuple[BlindReviewMapping, str]],
    submissions: list[tuple[ReviewSubmission, str]],
) -> str:
    """Bind adjudication to the exact bytes of every preceding review artifact."""

    records: list[dict[str, str]] = []
    for role, artifacts in (
        ("package", packages),
        ("mapping", mappings),
        ("submission", submissions),
    ):
        for artifact, digest in artifacts:
            digest = _validate_sha256(digest, field_name=f"{role}_sha256")
            artifact_id = (
                artifact.submission_id
                if isinstance(artifact, ReviewSubmission)
                else artifact.package_id
            )
            records.append({"role": role, "artifact_id": artifact_id, "sha256": digest})
    material = json.dumps(
        sorted(records, key=lambda item: (item["role"], item["artifact_id"], item["sha256"])),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _text_sha256(material)


def _verify_blind_adjudication(
    source: SemanticEntailmentSource,
    *,
    source_sha256: str,
    review_bundle_sha256: str,
    decoded_reviews: dict[tuple[str, str], list[ReviewerDecision]],
    reviewer_ids: set[str],
    latest_review_submitted_at: datetime,
    required_keys: set[tuple[str, str]],
    package_artifact: tuple[BlindAdjudicationPackage, str],
    mapping_artifact: tuple[BlindAdjudicationMapping, str],
    submission_artifact: tuple[BlindAdjudicationSubmission, str],
) -> tuple[
    dict[tuple[str, str], AdjudicationSubmissionDecision],
    ArtifactHash,
    ArtifactHash,
    ArtifactHash,
]:
    package, package_digest = package_artifact
    mapping, mapping_digest = mapping_artifact
    submission, submission_digest = submission_artifact
    package_digest = _validate_sha256(package_digest, field_name="adjudication_package_sha256")
    mapping_digest = _validate_sha256(mapping_digest, field_name="adjudication_mapping_sha256")
    submission_digest = _validate_sha256(
        submission_digest, field_name="adjudication_submission_sha256"
    )

    if mapping.package_id != package.package_id or submission.package_id != package.package_id:
        raise ValueError("adjudication package, mapping, and submission IDs do not match")
    if mapping.package_sha256 != package_digest or submission.package_sha256 != package_digest:
        raise ValueError("adjudication artifacts do not bind to exact package bytes")
    if mapping.source_sha256 != source_sha256:
        raise ValueError("adjudication mapping source_sha256 does not match source bytes")
    if mapping.prediction_sha256 != source.prediction_sha256:
        raise ValueError("adjudication mapping prediction hash does not match source")
    if mapping.review_bundle_sha256 != review_bundle_sha256:
        raise ValueError("adjudication mapping is stale for the supplied review artifacts")
    if mapping.adjudicator_id_sha256 != package.adjudicator_id_sha256:
        raise ValueError("adjudication package and mapping assignments differ")
    if _text_sha256(submission.adjudicator_id) != package.adjudicator_id_sha256:
        raise ValueError("adjudication submission does not match the assigned package")
    if submission.adjudicator_id in reviewer_ids:
        raise ValueError("adjudicator pseudonym must differ from reviewer pseudonyms")
    if mapping.generated_at != package.generated_at:
        raise ValueError("adjudication package and mapping timestamps differ")
    if package.generated_at < latest_review_submitted_at:
        raise ValueError("adjudication package predates the supplied review submissions")
    if mapping.blind_ordering_sha256 != package.blind_ordering_sha256:
        raise ValueError("adjudication package and mapping ordering hashes differ")
    if submission.adjudicator_attestation.attested_at < package.generated_at:
        raise ValueError("adjudicator attestation predates the dispute package")
    if submission.submitted_at < package.generated_at:
        raise ValueError("adjudication submission predates the dispute package")

    package_by_id = {item.blind_dispute_id: item for item in package.items}
    mapping_by_id = {entry.blind_dispute_id: entry for entry in mapping.entries}
    decisions_by_id = {item.blind_dispute_id: item for item in submission.decisions}
    if set(package_by_id) != set(mapping_by_id):
        raise ValueError("adjudication package and private mapping dispute IDs differ")
    if set(decisions_by_id) != set(package_by_id):
        raise ValueError("adjudication submission must resolve every packaged dispute exactly once")
    if [item.blind_dispute_id for item in package.items] != [
        item.blind_dispute_id for item in mapping.entries
    ]:
        raise ValueError("adjudication package and mapping order differ")

    mapped_keys = {entry.key for entry in mapping.entries}
    if mapped_keys != required_keys:
        raise ValueError("adjudication mapping must contain exactly the current required disputes")
    source_by_key = {item.key: item for item in source.items}
    adjudication_by_key: dict[tuple[str, str], AdjudicationSubmissionDecision] = {}
    for mapping_entry in mapping.entries:
        source_item = source_by_key.get(mapping_entry.key)
        if source_item is None:
            raise ValueError("adjudication mapping references an unknown source claim")
        package_item = package_by_id[mapping_entry.blind_dispute_id]
        if package_item.ordinal != mapping_entry.ordinal:
            raise ValueError("adjudication package and mapping ordinals differ")
        if package_item.claim != source_item.claim:
            raise ValueError("adjudication claim text drifted from source")
        expected_evidence = sorted(source_item.evidence, key=lambda item: item.section_id)
        if mapping_entry.evidence_section_ids != [item.section_id for item in expected_evidence]:
            raise ValueError("adjudication evidence identity mapping drifted from source")
        actual_evidence = [(item.content, item.content_sha256) for item in package_item.evidence]
        if actual_evidence != [(item.content, item.content_sha256) for item in expected_evidence]:
            raise ValueError("adjudication evidence content drifted from source")
        actual_prior = sorted(item.decision.value for item in package_item.prior_decisions)
        expected_prior = sorted(item.decision.value for item in decoded_reviews[mapping_entry.key])
        if actual_prior != expected_prior:
            raise ValueError("anonymous prior decisions drifted from review submissions")
        adjudication_by_key[mapping_entry.key] = decisions_by_id[mapping_entry.blind_dispute_id]

    public_payload = serialize_json_model(package).decode("utf-8")
    if any(reviewer_id in public_payload for reviewer_id in reviewer_ids):
        raise ValueError("public adjudication package contains a reviewer pseudonym")

    return (
        adjudication_by_key,
        ArtifactHash(artifact_id=package.package_id, sha256=package_digest),
        ArtifactHash(artifact_id=package.package_id, sha256=mapping_digest),
        ArtifactHash(artifact_id=submission.submission_id, sha256=submission_digest),
    )


def evaluate_entailment_reviews(
    source: SemanticEntailmentSource,
    *,
    source_sha256: str,
    packages: list[tuple[BlindReviewPackage, str]],
    mappings: list[tuple[BlindReviewMapping, str]],
    submissions: list[tuple[ReviewSubmission, str]],
    adjudication_package: tuple[BlindAdjudicationPackage, str] | None = None,
    adjudication_mapping: tuple[BlindAdjudicationMapping, str] | None = None,
    adjudication_submission: tuple[BlindAdjudicationSubmission, str] | None = None,
    generated_at: datetime | None = None,
) -> SemanticEntailmentReport:
    """Validate provenance and aggregate independent human review artifacts."""

    source_sha256 = _validate_sha256(source_sha256, field_name="source_sha256")
    generated_at = generated_at or datetime.now(UTC)
    _validate_aware(generated_at, field_name="generated_at")
    if not packages:
        raise ValueError("at least one blind package is required")
    if len(packages) != len(mappings):
        raise ValueError("every blind package requires exactly one mapping")
    adjudication_artifacts = (
        adjudication_package,
        adjudication_mapping,
        adjudication_submission,
    )
    if any(item is not None for item in adjudication_artifacts) and not all(
        item is not None for item in adjudication_artifacts
    ):
        raise ValueError("adjudication package, mapping, and submission are all-or-none")

    package_by_id: dict[str, tuple[BlindReviewPackage, str]] = {}
    for package, digest in packages:
        digest = _validate_sha256(digest, field_name="package_sha256")
        if package.package_id in package_by_id:
            raise ValueError("package IDs must be unique")
        package_by_id[package.package_id] = (package, digest)
    mapping_by_id: dict[str, tuple[BlindReviewMapping, str]] = {}
    for mapping, digest in mappings:
        digest = _validate_sha256(digest, field_name="mapping_sha256")
        if mapping.package_id in mapping_by_id:
            raise ValueError("mapping package IDs must be unique")
        mapping_by_id[mapping.package_id] = (mapping, digest)
    if set(package_by_id) != set(mapping_by_id):
        raise ValueError("package and mapping IDs do not match")

    for package_id, (package, _digest) in package_by_id.items():
        mapping = mapping_by_id[package_id][0]
        _verify_package_mapping(
            source,
            source_sha256=source_sha256,
            package=package,
            mapping=mapping,
        )

    submission_ids: set[str] = set()
    reviewer_ids: set[str] = set()
    decoded_reviews: dict[tuple[str, str], list[ReviewerDecision]] = {
        item.key: [] for item in source.items
    }
    reviewer_labels: dict[str, dict[tuple[str, str], EntailmentLabel]] = {}
    submission_artifacts: list[ArtifactHash] = []
    used_package_ids: set[str] = set()
    for submission, digest in submissions:
        digest = _validate_sha256(digest, field_name="submission_sha256")
        if submission.submission_id in submission_ids:
            raise ValueError("submission IDs must be unique")
        if submission.reviewer_id in reviewer_ids:
            raise ValueError("each reviewer may submit only one independent review")
        submission_ids.add(submission.submission_id)
        reviewer_ids.add(submission.reviewer_id)
        reviewer_labels[submission.reviewer_id] = {}
        if submission.package_id not in package_by_id:
            raise ValueError("submission references an unknown package")
        if submission.package_id in used_package_ids:
            raise ValueError("each blind package may be submitted only once")
        used_package_ids.add(submission.package_id)
        package, actual_package_hash = package_by_id[submission.package_id]
        mapping = mapping_by_id[submission.package_id][0]
        if submission.package_sha256 != actual_package_hash:
            raise ValueError("submission package_sha256 does not match package bytes")
        if _text_sha256(submission.reviewer_id) != package.reviewer_id_sha256:
            raise ValueError("submission reviewer does not match the assigned blind package")
        if submission.reviewer_attestation.attested_at < package.generated_at:
            raise ValueError("reviewer attestation predates its blind package")
        if submission.submitted_at < package.generated_at:
            raise ValueError("review submission predates its blind package")
        map_by_blind_id = {entry.blind_item_id: entry for entry in mapping.entries}
        for review in submission.reviews:
            entry = map_by_blind_id.get(review.blind_item_id)
            if entry is None:
                raise ValueError("submission contains an unknown blind_item_id")
            decoded_reviews[entry.key].append(
                ReviewerDecision(reviewer_id=submission.reviewer_id, decision=review.decision)
            )
            semantic_label = review.decision.semantic_label
            if semantic_label is not None:
                reviewer_labels[submission.reviewer_id][entry.key] = semantic_label
        submission_artifacts.append(
            ArtifactHash(artifact_id=submission.submission_id, sha256=digest)
        )

    adjudication_needed_keys = {
        key
        for key, decisions in decoded_reviews.items()
        if len(decisions) == 2 and _strict_two_review_consensus(decisions) is None
    }
    review_bundle_hash = _review_bundle_sha256(packages, mappings, submissions)
    adjudication_by_key: dict[tuple[str, str], AdjudicationSubmissionDecision] = {}
    adjudication_package_hash: ArtifactHash | None = None
    adjudication_mapping_hash: ArtifactHash | None = None
    adjudication_submission_hash: ArtifactHash | None = None
    if all(item is not None for item in adjudication_artifacts):
        assert adjudication_package is not None
        assert adjudication_mapping is not None
        assert adjudication_submission is not None
        (
            adjudication_by_key,
            adjudication_package_hash,
            adjudication_mapping_hash,
            adjudication_submission_hash,
        ) = _verify_blind_adjudication(
            source,
            source_sha256=source_sha256,
            review_bundle_sha256=review_bundle_hash,
            decoded_reviews=decoded_reviews,
            reviewer_ids=reviewer_ids,
            latest_review_submitted_at=max(
                submission.submitted_at for submission, _digest in submissions
            ),
            required_keys=adjudication_needed_keys,
            package_artifact=adjudication_package,
            mapping_artifact=adjudication_mapping,
            submission_artifact=adjudication_submission,
        )

    claims: list[ClaimEntailmentScore] = []
    micro_counts = {label: 0 for label in EntailmentLabel}
    macro_rate_sums = {label: 0.0 for label in EntailmentLabel}
    semantic_claims = 0

    for item in source.items:
        decisions = sorted(decoded_reviews[item.key], key=lambda value: value.reviewer_id)
        semantic_labels = [
            label
            for decision in decisions
            if (label := decision.decision.semantic_label) is not None
        ]
        counts = {label: semantic_labels.count(label) for label in EntailmentLabel}
        for label, count in counts.items():
            micro_counts[label] += count
        if semantic_labels:
            semantic_claims += 1
            for label, count in counts.items():
                macro_rate_sums[label] += _rate(count, len(semantic_labels))

        skip_count = sum(value.decision is ReviewDecision.SKIP for value in decisions)
        conflict_count = sum(value.decision is ReviewDecision.CONFLICT for value in decisions)
        distinct_labels = set(semantic_labels)
        consensus_label = _strict_two_review_consensus(decisions)
        needs_adjudication = item.key in adjudication_needed_keys
        adjudicated_entry = adjudication_by_key.get(item.key)
        if adjudicated_entry is not None and not needs_adjudication:
            raise ValueError("adjudication may only resolve a two-review non-consensus claim")
        final_label = consensus_label
        excluded = False
        if adjudicated_entry is not None:
            final_label = adjudicated_entry.decision.semantic_label
            excluded = adjudicated_entry.decision is AdjudicationDecision.EXCLUDE
            status = ClaimAgreementStatus.EXCLUDED if excluded else ClaimAgreementStatus.ADJUDICATED
        elif conflict_count:
            status = ClaimAgreementStatus.CONFLICT_FLAGGED
        elif len(distinct_labels) > 1:
            status = ClaimAgreementStatus.DISAGREEMENT
        elif needs_adjudication:
            status = ClaimAgreementStatus.ADJUDICATION_REQUIRED
        elif consensus_label is not None:
            status = ClaimAgreementStatus.CONSENSUS
        elif len(semantic_labels) == 1:
            status = ClaimAgreementStatus.SINGLE_RATING
        else:
            status = ClaimAgreementStatus.NO_SEMANTIC_RATING

        claims.append(
            ClaimEntailmentScore(
                question_id=item.question_id,
                claim_id=item.claim_id,
                assigned_reviewers=len(packages),
                received_reviews=len(decisions),
                semantic_reviews=len(semantic_labels),
                skipped_reviews=skip_count,
                conflict_reviews=conflict_count,
                entailed_reviews=counts[EntailmentLabel.ENTAILED],
                contradicted_reviews=counts[EntailmentLabel.CONTRADICTED],
                not_enough_info_reviews=counts[EntailmentLabel.NOT_ENOUGH_INFO],
                reviewer_decisions=decisions,
                agreement_status=status,
                consensus_label=consensus_label,
                final_label=final_label,
                adjudication_needed=needs_adjudication,
                adjudicated=adjudicated_entry is not None,
                excluded=excluded,
            )
        )

    unexpected_adjudications = set(adjudication_by_key) - adjudication_needed_keys
    if unexpected_adjudications:
        raise ValueError("adjudication contains claims that did not require resolution")

    total_claims = len(source.items)
    assigned_ratings = total_claims * len(packages)
    received_ratings = sum(item.received_reviews for item in claims)
    semantic_ratings = sum(item.semantic_reviews for item in claims)
    skipped_ratings = sum(item.skipped_reviews for item in claims)
    conflict_ratings = sum(item.conflict_reviews for item in claims)
    consensus_claims = sum(
        item.agreement_status is ClaimAgreementStatus.CONSENSUS for item in claims
    )
    resolved_claims = sum(item.final_label is not None for item in claims)
    excluded_claims = sum(item.excluded for item in claims)
    adjudicated_claims = sum(item.adjudicated for item in claims)
    unresolved_claims = total_claims - resolved_claims - excluded_claims
    final_counts = {
        label: sum(item.final_label is label for item in claims) for label in EntailmentLabel
    }
    final_label_counts = LabelCounts(
        entailed=final_counts[EntailmentLabel.ENTAILED],
        contradicted=final_counts[EntailmentLabel.CONTRADICTED],
        not_enough_info=final_counts[EntailmentLabel.NOT_ENOUGH_INFO],
    )

    macro_rates = LabelRates(
        entailed=_rate(macro_rate_sums[EntailmentLabel.ENTAILED], semantic_claims),
        contradicted=_rate(macro_rate_sums[EntailmentLabel.CONTRADICTED], semantic_claims),
        not_enough_info=_rate(macro_rate_sums[EntailmentLabel.NOT_ENOUGH_INFO], semantic_claims),
    )
    summary = EntailmentSummary(
        total_claims=total_claims,
        assigned_ratings=assigned_ratings,
        received_ratings=received_ratings,
        semantic_ratings=semantic_ratings,
        skipped_ratings=skipped_ratings,
        conflict_ratings=conflict_ratings,
        rating_coverage=_rate(received_ratings, assigned_ratings),
        semantic_rating_coverage=_rate(semantic_ratings, assigned_ratings),
        claims_with_semantic_rating=semantic_claims,
        claim_semantic_coverage=_rate(semantic_claims, total_claims),
        micro_label_rates=_label_rates(micro_counts, semantic_ratings),
        macro_label_rates=macro_rates,
        final_labeled_claims=resolved_claims,
        final_label_counts=final_label_counts,
        final_label_rates=_label_rates(final_counts, resolved_claims),
        final_label_coverage=_rate(resolved_claims, total_claims),
        consensus_claims=consensus_claims,
        resolved_claims=resolved_claims,
        unresolved_claims=unresolved_claims,
        excluded_claims=excluded_claims,
        adjudication_needed_claims=len(adjudication_needed_keys),
        adjudicated_claims=adjudicated_claims,
        adjudication_coverage=(
            1.0
            if not adjudication_needed_keys
            else _rate(adjudicated_claims, len(adjudication_needed_keys))
        ),
        agreement=_agreement_metrics(reviewer_labels),
    )

    expected_ratings = total_claims * 2
    blockers: list[FormalCompletionBlocker] = []
    if len(packages) != 2:
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.REVIEW_PACKAGE_COUNT_NOT_2,
                message="formal review requires exactly two blind packages",
                expected=2,
                actual=len(packages),
            )
        )
    if len(reviewer_ids) != 2:
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.REVIEWER_COUNT_NOT_2,
                message="formal review requires exactly two independent reviewers",
                expected=2,
                actual=len(reviewer_ids),
            )
        )
    if received_ratings != expected_ratings:
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.RECEIVED_RATING_COUNT_MISMATCH,
                message="received rating count differs from two ratings per source claim",
                expected=expected_ratings,
                actual=received_ratings,
            )
        )
    claims_with_two_reviews = sum(item.received_reviews == 2 for item in claims)
    if claims_with_two_reviews != total_claims:
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.PER_CLAIM_REVIEW_COVERAGE_INCOMPLETE,
                message="every source claim must receive exactly two reviews",
                expected=total_claims,
                actual=claims_with_two_reviews,
            )
        )
    if adjudicated_claims != len(adjudication_needed_keys):
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.ADJUDICATION_INCOMPLETE,
                message="every two-review non-consensus claim must be adjudicated",
                expected=len(adjudication_needed_keys),
                actual=adjudicated_claims,
            )
        )
    if unresolved_claims:
        blockers.append(
            FormalCompletionBlocker(
                code=FormalCompletionBlockerCode.UNRESOLVED_CLAIMS,
                message="every claim must have a final label or be explicitly excluded",
                expected=total_claims,
                actual=resolved_claims + excluded_claims,
            )
        )
    formal_completion = FormalCompletion(
        status=(
            FormalCompletionStatus.COMPLETE if not blockers else FormalCompletionStatus.INCOMPLETE
        ),
        expected_claims=total_claims,
        expected_ratings=expected_ratings,
        actual_review_packages=len(packages),
        actual_reviewers=len(reviewer_ids),
        actual_ratings=received_ratings,
        resolved_or_excluded_claims=resolved_claims + excluded_claims,
        required_adjudications=len(adjudication_needed_keys),
        completed_adjudications=adjudicated_claims,
        blockers=blockers,
    )
    return SemanticEntailmentReport(
        generated_at=generated_at,
        source_id=source.source_id,
        prediction_set_id=source.prediction_set_id,
        prediction_sha256=source.prediction_sha256,
        source_sha256=source_sha256,
        package_artifacts=[
            ArtifactHash(artifact_id=package.package_id, sha256=digest)
            for package, digest in packages
        ],
        mapping_artifacts=[
            ArtifactHash(artifact_id=mapping.package_id, sha256=digest)
            for mapping, digest in mappings
        ],
        submission_artifacts=submission_artifacts,
        adjudication_package_artifact=adjudication_package_hash,
        adjudication_mapping_artifact=adjudication_mapping_hash,
        adjudication_submission_artifact=adjudication_submission_hash,
        formal_completion=formal_completion,
        summary=summary,
        claims=claims,
    )


def build_blind_adjudication_package(
    source: SemanticEntailmentSource,
    *,
    source_sha256: str,
    packages: list[tuple[BlindReviewPackage, str]],
    mappings: list[tuple[BlindReviewMapping, str]],
    submissions: list[tuple[ReviewSubmission, str]],
    adjudicator_id: str,
    ordering_seed: str,
    generated_at: datetime | None = None,
) -> tuple[BlindAdjudicationPackage, BlindAdjudicationMapping]:
    """Build a randomized dispute-only package plus a separately held map."""

    source_sha256 = _validate_sha256(source_sha256, field_name="source_sha256")
    adjudicator_id = _validate_reviewer_pseudonym(adjudicator_id, field_name="adjudicator_id")
    ordering_seed = _non_blank(ordering_seed, field_name="ordering_seed")
    generated_at = generated_at or datetime.now(UTC)
    _validate_aware(generated_at, field_name="generated_at")
    reviewer_ids = {submission.reviewer_id for submission, _digest in submissions}
    if adjudicator_id in reviewer_ids:
        raise ValueError("adjudicator pseudonym must differ from reviewer pseudonyms")

    preliminary = evaluate_entailment_reviews(
        source,
        source_sha256=source_sha256,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        generated_at=generated_at,
    )
    if len(packages) != 2 or len(submissions) != 2 or len(reviewer_ids) != 2:
        raise ValueError("adjudication package requires exactly two independent review submissions")
    if any(item.received_reviews != 2 for item in preliminary.claims):
        raise ValueError("adjudication package requires two reviews for every source claim")
    if generated_at < max(submission.submitted_at for submission, _digest in submissions):
        raise ValueError("adjudication package cannot predate the supplied review submissions")

    required_keys = {
        (item.question_id, item.claim_id) for item in preliminary.claims if item.adjudication_needed
    }
    if not required_keys:
        raise ValueError("no two-review non-consensus claims require adjudication")

    report_by_key = {(item.question_id, item.claim_id): item for item in preliminary.claims}
    adjudicator_hash = _text_sha256(adjudicator_id)
    review_bundle_hash = _review_bundle_sha256(packages, mappings, submissions)
    package_material = "\0".join(
        (source_sha256, review_bundle_hash, adjudicator_hash, ordering_seed)
    )
    package_id = f"adjudication-package-{_text_sha256(package_material)[:24]}"

    dispute_rows: list[tuple[str, SourceClaim]] = []
    for source_item in source.items:
        if source_item.key not in required_keys:
            continue
        blind_material = "\0".join(
            (
                ordering_seed,
                source_sha256,
                source_item.question_id,
                source_item.claim_id,
            )
        )
        dispute_rows.append((f"dispute-{_text_sha256(blind_material)[:24]}", source_item))
    _seeded_random(source_sha256, review_bundle_hash, adjudicator_hash, ordering_seed).shuffle(
        dispute_rows
    )

    public_items: list[BlindAdjudicationItem] = []
    private_entries: list[BlindAdjudicationMapEntry] = []
    for ordinal, (blind_dispute_id, source_item) in enumerate(dispute_rows, start=1):
        evidence = sorted(source_item.evidence, key=lambda item: item.section_id)
        decisions = [
            BlindPriorDecision(ordinal=index, decision=value.decision)
            for index, value in enumerate(
                report_by_key[source_item.key].reviewer_decisions, start=1
            )
        ]
        _seeded_random(ordering_seed, blind_dispute_id, "prior-decisions").shuffle(decisions)
        decisions = [
            item.model_copy(update={"ordinal": index})
            for index, item in enumerate(decisions, start=1)
        ]
        public_items.append(
            BlindAdjudicationItem(
                ordinal=ordinal,
                blind_dispute_id=blind_dispute_id,
                claim=source_item.claim,
                evidence=[
                    BlindAdjudicationEvidence(
                        ordinal=index,
                        content=item.content,
                        content_sha256=item.content_sha256,
                    )
                    for index, item in enumerate(evidence, start=1)
                ],
                prior_decisions=decisions,
            )
        )
        private_entries.append(
            BlindAdjudicationMapEntry(
                ordinal=ordinal,
                blind_dispute_id=blind_dispute_id,
                question_id=source_item.question_id,
                claim_id=source_item.claim_id,
                evidence_section_ids=[item.section_id for item in evidence],
            )
        )

    ordering_hash = _adjudication_ordering_sha256(public_items)
    package = BlindAdjudicationPackage(
        package_id=package_id,
        adjudicator_id_sha256=adjudicator_hash,
        generated_at=generated_at,
        blind_ordering_sha256=ordering_hash,
        items=public_items,
    )
    package_bytes = serialize_json_model(package)
    public_payload = package_bytes.decode("utf-8")
    if any(reviewer_id in public_payload for reviewer_id in reviewer_ids):
        raise ValueError("public adjudication package contains a reviewer pseudonym")
    mapping = BlindAdjudicationMapping(
        package_id=package_id,
        package_sha256=sha256_bytes(package_bytes),
        source_sha256=source_sha256,
        prediction_sha256=source.prediction_sha256,
        adjudicator_id_sha256=adjudicator_hash,
        review_bundle_sha256=review_bundle_hash,
        generated_at=generated_at,
        blind_ordering_sha256=ordering_hash,
        entries=private_entries,
    )
    return package, mapping
