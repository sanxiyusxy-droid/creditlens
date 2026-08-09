"""Generate frozen predictions through the real Grounded QA service.

The runner deliberately separates the online QA phase from the gold-mapping
phase.  No source-gold bytes are loaded until every selected question has
finished its QA call.  A raw sidecar is atomically checkpointed after each QA
call, then the public prediction file is atomically checkpointed after each
gold-mapping step.  Scoring remains a separate, deterministic command.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from creditlens.agents.auditor import (  # noqa: E402
    GROUNDING_AUDIT_IMPLEMENTATION_VERSION,
)
from creditlens.agents.contracts import GROUNDED_ANSWER_CONTRACT_VERSION  # noqa: E402
from creditlens.application.qa_service import QAService, QAServiceError  # noqa: E402
from creditlens.application.snapshot_service import load_snapshot_context  # noqa: E402
from creditlens.common.config import get_settings  # noqa: E402
from creditlens.evaluation.answer_metrics import (  # noqa: E402
    ARABIC_NUMERAL_PATTERN,
    CHINESE_NUMERAL_PATTERN,
    AnswerEvalDataset,
    AnswerPrediction,
    AnswerPredictionProvenance,
    AnswerPredictionSet,
    PredictedNumericFact,
    PredictionStatus,
    TechnicalFailureProvenance,
    parse_restricted_chinese_number,
)
from creditlens.evaluation.gold_schema import GoldDataset, GoldQuestion  # noqa: E402
from creditlens.evaluation.recall import GoldMappingScope, map_anchor_to_section_ids  # noqa: E402
from creditlens.infrastructure.llm.chat import build_chat_provider  # noqa: E402
from creditlens.infrastructure.llm.embedding import build_embedding_provider  # noqa: E402
from creditlens.infrastructure.objectstore import build_object_store  # noqa: E402
from creditlens.infrastructure.postgres.models import Base  # noqa: E402
from creditlens.infrastructure.postgres.session import (  # noqa: E402
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client  # noqa: E402
from creditlens.retrieval.orchestrator import RetrievalOrchestrator  # noqa: E402
from creditlens.retrieval.rerank import build_reranker  # noqa: E402
from seed_synthetic_data import (  # noqa: E402
    CASE_ID,
    CASE_ID_002,
    CASE_ID_003,
    DEMO_USER_ID,
    TENANT_ID,
    seed_environment,
)

DEFAULT_QUERY_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "answer_eval_queries_v1.json"
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "answer_eval_v1.json"
DEFAULT_GOLD = PROJECT_ROOT / "evaluation" / "datasets" / "frozen_v2.json"
CASE_KEY_MAP = {
    "golden_case_001": CASE_ID,
    "golden_case_002": CASE_ID_002,
    "golden_case_003": CASE_ID_003,
}

PREDICTION_ADAPTER_VERSION = "1.0.0"
_NUMBER = re.compile(
    rf"百分之(?P<percent_chinese>{CHINESE_NUMERAL_PATTERN})"
    rf"|百分之(?P<percent_arabic>{ARABIC_NUMERAL_PATTERN})"
    rf"|(?P<arabic>{ARABIC_NUMERAL_PATTERN})\s*"
    r"(?P<arabic_unit>个百分点|%|亿元|万元|元|个月|月|年|天|倍|家|户)"
    rf"|(?P<chinese>{CHINESE_NUMERAL_PATTERN})\s*"
    r"(?P<chinese_unit>个百分点|%|亿元|万元|元|个月|月|年|天|倍|家|户)"
)
_CLAUSE_BOUNDARY = re.compile(r"[。！？!?；;，,/\n]")
_LABEL_TRAILING_BRIDGE = re.compile(
    r"(?:为人民币|人民币|为|是|达|达到|约|不高于|不得高于|不低于|不得低于|"
    r"不超过|不得超过|超过)$"
)
_LABEL_UPPER_BOUND = re.compile(r"(?:不高于|不得高于|不超过|不得超过)$")
_LABEL_LOWER_BOUND = re.compile(r"(?:不低于|不得低于)$")
_PLAUSIBLE_YEAR_MIN = 1900
_PLAUSIBLE_YEAR_MAX = 2200


class _StrictQueryModel(BaseModel):
    """Strict schema used by the gold-free online generation phase."""

    model_config = ConfigDict(extra="forbid")


class AnswerQueryQuestion(_StrictQueryModel):
    """The complete per-question input allowed to enter Phase 1.

    Deliberately absent are answerability, expected answers, refusal labels,
    citations, tags, intent and split.  ``extra=forbid`` turns an accidental
    gold-field addition into a hard failure instead of silently ignoring it.
    """

    question_id: str
    case_key: str
    question: str
    as_of_date: date
    decision_cutoff_at: datetime

    @field_validator("question_id", "case_key", "question")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("decision_cutoff_at")
    @classmethod
    def validate_aware_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_cutoff_at must include a timezone")
        return value


class AnswerQueryDataset(_StrictQueryModel):
    """Frozen, non-gold query projection consumed before any answer labels."""

    dataset_id: str
    dataset_version: str
    answer_eval_dataset_id: str
    answer_eval_dataset_version: str
    source_dataset_id: str
    source_dataset_version: str
    source_split: str
    ordering: Literal["sha256(question_id)"]
    frozen: Literal[True]
    questions: list[AnswerQueryQuestion] = Field(min_length=1)

    @field_validator(
        "dataset_id",
        "dataset_version",
        "answer_eval_dataset_id",
        "answer_eval_dataset_version",
        "source_dataset_id",
        "source_dataset_version",
        "source_split",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("frozen", mode="before")
    @classmethod
    def validate_literal_true(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError("frozen must be exactly true")
        return value

    @model_validator(mode="after")
    def validate_question_ids(self) -> Self:
        question_ids = [item.question_id for item in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        expected_order = sorted(
            question_ids,
            key=lambda question_id: hashlib.sha256(question_id.encode("utf-8")).hexdigest(),
        )
        if question_ids != expected_order:
            raise ValueError("questions must use sha256(question_id) ordering")
        return self


@dataclass
class CitationMappingStats:
    """Conservative mapping diagnostics accumulated across predictions."""

    ambiguous_citation_sections: int = 0
    unmapped_citation_sections: int = 0


@dataclass
class _RawQuestionResult:
    question: AnswerQueryQuestion
    response: Any | None = None
    failure: AnswerPrediction | None = None


@dataclass
class _RuntimeResources:
    """Constructed resources in creation order for best-effort cleanup."""

    values: list[tuple[str, Any]] = field(default_factory=list)

    def add(self, name: str, value: Any) -> Any:
        if value is not None:
            self.values.append((name, value))
        return value


def _read_bytes_and_sha256(path: Path) -> tuple[bytes, str]:
    """Read a frozen input exactly once and hash those exact bytes."""

    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_metadata() -> tuple[str | None, bool | None]:
    commit = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain", "--untracked-files=all")
    return commit, None if status is None else bool(status)


def _prompt_fingerprint(settings: Any) -> tuple[str, str]:
    version = str(settings.qa_prompt_version)
    prompt_path = PROJECT_ROOT / "config" / "prompts" / f"{version}.yaml"
    prompt_bytes = prompt_path.read_bytes()
    return version, hashlib.sha256(prompt_bytes).hexdigest()


def _experiment_contract(
    *,
    query_dataset_sha256: str,
    top_k: int,
    prompt_version: str,
    prompt_sha256: str,
    settings: Any,
    orchestrator: Any | None = None,
    audit_implementation_version: str = GROUNDING_AUDIT_IMPLEMENTATION_VERSION,
    grounded_answer_contract_version: str = GROUNDED_ANSWER_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Return every generation-affecting dimension required for idempotency."""

    runtime_embedder = getattr(orchestrator, "embedder", None)
    runtime_reranker = getattr(orchestrator, "reranker", None)
    return {
        # This is intentionally the non-gold query projection hash.  Neither
        # answer expectations nor source-gold bytes may influence Phase-1
        # idempotency keys.
        "query_dataset_sha256": query_dataset_sha256,
        "prediction_adapter_version": PREDICTION_ADAPTER_VERSION,
        "top_k": top_k,
        "prompt": {"version": prompt_version, "sha256": prompt_sha256},
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model or None,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model or None,
            "version": settings.effective_embedding_version,
            "dimension": settings.embedding_dim,
        },
        "rerank": {
            "provider": settings.rerank_provider,
            "model": settings.rerank_model or None,
            "enabled": settings.orchestrator_enable_rerank,
        },
        "qa": {
            "allow_extractive_fallback": getattr(settings, "qa_allow_extractive_fallback", False),
            "max_claims": getattr(settings, "qa_max_claims", None),
            "max_generation_tokens": getattr(settings, "qa_max_generation_tokens", None),
            "max_audit_repairs": getattr(settings, "qa_max_audit_repairs", None),
        },
        "audit": {
            "implementation_version": audit_implementation_version,
            "grounded_answer_contract_version": grounded_answer_contract_version,
        },
        # Keep these effective runtime values aligned with QAService's request
        # hash.  Provider labels alone are insufficient when a model deployment
        # or fusion configuration changes behind the same provider name.
        "orchestrator_runtime": {
            "rrf_k": getattr(
                orchestrator,
                "rrf_k",
                getattr(settings, "rrf_k", None),
            ),
            "route_weights": getattr(orchestrator, "route_weights", None),
            "embedding_version": getattr(
                runtime_embedder,
                "version",
                settings.effective_embedding_version,
            ),
            "reranker_version": getattr(
                runtime_reranker,
                "version",
                settings.rerank_model or None,
            ),
        },
    }


def _idempotency_key(experiment_sha256: str, question_id: str) -> str:
    question_hash = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16]
    key = f"answer-eval:{experiment_sha256[:32]}:{question_hash}"
    if len(key) > 128:  # Defensive assertion for the API/database contract.
        raise ValueError("answer-evaluation idempotency key exceeds 128 characters")
    return key


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one checkpoint atomically; never expose a truncated JSON file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode(
        "utf-8"
    )
    _atomic_write(path, encoded)


def _atomic_write_prediction_set(path: Path, prediction_set: AnswerPredictionSet) -> None:
    _atomic_write(path, (prediction_set.model_dump_json(indent=2) + "\n").encode("utf-8"))


def _raw_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.raw.checkpoint.json")


def _jsonable_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "__dict__"):
        return {
            key: _jsonable_response(value)
            for key, value in vars(response).items()
            if not key.startswith("_")
        }
    if isinstance(response, dict):
        return {str(key): _jsonable_response(value) for key, value in response.items()}
    if isinstance(response, (list, tuple)):
        return [_jsonable_response(value) for value in response]
    return response


def _checkpoint_raw_results(
    path: Path,
    *,
    query_dataset: AnswerQueryDataset,
    query_dataset_sha256: str,
    experiment_sha256: str,
    selected_count: int,
    raw_results: list[_RawQuestionResult],
) -> None:
    records: list[dict[str, Any]] = []
    for item in raw_results:
        record: dict[str, Any] = {"question_id": item.question.question_id}
        if item.failure is not None:
            record["failure"] = item.failure.model_dump(mode="json")
        else:
            record["response"] = _jsonable_response(item.response)
        records.append(record)
    _atomic_write_json(
        path,
        {
            "checkpoint_type": "grounded_qa_raw_phase",
            "query_dataset_id": query_dataset.dataset_id,
            "query_dataset_version": query_dataset.dataset_version,
            "query_dataset_sha256": query_dataset_sha256,
            "experiment_sha256": experiment_sha256,
            "completed_questions": len(raw_results),
            "selected_questions": selected_count,
            "qa_phase_complete": len(raw_results) == selected_count,
            "results": records,
        },
    )


def extract_numeric_facts(answer: str) -> list[PredictedNumericFact]:
    """Extract value/unit and bounded left context without consulting gold."""

    answer = unicodedata.normalize("NFKC", answer)
    facts: list[PredictedNumericFact] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _NUMBER.finditer(answer):
        raw_value = (
            match.group("percent_chinese")
            or match.group("percent_arabic")
            or match.group("arabic")
            or match.group("chinese")
        )
        unit = (
            "%"
            if match.group("percent_chinese") or match.group("percent_arabic")
            else match.group("arabic_unit") or match.group("chinese_unit")
        )
        if match.group("percent_chinese") or match.group("chinese"):
            parsed_value = parse_restricted_chinese_number(raw_value)
            if parsed_value is None:
                continue
            normalized_value = format(parsed_value, "f")
        else:
            normalized_value = raw_value.replace(",", "")
        if "." in normalized_value:
            normalized_value = normalized_value.rstrip("0").rstrip(".")
        if not normalized_value:
            continue
        if unit == "年":
            try:
                numeric_year = int(normalized_value)
            except ValueError:
                numeric_year = 0
            if _PLAUSIBLE_YEAR_MIN <= numeric_year <= _PLAUSIBLE_YEAR_MAX:
                continue
        # “3个月后” is a duration, but “3月营收” is a calendar modifier.
        if unit == "月" and match.end() < len(answer):
            following = answer[match.end()]
            if following not in "。！？!?；;，,、 \t\n":
                continue
        left = answer[max(0, match.start() - 48) : match.start()]
        label = _CLAUSE_BOUNDARY.split(left)[-1].strip(" ：:（）()[]【】\t")
        label = _LABEL_UPPER_BOUND.sub("上限", label)
        label = _LABEL_LOWER_BOUND.sub("下限", label)
        previous_label = None
        while label != previous_label:
            previous_label = label
            label = _LABEL_TRAILING_BRIDGE.sub("", label).rstrip()
        label = label[-40:].strip() or "未命名数值"
        key = (label, normalized_value, unit)
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            PredictedNumericFact(
                name=label,
                value=normalized_value,
                unit=unit,
            )
        )
    return facts


def _validate_gold_provenance(
    dataset: AnswerEvalDataset,
    gold: GoldDataset,
) -> dict[str, GoldQuestion]:
    """Fail closed when the answer set is not an exact projection of its source."""

    mismatches: list[str] = []
    if gold.dataset_id != dataset.source_dataset_id:
        mismatches.append("source_dataset_id")
    if gold.dataset_version != dataset.source_dataset_version:
        mismatches.append("source_dataset_version")

    gold_questions: dict[str, GoldQuestion] = {}
    duplicate_ids: set[str] = set()
    for question in gold.questions:
        if question.question_id in gold_questions:
            duplicate_ids.add(question.question_id)
        gold_questions[question.question_id] = question
    if duplicate_ids:
        mismatches.append(f"duplicate_gold_question_ids={sorted(duplicate_ids)!r}")

    for question in dataset.questions:
        source = gold_questions.get(question.question_id)
        if source is None:
            mismatches.append(f"{question.question_id}:missing")
            continue
        expected_fields = {
            "question": question.question,
            "case_key": question.case_key,
            "as_of_date": question.as_of_date,
            "decision_cutoff_at": question.decision_cutoff_at,
            "answerable": question.answerable,
            "split": dataset.source_split,
        }
        source_fields = {
            "question": source.question,
            "case_key": source.case_key,
            "as_of_date": source.as_of_date,
            "decision_cutoff_at": source.decision_cutoff_at,
            "answerable": source.answerable,
            "split": source.split,
        }
        if (
            source.decision_cutoff_at.tzinfo is None
            or source.decision_cutoff_at.utcoffset() is None
        ):
            mismatches.append(f"{question.question_id}:decision_cutoff_at_timezone")
        for field_name, expected in expected_fields.items():
            if source_fields[field_name] != expected:
                mismatches.append(f"{question.question_id}:{field_name}")

    if mismatches:
        joined = ", ".join(mismatches[:20])
        suffix = f" (+{len(mismatches) - 20} more)" if len(mismatches) > 20 else ""
        raise RuntimeError(f"source gold provenance mismatch: {joined}{suffix}")
    return gold_questions


def _validate_query_projection(
    query_dataset: AnswerQueryDataset,
    answer_dataset: AnswerEvalDataset,
) -> None:
    """Verify Phase-1 queries are an exact, ordered projection of Phase-2 eval data."""

    mismatches: list[str] = []
    metadata_pairs = {
        "answer_eval_dataset_id": (
            query_dataset.answer_eval_dataset_id,
            answer_dataset.dataset_id,
        ),
        "answer_eval_dataset_version": (
            query_dataset.answer_eval_dataset_version,
            answer_dataset.dataset_version,
        ),
        "source_dataset_id": (
            query_dataset.source_dataset_id,
            answer_dataset.source_dataset_id,
        ),
        "source_dataset_version": (
            query_dataset.source_dataset_version,
            answer_dataset.source_dataset_version,
        ),
        "source_split": (
            query_dataset.source_split,
            answer_dataset.source_split,
        ),
    }
    for field_name, (projected, expected) in metadata_pairs.items():
        if projected != expected:
            mismatches.append(field_name)
    if answer_dataset.frozen is not True:
        mismatches.append("answer_eval_dataset:frozen")

    query_ids = [item.question_id for item in query_dataset.questions]
    expected_query_ids = sorted(
        query_ids,
        key=lambda question_id: hashlib.sha256(question_id.encode("utf-8")).hexdigest(),
    )
    if query_ids != expected_query_ids:
        mismatches.append("query_ordering")
    eval_ids = [item.question_id for item in answer_dataset.questions]
    if set(query_ids) != set(eval_ids) or len(query_ids) != len(eval_ids):
        mismatches.append("question_membership")

    eval_by_id = {item.question_id: item for item in answer_dataset.questions}
    for query in query_dataset.questions:
        expected = eval_by_id.get(query.question_id)
        if expected is None:
            mismatches.append(f"{query.question_id}:missing_from_answer_eval")
            continue
        projected_fields = {
            "case_key": query.case_key,
            "question": query.question,
            "as_of_date": query.as_of_date,
            "decision_cutoff_at": query.decision_cutoff_at,
        }
        expected_fields = {
            "case_key": expected.case_key,
            "question": expected.question,
            "as_of_date": expected.as_of_date,
            "decision_cutoff_at": expected.decision_cutoff_at,
        }
        for field_name, projected in projected_fields.items():
            if projected != expected_fields[field_name]:
                mismatches.append(f"{query.question_id}:{field_name}")

    if mismatches:
        joined = ", ".join(mismatches[:20])
        suffix = f" (+{len(mismatches) - 20} more)" if len(mismatches) > 20 else ""
        raise RuntimeError(f"query projection mismatch: {joined}{suffix}")


async def _citation_key_map(
    session: Any,
    *,
    gold: GoldDataset,
    case_id: Any,
    snapshot: Any,
) -> dict[str, set[str]]:
    """Map runtime section UUIDs to stable keys after the QA phase only."""

    scope = GoldMappingScope(
        tenant_id=TENANT_ID,
        case_id=case_id,
        allowed_parse_run_ids=frozenset(snapshot.allowed_parse_run_ids),
    )
    by_section: dict[str, set[str]] = {}
    for anchor in gold.anchors:
        for section_id in await map_anchor_to_section_ids(session, anchor, scope):
            by_section.setdefault(str(section_id), set()).add(anchor.gold_evidence_key)
    return by_section


def _response_section_ids(response: Any) -> set[str]:
    section_ids: set[str] = set()
    for claim in response.claims:
        for citation in [*claim.citations, *claim.opposing_citations]:
            section_id = citation.get("section_id")
            if section_id:
                section_ids.add(str(section_id))
    return section_ids


def _provenance_from_response(response: Any) -> AnswerPredictionProvenance:
    """Copy only the allow-listed run attribution supplied by QAService."""

    return AnswerPredictionProvenance(
        run_id=response.run_id,
        snapshot_id=response.snapshot_id,
        generation_mode=response.generation_mode,
        model_invocation_ids=list(response.model_invocation_ids),
        idempotent_replay=response.idempotent_replay,
    )


def _prediction_from_response(
    question_id: str,
    response: Any,
    citation_map: Mapping[str, set[str]],
    mapping_stats: CitationMappingStats | None = None,
) -> AnswerPrediction:
    if response.answer_status == "ANSWERED":
        citation_refs: set[str] = set()
        for section_id in _response_section_ids(response):
            mapped = citation_map.get(section_id, set())
            if len(mapped) == 1:
                citation_refs.add(next(iter(mapped)))
            elif len(mapped) > 1:
                citation_refs.add(f"ambiguous:section:{section_id}")
                if mapping_stats is not None:
                    mapping_stats.ambiguous_citation_sections += 1
            else:
                citation_refs.add(f"unmapped:section:{section_id}")
                if mapping_stats is not None:
                    mapping_stats.unmapped_citation_sections += 1
        return AnswerPrediction(
            question_id=question_id,
            status=PredictionStatus.ANSWERED,
            answer=response.answer,
            numeric_facts=extract_numeric_facts(response.answer),
            citation_refs=sorted(citation_refs),
            provenance=_provenance_from_response(response),
        )
    if response.answer_status == "ABSTAINED":
        reason_code = getattr(response, "refusal_reason_code", None) or "UNSPECIFIED"
        return AnswerPrediction(
            question_id=question_id,
            status=PredictionStatus.REFUSED,
            refusal_reason_code=reason_code,
            provenance=_provenance_from_response(response),
        )
    if response.answer_status == "NEEDS_REVIEW":
        return AnswerPrediction(
            question_id=question_id,
            status=PredictionStatus.NEEDS_REVIEW,
            provenance=_provenance_from_response(response),
        )
    failure_run_id = getattr(response, "run_id", None)
    return AnswerPrediction(
        question_id=question_id,
        status=PredictionStatus.TECHNICAL_FAILURE,
        error_type="UNKNOWN_ANSWER_STATUS",
        error_message="The Grounded QA workflow returned an unknown answer status.",
        provenance=(
            TechnicalFailureProvenance(run_id=failure_run_id)
            if failure_run_id is not None
            else None
        ),
    )


def _technical_failure(
    question_id: str,
    error_type: str,
    message: str,
    *,
    run_id: uuid.UUID | None = None,
) -> AnswerPrediction:
    return AnswerPrediction(
        question_id=question_id,
        status=PredictionStatus.TECHNICAL_FAILURE,
        error_type=error_type or "UNSPECIFIED_TECHNICAL_FAILURE",
        error_message=message,
        provenance=TechnicalFailureProvenance(run_id=run_id) if run_id else None,
    )


async def _close_one(resource: Any) -> None:
    target = resource
    closer = None
    for method_name in ("aclose", "close", "dispose"):
        candidate = getattr(target, method_name, None)
        if callable(candidate):
            closer = candidate
            break
    if closer is None:
        target = getattr(resource, "_client", None)
        if target is not None:
            for method_name in ("aclose", "close"):
                candidate = getattr(target, method_name, None)
                if callable(candidate):
                    closer = candidate
                    break
    if closer is None:
        return
    result = closer()
    if inspect.isawaitable(result):
        await result


async def _close_runtime_resources(resources: _RuntimeResources) -> list[str]:
    """Close every constructed resource even if individual close calls fail."""

    failures: list[str] = []
    seen: set[int] = set()
    for name, resource in reversed(resources.values):
        if id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            await _close_one(resource)
        except Exception as exc:  # Cleanup is best-effort; later resources must still close.
            failures.append(f"{name}:{type(exc).__name__}")
    return failures


def _prediction_metadata(
    *,
    query_dataset_sha256: str,
    answer_eval_dataset_sha256: str,
    source_gold_sha256: str,
    experiment_sha256: str,
    prompt_sha256: str,
    settings: Any,
    top_k: int,
    selected_questions: int,
    mapping_stats: CitationMappingStats,
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, str | int | float | bool | None]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "query_dataset_sha256": query_dataset_sha256,
        "answer_eval_dataset_sha256": answer_eval_dataset_sha256,
        "source_gold_sha256": source_gold_sha256,
        "experiment_sha256": experiment_sha256,
        "prediction_adapter_version": PREDICTION_ADAPTER_VERSION,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model or None,
        "prompt_version": settings.qa_prompt_version,
        "prompt_sha256": prompt_sha256,
        "embedding_provider": settings.embedding_provider,
        "embedding_version": settings.effective_embedding_version,
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model or None,
        "top_k": top_k,
        "selected_questions": selected_questions,
        "ambiguous_citation_sections": mapping_stats.ambiguous_citation_sections,
        "unmapped_citation_sections": mapping_stats.unmapped_citation_sections,
        "cleanup_failure_count": 0,
        "cleanup_failures": None,
    }


def _build_prediction_set(
    *,
    dataset: AnswerEvalDataset,
    experiment_sha256: str,
    predictions: list[AnswerPrediction],
    metadata: dict[str, str | int | float | bool | None],
) -> AnswerPredictionSet:
    return AnswerPredictionSet(
        prediction_set_id=f"grounded-qa-{experiment_sha256[:24]}",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        prediction_adapter_version=PREDICTION_ADAPTER_VERSION,
        predictions=predictions,
        metadata=metadata,
    )


async def generate(args: argparse.Namespace) -> AnswerPredictionSet:
    settings = get_settings()
    if settings.llm_provider == "disabled" and not args.allow_disabled_llm:
        raise RuntimeError(
            "LLM_PROVIDER=disabled; configure a real provider or pass --allow-disabled-llm "
            "to record technical failures explicitly."
        )

    # Phase 1 reads exactly one frozen input: the non-gold query projection.
    # The complete answer-evaluation set and source gold are not opened until
    # all selected QA calls have been atomically checkpointed.
    query_path = Path(getattr(args, "query_dataset", DEFAULT_QUERY_DATASET))
    query_bytes, query_dataset_sha256 = _read_bytes_and_sha256(query_path)
    query_dataset = AnswerQueryDataset.model_validate_json(query_bytes)
    selected = query_dataset.questions[: args.limit] if args.limit else query_dataset.questions
    prompt_version, prompt_sha256 = _prompt_fingerprint(settings)
    git_commit, git_dirty = _git_metadata()
    output_path = Path(args.output)
    raw_checkpoint = _raw_checkpoint_path(output_path)

    resources = _RuntimeResources()
    cleanup_failures: list[str] = []
    prediction_set: AnswerPredictionSet | None = None
    try:
        engine = resources.add("engine", create_engine())
        factory = create_session_factory(engine)
        store = resources.add("object_store", build_object_store(settings))
        qdrant = resources.add("qdrant", build_qdrant_client(settings))
        embedder = resources.add("embedding", build_embedding_provider(settings))
        reranker = resources.add("reranker", build_reranker(settings))
        chat = resources.add("chat", build_chat_provider(settings))
        orchestrator = RetrievalOrchestrator(
            qdrant=qdrant,
            embedder=embedder,
            reranker=reranker,
            rrf_k=settings.rrf_k,
        )
        experiment_contract = _experiment_contract(
            query_dataset_sha256=query_dataset_sha256,
            top_k=args.top_k,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            settings=settings,
            orchestrator=orchestrator,
        )
        experiment_sha256 = _sha256_json(experiment_contract)

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await seed_environment(factory, store, qdrant, settings)

        service = QAService(
            session_factory=factory,
            orchestrator=orchestrator,
            settings=settings,
            chat=chat,
            tenant_id=TENANT_ID,
            user_id=DEMO_USER_ID,
        )

        # Phase 1: online QA only.  Do not load or inspect the source gold here.
        raw_results: list[_RawQuestionResult] = []
        for index, question in enumerate(selected, start=1):
            case_id = CASE_KEY_MAP.get(question.case_key)
            print(f"[{index}/{len(selected)}] QA {question.question_id}", flush=True)
            if case_id is None:
                raw_results.append(
                    _RawQuestionResult(
                        question=question,
                        failure=_technical_failure(
                            question.question_id,
                            "UNKNOWN_CASE_KEY",
                            "The frozen question references an unknown case key.",
                        ),
                    )
                )
            else:
                try:
                    response = await service.ask(
                        case_id=case_id,
                        question=question.question,
                        top_k=args.top_k,
                        as_of_date=question.as_of_date,
                        decision_cutoff_at=question.decision_cutoff_at.astimezone(UTC),
                        idempotency_key=_idempotency_key(experiment_sha256, question.question_id),
                    )
                    raw_results.append(_RawQuestionResult(question=question, response=response))
                except QAServiceError as exc:
                    raw_results.append(
                        _RawQuestionResult(
                            question=question,
                            failure=_technical_failure(
                                question.question_id,
                                exc.error_type,
                                f"Grounded QA run {exc.run_id} failed.",
                                run_id=exc.run_id,
                            ),
                        )
                    )
                except Exception as exc:
                    raw_results.append(
                        _RawQuestionResult(
                            question=question,
                            failure=_technical_failure(
                                question.question_id,
                                type(exc).__name__,
                                "The question failed before a Grounded QA response was produced.",
                            ),
                        )
                    )
            _checkpoint_raw_results(
                raw_checkpoint,
                query_dataset=query_dataset,
                query_dataset_sha256=query_dataset_sha256,
                experiment_sha256=experiment_sha256,
                selected_count=len(selected),
                raw_results=raw_results,
            )

        # Phase 2 starts only after the complete raw checkpoint exists.  Load
        # and verify the answer evaluation before opening its source gold.
        dataset_bytes, answer_eval_dataset_sha256 = _read_bytes_and_sha256(Path(args.dataset))
        dataset = AnswerEvalDataset.model_validate_json(dataset_bytes)
        _validate_query_projection(query_dataset, dataset)
        gold_bytes, source_gold_sha256 = _read_bytes_and_sha256(Path(args.gold_dataset))
        gold = GoldDataset.model_validate_json(gold_bytes)
        _validate_gold_provenance(dataset, gold)

        predictions: list[AnswerPrediction] = []
        mapping_stats = CitationMappingStats()
        citation_cache: dict[tuple[str, ...], dict[str, set[str]]] = {}
        metadata = _prediction_metadata(
            query_dataset_sha256=query_dataset_sha256,
            answer_eval_dataset_sha256=answer_eval_dataset_sha256,
            source_gold_sha256=source_gold_sha256,
            experiment_sha256=experiment_sha256,
            prompt_sha256=prompt_sha256,
            settings=settings,
            top_k=args.top_k,
            selected_questions=len(selected),
            mapping_stats=mapping_stats,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
        for index, raw in enumerate(raw_results, start=1):
            print(f"[{index}/{len(raw_results)}] MAP {raw.question.question_id}", flush=True)
            if raw.failure is not None:
                prediction = raw.failure
            elif raw.response.answer_status != "ANSWERED":
                # Refusal/review states carry no citations, so a database or
                # anchor-mapping failure must not turn them into technical failures.
                prediction = _prediction_from_response(
                    raw.question.question_id,
                    raw.response,
                    {},
                    mapping_stats,
                )
            else:
                case_id = CASE_KEY_MAP[raw.question.case_key]
                try:
                    question_mapping_stats = CitationMappingStats()
                    async with session_scope(
                        factory,
                        tenant_id=TENANT_ID,
                        user_id=DEMO_USER_ID,
                    ) as session:
                        snapshot = await load_snapshot_context(session, raw.response.snapshot_id)
                        cache_key = (
                            str(case_id),
                            *(
                                str(item)
                                for item in sorted(snapshot.allowed_parse_run_ids, key=str)
                            ),
                        )
                        if cache_key not in citation_cache:
                            citation_cache[cache_key] = await _citation_key_map(
                                session,
                                gold=gold,
                                case_id=case_id,
                                snapshot=snapshot,
                            )
                        citation_map = citation_cache[cache_key]
                    prediction = _prediction_from_response(
                        raw.question.question_id,
                        raw.response,
                        citation_map,
                        question_mapping_stats,
                    )
                    mapping_stats.ambiguous_citation_sections += (
                        question_mapping_stats.ambiguous_citation_sections
                    )
                    mapping_stats.unmapped_citation_sections += (
                        question_mapping_stats.unmapped_citation_sections
                    )
                except Exception as exc:
                    prediction = _technical_failure(
                        raw.question.question_id,
                        f"GOLD_MAPPING_{type(exc).__name__}",
                        "The QA response could not be mapped to stable citation keys.",
                        run_id=raw.response.run_id,
                    )
            predictions.append(prediction)
            metadata["ambiguous_citation_sections"] = mapping_stats.ambiguous_citation_sections
            metadata["unmapped_citation_sections"] = mapping_stats.unmapped_citation_sections
            prediction_set = _build_prediction_set(
                dataset=dataset,
                experiment_sha256=experiment_sha256,
                predictions=predictions,
                metadata=metadata,
            )
            _atomic_write_prediction_set(output_path, prediction_set)
    finally:
        cleanup_failures = await _close_runtime_resources(resources)

    if prediction_set is None:
        raise RuntimeError("answer prediction generation produced no checkpoint")
    prediction_set.metadata["cleanup_failure_count"] = len(cleanup_failures)
    prediction_set.metadata["cleanup_failures"] = (
        ",".join(cleanup_failures) if cleanup_failures else None
    )
    _atomic_write_prediction_set(output_path, prediction_set)
    return prediction_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen questions through Grounded QA.")
    parser.add_argument(
        "--query-dataset",
        type=Path,
        default=DEFAULT_QUERY_DATASET,
        help="Gold-free frozen projection consumed by the online QA phase.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gold-dataset", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--limit", type=int, default=0, help="0 means the full frozen set.")
    parser.add_argument("--allow-disabled-llm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    import asyncio

    prediction_set = asyncio.run(generate(args))
    output = args.output.resolve()
    _atomic_write_prediction_set(output, prediction_set)
    print(
        json.dumps(
            {
                "predictions": str(output),
                "raw_checkpoint": str(_raw_checkpoint_path(output)),
                "count": len(prediction_set.predictions),
                "metadata": prediction_set.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
