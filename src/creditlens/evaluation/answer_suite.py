"""Contracts and helpers for the reproducible v1.6 answer re-evaluation suite.

The online generator remains the authority for the two-phase gold boundary.
This module deliberately contains no model/provider code and never opens the
source-gold file.  It only validates the public manifest and builds a scoring
projection *after* a prediction checkpoint has been produced.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creditlens.evaluation.answer_metrics import (
    AnswerEvalDataset,
    AnswerEvaluationSummary,
    AnswerPredictionSet,
)
from creditlens.evaluation.source_state import (
    SOURCE_STATE_ALGORITHM,
    SOURCE_STATE_SCOPE,
    EvidenceMaturity,
    validate_source_state_binding,
)

ANSWER_REEVALUATION_PROTOCOL_VERSION = "1.0.0"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerSuiteStageName(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


class AnswerSuiteStage(_StrictModel):
    name: AnswerSuiteStageName
    question_limit: int = Field(gt=0)
    expected_question_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_count_contract(self) -> Self:
        if self.question_limit != self.expected_question_count:
            raise ValueError("question_limit must equal expected_question_count")
        return self


class AnswerReevaluationManifest(_StrictModel):
    protocol_id: Literal["creditlens_answer_reevaluation_v1"]
    protocol_version: Literal["1.0.0"]
    query_dataset: str
    answer_dataset: str
    source_gold_dataset: str
    top_k: int = Field(default=8, ge=1, le=20)
    stages: list[AnswerSuiteStage] = Field(min_length=2, max_length=2)
    gold_boundary: Literal["ONLINE_QUERY_ONLY_THEN_OFFLINE_GOLD_MAPPING"]

    @field_validator("query_dataset", "answer_dataset", "source_gold_dataset")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("manifest dataset paths must be project-relative and traversal-free")
        if candidate.suffix.lower() != ".json":
            raise ValueError("manifest dataset paths must point to JSON files")
        return candidate.as_posix()

    @model_validator(mode="after")
    def validate_stage_contract(self) -> Self:
        if [stage.name for stage in self.stages] != [
            AnswerSuiteStageName.SMOKE,
            AnswerSuiteStageName.FULL,
        ]:
            raise ValueError("manifest stages must be ordered exactly as smoke then full")
        stages = {stage.name: stage for stage in self.stages}
        if set(stages) != {AnswerSuiteStageName.SMOKE, AnswerSuiteStageName.FULL}:
            raise ValueError("manifest must define exactly smoke and full stages")
        if stages[AnswerSuiteStageName.SMOKE].expected_question_count != 3:
            raise ValueError("the smoke stage is frozen at exactly 3 questions")
        if stages[AnswerSuiteStageName.FULL].expected_question_count != 41:
            raise ValueError("the full stage is frozen at exactly 41 questions")
        return self


_COMPLETED_STAGE_FIELDS = (
    "prediction_count",
    "idempotent_replay_count",
    "generation_return_code",
    "scoring_return_code",
    "predictions_path",
    "scoring_dataset_path",
    "report_path",
    "query_dataset_sha256",
    "answer_dataset_sha256",
    "source_gold_sha256",
    "predictions_sha256",
    "scoring_dataset_sha256",
    "report_sha256",
    "runtime_profile_sha256",
    "git_commit",
    "git_dirty",
    "source_state_sha256",
    "source_state_algorithm",
    "source_state_scope",
    "source_state_file_count",
    "evidence_maturity",
    "summary",
)
_SHA256_STAGE_FIELDS = (
    "query_dataset_sha256",
    "answer_dataset_sha256",
    "source_gold_sha256",
    "predictions_sha256",
    "scoring_dataset_sha256",
    "report_sha256",
    "runtime_profile_sha256",
    "source_state_sha256",
)
_PATH_STAGE_FIELDS = ("predictions_path", "scoring_dataset_path", "report_path")

_COMPLETED_STAGE_PROPERTIES: dict[str, object] = {
    **{field: {"type": "string", "minLength": 1} for field in _PATH_STAGE_FIELDS},
    **{field: {"type": "string", "pattern": r"^[0-9a-f]{64}$"} for field in _SHA256_STAGE_FIELDS},
    "prediction_count": {"type": "integer", "minimum": 0},
    "idempotent_replay_count": {"const": 0, "type": "integer"},
    "generation_return_code": {"const": 0, "type": "integer"},
    "scoring_return_code": {"const": 0, "type": "integer"},
    "git_commit": {"type": "string", "pattern": r"^[0-9a-f]{7,64}$"},
    "git_dirty": {"type": "boolean"},
    "source_state_algorithm": {"const": SOURCE_STATE_ALGORITHM, "type": "string"},
    "source_state_scope": {"const": SOURCE_STATE_SCOPE, "type": "string"},
    "source_state_file_count": {"type": "integer", "minimum": 1},
    "evidence_maturity": {
        "enum": [
            EvidenceMaturity.DEVELOPMENT_SOURCE_BOUND.value,
            EvidenceMaturity.RELEASE_CANDIDATE.value,
        ],
        "type": "string",
    },
    "summary": {
        "properties": {"missing_predictions": {"const": 0, "type": "integer"}},
        "required": ["missing_predictions"],
        "type": "object",
    },
    "failure_phase": {"type": "null"},
}

_ANSWER_SUITE_STAGE_SCHEMA_EXTRA = {
    "allOf": [
        {
            "if": {
                "properties": {"name": {"const": "smoke"}},
                "required": ["name"],
            },
            "then": {"properties": {"expected_question_count": {"const": 3}}},
        },
        {
            "if": {
                "properties": {"name": {"const": "full"}},
                "required": ["name"],
            },
            "then": {"properties": {"expected_question_count": {"const": 41}}},
        },
        {
            "if": {
                "properties": {"status": {"const": "PLANNED"}},
                "required": ["status"],
            },
            "then": {
                "properties": {
                    **{field: {"type": "null"} for field in _COMPLETED_STAGE_FIELDS},
                    "failure_phase": {"type": "null"},
                }
            },
        },
        {
            "if": {
                "properties": {"status": {"const": "RUNNING"}},
                "required": ["status"],
            },
            "then": {"properties": {"failure_phase": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {"status": {"const": "FAILED"}},
                "required": ["status"],
            },
            "then": {
                "properties": {
                    "failure_phase": {
                        "enum": ["GENERATION", "SCORING", "VALIDATION", "INTERRUPTED"],
                        "type": "string",
                    }
                },
                "required": ["failure_phase"],
            },
        },
        {
            "if": {
                "properties": {
                    "status": {"const": "FAILED"},
                    "failure_phase": {"const": "GENERATION"},
                },
                "required": ["status", "failure_phase"],
            },
            "then": {
                "properties": {
                    "generation_return_code": {
                        "not": {"const": 0},
                        "type": "integer",
                    }
                },
                "required": ["generation_return_code"],
            },
        },
        {
            "if": {
                "properties": {
                    "status": {"const": "FAILED"},
                    "failure_phase": {"const": "SCORING"},
                },
                "required": ["status", "failure_phase"],
            },
            "then": {
                "properties": {
                    "scoring_return_code": {
                        "not": {"const": 0},
                        "type": "integer",
                    }
                },
                "required": ["scoring_return_code"],
            },
        },
        {
            "if": {
                "properties": {"status": {"const": "COMPLETED"}},
                "required": ["status"],
            },
            "then": {
                "properties": _COMPLETED_STAGE_PROPERTIES,
                "required": list(_COMPLETED_STAGE_FIELDS),
            },
        },
        {
            "if": {
                "properties": {
                    "name": {"const": "smoke"},
                    "status": {"const": "COMPLETED"},
                },
                "required": ["name", "status"],
            },
            "then": {
                "properties": {
                    "prediction_count": {"const": 3, "type": "integer"},
                    "summary": {
                        "properties": {
                            "technical_failures": {"const": 0, "type": "integer"},
                            "total_questions": {"const": 3, "type": "integer"},
                        },
                        "required": ["technical_failures", "total_questions"],
                        "type": "object",
                    },
                }
            },
        },
        {
            "if": {
                "properties": {
                    "name": {"const": "full"},
                    "status": {"const": "COMPLETED"},
                },
                "required": ["name", "status"],
            },
            "then": {
                "properties": {
                    "prediction_count": {"const": 41, "type": "integer"},
                    "summary": {
                        "properties": {"total_questions": {"const": 41, "type": "integer"}},
                        "required": ["total_questions"],
                        "type": "object",
                    },
                }
            },
        },
    ]
}


class AnswerSuiteStageRecord(_StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=_ANSWER_SUITE_STAGE_SCHEMA_EXTRA,
    )

    name: AnswerSuiteStageName
    status: Literal["PLANNED", "RUNNING", "FAILED", "COMPLETED"]
    expected_question_count: int = Field(gt=0)
    execution_nonce: str
    prediction_count: int | None = Field(default=None, ge=0)
    idempotent_replay_count: int | None = Field(default=None, ge=0)
    generation_return_code: int | None = None
    scoring_return_code: int | None = None
    predictions_path: str | None = None
    scoring_dataset_path: str | None = None
    report_path: str | None = None
    query_dataset_sha256: str | None = None
    answer_dataset_sha256: str | None = None
    source_gold_sha256: str | None = None
    predictions_sha256: str | None = None
    scoring_dataset_sha256: str | None = None
    report_sha256: str | None = None
    runtime_profile_sha256: str | None = None
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,64}$")
    git_dirty: bool | None = None
    source_state_sha256: str | None = None
    source_state_algorithm: Literal["sha256-canonical-file-manifest-v1"] | None = None
    source_state_scope: Literal["creditlens-runtime-evidence-v1"] | None = None
    source_state_file_count: int | None = Field(default=None, gt=0)
    evidence_maturity: EvidenceMaturity | None = None
    summary: AnswerEvaluationSummary | None = None
    failure_phase: Literal["GENERATION", "SCORING", "VALIDATION", "INTERRUPTED"] | None = None

    @field_validator("execution_nonce")
    @classmethod
    def validate_execution_nonce(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 8:
            raise ValueError("execution_nonce must contain at least 8 non-whitespace characters")
        return normalized

    @field_validator(
        "query_dataset_sha256",
        "answer_dataset_sha256",
        "source_gold_sha256",
        "predictions_sha256",
        "scoring_dataset_sha256",
        "report_sha256",
        "source_state_sha256",
        "runtime_profile_sha256",
    )
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("artifact hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_state_contract(self) -> Self:
        frozen_count = {
            AnswerSuiteStageName.SMOKE: 3,
            AnswerSuiteStageName.FULL: 41,
        }[self.name]
        if self.expected_question_count != frozen_count:
            raise ValueError(f"{self.name.value} expected_question_count must equal {frozen_count}")

        result_fields = (
            self.prediction_count,
            self.idempotent_replay_count,
            self.generation_return_code,
            self.scoring_return_code,
            self.predictions_path,
            self.scoring_dataset_path,
            self.report_path,
            self.query_dataset_sha256,
            self.answer_dataset_sha256,
            self.source_gold_sha256,
            self.predictions_sha256,
            self.scoring_dataset_sha256,
            self.report_sha256,
            self.runtime_profile_sha256,
            self.git_commit,
            self.git_dirty,
            self.source_state_sha256,
            self.source_state_algorithm,
            self.source_state_scope,
            self.source_state_file_count,
            self.evidence_maturity,
            self.summary,
        )
        source_fields = (
            self.git_commit,
            self.git_dirty,
            self.source_state_sha256,
            self.source_state_algorithm,
            self.source_state_scope,
            self.source_state_file_count,
            self.evidence_maturity,
        )
        if any(value is not None for value in source_fields):
            if any(value is None for value in source_fields):
                raise ValueError("source-state evidence fields must be present as one complete set")
            validate_source_state_binding(
                git_commit=self.git_commit,
                git_dirty=self.git_dirty,
                source_state_sha256=self.source_state_sha256 or "",
                source_state_algorithm=self.source_state_algorithm or SOURCE_STATE_ALGORITHM,
                source_state_scope=self.source_state_scope or SOURCE_STATE_SCOPE,
                source_state_file_count=self.source_state_file_count or 0,
                evidence_maturity=self.evidence_maturity
                or EvidenceMaturity.DEVELOPMENT_SOURCE_BOUND,
            )
        if self.status == "PLANNED":
            if any(value is not None for value in result_fields):
                raise ValueError("PLANNED stages cannot contain execution artifacts or results")
            if self.failure_phase is not None:
                raise ValueError("PLANNED stages cannot contain a failure_phase")
            return self

        if self.status == "RUNNING":
            if self.failure_phase is not None:
                raise ValueError("RUNNING stages cannot contain a failure_phase")
            return self

        if self.status == "FAILED":
            if self.failure_phase is None:
                raise ValueError("FAILED stages require a failure_phase")
            if self.failure_phase == "GENERATION" and self.generation_return_code in {None, 0}:
                raise ValueError("GENERATION failures require a non-zero generation_return_code")
            if self.failure_phase == "SCORING" and self.scoring_return_code in {None, 0}:
                raise ValueError("SCORING failures require a non-zero scoring_return_code")
            return self

        required_paths_and_hashes = {
            "predictions_path": self.predictions_path,
            "scoring_dataset_path": self.scoring_dataset_path,
            "report_path": self.report_path,
            "query_dataset_sha256": self.query_dataset_sha256,
            "answer_dataset_sha256": self.answer_dataset_sha256,
            "source_gold_sha256": self.source_gold_sha256,
            "predictions_sha256": self.predictions_sha256,
            "scoring_dataset_sha256": self.scoring_dataset_sha256,
            "report_sha256": self.report_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "git_commit": self.git_commit,
            "source_state_sha256": self.source_state_sha256,
            "source_state_algorithm": self.source_state_algorithm,
            "source_state_scope": self.source_state_scope,
            "source_state_file_count": self.source_state_file_count,
            "evidence_maturity": self.evidence_maturity,
        }
        missing = [key for key, value in required_paths_and_hashes.items() if not value]
        if missing:
            raise ValueError(f"COMPLETED stages require artifacts and hashes: missing={missing}")
        if self.git_dirty is None:
            raise ValueError("COMPLETED stages require captured git_dirty state")
        if self.failure_phase is not None:
            raise ValueError("COMPLETED stages cannot contain a failure_phase")
        if self.prediction_count != self.expected_question_count:
            raise ValueError("COMPLETED prediction_count must equal expected_question_count")
        if self.idempotent_replay_count != 0:
            raise ValueError("COMPLETED stages require zero idempotent replays")
        if self.generation_return_code != 0 or self.scoring_return_code != 0:
            raise ValueError("COMPLETED stages require zero generation and scoring return codes")
        if self.summary is None:
            raise ValueError("COMPLETED stages require a validated evaluation summary")
        if self.summary.total_questions != self.expected_question_count:
            raise ValueError("COMPLETED summary total_questions must equal the frozen stage count")
        if self.summary.missing_predictions != 0:
            raise ValueError("COMPLETED stages require zero missing predictions")
        if self.name is AnswerSuiteStageName.SMOKE and self.summary.technical_failures != 0:
            raise ValueError("COMPLETED smoke stage requires zero technical failures")
        return self


class AnswerSuiteRunRecord(_StrictModel):
    protocol_id: Literal["creditlens_answer_reevaluation_v1"]
    protocol_version: Literal["1.0.0"]
    run_id: str
    generated_at: str
    execution_requested: bool
    gold_boundary: Literal["ONLINE_QUERY_ONLY_THEN_OFFLINE_GOLD_MAPPING"]
    manifest_sha256: str
    stages: list[AnswerSuiteStageRecord] = Field(min_length=1, max_length=2)

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_run_contract(self) -> Self:
        stage_names = [stage.name for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("suite stage names must be unique")
        if not self.execution_requested and any(stage.status != "PLANNED" for stage in self.stages):
            raise ValueError("plan-only suite records may contain only PLANNED stages")
        return self


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve an already validated manifest path and retain the root boundary."""

    root = project_root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("resolved manifest path escapes the project root")
    return resolved


def build_scoring_dataset(
    dataset: AnswerEvalDataset,
    predictions: AnswerPredictionSet,
    *,
    expected_count: int,
) -> AnswerEvalDataset:
    """Return the exact gold projection corresponding to completed predictions.

    This helper is intentionally called only after the generator has completed
    its gold-free online phase.  Ordering follows the frozen answer dataset,
    while membership must exactly match the prediction checkpoint.
    """

    predicted_ids = [prediction.question_id for prediction in predictions.predictions]
    if len(predicted_ids) != expected_count:
        raise ValueError(
            f"prediction checkpoint has {len(predicted_ids)} questions; expected {expected_count}"
        )
    predicted_id_set = set(predicted_ids)
    selected = [item for item in dataset.questions if item.question_id in predicted_id_set]
    if len(selected) != expected_count:
        unknown = sorted(predicted_id_set - {item.question_id for item in dataset.questions})
        raise ValueError(f"prediction IDs are absent from the answer dataset: {unknown}")
    return dataset.model_copy(update={"questions": selected}, deep=True)
