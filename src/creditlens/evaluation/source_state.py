"""Deterministically bind development evidence to the source tree that produced it.

The digest is intentionally narrower than a tarball of the repository: it covers
runtime code, deployment/configuration inputs, database migrations, frozen
evaluation datasets, and synthetic demo source assets.  Generated reports,
predictions, schemas, caches, virtual environments, local object data, secrets,
documentation, and tests are outside the digest.  Git dirty state is recorded
separately, so a clean digest can never make a dirty checkout look like release
evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SOURCE_STATE_ALGORITHM = "sha256-canonical-file-manifest-v1"
SOURCE_STATE_SCOPE = "creditlens-runtime-evidence-v1"

# Every regular file below these roots is covered, except the explicitly listed
# transient files/directories.  ``evaluation/datasets`` is deliberately used
# instead of ``evaluation`` so generated reports, predictions, review packages,
# and exported JSON schemas cannot perturb a run's source identity.
SOURCE_STATE_INCLUDED_TREES = (
    "src",
    "apps",
    "scripts",
    "infra",
    "migrations",
    "config",
    "evaluation/datasets",
    "data/synthetic",
)
SOURCE_STATE_INCLUDED_FILES = (
    ".env.example",
    "alembic.ini",
    "compose.yaml",
    "docker-compose.test.yml",
    "pyproject.toml",
    "uv.lock",
)
SOURCE_STATE_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
)
SOURCE_STATE_EXCLUDED_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})
SOURCE_STATE_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".tmp"})
SOURCE_STATE_METADATA_FIELDS = frozenset(
    {
        "git_commit",
        "git_dirty",
        "source_state_sha256",
        "source_state_algorithm",
        "source_state_scope",
        "source_state_file_count",
        "evidence_maturity",
    }
)


class EvidenceMaturity(StrEnum):
    """How strongly an artifact may be presented.

    ``RELEASE_CANDIDATE`` means only that the complete Git checkout was clean at
    capture time and had a measured commit.  It does not claim that the commit
    was tagged, reviewed, published, or released.
    """

    DEVELOPMENT_SOURCE_BOUND = "DEVELOPMENT_SOURCE_BOUND"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


@dataclass(frozen=True, slots=True)
class SourceStateEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceStateSnapshot:
    algorithm: str
    scope: str
    sha256: str
    file_count: int
    entries: tuple[SourceStateEntry, ...]


@dataclass(frozen=True, slots=True)
class SourceStateEvidence:
    git_commit: str | None
    git_dirty: bool | None
    source_state_sha256: str
    source_state_algorithm: str
    source_state_scope: str
    source_state_file_count: int
    evidence_maturity: EvidenceMaturity

    def as_metadata(self) -> dict[str, str | int | bool | None]:
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "source_state_sha256": self.source_state_sha256,
            "source_state_algorithm": self.source_state_algorithm,
            "source_state_scope": self.source_state_scope,
            "source_state_file_count": self.source_state_file_count,
            "evidence_maturity": self.evidence_maturity.value,
        }


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    if value != normalized:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


def _validate_git_commit(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        value != value.strip().lower()
        or len(value) < 7
        or len(value) > 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_commit must be a lowercase hexadecimal revision")
    return value


def classify_evidence_maturity(
    *,
    git_commit: str | None,
    git_dirty: bool | None,
) -> EvidenceMaturity:
    """Never elevate unknown or dirty Git state to release-candidate evidence."""

    commit = _validate_git_commit(git_commit)
    if commit is not None and git_dirty is False:
        return EvidenceMaturity.RELEASE_CANDIDATE
    return EvidenceMaturity.DEVELOPMENT_SOURCE_BOUND


def validate_source_state_binding(
    *,
    git_commit: str | None,
    git_dirty: bool | None,
    source_state_sha256: str,
    source_state_algorithm: str,
    source_state_scope: str,
    source_state_file_count: int,
    evidence_maturity: EvidenceMaturity | str,
) -> None:
    """Validate provenance fields embedded in a schema-backed evidence artifact."""

    _validate_git_commit(git_commit)
    _validate_sha256(source_state_sha256, field_name="source_state_sha256")
    if source_state_algorithm != SOURCE_STATE_ALGORITHM:
        raise ValueError("source_state_algorithm does not match the supported algorithm")
    if source_state_scope != SOURCE_STATE_SCOPE:
        raise ValueError("source_state_scope does not match the supported scope")
    if source_state_file_count <= 0:
        raise ValueError("source_state_file_count must be positive")
    try:
        maturity = EvidenceMaturity(evidence_maturity)
    except ValueError as exc:
        raise ValueError("evidence_maturity is not supported") from exc
    expected = classify_evidence_maturity(git_commit=git_commit, git_dirty=git_dirty)
    if maturity is not expected:
        raise ValueError("evidence_maturity does not match git_commit/git_dirty")


def source_state_evidence_from_metadata(
    metadata: Mapping[str, object],
) -> SourceStateEvidence:
    """Strictly parse the seven provenance fields from an existing artifact."""

    missing = sorted(SOURCE_STATE_METADATA_FIELDS - set(metadata))
    if missing:
        raise ValueError(f"source-state metadata is incomplete: missing={missing}")
    git_commit = metadata["git_commit"]
    git_dirty = metadata["git_dirty"]
    source_state_sha256 = metadata["source_state_sha256"]
    source_state_algorithm = metadata["source_state_algorithm"]
    source_state_scope = metadata["source_state_scope"]
    source_state_file_count = metadata["source_state_file_count"]
    evidence_maturity = metadata["evidence_maturity"]
    if git_commit is not None and not isinstance(git_commit, str):
        raise ValueError("git_commit must be a string or null")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise ValueError("git_dirty must be a boolean or null")
    if not isinstance(source_state_sha256, str):
        raise ValueError("source_state_sha256 must be a string")
    if not isinstance(source_state_algorithm, str):
        raise ValueError("source_state_algorithm must be a string")
    if not isinstance(source_state_scope, str):
        raise ValueError("source_state_scope must be a string")
    if isinstance(source_state_file_count, bool) or not isinstance(source_state_file_count, int):
        raise ValueError("source_state_file_count must be an integer")
    if not isinstance(evidence_maturity, str):
        raise ValueError("evidence_maturity must be a string")
    evidence = SourceStateEvidence(
        git_commit=git_commit,
        git_dirty=git_dirty,
        source_state_sha256=source_state_sha256,
        source_state_algorithm=source_state_algorithm,
        source_state_scope=source_state_scope,
        source_state_file_count=source_state_file_count,
        evidence_maturity=EvidenceMaturity(evidence_maturity),
    )
    validate_source_state_binding(**evidence.as_metadata())
    return evidence


def _included_paths(project_root: Path) -> tuple[Path, ...]:
    root = project_root.resolve()
    if not root.is_dir():
        raise ValueError(f"project root is not a directory: {root}")

    included: list[Path] = []
    for relative_name in SOURCE_STATE_INCLUDED_FILES:
        path = root / relative_name
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"required source-state file is missing or not regular: {relative_name}"
            )
        included.append(path)

    for relative_name in SOURCE_STATE_INCLUDED_TREES:
        tree = root / Path(*relative_name.split("/"))
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError(
                f"required source-state tree is missing or not regular: {relative_name}"
            )
        for current, directory_names, file_names in os.walk(tree, topdown=True, followlinks=False):
            current_path = Path(current)
            retained_directories: list[str] = []
            for directory_name in sorted(directory_names):
                child = current_path / directory_name
                if directory_name in SOURCE_STATE_EXCLUDED_DIRECTORY_NAMES:
                    continue
                if child.is_symlink():
                    raise ValueError(
                        "source-state scope contains a directory symlink: "
                        f"{child.relative_to(root).as_posix()}"
                    )
                retained_directories.append(directory_name)
            directory_names[:] = retained_directories

            for file_name in sorted(file_names):
                path = current_path / file_name
                if file_name in SOURCE_STATE_EXCLUDED_FILE_NAMES:
                    continue
                if path.suffix.lower() in SOURCE_STATE_EXCLUDED_SUFFIXES:
                    continue
                # PDFs in data/synthetic are generated derivatives of the
                # committed textual sources and are intentionally excluded.
                if relative_name == "data/synthetic" and path.suffix.lower() == ".pdf":
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        "source-state scope contains a non-regular file: "
                        f"{path.relative_to(root).as_posix()}"
                    )
                included.append(path)

    relative_names = [path.relative_to(root).as_posix() for path in included]
    if len(set(relative_names)) != len(relative_names):
        raise ValueError("source-state scope contains duplicate paths")
    return tuple(path for _, path in sorted(zip(relative_names, included, strict=True)))


def compute_source_state(project_root: Path) -> SourceStateSnapshot:
    """Hash a canonical manifest of path, size, and content digest records."""

    root = project_root.resolve()
    entries: list[SourceStateEntry] = []
    for path in _included_paths(root):
        payload = path.read_bytes()
        entries.append(
            SourceStateEntry(
                path=path.relative_to(root).as_posix(),
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )

    canonical_records = [
        {
            "algorithm": SOURCE_STATE_ALGORITHM,
            "scope": SOURCE_STATE_SCOPE,
        },
        *[{"path": entry.path, "sha256": entry.sha256, "size": entry.size} for entry in entries],
    ]
    manifest = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in canonical_records
    ).encode("utf-8")
    return SourceStateSnapshot(
        algorithm=SOURCE_STATE_ALGORITHM,
        scope=SOURCE_STATE_SCOPE,
        sha256=hashlib.sha256(manifest).hexdigest(),
        file_count=len(entries),
        entries=tuple(entries),
    )


def verify_source_state(
    project_root: Path,
    *,
    expected_sha256: str,
    expected_file_count: int | None = None,
) -> SourceStateSnapshot:
    """Recompute the digest and fail closed on any mismatch."""

    expected = _validate_sha256(expected_sha256, field_name="expected_sha256")
    snapshot = compute_source_state(project_root)
    if snapshot.sha256 != expected:
        raise ValueError(
            f"source state SHA-256 mismatch: expected={expected}, actual={snapshot.sha256}"
        )
    if expected_file_count is not None and snapshot.file_count != expected_file_count:
        raise ValueError(
            "source state file count mismatch: "
            f"expected={expected_file_count}, actual={snapshot.file_count}"
        )
    return snapshot


def verify_captured_source_state(
    project_root: Path,
    evidence: SourceStateEvidence,
    *,
    strict_git: bool = False,
) -> SourceStateSnapshot:
    """Prove that source files and Git state did not change during a run."""

    validate_source_state_binding(**evidence.as_metadata())
    snapshot = verify_source_state(
        project_root,
        expected_sha256=evidence.source_state_sha256,
        expected_file_count=evidence.source_state_file_count,
    )
    current_git = discover_git_state(project_root, strict=strict_git)
    captured_git = (evidence.git_commit, evidence.git_dirty)
    if current_git != captured_git:
        raise ValueError(
            "git revision/dirty state changed after source-state capture: "
            f"captured={captured_git}, current={current_git}"
        )
    return snapshot


def discover_git_state(
    project_root: Path,
    *,
    strict: bool = False,
) -> tuple[str | None, bool | None]:
    """Measure the complete checkout state; optionally fail if Git is unavailable."""

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if strict:
            raise ValueError(f"unable to measure git revision/dirty state: {exc}") from exc
        return None, None
    if commit_result.returncode != 0 or status_result.returncode != 0:
        if strict:
            raise ValueError("unable to measure git revision/dirty state")
        return None, None
    commit = commit_result.stdout.strip().lower()
    try:
        validated_commit = _validate_git_commit(commit)
    except ValueError:
        if strict:
            raise
        return None, None
    return validated_commit, bool(status_result.stdout.strip())


def build_source_state_evidence(
    project_root: Path,
    *,
    git_commit: str | None,
    git_dirty: bool | None,
) -> SourceStateEvidence:
    snapshot = compute_source_state(project_root)
    maturity = classify_evidence_maturity(git_commit=git_commit, git_dirty=git_dirty)
    evidence = SourceStateEvidence(
        git_commit=_validate_git_commit(git_commit),
        git_dirty=git_dirty,
        source_state_sha256=snapshot.sha256,
        source_state_algorithm=snapshot.algorithm,
        source_state_scope=snapshot.scope,
        source_state_file_count=snapshot.file_count,
        evidence_maturity=maturity,
    )
    validate_source_state_binding(**evidence.as_metadata())
    return evidence


def capture_source_state_evidence(
    project_root: Path,
    *,
    strict_git: bool = False,
) -> SourceStateEvidence:
    git_commit, git_dirty = discover_git_state(project_root, strict=strict_git)
    return build_source_state_evidence(
        project_root,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
