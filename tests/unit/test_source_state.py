"""Deterministic source-state evidence binding tests."""

import json
from pathlib import Path

import pytest
from scripts.verify_source_state import main as verify_cli_main

from creditlens.evaluation.source_state import (
    SOURCE_STATE_ALGORITHM,
    SOURCE_STATE_INCLUDED_FILES,
    SOURCE_STATE_INCLUDED_TREES,
    SOURCE_STATE_SCOPE,
    EvidenceMaturity,
    build_source_state_evidence,
    compute_source_state,
    source_state_evidence_from_metadata,
    validate_source_state_binding,
    verify_source_state,
)


def _project(root: Path) -> Path:
    for relative_name in SOURCE_STATE_INCLUDED_FILES:
        path = root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"top:{relative_name}\n", encoding="utf-8")
    for relative_name in SOURCE_STATE_INCLUDED_TREES:
        path = root / Path(*relative_name.split("/")) / "covered.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"tree:{relative_name}\n", encoding="utf-8")
    return root


def test_source_state_is_deterministic_and_path_sensitive(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")

    first = compute_source_state(project)
    second = compute_source_state(project)
    assert first == second

    covered = project / "src" / "covered.txt"
    covered.rename(project / "src" / "renamed.txt")
    renamed = compute_source_state(project)
    assert renamed.sha256 != first.sha256
    assert renamed.file_count == first.file_count


def test_generated_outputs_caches_and_synthetic_pdfs_are_excluded(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    expected = compute_source_state(project)

    generated = {
        project / "evaluation" / "reports" / "local" / "report.json": "report",
        project / "evaluation" / "predictions" / "local" / "prediction.json": "prediction",
        project / "src" / "__pycache__" / "module.pyc": "cache",
        project / "data" / "synthetic" / "generated.pdf": "derived",
        project / ".venv" / "ignored.txt": "venv",
    }
    for path, content in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert compute_source_state(project) == expected


def test_in_scope_content_change_fails_verification(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    expected = compute_source_state(project)
    (project / "config" / "covered.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source state SHA-256 mismatch"):
        verify_source_state(
            project,
            expected_sha256=expected.sha256,
            expected_file_count=expected.file_count,
        )


@pytest.mark.parametrize(
    ("git_commit", "git_dirty", "expected"),
    [
        ("a" * 40, False, EvidenceMaturity.RELEASE_CANDIDATE),
        ("a" * 40, True, EvidenceMaturity.DEVELOPMENT_SOURCE_BOUND),
        (None, None, EvidenceMaturity.DEVELOPMENT_SOURCE_BOUND),
    ],
)
def test_evidence_maturity_never_promotes_dirty_or_unknown_git_state(
    tmp_path: Path,
    git_commit: str | None,
    git_dirty: bool | None,
    expected: EvidenceMaturity,
) -> None:
    evidence = build_source_state_evidence(
        _project(tmp_path / "project"),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    assert evidence.evidence_maturity is expected
    validate_source_state_binding(**evidence.as_metadata())


def test_maturity_tampering_is_rejected(tmp_path: Path) -> None:
    evidence = build_source_state_evidence(
        _project(tmp_path / "project"),
        git_commit="b" * 40,
        git_dirty=True,
    )
    payload = evidence.as_metadata()
    payload["evidence_maturity"] = EvidenceMaturity.RELEASE_CANDIDATE.value

    with pytest.raises(ValueError, match="does not match git_commit/git_dirty"):
        validate_source_state_binding(**payload)  # type: ignore[arg-type]


def test_existing_artifact_metadata_requires_complete_typed_binding(tmp_path: Path) -> None:
    evidence = build_source_state_evidence(
        _project(tmp_path / "project"),
        git_commit="d" * 40,
        git_dirty=True,
    )
    payload = evidence.as_metadata()
    assert source_state_evidence_from_metadata(payload) == evidence

    incomplete = dict(payload)
    incomplete.pop("source_state_scope")
    with pytest.raises(ValueError, match="metadata is incomplete"):
        source_state_evidence_from_metadata(incomplete)

    wrong_type = dict(payload)
    wrong_type["source_state_file_count"] = True
    with pytest.raises(ValueError, match="file_count must be an integer"):
        source_state_evidence_from_metadata(wrong_type)


def test_artifact_cli_recomputes_and_verifies_binding(
    tmp_path: Path,
    capsys,
) -> None:
    project = _project(tmp_path / "project")
    evidence = build_source_state_evidence(
        project,
        git_commit="c" * 40,
        git_dirty=True,
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"metadata": evidence.as_metadata()}),
        encoding="utf-8",
    )

    assert verify_cli_main(["--project-root", str(project), "--artifact", str(artifact)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "source_state_algorithm": SOURCE_STATE_ALGORITHM,
        "source_state_file_count": evidence.source_state_file_count,
        "source_state_scope": SOURCE_STATE_SCOPE,
        "source_state_sha256": evidence.source_state_sha256,
        "verified": True,
    }
