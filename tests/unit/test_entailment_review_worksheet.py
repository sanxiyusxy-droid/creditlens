from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import scripts.build_entailment_review_worksheet as review_worksheet_cli
import scripts.compile_entailment_submission as review_submission_cli
from pydantic import ValidationError
from scripts.build_entailment_review_worksheet import main as build_worksheet_main
from scripts.compile_entailment_submission import main as compile_submission_main

from creditlens.evaluation.semantic_entailment import (
    ReviewDecision,
    ReviewerAttestationV1,
    ReviewSubmission,
    ReviewWorksheet,
    SemanticEntailmentSource,
    SourceClaim,
    SourceEvidence,
    SourceInputArtifact,
    build_blind_review_package,
    build_review_worksheet,
    compile_review_submission,
    serialize_json_model,
    sha256_bytes,
)

NOW = datetime(2026, 8, 16, 9, 30, tzinfo=UTC)
REVIEWER_ID = "rvw_H7Lz4Jq9mN2xP8sK5cT1vW6y"
OTHER_REVIEWER_ID = "rvw_Q3bR8uD2nF7kM5pX9sL4zC6a"


def _digest(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _symlink_or_skip(link, target) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this test environment: {exc}")


@pytest.mark.parametrize("module", [review_worksheet_cli, review_submission_cli])
def test_review_publish_rechecks_output_reparse(tmp_path, monkeypatch, module):
    output = tmp_path / f"{module.__name__.rsplit('.', 1)[-1]}.json"
    checks = 0

    def reject_on_publish(_path):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("simulated output reparse race")

    monkeypatch.setattr(module, "_reject_output_reparse", reject_on_publish)
    with pytest.raises(ValueError, match="reparse race"):
        module._atomic_write(output, b"new artifact", overwrite=True)

    assert checks == 2
    assert not output.exists()


def _package():
    source = SemanticEntailmentSource(
        source_id="source-worksheet-test",
        prediction_set_id="predictions-worksheet-test",
        prediction_sha256="a" * 64,
        created_at=NOW,
        input_artifacts=[
            SourceInputArtifact(
                role="prediction_set", artifact_id="predictions.json", sha256="a" * 64
            ),
            SourceInputArtifact(role="raw_checkpoint", artifact_id="raw.json", sha256="b" * 64),
        ],
        items=[
            SourceClaim(
                question_id=f"q-{ordinal}",
                claim_id=f"claim-{ordinal}",
                claim=f"Blind claim {ordinal}",
                evidence=[
                    SourceEvidence(
                        section_id=f"section-{ordinal}",
                        content=f"Frozen evidence {ordinal}",
                        content_sha256=_digest(f"Frozen evidence {ordinal}"),
                    )
                ],
            )
            for ordinal in range(1, 4)
        ],
    )
    source_sha256 = sha256_bytes(serialize_json_model(source))
    package, _mapping = build_blind_review_package(
        source,
        source_sha256=source_sha256,
        reviewer_id=REVIEWER_ID,
        ordering_seed="sealed-worksheet-seed",
        generated_at=NOW,
    )
    package_bytes = serialize_json_model(package)
    return package, package_bytes, sha256_bytes(package_bytes)


def _blank_worksheet():
    package, package_bytes, package_sha256 = _package()
    worksheet = build_review_worksheet(
        package,
        package_sha256=package_sha256,
        reviewer_id=REVIEWER_ID,
        created_at=NOW,
    )
    return package, package_bytes, package_sha256, worksheet


def _filled_worksheet(*, decision: ReviewDecision = ReviewDecision.ENTAILED) -> ReviewWorksheet:
    _package_model, _package_bytes, _package_sha256, worksheet = _blank_worksheet()
    payload = worksheet.model_dump(mode="json")
    for item in payload["items"]:
        item["decision"] = decision.value
        item["rationale"] = (
            "Required exception rationale."
            if decision in {ReviewDecision.SKIP, ReviewDecision.CONFLICT}
            else ""
        )
    return ReviewWorksheet.model_validate(payload)


def _attestation(*, attested_at: datetime = NOW) -> ReviewerAttestationV1:
    return ReviewerAttestationV1(
        actor_type="HUMAN",
        model_assistance_used=False,
        accessed_repo_or_gold=False,
        accessed_source_or_mapping=False,
        coordinated_with_other_reviewers=False,
        attested_at=attested_at,
    )


def _compile_args(package_path, worksheet_path, output_path):
    return [
        "--package",
        str(package_path),
        "--worksheet",
        str(worksheet_path),
        "--reviewer-id",
        REVIEWER_ID,
        "--submitted-at",
        NOW.isoformat(),
        "--attest-human",
        "--attest-no-model-assistance",
        "--attest-no-repo-or-gold-access",
        "--attest-no-source-or-mapping-access",
        "--attest-no-reviewer-coordination",
        "--output",
        str(output_path),
    ]


def test_worksheet_to_submission_is_blind_bound_complete_and_reproducible():
    package, _package_bytes, package_sha256, worksheet = _blank_worksheet()
    worksheet_json = worksheet.model_dump_json()

    assert worksheet.package_id == package.package_id
    assert worksheet.package_sha256 == package_sha256
    assert worksheet.reviewer_id_sha256 == _digest(REVIEWER_ID)
    assert [item.ordinal for item in worksheet.items] == [1, 2, 3]
    assert all(item.decision is None for item in worksheet.items)
    for forbidden in (REVIEWER_ID, "question_id", "claim_id", "gold", "source_sha256"):
        assert forbidden not in worksheet_json.lower()

    filled = _filled_worksheet()
    first = compile_review_submission(
        package,
        filled,
        package_sha256=package_sha256,
        reviewer_id=REVIEWER_ID,
        reviewer_attestation=_attestation(),
        submitted_at=NOW,
    )
    repeated = compile_review_submission(
        package,
        filled,
        package_sha256=package_sha256,
        reviewer_id=REVIEWER_ID,
        reviewer_attestation=_attestation(),
        submitted_at=NOW,
    )

    assert serialize_json_model(first) == serialize_json_model(repeated)
    assert len(first.reviews) == len(package.items)
    assert first.reviewer_attestation.actor_type == "HUMAN"
    assert first.reviewer_attestation.model_assistance_used is False


@pytest.mark.parametrize("mutation", ["missing", "unknown", "ordinal"])
def test_compiler_rejects_incomplete_unknown_or_ordinal_drift(mutation):
    package, _package_bytes, package_sha256, _worksheet = _blank_worksheet()
    payload = _filled_worksheet().model_dump(mode="json")
    if mutation == "missing":
        payload["items"].pop()
    elif mutation == "unknown":
        payload["items"][0]["blind_item_id"] = "item-unknown"
    else:
        first_id = payload["items"][0]["blind_item_id"]
        payload["items"][0]["blind_item_id"] = payload["items"][1]["blind_item_id"]
        payload["items"][1]["blind_item_id"] = first_id
    worksheet = ReviewWorksheet.model_validate(payload)

    with pytest.raises(ValueError, match=r"unknown|cover every|ordinal"):
        compile_review_submission(
            package,
            worksheet,
            package_sha256=package_sha256,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )


def test_worksheet_schema_rejects_duplicate_ids_and_null_decisions_do_not_compile():
    package, _package_bytes, package_sha256, worksheet = _blank_worksheet()
    duplicate = worksheet.model_dump(mode="json")
    duplicate["items"][1]["blind_item_id"] = duplicate["items"][0]["blind_item_id"]
    with pytest.raises(ValidationError, match="must be unique"):
        ReviewWorksheet.model_validate(duplicate)

    with pytest.raises(ValueError, match="non-null"):
        compile_review_submission(
            package,
            worksheet,
            package_sha256=package_sha256,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )


@pytest.mark.parametrize("decision", [ReviewDecision.SKIP, ReviewDecision.CONFLICT])
def test_compiler_requires_exception_rationale(decision):
    package, _package_bytes, package_sha256, _worksheet = _blank_worksheet()
    payload = _filled_worksheet().model_dump(mode="json")
    payload["items"][0]["decision"] = decision.value
    payload["items"][0]["rationale"] = "  "
    worksheet = ReviewWorksheet.model_validate(payload)

    with pytest.raises(ValueError, match="require a rationale"):
        compile_review_submission(
            package,
            worksheet,
            package_sha256=package_sha256,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )


def test_compiler_rejects_package_hash_id_and_reviewer_identity_drift():
    package, _package_bytes, package_sha256, _worksheet = _blank_worksheet()
    filled = _filled_worksheet()

    drifted_id = filled.model_copy(update={"package_id": "another-package"})
    with pytest.raises(ValueError, match="package_id"):
        compile_review_submission(
            package,
            drifted_id,
            package_sha256=package_sha256,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )

    with pytest.raises(ValueError, match="package_sha256"):
        compile_review_submission(
            package,
            filled,
            package_sha256="f" * 64,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )

    with pytest.raises(ValueError, match="reviewer_id hash"):
        compile_review_submission(
            package,
            filled,
            package_sha256=package_sha256,
            reviewer_id=OTHER_REVIEWER_ID,
            reviewer_attestation=_attestation(),
            submitted_at=NOW,
        )


def test_reviewer_identity_requires_a_high_entropy_pseudonym():
    package, _package_bytes, package_sha256 = _package()
    with pytest.raises(ValueError, match=r"high-entropy pseudonym.*real name"):
        build_review_worksheet(
            package,
            package_sha256=package_sha256,
            reviewer_id="Alice Smith",
            created_at=NOW,
        )


def test_review_submission_model_rejects_low_entropy_pseudonym():
    package, _package_bytes, package_sha256, _worksheet = _blank_worksheet()
    submission = compile_review_submission(
        package,
        _filled_worksheet(),
        package_sha256=package_sha256,
        reviewer_id=REVIEWER_ID,
        reviewer_attestation=_attestation(),
        submitted_at=NOW,
    )
    payload = submission.model_dump(mode="json")
    payload["reviewer_id"] = "x"

    with pytest.raises(ValidationError, match="high-entropy pseudonym"):
        ReviewSubmission.model_validate(payload)


def test_attestation_is_strict_aware_and_temporally_valid():
    with pytest.raises(ValidationError, match="boolean false"):
        ReviewerAttestationV1.model_validate(
            {
                "actor_type": "HUMAN",
                "model_assistance_used": True,
                "accessed_repo_or_gold": False,
                "accessed_source_or_mapping": False,
                "coordinated_with_other_reviewers": False,
                "attested_at": NOW.isoformat(),
            }
        )
    with pytest.raises(ValidationError, match="HUMAN"):
        ReviewerAttestationV1.model_validate(
            {
                "actor_type": "MODEL",
                "model_assistance_used": False,
                "accessed_repo_or_gold": False,
                "accessed_source_or_mapping": False,
                "coordinated_with_other_reviewers": False,
                "attested_at": NOW.isoformat(),
            }
        )
    with pytest.raises(ValidationError, match="timezone"):
        ReviewerAttestationV1(
            actor_type="HUMAN",
            model_assistance_used=False,
            accessed_repo_or_gold=False,
            accessed_source_or_mapping=False,
            coordinated_with_other_reviewers=False,
            attested_at=datetime(2026, 8, 16, 9, 30),
        )

    package, _package_bytes, package_sha256, _worksheet = _blank_worksheet()
    with pytest.raises(ValueError, match="later than submitted_at"):
        compile_review_submission(
            package,
            _filled_worksheet(),
            package_sha256=package_sha256,
            reviewer_id=REVIEWER_ID,
            reviewer_attestation=_attestation(attested_at=NOW + timedelta(seconds=1)),
            submitted_at=NOW,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "actor_type",
        "model_assistance_used",
        "accessed_repo_or_gold",
        "accessed_source_or_mapping",
        "coordinated_with_other_reviewers",
        "attested_at",
    ],
)
def test_every_attestation_field_is_explicitly_required(missing_field):
    payload = {
        "actor_type": "HUMAN",
        "model_assistance_used": False,
        "accessed_repo_or_gold": False,
        "accessed_source_or_mapping": False,
        "coordinated_with_other_reviewers": False,
        "attested_at": NOW.isoformat(),
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError, match="Field required"):
        ReviewerAttestationV1.model_validate(payload)


def test_build_worksheet_cli_is_reproducible_no_clobber_and_alias_safe(tmp_path):
    _package_model, package_bytes, _package_sha256 = _package()
    package_path = tmp_path / "package.json"
    first_output = tmp_path / "worksheet-a.json"
    second_output = tmp_path / "worksheet-b.json"
    package_path.write_bytes(package_bytes)
    common = [
        "--package",
        str(package_path),
        "--reviewer-id",
        REVIEWER_ID,
        "--created-at",
        NOW.isoformat(),
    ]

    assert build_worksheet_main([*common, "--output", str(first_output)]) == 0
    assert build_worksheet_main([*common, "--output", str(second_output)]) == 0
    assert first_output.read_bytes() == second_output.read_bytes()
    original = first_output.read_bytes()
    with pytest.raises(SystemExit):
        build_worksheet_main([*common, "--output", str(first_output)])
    assert first_output.read_bytes() == original

    alias = tmp_path / "package-hard-link.json"
    alias.hardlink_to(package_path)
    with pytest.raises(SystemExit):
        build_worksheet_main([*common, "--output", str(alias), "--overwrite"])
    assert package_path.read_bytes() == package_bytes


def test_build_worksheet_cli_rejects_output_symlink_without_touching_victim(tmp_path, capsys):
    _package_model, package_bytes, _package_sha256 = _package()
    package_path = tmp_path / "package.json"
    victim_path = tmp_path / "unrelated-victim.json"
    output_link = tmp_path / "worksheet.json"
    package_path.write_bytes(package_bytes)
    victim_bytes = b"unrelated-victim-must-survive"
    victim_path.write_bytes(victim_bytes)
    _symlink_or_skip(output_link, victim_path)

    with pytest.raises(SystemExit):
        build_worksheet_main(
            [
                "--package",
                str(package_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--created-at",
                NOW.isoformat(),
                "--output",
                str(output_link),
                "--overwrite",
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert victim_path.read_bytes() == victim_bytes
    assert output_link.is_symlink()


def test_build_worksheet_cli_rejects_dangling_output_symlink(tmp_path, capsys):
    _package_model, package_bytes, _package_sha256 = _package()
    package_path = tmp_path / "package.json"
    missing_target = tmp_path / "must-not-be-created.json"
    output_link = tmp_path / "worksheet.json"
    package_path.write_bytes(package_bytes)
    _symlink_or_skip(output_link, missing_target)

    with pytest.raises(SystemExit):
        build_worksheet_main(
            [
                "--package",
                str(package_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--created-at",
                NOW.isoformat(),
                "--output",
                str(output_link),
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert output_link.is_symlink()
    assert not missing_target.exists()


def test_compile_cli_is_reproducible_strict_and_alias_safe(tmp_path):
    _package_model, package_bytes, _package_sha256, _worksheet = _blank_worksheet()
    package_path = tmp_path / "package.json"
    worksheet_path = tmp_path / "worksheet.json"
    first_output = tmp_path / "submission-a.json"
    second_output = tmp_path / "submission-b.json"
    package_path.write_bytes(package_bytes)
    worksheet_path.write_bytes(serialize_json_model(_filled_worksheet()))

    first_args = _compile_args(package_path, worksheet_path, first_output)
    assert compile_submission_main(first_args) == 0
    assert compile_submission_main(_compile_args(package_path, worksheet_path, second_output)) == 0
    assert first_output.read_bytes() == second_output.read_bytes()
    submission = ReviewSubmission.model_validate_json(first_output.read_bytes())
    assert submission.reviewer_attestation.attested_at == NOW
    assert submission.reviewer_attestation.actor_type == "HUMAN"
    assert submission.reviewer_attestation.model_assistance_used is False
    assert submission.reviewer_attestation.accessed_repo_or_gold is False
    assert submission.reviewer_attestation.accessed_source_or_mapping is False
    assert submission.reviewer_attestation.coordinated_with_other_reviewers is False

    attestation_flags = (
        "--attest-human",
        "--attest-no-model-assistance",
        "--attest-no-repo-or-gold-access",
        "--attest-no-source-or-mapping-access",
        "--attest-no-reviewer-coordination",
    )
    for index, omitted_flag in enumerate(attestation_flags, start=1):
        missing_attestation = [value for value in first_args if value != omitted_flag]
        output_index = missing_attestation.index("--output") + 1
        missing_attestation[output_index] = str(tmp_path / f"missing-attestation-{index}.json")
        with pytest.raises(SystemExit):
            compile_submission_main(missing_attestation)

    alias = tmp_path / "worksheet-hard-link.json"
    alias.hardlink_to(worksheet_path)
    with pytest.raises(SystemExit):
        compile_submission_main(
            [*_compile_args(package_path, worksheet_path, alias), "--overwrite"]
        )
    assert alias.read_bytes() == worksheet_path.read_bytes()


def test_compile_cli_rejects_output_symlink_without_touching_victim(tmp_path, capsys):
    _package_model, package_bytes, _package_sha256, _worksheet = _blank_worksheet()
    package_path = tmp_path / "package.json"
    worksheet_path = tmp_path / "worksheet.json"
    victim_path = tmp_path / "unrelated-victim.json"
    output_link = tmp_path / "submission.json"
    package_path.write_bytes(package_bytes)
    worksheet_path.write_bytes(serialize_json_model(_filled_worksheet()))
    victim_bytes = b"unrelated-victim-must-survive"
    victim_path.write_bytes(victim_bytes)
    _symlink_or_skip(output_link, victim_path)

    with pytest.raises(SystemExit):
        compile_submission_main(
            [*_compile_args(package_path, worksheet_path, output_link), "--overwrite"]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert victim_path.read_bytes() == victim_bytes
    assert output_link.is_symlink()


def test_compile_cli_rejects_dangling_output_symlink(tmp_path, capsys):
    _package_model, package_bytes, _package_sha256, _worksheet = _blank_worksheet()
    package_path = tmp_path / "package.json"
    worksheet_path = tmp_path / "worksheet.json"
    missing_target = tmp_path / "must-not-be-created.json"
    output_link = tmp_path / "submission.json"
    package_path.write_bytes(package_bytes)
    worksheet_path.write_bytes(serialize_json_model(_filled_worksheet()))
    _symlink_or_skip(output_link, missing_target)

    with pytest.raises(SystemExit):
        compile_submission_main(_compile_args(package_path, worksheet_path, output_link))

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert output_link.is_symlink()
    assert not missing_target.exists()


def test_cli_rejects_naive_reproducibility_timestamps(tmp_path):
    _package_model, package_bytes, _package_sha256 = _package()
    package_path = tmp_path / "package.json"
    package_path.write_bytes(package_bytes)
    with pytest.raises(SystemExit):
        build_worksheet_main(
            [
                "--package",
                str(package_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--created-at",
                "2026-08-16T09:30:00",
                "--output",
                str(tmp_path / "worksheet.json"),
            ]
        )
