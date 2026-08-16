from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.build_entailment_adjudication_package import main as build_package_main
from scripts.build_entailment_adjudication_worksheet import main as build_worksheet_main
from scripts.compile_entailment_adjudication_submission import main as compile_submission_main
from scripts.score_entailment_review import main as score_reviews_main

from creditlens.evaluation.semantic_entailment import (
    AdjudicationDecision,
    AdjudicationWorksheet,
    AdjudicatorAttestationV1,
    BlindReview,
    FormalCompletionBlockerCode,
    FormalCompletionStatus,
    ReviewDecision,
    ReviewerAttestationV1,
    ReviewSubmission,
    SemanticEntailmentSource,
    SourceClaim,
    SourceEvidence,
    SourceInputArtifact,
    build_adjudication_worksheet,
    build_blind_adjudication_package,
    build_blind_review_package,
    compile_adjudication_submission,
    evaluate_entailment_reviews,
    serialize_json_model,
    sha256_bytes,
)

NOW = datetime(2026, 8, 16, 10, 0, tzinfo=UTC)
REVIEWER_A = "rvw_H7Lz4Jq9mN2xP8sK5cT1vW6y"
REVIEWER_B = "rvw_Q3bR8uD2nF7kM5pX9sL4zC6a"
ADJUDICATOR = "adj_R4tY8uI2oP6aS9dF3gH7jK1z"


def _digest(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _reviewer_attestation(at: datetime = NOW) -> ReviewerAttestationV1:
    return ReviewerAttestationV1(
        actor_type="HUMAN",
        model_assistance_used=False,
        accessed_repo_or_gold=False,
        accessed_source_or_mapping=False,
        coordinated_with_other_reviewers=False,
        attested_at=at,
    )


def _adjudicator_attestation(at: datetime = NOW) -> AdjudicatorAttestationV1:
    return AdjudicatorAttestationV1(
        actor_type="HUMAN",
        model_assistance_used=False,
        accessed_repository=False,
        accessed_gold=False,
        accessed_source_or_private_mapping=False,
        accessed_raw_submissions_or_reviewer_identity=False,
        attested_at=at,
    )


def _source(claim_count: int = 1) -> tuple[SemanticEntailmentSource, str]:
    claims: list[SourceClaim] = []
    for ordinal in range(1, claim_count + 1):
        content = f"Frozen evidence statement {ordinal}."
        claims.append(
            SourceClaim(
                question_id=f"private-question-{ordinal}",
                claim_id=f"private-claim-{ordinal}",
                claim=f"Public semantic statement {ordinal}.",
                evidence=[
                    SourceEvidence(
                        section_id=f"private-section-{ordinal}",
                        content=content,
                        content_sha256=_digest(content),
                    )
                ],
            )
        )
    source = SemanticEntailmentSource(
        source_id="private-source",
        prediction_set_id="prediction-set",
        prediction_sha256="a" * 64,
        created_at=NOW,
        input_artifacts=[
            SourceInputArtifact(
                role="prediction_set", artifact_id="predictions.json", sha256="a" * 64
            ),
            SourceInputArtifact(role="raw_checkpoint", artifact_id="raw.json", sha256="b" * 64),
        ],
        items=claims,
    )
    return source, sha256_bytes(serialize_json_model(source))


def _review_artifacts(
    source: SemanticEntailmentSource,
    source_hash: str,
    reviewer_id: str,
    seed: str,
    decisions_by_key: dict[tuple[str, str], ReviewDecision],
):
    package, mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id=reviewer_id,
        ordering_seed=seed,
        generated_at=NOW,
    )
    package_hash = sha256_bytes(serialize_json_model(package))
    reviews = []
    for package_item, map_entry in zip(package.items, mapping.entries, strict=True):
        decision = decisions_by_key[map_entry.key]
        rationale = (
            "Reviewer could not assign a semantic label."
            if decision in {ReviewDecision.SKIP, ReviewDecision.CONFLICT}
            else ""
        )
        reviews.append(
            BlindReview(
                blind_item_id=package_item.blind_item_id,
                decision=decision,
                rationale=rationale,
            )
        )
    submission = ReviewSubmission(
        submission_id=f"submission-{_digest(reviewer_id)[:16]}",
        package_id=package.package_id,
        package_sha256=package_hash,
        reviewer_id=reviewer_id,
        submitted_at=NOW,
        reviewer_attestation=_reviewer_attestation(),
        reviews=reviews,
    )
    return (
        (package, package_hash),
        (mapping, sha256_bytes(serialize_json_model(mapping))),
        (submission, sha256_bytes(serialize_json_model(submission))),
    )


def _two_reviews(
    source: SemanticEntailmentSource,
    source_hash: str,
    first: dict[tuple[str, str], ReviewDecision],
    second: dict[tuple[str, str], ReviewDecision],
):
    left = _review_artifacts(source, source_hash, REVIEWER_A, "review-seed-a", first)
    right = _review_artifacts(source, source_hash, REVIEWER_B, "review-seed-b", second)
    return (
        [left[0], right[0]],
        [left[1], right[1]],
        [left[2], right[2]],
    )


def _adjudication_artifacts(source, source_hash, packages, mappings, submissions, decisions):
    package, mapping = build_blind_adjudication_package(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        adjudicator_id=ADJUDICATOR,
        ordering_seed="sealed-adjudication-seed",
        generated_at=NOW,
    )
    package_hash = sha256_bytes(serialize_json_model(package))
    worksheet = build_adjudication_worksheet(
        package,
        package_sha256=package_hash,
        adjudicator_id=ADJUDICATOR,
        created_at=NOW,
    )
    payload = worksheet.model_dump(mode="json")
    for item in payload["items"]:
        item["decision"] = decisions[item["blind_dispute_id"]].value
        item["rationale"] = "Independent human resolution from frozen evidence."
    worksheet = AdjudicationWorksheet.model_validate(payload)
    submission = compile_adjudication_submission(
        package,
        worksheet,
        package_sha256=package_hash,
        adjudicator_id=ADJUDICATOR,
        adjudicator_attestation=_adjudicator_attestation(),
        submitted_at=NOW,
    )
    return (
        (package, package_hash),
        (mapping, sha256_bytes(serialize_json_model(mapping))),
        (submission, sha256_bytes(serialize_json_model(submission))),
    )


@pytest.mark.parametrize(
    ("left", "right", "needs_adjudication"),
    [
        (ReviewDecision.ENTAILED, ReviewDecision.ENTAILED, False),
        (ReviewDecision.ENTAILED, ReviewDecision.CONTRADICTED, True),
        (ReviewDecision.ENTAILED, ReviewDecision.SKIP, True),
        (ReviewDecision.SKIP, ReviewDecision.SKIP, True),
        (ReviewDecision.CONFLICT, ReviewDecision.ENTAILED, True),
    ],
)
def test_exactly_two_reviews_require_strict_semantic_consensus(left, right, needs_adjudication):
    source, source_hash = _source()
    key = source.items[0].key
    packages, mappings, submissions = _two_reviews(source, source_hash, {key: left}, {key: right})
    report = evaluate_entailment_reviews(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        generated_at=NOW,
    )

    assert report.claims[0].adjudication_needed is needs_adjudication
    assert (report.claims[0].consensus_label is not None) is (not needs_adjudication)
    expected_status = (
        FormalCompletionStatus.INCOMPLETE if needs_adjudication else FormalCompletionStatus.COMPLETE
    )
    assert report.formal_completion.status is expected_status


def test_one_review_is_incomplete_but_not_a_dispute():
    source, source_hash = _source()
    key = source.items[0].key
    package, mapping, submission = _review_artifacts(
        source, source_hash, REVIEWER_A, "one-review", {key: ReviewDecision.ENTAILED}
    )
    report = evaluate_entailment_reviews(
        source,
        source_sha256=source_hash,
        packages=[package],
        mappings=[mapping],
        submissions=[submission],
        generated_at=NOW,
    )

    assert report.claims[0].adjudication_needed is False
    assert report.formal_completion.status is FormalCompletionStatus.INCOMPLETE
    codes = {item.code for item in report.formal_completion.blockers}
    assert FormalCompletionBlockerCode.REVIEWER_COUNT_NOT_2 in codes
    assert FormalCompletionBlockerCode.PER_CLAIM_REVIEW_COVERAGE_INCOMPLETE in codes
    assert FormalCompletionBlockerCode.ADJUDICATION_INCOMPLETE not in codes


def test_blind_dispute_package_private_mapping_and_final_claim_aggregate():
    source, source_hash = _source(claim_count=3)
    first = {
        source.items[0].key: ReviewDecision.ENTAILED,
        source.items[1].key: ReviewDecision.ENTAILED,
        source.items[2].key: ReviewDecision.SKIP,
    }
    second = {
        source.items[0].key: ReviewDecision.ENTAILED,
        source.items[1].key: ReviewDecision.CONTRADICTED,
        source.items[2].key: ReviewDecision.ENTAILED,
    }
    packages, mappings, submissions = _two_reviews(source, source_hash, first, second)
    package, mapping = build_blind_adjudication_package(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        adjudicator_id=ADJUDICATOR,
        ordering_seed="sealed-adjudication-seed",
        generated_at=NOW,
    )
    repeated, repeated_mapping = build_blind_adjudication_package(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        adjudicator_id=ADJUDICATOR,
        ordering_seed="sealed-adjudication-seed",
        generated_at=NOW,
    )
    assert package == repeated
    assert mapping == repeated_mapping
    assert len(package.items) == 2

    public = package.model_dump(mode="json")
    public_text = json.dumps(public, ensure_ascii=False)
    forbidden_keys = {"question_id", "claim_id", "reviewer_id", "gold"}

    def assert_private_keys_absent(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for child in value.values():
                assert_private_keys_absent(child)
        elif isinstance(value, list):
            for child in value:
                assert_private_keys_absent(child)

    assert_private_keys_absent(public)
    for forbidden_value in (
        REVIEWER_A,
        REVIEWER_B,
        *[item.question_id for item in source.items],
        *[item.claim_id for item in source.items],
    ):
        assert forbidden_value not in public_text
    assert {entry.key for entry in mapping.entries} == {
        source.items[1].key,
        source.items[2].key,
    }

    decisions = {}
    for entry in mapping.entries:
        decisions[entry.blind_dispute_id] = (
            AdjudicationDecision.CONTRADICTED
            if entry.key == source.items[1].key
            else AdjudicationDecision.EXCLUDE
        )
    adjudication = _adjudication_artifacts(
        source, source_hash, packages, mappings, submissions, decisions
    )
    report = evaluate_entailment_reviews(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        adjudication_package=adjudication[0],
        adjudication_mapping=adjudication[1],
        adjudication_submission=adjudication[2],
        generated_at=NOW,
    )

    assert report.formal_completion.status is FormalCompletionStatus.COMPLETE
    assert report.formal_completion.blockers == []
    assert report.formal_completion.expected_claims == 3
    assert report.formal_completion.expected_ratings == 6
    assert report.summary.final_labeled_claims == 2
    assert report.summary.final_label_counts.entailed == 1
    assert report.summary.final_label_counts.contradicted == 1
    assert report.summary.final_label_counts.not_enough_info == 0
    assert report.summary.final_label_rates.entailed == pytest.approx(0.5)
    assert report.summary.final_label_rates.contradicted == pytest.approx(0.5)
    assert report.summary.excluded_claims == 1


def test_adjudication_bundle_hash_rejects_stale_review_material():
    source, source_hash = _source()
    key = source.items[0].key
    packages, mappings, submissions = _two_reviews(
        source,
        source_hash,
        {key: ReviewDecision.ENTAILED},
        {key: ReviewDecision.CONTRADICTED},
    )
    package, mapping = build_blind_adjudication_package(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=submissions,
        adjudicator_id=ADJUDICATOR,
        ordering_seed="sealed-adjudication-seed",
        generated_at=NOW,
    )
    decisions = {package.items[0].blind_dispute_id: AdjudicationDecision.ENTAILED}
    adjudication = _adjudication_artifacts(
        source, source_hash, packages, mappings, submissions, decisions
    )
    stale_submissions = [submissions[0], (submissions[1][0], "f" * 64)]

    with pytest.raises(ValueError, match="stale"):
        evaluate_entailment_reviews(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=stale_submissions,
            adjudication_package=adjudication[0],
            adjudication_mapping=adjudication[1],
            adjudication_submission=adjudication[2],
            generated_at=NOW,
        )
    assert mapping.review_bundle_sha256 == adjudication[1][0].review_bundle_sha256


def test_attestation_fields_are_required_and_time_chain_is_enforced():
    with pytest.raises(ValidationError, match="Field required"):
        ReviewerAttestationV1(attested_at=NOW)
    with pytest.raises(ValidationError, match="Field required"):
        AdjudicatorAttestationV1(attested_at=NOW)

    source, source_hash = _source()
    package, _mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id=REVIEWER_A,
        ordering_seed="time-chain",
        generated_at=NOW,
    )
    package_hash = sha256_bytes(serialize_json_model(package))
    from creditlens.evaluation.semantic_entailment import (  # local to keep imports focused
        ReviewWorksheet,
        build_review_worksheet,
        compile_review_submission,
    )

    worksheet = build_review_worksheet(
        package,
        package_sha256=package_hash,
        reviewer_id=REVIEWER_A,
        created_at=NOW + timedelta(seconds=1),
    )
    payload = worksheet.model_dump(mode="json")
    payload["items"][0]["decision"] = ReviewDecision.ENTAILED.value
    filled = ReviewWorksheet.model_validate(payload)
    with pytest.raises(ValueError, match="completed worksheet"):
        compile_review_submission(
            package,
            filled,
            package_sha256=package_hash,
            reviewer_id=REVIEWER_A,
            reviewer_attestation=_reviewer_attestation(NOW),
            submitted_at=NOW + timedelta(seconds=2),
        )


def test_adjudication_builder_rejects_timestamp_before_review_submissions():
    source, source_hash = _source()
    key = source.items[0].key
    packages, mappings, submissions = _two_reviews(
        source,
        source_hash,
        {key: ReviewDecision.ENTAILED},
        {key: ReviewDecision.CONTRADICTED},
    )

    with pytest.raises(ValueError, match=r"cannot predate.*review submissions"):
        build_blind_adjudication_package(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=submissions,
            adjudicator_id=ADJUDICATOR,
            ordering_seed="time-chain-regression",
            generated_at=NOW - timedelta(seconds=1),
        )


def test_scorer_defensively_rejects_adjudication_predating_review_submissions():
    source, source_hash = _source()
    key = source.items[0].key
    packages, mappings, submissions = _two_reviews(
        source,
        source_hash,
        {key: ReviewDecision.ENTAILED},
        {key: ReviewDecision.CONTRADICTED},
    )
    later_review_time = NOW + timedelta(seconds=1)
    updated_submissions = []
    for submission, _digest_value in submissions:
        payload = submission.model_dump(mode="json")
        payload["submitted_at"] = later_review_time.isoformat()
        payload["reviewer_attestation"]["attested_at"] = later_review_time.isoformat()
        updated = ReviewSubmission.model_validate(payload)
        updated_submissions.append((updated, sha256_bytes(serialize_json_model(updated))))

    valid_package, valid_mapping = build_blind_adjudication_package(
        source,
        source_sha256=source_hash,
        packages=packages,
        mappings=mappings,
        submissions=updated_submissions,
        adjudicator_id=ADJUDICATOR,
        ordering_seed="scorer-time-chain-regression",
        generated_at=NOW + timedelta(seconds=2),
    )
    valid_package_hash = sha256_bytes(serialize_json_model(valid_package))
    worksheet = build_adjudication_worksheet(
        valid_package,
        package_sha256=valid_package_hash,
        adjudicator_id=ADJUDICATOR,
        created_at=NOW + timedelta(seconds=2),
    )
    worksheet_payload = worksheet.model_dump(mode="json")
    worksheet_payload["items"][0]["decision"] = AdjudicationDecision.ENTAILED.value
    worksheet_payload["items"][0]["rationale"] = "Independent human resolution."
    filled = AdjudicationWorksheet.model_validate(worksheet_payload)
    valid_submission = compile_adjudication_submission(
        valid_package,
        filled,
        package_sha256=valid_package_hash,
        adjudicator_id=ADJUDICATOR,
        adjudicator_attestation=_adjudicator_attestation(NOW + timedelta(seconds=2)),
        submitted_at=NOW + timedelta(seconds=2),
    )

    stale_time = NOW
    stale_package = valid_package.model_copy(update={"generated_at": stale_time})
    stale_package_hash = sha256_bytes(serialize_json_model(stale_package))
    stale_mapping = valid_mapping.model_copy(
        update={"generated_at": stale_time, "package_sha256": stale_package_hash}
    )
    stale_submission = valid_submission.model_copy(update={"package_sha256": stale_package_hash})

    with pytest.raises(ValueError, match=r"predates.*review submissions"):
        evaluate_entailment_reviews(
            source,
            source_sha256=source_hash,
            packages=packages,
            mappings=mappings,
            submissions=updated_submissions,
            adjudication_package=(stale_package, stale_package_hash),
            adjudication_mapping=(
                stale_mapping,
                sha256_bytes(serialize_json_model(stale_mapping)),
            ),
            adjudication_submission=(
                stale_submission,
                sha256_bytes(serialize_json_model(stale_submission)),
            ),
            generated_at=NOW + timedelta(seconds=3),
        )


def _write_review_files(tmp_path: Path):
    source, source_hash = _source()
    key = source.items[0].key
    packages, mappings, submissions = _two_reviews(
        source,
        source_hash,
        {key: ReviewDecision.ENTAILED},
        {key: ReviewDecision.CONTRADICTED},
    )
    source_path = tmp_path / "source.json"
    source_path.write_bytes(serialize_json_model(source))
    package_paths = []
    mapping_paths = []
    submission_paths = []
    for index in range(2):
        package_path = tmp_path / f"review-package-{index}.json"
        mapping_path = tmp_path / f"review-mapping-{index}.json"
        submission_path = tmp_path / f"review-submission-{index}.json"
        package_path.write_bytes(serialize_json_model(packages[index][0]))
        mapping_path.write_bytes(serialize_json_model(mappings[index][0]))
        submission_path.write_bytes(serialize_json_model(submissions[index][0]))
        package_paths.append(package_path)
        mapping_paths.append(mapping_path)
        submission_paths.append(submission_path)
    return source_path, package_paths, mapping_paths, submission_paths


def _score_args(source_path, packages, mappings, submissions, output):
    args = ["--source", str(source_path)]
    for path in packages:
        args.extend(("--package", str(path)))
    for path in mappings:
        args.extend(("--mapping", str(path)))
    for path in submissions:
        args.extend(("--submission", str(path)))
    args.extend(("--output", str(output)))
    return args


def test_adjudication_cli_workflow_reaches_formal_complete(tmp_path):
    source, packages, mappings, submissions = _write_review_files(tmp_path)
    dispute_package = tmp_path / "adjudication-package.json"
    private_mapping = tmp_path / "adjudication-mapping.json"
    build_args = ["--source", str(source)]
    for path in packages:
        build_args.extend(("--package", str(path)))
    for path in mappings:
        build_args.extend(("--mapping", str(path)))
    for path in submissions:
        build_args.extend(("--submission", str(path)))
    build_args.extend(
        (
            "--adjudicator-id",
            ADJUDICATOR,
            "--ordering-seed",
            "sealed-cli-seed",
            "--generated-at",
            NOW.isoformat(),
            "--output",
            str(dispute_package),
            "--mapping-output",
            str(private_mapping),
        )
    )
    assert build_package_main(build_args) == 0

    worksheet_path = tmp_path / "adjudication-worksheet.json"
    assert (
        build_worksheet_main(
            [
                "--package",
                str(dispute_package),
                "--adjudicator-id",
                ADJUDICATOR,
                "--created-at",
                NOW.isoformat(),
                "--output",
                str(worksheet_path),
            ]
        )
        == 0
    )
    worksheet = json.loads(worksheet_path.read_bytes())
    worksheet["items"][0]["decision"] = AdjudicationDecision.ENTAILED.value
    worksheet["items"][0]["rationale"] = "Human resolution from frozen evidence."
    worksheet_path.write_text(
        json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    adjudication_submission = tmp_path / "adjudication-submission.json"
    assert (
        compile_submission_main(
            [
                "--package",
                str(dispute_package),
                "--worksheet",
                str(worksheet_path),
                "--adjudicator-id",
                ADJUDICATOR,
                "--submitted-at",
                NOW.isoformat(),
                "--attest-human",
                "--attest-no-model-assistance",
                "--attest-no-repository-access",
                "--attest-no-gold-access",
                "--attest-no-source-or-private-mapping-access",
                "--attest-no-raw-submission-or-reviewer-identity-access",
                "--output",
                str(adjudication_submission),
            ]
        )
        == 0
    )

    report_path = tmp_path / "formal-report.json"
    score_args = _score_args(source, packages, mappings, submissions, report_path)
    score_args.extend(
        (
            "--adjudication-package",
            str(dispute_package),
            "--adjudication-mapping",
            str(private_mapping),
            "--adjudication-submission",
            str(adjudication_submission),
            "--require-complete",
        )
    )
    assert score_reviews_main(score_args) == 0
    report = json.loads(report_path.read_bytes())
    assert report["formal_completion"]["status"] == "COMPLETE"
    assert report["formal_completion"]["blockers"] == []
    assert report["summary"]["final_label_counts"]["entailed"] == 1


def test_require_complete_does_not_publish_or_overwrite_incomplete_report(tmp_path):
    source, packages, mappings, submissions = _write_review_files(tmp_path)
    output = tmp_path / "formal-report.json"
    victim = b"existing-formal-report"
    output.write_bytes(victim)
    args = _score_args(source, packages, mappings, submissions, output)

    with pytest.raises(SystemExit):
        score_reviews_main([*args, "--require-complete", "--overwrite"])
    assert output.read_bytes() == victim

    incomplete = tmp_path / "diagnostic-report.json"
    diagnostic_args = _score_args(source, packages, mappings, submissions, incomplete)
    assert score_reviews_main(diagnostic_args) == 0
    report = json.loads(incomplete.read_bytes())
    assert report["formal_completion"]["status"] == "INCOMPLETE"
    assert report["formal_completion"]["blockers"]


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")


@pytest.mark.parametrize("cli_name", ["build", "worksheet", "compile", "score"])
@pytest.mark.parametrize("dangling", [False, True])
def test_adjudication_clis_reject_output_symlink_without_touching_victim(
    tmp_path, cli_name, dangling
):
    dummy_paths = [tmp_path / f"input-{index}.json" for index in range(7)]
    for index, path in enumerate(dummy_paths):
        path.write_text(f"input-{index}", encoding="utf-8")
    victim = tmp_path / "victim.json"
    output_link = tmp_path / "output.json"
    victim_bytes = b"victim-must-remain-byte-identical"
    if dangling:
        target = tmp_path / "missing-target.json"
    else:
        target = victim
        victim.write_bytes(victim_bytes)
    _symlink_or_skip(output_link, target)

    if cli_name == "build":
        args = [
            "--source",
            str(dummy_paths[0]),
            "--package",
            str(dummy_paths[1]),
            "--package",
            str(dummy_paths[2]),
            "--mapping",
            str(dummy_paths[3]),
            "--mapping",
            str(dummy_paths[4]),
            "--submission",
            str(dummy_paths[5]),
            "--submission",
            str(dummy_paths[6]),
            "--adjudicator-id",
            ADJUDICATOR,
            "--ordering-seed",
            "seed",
            "--output",
            str(output_link),
            "--mapping-output",
            str(tmp_path / "private-map.json"),
            "--overwrite",
        ]
        entrypoint = build_package_main
    elif cli_name == "worksheet":
        args = [
            "--package",
            str(dummy_paths[0]),
            "--adjudicator-id",
            ADJUDICATOR,
            "--output",
            str(output_link),
            "--overwrite",
        ]
        entrypoint = build_worksheet_main
    elif cli_name == "compile":
        args = [
            "--package",
            str(dummy_paths[0]),
            "--worksheet",
            str(dummy_paths[1]),
            "--adjudicator-id",
            ADJUDICATOR,
            "--output",
            str(output_link),
            "--overwrite",
        ]
        entrypoint = compile_submission_main
    else:
        args = [
            "--source",
            str(dummy_paths[0]),
            "--package",
            str(dummy_paths[1]),
            "--mapping",
            str(dummy_paths[2]),
            "--output",
            str(output_link),
            "--overwrite",
        ]
        entrypoint = score_reviews_main

    with pytest.raises(SystemExit):
        entrypoint(args)
    assert output_link.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert victim.read_bytes() == victim_bytes
