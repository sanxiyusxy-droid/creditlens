from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from scripts.score_entailment_review import main as score_reviews_main

from creditlens.evaluation.semantic_entailment import (
    AdjudicationDecision,
    AdjudicationEntry,
    AdjudicationSet,
    BlindReview,
    ReviewDecision,
    ReviewSubmission,
    SemanticEntailmentSource,
    SourceClaim,
    SourceEvidence,
    SourceInputArtifact,
    build_blind_review_package,
    evaluate_entailment_reviews,
    serialize_json_model,
    sha256_bytes,
)

NOW = datetime(2026, 8, 12, tzinfo=UTC)


def _digest(value: str) -> str:
    return sha256_bytes(value.encode())


def _source() -> tuple[SemanticEntailmentSource, str]:
    items = []
    for number in range(1, 4):
        content = f"frozen evidence {number}"
        items.append(
            SourceClaim(
                question_id=f"q{number}",
                claim_id=f"c{number}",
                claim=f"claim {number}",
                evidence=[
                    SourceEvidence(
                        section_id=f"section-{number}",
                        content=content,
                        content_sha256=_digest(content),
                    )
                ],
            )
        )
    source = SemanticEntailmentSource(
        source_id="source-1",
        prediction_set_id="prediction-1",
        prediction_sha256="a" * 64,
        created_at=NOW,
        input_artifacts=[
            SourceInputArtifact(
                role="prediction_set", artifact_id="predictions.json", sha256="a" * 64
            ),
            SourceInputArtifact(role="raw_checkpoint", artifact_id="raw.json", sha256="b" * 64),
        ],
        items=items,
    )
    return source, sha256_bytes(serialize_json_model(source))


def _reviewer_artifacts(source, source_hash, reviewer_id, seed, decisions):
    package, mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id=reviewer_id,
        ordering_seed=seed,
        generated_at=NOW,
    )
    package_hash = sha256_bytes(serialize_json_model(package))
    submission = ReviewSubmission(
        submission_id=f"submission-{reviewer_id}",
        package_id=package.package_id,
        package_sha256=package_hash,
        reviewer_id=reviewer_id,
        submitted_at=NOW,
        reviews=[
            BlindReview(blind_item_id=item.blind_item_id, decision=decisions[index])
            for index, item in enumerate(package.items)
        ],
    )
    return (
        (package, package_hash),
        (mapping, sha256_bytes(serialize_json_model(mapping))),
        (submission, sha256_bytes(serialize_json_model(submission))),
    )


def _write_score_inputs(tmp_path):
    source, source_hash = _source()
    package, mapping, submission = _reviewer_artifacts(
        source,
        source_hash,
        "reviewer-a",
        "seed-a",
        [ReviewDecision.ENTAILED] * len(source.items),
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    submission_path = tmp_path / "submission.json"
    source_path.write_bytes(serialize_json_model(source))
    package_path.write_bytes(serialize_json_model(package[0]))
    mapping_path.write_bytes(serialize_json_model(mapping[0]))
    submission_path.write_bytes(serialize_json_model(submission[0]))
    return source_path, package_path, mapping_path, submission_path


def _score_arguments(input_paths, output_path):
    source_path, package_path, mapping_path, submission_path = input_paths
    return [
        "--source",
        str(source_path),
        "--package",
        str(package_path),
        "--mapping",
        str(mapping_path),
        "--submission",
        str(submission_path),
        "--output",
        str(output_path),
    ]


def test_source_contract_rejects_extra_gold_naive_time_and_bad_content_hash():
    source, _source_hash = _source()
    payload = source.model_dump(mode="json")
    payload["gold_labels"] = ["ENTAILED"]
    with pytest.raises(ValidationError, match="Extra inputs"):
        SemanticEntailmentSource.model_validate(payload)

    payload = source.model_dump(mode="json")
    payload["created_at"] = "2026-08-12T00:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        SemanticEntailmentSource.model_validate(payload)

    payload = source.model_dump(mode="json")
    payload["items"][0]["evidence"][0]["content_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="exact UTF-8 evidence content"):
        SemanticEntailmentSource.model_validate(payload)


def test_blind_packages_are_reproducible_and_keep_strict_provenance():
    source, source_hash = _source()
    package, mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id="reviewer-a",
        ordering_seed="sealed-study-seed",
        generated_at=NOW,
    )
    repeated, repeated_mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id="reviewer-a",
        ordering_seed="sealed-study-seed",
        generated_at=NOW,
    )

    assert package == repeated
    assert mapping == repeated_mapping
    assert package.prediction_sha256 == source.prediction_sha256
    assert package.source_sha256 == source_hash
    assert [item.blind_item_id for item in package.items] == [
        item.blind_item_id for item in mapping.entries
    ]
    assert not hasattr(package.items[0], "question_id")
    assert not hasattr(package.items[0], "claim_id")


def test_aggregate_reports_macro_micro_kappa_coverage_and_adjudication():
    source, source_hash = _source()
    # Labels are specified in each independently randomized package order.
    first = _reviewer_artifacts(
        source,
        source_hash,
        "reviewer-a",
        "seed-a",
        [ReviewDecision.ENTAILED, ReviewDecision.CONTRADICTED, ReviewDecision.NOT_ENOUGH_INFO],
    )
    mapping_a = {entry.blind_item_id: entry.key for entry in first[1][0].entries}
    label_by_key = {
        mapping_a[review.blind_item_id]: review.decision for review in first[2][0].reviews
    }

    package_b, mapping_b = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id="reviewer-b",
        ordering_seed="seed-b",
        generated_at=NOW,
    )
    decision_b = []
    conflict_key = first[1][0].entries[0].key
    for entry in mapping_b.entries:
        decision_b.append(
            ReviewDecision.CONTRADICTED
            if entry.key == conflict_key
            and label_by_key[entry.key] is not ReviewDecision.CONTRADICTED
            else label_by_key[entry.key]
        )
    package_b_hash = sha256_bytes(serialize_json_model(package_b))
    submission_b = ReviewSubmission(
        submission_id="submission-reviewer-b",
        package_id=package_b.package_id,
        package_sha256=package_b_hash,
        reviewer_id="reviewer-b",
        submitted_at=NOW,
        reviews=[
            BlindReview(blind_item_id=item.blind_item_id, decision=decision_b[index])
            for index, item in enumerate(package_b.items)
        ],
    )
    second = (
        (package_b, package_b_hash),
        (mapping_b, sha256_bytes(serialize_json_model(mapping_b))),
        (submission_b, sha256_bytes(serialize_json_model(submission_b))),
    )
    adjudication = AdjudicationSet(
        adjudication_id="adjudication-1",
        source_sha256=source_hash,
        adjudicator_id="senior-reviewer",
        adjudicated_at=NOW,
        entries=[
            AdjudicationEntry(
                question_id=conflict_key[0],
                claim_id=conflict_key[1],
                decision=AdjudicationDecision.ENTAILED,
                rationale="Resolved after reading the same frozen evidence.",
            )
        ],
    )

    report = evaluate_entailment_reviews(
        source,
        source_sha256=source_hash,
        packages=[first[0], second[0]],
        mappings=[first[1], second[1]],
        submissions=[first[2], second[2]],
        adjudication=(adjudication, sha256_bytes(serialize_json_model(adjudication))),
        generated_at=NOW,
    )

    assert report.model_judge_used is False
    assert report.citation_set_used_as_faithfulness is False
    assert report.summary.rating_coverage == 1.0
    assert report.summary.semantic_rating_coverage == 1.0
    assert report.summary.claim_semantic_coverage == 1.0
    assert report.summary.adjudication_needed_claims == 1
    assert report.summary.adjudication_coverage == 1.0
    assert report.summary.resolved_claims == 3
    assert report.summary.agreement.cohen_kappa_comparable_items == 3
    assert report.summary.agreement.cohen_kappa is not None
    assert report.summary.micro_label_rates.entailed == pytest.approx(1 / 6)
    assert report.summary.macro_label_rates.entailed == pytest.approx(1 / 6)


def test_skip_and_conflict_require_rationale_and_are_not_semantic_labels():
    with pytest.raises(ValidationError, match="require a rationale"):
        BlindReview(blind_item_id="x", decision=ReviewDecision.SKIP)

    review = BlindReview(
        blind_item_id="x",
        decision=ReviewDecision.CONFLICT,
        rationale="Evidence sections internally conflict.",
    )
    assert review.decision.semantic_label is None


def test_submission_rejects_package_hash_or_reviewer_assignment_drift():
    source, source_hash = _source()
    package, mapping = build_blind_review_package(
        source,
        source_sha256=source_hash,
        reviewer_id="reviewer-a",
        ordering_seed="seed",
        generated_at=NOW,
    )
    submission = ReviewSubmission(
        submission_id="submission-a",
        package_id=package.package_id,
        package_sha256="b" * 64,
        reviewer_id="reviewer-b",
        submitted_at=NOW,
        reviews=[],
    )

    with pytest.raises(ValueError, match="package_sha256"):
        evaluate_entailment_reviews(
            source,
            source_sha256=source_hash,
            packages=[(package, "a" * 64)],
            mappings=[(mapping, "c" * 64)],
            submissions=[(submission, "d" * 64)],
            generated_at=NOW,
        )


@pytest.mark.parametrize("input_index", range(4))
def test_score_cli_rejects_output_equal_to_every_input(tmp_path, input_index):
    inputs = _write_score_inputs(tmp_path)
    protected_path = inputs[input_index]
    protected_bytes = protected_path.read_bytes()

    with pytest.raises(SystemExit):
        score_reviews_main(_score_arguments(inputs, protected_path))

    assert protected_path.read_bytes() == protected_bytes


def test_score_cli_rejects_output_aliasing_adjudication_input(tmp_path):
    inputs = _write_score_inputs(tmp_path)
    adjudication_path = tmp_path / "adjudication.json"
    adjudication_path.write_text("do-not-touch", encoding="utf-8")
    arguments = [
        *_score_arguments(inputs, adjudication_path),
        "--adjudication",
        str(adjudication_path),
        "--overwrite",
    ]

    with pytest.raises(SystemExit):
        score_reviews_main(arguments)

    assert adjudication_path.read_text(encoding="utf-8") == "do-not-touch"


def test_score_cli_rejects_hard_link_alias_even_with_overwrite(tmp_path):
    inputs = _write_score_inputs(tmp_path)
    source_path = inputs[0]
    output_alias = tmp_path / "source-hard-link.json"
    output_alias.hardlink_to(source_path)
    protected_bytes = source_path.read_bytes()

    with pytest.raises(SystemExit):
        score_reviews_main([*_score_arguments(inputs, output_alias), "--overwrite"])

    assert source_path.read_bytes() == protected_bytes
    assert output_alias.read_bytes() == protected_bytes


def test_score_cli_is_no_clobber_by_default_and_overwrite_is_explicit(tmp_path):
    inputs = _write_score_inputs(tmp_path)
    output_path = tmp_path / "report.json"
    output_path.write_text("keep-existing-report", encoding="utf-8")
    arguments = _score_arguments(inputs, output_path)

    with pytest.raises(SystemExit):
        score_reviews_main(arguments)
    assert output_path.read_text(encoding="utf-8") == "keep-existing-report"

    assert score_reviews_main([*arguments, "--overwrite"]) == 0
    report = json.loads(output_path.read_bytes())
    assert report["report_type"] == "manual_claim_evidence_semantic_entailment"
    assert report["summary"]["total_claims"] == 3


def test_score_cli_atomic_overwrite_failure_keeps_existing_report(tmp_path, monkeypatch):
    inputs = _write_score_inputs(tmp_path)
    output_path = tmp_path / "report.json"
    output_path.write_text("keep-existing-report", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("scripts.score_entailment_review.os.replace", fail_replace)
    with pytest.raises(SystemExit):
        score_reviews_main([*_score_arguments(inputs, output_path), "--overwrite"])

    assert output_path.read_text(encoding="utf-8") == "keep-existing-report"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []
