from __future__ import annotations

import hashlib
import json
import os
import uuid

import pytest
import scripts.build_entailment_review_package as review_package_cli
import scripts.build_entailment_source as source_cli
from scripts.build_entailment_review_package import main as build_package_main
from scripts.build_entailment_source import main as build_source_main

from creditlens.evaluation.semantic_entailment import (
    build_source_from_grounded_qa_artifacts,
    sha256_bytes,
)

REVIEWER_ID = "rvw_H7Lz4Jq9mN2xP8sK5cT1vW6y"


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _symlink_or_skip(link, target) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this test environment: {exc}")


@pytest.mark.parametrize("module", [source_cli, review_package_cli])
def test_source_and_package_publish_recheck_output_reparse(tmp_path, monkeypatch, module):
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


def _artifacts():
    text = "Frozen evidence text."
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    section_id = "11111111-1111-1111-1111-111111111111"
    evidence_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"evidence:{section_id}:{text_hash}"))
    run_id = "22222222-2222-2222-2222-222222222222"
    snapshot_id = "33333333-3333-3333-3333-333333333333"
    invocation_id = "44444444-4444-4444-4444-444444444444"
    prediction = {
        "prediction_set_id": "prediction-1",
        "dataset_id": "dataset-1",
        "dataset_version": "1",
        "prediction_adapter_version": "1",
        "predictions": [
            {
                "question_id": "q1",
                "status": "ANSWERED",
                "answer": "An answer.",
                "numeric_facts": [],
                "citation_refs": ["stable-ref"],
                "refusal_reason_code": None,
                "error_type": None,
                "error_message": None,
                "provenance": {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "generation_mode": "llm",
                    "model_invocation_ids": [invocation_id],
                    "idempotent_replay": True,
                },
            }
        ],
        "metadata": {},
    }
    response = {
        "question": "This must not enter the blind package.",
        "answer_status": "ANSWERED",
        "answer": "An answer.",
        "claims": [
            {
                "claim_id": "claim-1",
                "statement": "A generated claim.",
                "citations": [
                    {
                        "evidence_id": evidence_id,
                        "section_id": section_id,
                        "content_hash": text_hash,
                    }
                ],
                "opposing_citations": [],
            }
        ],
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "generation_mode": "llm",
        "model_invocation_ids": [invocation_id],
        "idempotent_replay": True,
        "candidates": [],
    }
    raw = {
        "checkpoint_type": "grounded_qa_raw_phase",
        "completed_questions": 1,
        "selected_questions": 1,
        "qa_phase_complete": True,
        "results": [{"question_id": "q1", "response": response}],
    }
    evidence_response = dict(response)
    evidence_response["candidates"] = [
        {"section_id": section_id, "text": text, "text_hash": text_hash}
    ]
    evidence = {
        "checkpoint_type": "grounded_qa_raw_phase",
        "completed_questions": 1,
        "selected_questions": 1,
        "qa_phase_complete": True,
        "results": [{"question_id": "other-run", "response": evidence_response}],
    }
    return _json_bytes(prediction), _json_bytes(raw), _json_bytes(evidence)


def test_adapter_binds_actual_bytes_and_uses_auxiliary_candidate_text_only():
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )

    assert source.prediction_sha256 == sha256_bytes(prediction)
    assert [item.role for item in source.input_artifacts] == [
        "prediction_set",
        "raw_checkpoint",
        "evidence_checkpoint",
    ]
    assert source.items[0].claim == "A generated claim."
    assert source.items[0].evidence[0].content == "Frozen evidence text."
    serialized = source.model_dump_json()
    assert "This must not enter the blind package." not in serialized
    assert "stable-ref" not in serialized


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda prediction, raw: raw["results"][0]["response"].update(
                run_id="55555555-5555-5555-5555-555555555555"
            ),
            "run_id differs",
        ),
        (
            lambda prediction, raw: raw["results"][0]["response"]["claims"][0]["citations"][
                0
            ].update(evidence_id="66666666-6666-6666-6666-666666666666"),
            "evidence_id does not match",
        ),
    ],
)
def test_adapter_fails_closed_on_provenance_or_evidence_identity_drift(mutator, message):
    prediction_bytes, raw_bytes, evidence = _artifacts()
    prediction = json.loads(prediction_bytes)
    raw = json.loads(raw_bytes)
    mutator(prediction, raw)
    with pytest.raises(ValueError, match=message):
        build_source_from_grounded_qa_artifacts(
            prediction_bytes=_json_bytes(prediction),
            raw_checkpoint_bytes=_json_bytes(raw),
            evidence_checkpoints=[("evidence.json", evidence)],
            source_id="source-1",
        )


def test_adapter_requires_frozen_text_for_every_supporting_citation():
    prediction, raw, _evidence = _artifacts()
    with pytest.raises(ValueError, match="lacks frozen candidate text"):
        build_source_from_grounded_qa_artifacts(
            prediction_bytes=prediction,
            raw_checkpoint_bytes=raw,
            source_id="source-1",
        )


def test_adapter_rejects_tampered_auxiliary_candidate_text():
    prediction, raw, evidence_bytes = _artifacts()
    evidence = json.loads(evidence_bytes)
    evidence["results"][0]["response"]["candidates"][0]["text"] = "Tampered text."
    with pytest.raises(ValueError, match="text_hash does not match exact text"):
        build_source_from_grounded_qa_artifacts(
            prediction_bytes=prediction,
            raw_checkpoint_bytes=raw,
            evidence_checkpoints=[("evidence.json", _json_bytes(evidence))],
            source_id="source-1",
        )


def test_adapter_rejects_auxiliary_candidate_hash_drift():
    prediction, raw, evidence_bytes = _artifacts()
    evidence = json.loads(evidence_bytes)
    candidate = evidence["results"][0]["response"]["candidates"][0]
    candidate["text"] = "Different but internally hashed text."
    candidate["text_hash"] = hashlib.sha256(candidate["text"].encode()).hexdigest()
    with pytest.raises(ValueError, match="lacks frozen candidate text"):
        build_source_from_grounded_qa_artifacts(
            prediction_bytes=prediction,
            raw_checkpoint_bytes=raw,
            evidence_checkpoints=[("evidence.json", _json_bytes(evidence))],
            source_id="source-1",
        )


def test_source_cli_rejects_input_output_alias(tmp_path):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--source-id",
                "source-1",
                "--output",
                str(prediction_path),
            ]
        )
    assert prediction_path.read_bytes() == prediction


def test_source_cli_is_no_clobber_by_default_and_overwrite_is_explicit(tmp_path):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    output_path.write_text("keep-me", encoding="utf-8")
    arguments = [
        "--prediction",
        str(prediction_path),
        "--raw-checkpoint",
        str(raw_path),
        "--evidence-checkpoint",
        str(evidence_path),
        "--source-id",
        "source-1",
        "--output",
        str(output_path),
    ]

    with pytest.raises(SystemExit):
        build_source_main(arguments)
    assert output_path.read_text(encoding="utf-8") == "keep-me"

    assert build_source_main([*arguments, "--overwrite"]) == 0
    assert json.loads(output_path.read_bytes())["source_id"] == "source-1"


def test_source_cli_explicit_created_at_makes_output_bytes_reproducible(tmp_path):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    first_output = tmp_path / "source-first.json"
    second_output = tmp_path / "source-second.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    common_arguments = [
        "--prediction",
        str(prediction_path),
        "--raw-checkpoint",
        str(raw_path),
        "--evidence-checkpoint",
        str(evidence_path),
        "--source-id",
        "source-1",
        "--created-at",
        "2026-08-12T09:30:00+08:00",
    ]

    assert build_source_main([*common_arguments, "--output", str(first_output)]) == 0
    assert build_source_main([*common_arguments, "--output", str(second_output)]) == 0

    first_bytes = first_output.read_bytes()
    second_bytes = second_output.read_bytes()
    assert first_bytes == second_bytes
    assert sha256_bytes(first_bytes) == sha256_bytes(second_bytes)
    assert json.loads(first_bytes)["created_at"] == "2026-08-12T09:30:00+08:00"


@pytest.mark.parametrize("created_at", ["2026-08-12T09:30:00", "not-a-timestamp"])
def test_source_cli_rejects_naive_or_invalid_created_at(tmp_path, created_at):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)

    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--source-id",
                "source-1",
                "--created-at",
                created_at,
                "--output",
                str(output_path),
            ]
        )
    assert not output_path.exists()


def test_source_cli_rejects_input_output_hard_link_alias(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    os.link(prediction_path, output_path)

    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--source-id",
                "source-1",
                "--output",
                str(output_path),
                "--overwrite",
            ]
        )
    assert "must not refer to the same underlying file" in capsys.readouterr().err
    assert prediction_path.read_bytes() == prediction
    assert output_path.read_bytes() == prediction


def test_source_cli_rejects_output_symlink_without_touching_unrelated_victim(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    victim_path = tmp_path / "unrelated-victim.json"
    output_link = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    victim_bytes = b"unrelated-victim-must-survive"
    victim_path.write_bytes(victim_bytes)
    _symlink_or_skip(output_link, victim_path)

    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--source-id",
                "source-1",
                "--output",
                str(output_link),
                "--overwrite",
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert victim_path.read_bytes() == victim_bytes
    assert output_link.is_symlink()


def test_source_cli_rejects_dangling_output_symlink(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    missing_target = tmp_path / "must-not-be-created.json"
    output_link = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    _symlink_or_skip(output_link, missing_target)

    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--source-id",
                "source-1",
                "--output",
                str(output_link),
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert output_link.is_symlink()
    assert not missing_target.exists()


def test_source_cli_rejects_hard_link_alias_between_inputs(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    prediction_path = tmp_path / "prediction.json"
    raw_path = tmp_path / "raw.json"
    evidence_path = tmp_path / "evidence.json"
    evidence_alias_path = tmp_path / "evidence-alias.json"
    output_path = tmp_path / "source.json"
    prediction_path.write_bytes(prediction)
    raw_path.write_bytes(raw)
    evidence_path.write_bytes(evidence)
    os.link(evidence_path, evidence_alias_path)

    with pytest.raises(SystemExit):
        build_source_main(
            [
                "--prediction",
                str(prediction_path),
                "--raw-checkpoint",
                str(raw_path),
                "--evidence-checkpoint",
                str(evidence_path),
                "--evidence-checkpoint",
                str(evidence_alias_path),
                "--source-id",
                "source-1",
                "--output",
                str(output_path),
            ]
        )
    assert "must not refer to the same underlying file" in capsys.readouterr().err
    assert not output_path.exists()


def test_package_cli_rejects_same_package_and_mapping_path(tmp_path):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    same_path = tmp_path / "same.json"
    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--output",
                str(same_path),
                "--mapping-output",
                str(same_path),
            ]
        )
    assert not same_path.exists()


def test_package_cli_rejects_low_entropy_reviewer_identity_end_to_end(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")

    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                "short-pseudonym",
                "--ordering-seed",
                "seed",
                "--output",
                str(package_path),
                "--mapping-output",
                str(mapping_path),
            ]
        )

    assert "high-entropy pseudonym" in capsys.readouterr().err
    assert not package_path.exists()
    assert not mapping_path.exists()


def test_package_cli_explicit_generated_at_makes_both_artifacts_reproducible(tmp_path):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    common = [
        "--source",
        str(source_path),
        "--reviewer-id",
        REVIEWER_ID,
        "--ordering-seed",
        "sealed-review-seed",
        "--generated-at",
        "2026-08-16T09:30:00+08:00",
    ]
    first_package = tmp_path / "package-first.json"
    first_mapping = tmp_path / "mapping-first.json"
    second_package = tmp_path / "package-second.json"
    second_mapping = tmp_path / "mapping-second.json"

    assert (
        build_package_main(
            [
                *common,
                "--output",
                str(first_package),
                "--mapping-output",
                str(first_mapping),
            ]
        )
        == 0
    )
    assert (
        build_package_main(
            [
                *common,
                "--output",
                str(second_package),
                "--mapping-output",
                str(second_mapping),
            ]
        )
        == 0
    )

    assert first_package.read_bytes() == second_package.read_bytes()
    assert first_mapping.read_bytes() == second_mapping.read_bytes()
    assert json.loads(first_package.read_bytes())["generated_at"] == "2026-08-16T09:30:00+08:00"


@pytest.mark.parametrize("generated_at", ["2026-08-16T09:30:00", "not-a-timestamp"])
def test_package_cli_rejects_naive_or_invalid_generated_at(tmp_path, generated_at):
    output = tmp_path / "package.json"
    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(tmp_path / "source.json"),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--generated-at",
                generated_at,
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "alias_pair",
    [
        "source-output",
        "source-mapping",
        "output-mapping",
    ],
)
def test_package_cli_rejects_hard_link_aliases_even_with_overwrite(tmp_path, capsys, alias_pair):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")

    if alias_pair == "source-output":
        os.link(source_path, package_path)
    elif alias_pair == "source-mapping":
        os.link(source_path, mapping_path)
    else:
        package_path.write_text("existing-package", encoding="utf-8")
        os.link(package_path, mapping_path)

    paths = (source_path, package_path, mapping_path)
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--output",
                str(package_path),
                "--mapping-output",
                str(mapping_path),
                "--overwrite",
            ]
        )

    assert "must not refer to the same underlying file" in capsys.readouterr().err
    after = {path: path.read_bytes() if path.exists() else None for path in paths}
    assert after == before


def test_package_cli_does_not_partially_clobber_existing_outputs(tmp_path):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    mapping_path.write_text("keep-private-map", encoding="utf-8")
    arguments = [
        "--source",
        str(source_path),
        "--reviewer-id",
        REVIEWER_ID,
        "--ordering-seed",
        "seed",
        "--output",
        str(package_path),
        "--mapping-output",
        str(mapping_path),
    ]

    with pytest.raises(SystemExit):
        build_package_main(arguments)
    assert not package_path.exists()
    assert mapping_path.read_text(encoding="utf-8") == "keep-private-map"

    assert build_package_main([*arguments, "--overwrite"]) == 0
    assert json.loads(package_path.read_bytes())["document_type"] == (
        "semantic_entailment_blind_review_package"
    )
    assert json.loads(mapping_path.read_bytes())["document_type"] == (
        "semantic_entailment_blind_mapping"
    )


def test_package_cli_rejects_output_symlink_without_touching_unrelated_victim(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_link = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    victim_path = tmp_path / "unrelated-victim.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    victim_bytes = b"unrelated-victim-must-survive"
    victim_path.write_bytes(victim_bytes)
    _symlink_or_skip(package_link, victim_path)

    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--output",
                str(package_link),
                "--mapping-output",
                str(mapping_path),
                "--overwrite",
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert victim_path.read_bytes() == victim_bytes
    assert package_link.is_symlink()
    assert not mapping_path.exists()


def test_package_cli_rejects_dangling_mapping_symlink_before_writing_package(tmp_path, capsys):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_link = tmp_path / "mapping.json"
    missing_target = tmp_path / "must-not-be-created.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    _symlink_or_skip(mapping_link, missing_target)

    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--output",
                str(package_path),
                "--mapping-output",
                str(mapping_link),
            ]
        )

    assert "symbolic link or reparse point" in capsys.readouterr().err
    assert mapping_link.is_symlink()
    assert not missing_target.exists()
    assert not package_path.exists()


def test_package_cli_overwrite_failure_is_reported_not_silently_successful(
    tmp_path, monkeypatch, capsys
):
    prediction, raw, evidence = _artifacts()
    source = build_source_from_grounded_qa_artifacts(
        prediction_bytes=prediction,
        raw_checkpoint_bytes=raw,
        evidence_checkpoints=[("evidence.json", evidence)],
        source_id="source-1",
    )
    source_path = tmp_path / "source.json"
    package_path = tmp_path / "package.json"
    mapping_path = tmp_path / "mapping.json"
    source_path.write_text(source.model_dump_json(), encoding="utf-8")
    package_path.write_text("old-package", encoding="utf-8")
    mapping_path.write_text("old-mapping", encoding="utf-8")
    real_replace = os.replace
    replace_calls = 0

    def fail_second_replace(source_file, destination_file):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated second artifact publish failure")
        return real_replace(source_file, destination_file)

    monkeypatch.setattr("scripts.build_entailment_review_package.os.replace", fail_second_replace)
    with pytest.raises(SystemExit):
        build_package_main(
            [
                "--source",
                str(source_path),
                "--reviewer-id",
                REVIEWER_ID,
                "--ordering-seed",
                "seed",
                "--output",
                str(package_path),
                "--mapping-output",
                str(mapping_path),
                "--overwrite",
            ]
        )

    captured = capsys.readouterr()
    assert "simulated second artifact publish failure" in captured.err
    assert "package_sha256" not in captured.out
    assert mapping_path.read_text(encoding="utf-8") == "old-mapping"
