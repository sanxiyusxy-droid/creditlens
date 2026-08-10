"""Tests for the real-system answer prediction adapter."""

import argparse
import hashlib
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from scripts.generate_answer_predictions import (
    DEFAULT_DATASET,
    DEFAULT_GOLD,
    DEFAULT_QUERY_DATASET,
    PREDICTION_ADAPTER_VERSION,
    AnswerQueryDataset,
    CitationMappingStats,
    _close_runtime_resources,
    _experiment_contract,
    _git_metadata,
    _idempotency_key,
    _prediction_from_response,
    _read_bytes_and_sha256,
    _RuntimeResources,
    _sha256_json,
    _validate_gold_provenance,
    _validate_query_projection,
    extract_numeric_facts,
)

from creditlens.evaluation.answer_metrics import (
    AnswerEvalDataset,
    AnswerPrediction,
    AnswerPredictionProvenance,
    AnswerPredictionSet,
    PredictionStatus,
)
from creditlens.evaluation.gold_schema import GoldDataset


def _settings(**overrides):
    values = {
        "llm_provider": "openai_compatible",
        "llm_model": "chat-v1",
        "embedding_provider": "openai_compatible",
        "embedding_model": "embed-v1",
        "effective_embedding_version": "embed-v1@api",
        "embedding_dim": 1024,
        "rerank_provider": "http",
        "rerank_model": "rerank-v1",
        "orchestrator_enable_rerank": True,
        "qa_allow_extractive_fallback": False,
        "qa_max_claims": 6,
        "qa_max_generation_tokens": 2048,
        "qa_max_audit_repairs": 1,
        "rrf_k": 60,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _response_provenance_kwargs(*, idempotent_replay: bool = False):
    return {
        "run_id": uuid.UUID("10000000-0000-0000-0000-000000000001"),
        "snapshot_id": uuid.UUID("20000000-0000-0000-0000-000000000001"),
        "generation_mode": "llm",
        "model_invocation_ids": [uuid.UUID("30000000-0000-0000-0000-000000000001")],
        "idempotent_replay": idempotent_replay,
    }


def _orchestrator(**overrides):
    values = {
        "rrf_k": 60,
        "route_weights": None,
        "embedder": SimpleNamespace(version="embed-runtime-v1"),
        "reranker": SimpleNamespace(version="rerank-runtime-v1"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _experiment_hash(settings=None, **overrides):
    arguments = {
        "query_dataset_sha256": "a" * 64,
        "top_k": 8,
        "prompt_version": "grounded_qa_v1",
        "prompt_sha256": "b" * 64,
        "settings": settings or _settings(),
        "orchestrator": _orchestrator(),
    }
    arguments.update(overrides)
    return _sha256_json(_experiment_contract(**arguments))


def _datasets():
    dataset = AnswerEvalDataset.model_validate_json(DEFAULT_DATASET.read_bytes())
    gold = GoldDataset.model_validate_json(DEFAULT_GOLD.read_bytes())
    return dataset, gold


def _query_dataset():
    return AnswerQueryDataset.model_validate_json(DEFAULT_QUERY_DATASET.read_bytes())


def test_extract_numeric_facts_keeps_local_label_value_and_unit():
    facts = extract_numeric_facts(
        "2025年营业收入为50000万元，同比增长12.5%。贷款期限不超过24个月。"
    )

    assert [(item.name, str(item.value), item.unit) for item in facts] == [
        ("2025年营业收入", "50000", "万元"),
        ("同比增长", "12.5", "%"),
        ("贷款期限上限", "24", "个月"),
    ]


def test_extract_numeric_facts_handles_q101_thousands_and_filters_calendar_years():
    facts = extract_numeric_facts(
        "星辰微电子2025年营业收入为18,500万元/同比增长23.3%。截至2026年，贷款期限为3年。"
    )

    assert [(item.name, str(item.value), item.unit) for item in facts] == [
        ("星辰微电子2025年营业收入", "18500", "万元"),
        ("同比增长", "23.3", "%"),
        ("贷款期限", "3", "年"),
    ]


def test_extract_numeric_facts_normalizes_restricted_chinese_numbers():
    facts = extract_numeric_facts(
        "资产负债率不高于百分之七十，担保覆盖率不低于百分之一百三十；"
        "余额上限为人民币一千万元，授信上限为五亿元，期限十二个月，覆盖零点九二倍。"
    )

    assert [(item.name, str(item.value), item.unit) for item in facts] == [
        ("资产负债率上限", "70", "%"),
        ("担保覆盖率下限", "130", "%"),
        ("余额上限", "1000", "万元"),
        ("授信上限", "5", "亿元"),
        ("期限", "12", "个月"),
        ("覆盖", "0.92", "倍"),
    ]


@pytest.mark.parametrize(
    ("question_id", "answer", "expected"),
    [
        (
            "q001",
            "借款人资产负债率不得高于百分之七十。",
            [("借款人资产负债率上限", "70", "%")],
        ),
        (
            "q003",
            "单一小微企业借款人的流动资金贷款余额上限为人民币一千万元。",
            [("单一小微企业借款人的流动资金贷款余额上限", "1000", "万元")],
        ),
        (
            "q081",
            "资产负债率不得高于百分之七十五。",
            [("资产负债率上限", "75", "%")],
        ),
        (
            "q101",
            "星辰微电子2025年度营业收入为人民币1.85亿元。",
            [("星辰微电子2025年度营业收入", "1.85", "亿元")],
        ),
        (
            "q086",
            "研发投入占营业收入比例不得低于百分之三。",
            [("研发投入占营业收入比例下限", "3", "%")],
        ),
        (
            "q153",
            "一般风险准备金余额不低于融资余额的百分之一。",
            [("一般风险准备金余额不低于融资余额的", "1", "%")],
        ),
    ],
)
def test_extract_numeric_facts_covers_first_run_positive_cases(
    question_id: str,
    answer: str,
    expected: list[tuple[str, str, str]],
) -> None:
    assert question_id.startswith("q")
    assert [(item.name, str(item.value), item.unit) for item in extract_numeric_facts(answer)] == (
        expected
    )


def test_extract_numeric_facts_does_not_emit_chinese_year_or_article_number():
    facts = extract_numeric_facts("二〇二五年修订的第十二条规定，贷款期限为十二个月。")

    assert [(item.name, str(item.value), item.unit) for item in facts] == [
        ("贷款期限", "12", "个月")
    ]


def test_extract_numeric_facts_keeps_non_revenue_metric_boundary():
    facts = extract_numeric_facts("非营业收入为人民币1.85亿元。")

    assert [(item.name, str(item.value), item.unit) for item in facts] == [
        ("非营业收入", "1.85", "亿元")
    ]


def test_answered_response_maps_exact_ambiguous_and_unmapped_sections_conservatively():
    response = SimpleNamespace(
        answer_status="ANSWERED",
        answer="营业收入为18,500万元。",
        claims=[
            SimpleNamespace(
                citations=[
                    {"section_id": "known"},
                    {"section_id": "ambiguous"},
                    {"section_id": "unknown"},
                ],
                opposing_citations=[],
            )
        ],
        **_response_provenance_kwargs(),
    )
    stats = CitationMappingStats()

    prediction = _prediction_from_response(
        "q101",
        response,
        {
            "known": {"ar_tech:sec2"},
            "ambiguous": {"ar_tech:sec2", "ar_tech:sec3"},
        },
        stats,
    )

    assert prediction.status is PredictionStatus.ANSWERED
    assert prediction.citation_refs == [
        "ambiguous:section:ambiguous",
        "ar_tech:sec2",
        "unmapped:section:unknown",
    ]
    assert "ar_tech:sec3" not in prediction.citation_refs
    assert stats.ambiguous_citation_sections == 1
    assert stats.unmapped_citation_sections == 1
    assert prediction.provenance == AnswerPredictionProvenance(**_response_provenance_kwargs())
    assert prediction.numeric_facts[0].name == "营业收入"


def test_abstention_uses_service_reason_and_missing_reason_is_explicit():
    refused = _prediction_from_response(
        "q069",
        SimpleNamespace(
            answer_status="ABSTAINED",
            refusal_reason_code="INSUFFICIENT_EVIDENCE",
            **_response_provenance_kwargs(idempotent_replay=True),
        ),
        {},
    )
    unspecified = _prediction_from_response(
        "q168",
        SimpleNamespace(
            answer_status="ABSTAINED",
            **_response_provenance_kwargs(),
        ),
        {},
    )

    assert refused.status is PredictionStatus.REFUSED
    assert refused.refusal_reason_code == "INSUFFICIENT_EVIDENCE"
    assert refused.provenance.idempotent_replay is True
    assert refused.provenance.model_invocation_ids == [
        uuid.UUID("30000000-0000-0000-0000-000000000001")
    ]
    assert unspecified.refusal_reason_code == "UNSPECIFIED"


def test_needs_review_is_not_misclassified_as_a_technical_failure():
    prediction = _prediction_from_response(
        "q001",
        SimpleNamespace(
            answer_status="NEEDS_REVIEW",
            **_response_provenance_kwargs(),
        ),
        {},
    )

    assert prediction.status is PredictionStatus.NEEDS_REVIEW
    assert prediction.error_type is None
    assert prediction.provenance.run_id == _response_provenance_kwargs()["run_id"]


def test_three_business_statuses_keep_response_provenance_in_json_roundtrip():
    predictions: list[AnswerPrediction] = [
        _prediction_from_response(
            "q-answered",
            SimpleNamespace(
                answer_status="ANSWERED",
                answer="grounded answer",
                claims=[],
                **_response_provenance_kwargs(),
            ),
            {},
        ),
        _prediction_from_response(
            "q-refused",
            SimpleNamespace(
                answer_status="ABSTAINED",
                refusal_reason_code="INSUFFICIENT_EVIDENCE",
                **_response_provenance_kwargs(idempotent_replay=True),
            ),
            {},
        ),
        _prediction_from_response(
            "q-review",
            SimpleNamespace(
                answer_status="NEEDS_REVIEW",
                **_response_provenance_kwargs(),
            ),
            {},
        ),
    ]
    prediction_set = AnswerPredictionSet(
        prediction_set_id="runner-provenance-roundtrip",
        dataset_id="answer_eval_v1",
        dataset_version="1.0.0",
        predictions=predictions,
    )

    restored = AnswerPredictionSet.model_validate_json(prediction_set.model_dump_json())

    assert [item.status for item in restored.predictions] == [
        PredictionStatus.ANSWERED,
        PredictionStatus.REFUSED,
        PredictionStatus.NEEDS_REVIEW,
    ]
    assert all(
        isinstance(item.provenance, AnswerPredictionProvenance) for item in restored.predictions
    )
    assert restored.predictions[1].provenance.idempotent_replay is True


def test_experiment_hash_covers_all_generation_dimensions():
    baseline = _experiment_hash()
    variants = {
        _experiment_hash(query_dataset_sha256="c" * 64),
        _experiment_hash(top_k=9),
        _experiment_hash(prompt_version="grounded_qa_v2"),
        _experiment_hash(prompt_sha256="d" * 64),
        _experiment_hash(_settings(llm_provider="disabled")),
        _experiment_hash(_settings(llm_model="chat-v2")),
        _experiment_hash(_settings(embedding_provider="hash_fallback")),
        _experiment_hash(_settings(embedding_model="embed-v2")),
        _experiment_hash(_settings(effective_embedding_version="embed-v2@api")),
        _experiment_hash(_settings(embedding_dim=2560)),
        _experiment_hash(_settings(rerank_provider="lexical_fallback")),
        _experiment_hash(_settings(rerank_model="rerank-v2")),
        _experiment_hash(_settings(orchestrator_enable_rerank=False)),
        _experiment_hash(_settings(qa_allow_extractive_fallback=True)),
        _experiment_hash(_settings(qa_max_claims=7)),
        _experiment_hash(_settings(qa_max_generation_tokens=4096)),
        _experiment_hash(_settings(qa_max_audit_repairs=2)),
        _experiment_hash(audit_implementation_version="structural_evidence_v3"),
        _experiment_hash(grounded_answer_contract_version="1.2"),
        _experiment_hash(orchestrator=_orchestrator(rrf_k=61)),
        _experiment_hash(orchestrator=_orchestrator(route_weights={"DENSE": 2.0, "SPARSE": 1.0})),
        _experiment_hash(
            orchestrator=_orchestrator(embedder=SimpleNamespace(version="embed-runtime-v2"))
        ),
        _experiment_hash(
            orchestrator=_orchestrator(reranker=SimpleNamespace(version="rerank-runtime-v2"))
        ),
    }

    assert baseline not in variants
    assert len(variants) == 23
    assert len(_idempotency_key(baseline, "q" + "x" * 10_000)) <= 128


def test_experiment_contract_versions_the_prediction_adapter():
    contract = _experiment_contract(
        query_dataset_sha256="a" * 64,
        top_k=8,
        prompt_version="grounded_qa_v1",
        prompt_sha256="b" * 64,
        settings=_settings(),
        orchestrator=_orchestrator(),
    )

    assert contract["prediction_adapter_version"] == PREDICTION_ADAPTER_VERSION


def test_frozen_input_is_read_once_and_hashes_the_same_bytes():
    class OneReadPath:
        calls = 0

        def read_bytes(self):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("frozen input was read more than once")
            return b"frozen bytes"

    path = OneReadPath()
    payload, digest = _read_bytes_and_sha256(path)  # type: ignore[arg-type]

    assert payload == b"frozen bytes"
    assert digest == "ed605a86d0d0e2ac7ff2c1910fc7bcf52ac5e52b212f0fe530a7933a7c99f940"
    assert path.calls == 1


def test_query_projection_contains_only_gold_free_phase_one_fields():
    projection = _query_dataset()
    answer_dataset, _gold = _datasets()

    assert projection.frozen is True
    assert projection.ordering == "sha256(question_id)"
    assert len(projection.questions) == 41
    assert set(projection.questions[0].model_dump()) == {
        "question_id",
        "case_key",
        "question",
        "as_of_date",
        "decision_cutoff_at",
    }
    raw = json.loads(DEFAULT_QUERY_DATASET.read_text(encoding="utf-8"))
    forbidden = {
        "expected",
        "answerable",
        "expected_refusal_reason",
        "tags",
        "citation_refs",
        "intent",
        "split",
    }
    assert all(forbidden.isdisjoint(item) for item in raw["questions"])
    assert [item.question_id for item in projection.questions] != [
        item.question_id for item in answer_dataset.questions
    ]
    _validate_query_projection(projection, answer_dataset)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("frozen", False), "frozen"),
        (
            lambda payload: payload["questions"][0].__setitem__(
                "decision_cutoff_at", "2026-06-30T15:59:59"
            ),
            "timezone",
        ),
        (
            lambda payload: payload["questions"][0].__setitem__("answerable", True),
            "Extra inputs",
        ),
        (
            lambda payload: payload["questions"][0].__setitem__("expected", {}),
            "Extra inputs",
        ),
        (lambda payload: payload["questions"].reverse(), "sha256"),
    ],
)
def test_query_projection_rejects_non_frozen_naive_or_gold_fields(mutation, message):
    payload = json.loads(DEFAULT_QUERY_DATASET.read_text(encoding="utf-8"))
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        AnswerQueryDataset.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("case_key", "golden_case_999"),
        ("question", "drifted question"),
        ("as_of_date", date(2026, 6, 29)),
        ("decision_cutoff_at", datetime(2026, 6, 30, 15, 59, 58, tzinfo=UTC)),
    ],
)
def test_query_projection_drift_fails_before_source_gold_mapping(field_name, replacement):
    projection = _query_dataset()
    dataset, _gold = _datasets()
    question = next(item for item in projection.questions if item.question_id == "q001")
    setattr(question, field_name, replacement)

    with pytest.raises(RuntimeError, match=rf"q001:{field_name}"):
        _validate_query_projection(projection, dataset)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("answer_eval_dataset_id", "other-eval"),
        ("answer_eval_dataset_version", "999.0.0"),
        ("source_dataset_id", "other-source"),
        ("source_dataset_version", "999.0.0"),
        ("source_split", "dev"),
    ],
)
def test_query_projection_metadata_must_match_answer_eval(field_name, replacement):
    projection = _query_dataset()
    dataset, _gold = _datasets()
    setattr(projection, field_name, replacement)

    with pytest.raises(RuntimeError, match=field_name):
        _validate_query_projection(projection, dataset)


def test_source_gold_provenance_matches_the_frozen_answer_projection():
    dataset, gold = _datasets()

    by_id = _validate_gold_provenance(dataset, gold)

    assert set(question.question_id for question in dataset.questions) <= set(by_id)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("question", "篡改的问题"),
        ("case_key", "golden_case_999"),
        ("as_of_date", None),
        ("decision_cutoff_at", None),
        ("answerable", None),
        ("split", "dev"),
    ],
)
def test_source_gold_question_fields_are_strictly_validated(field_name, replacement):
    dataset, gold = _datasets()
    source = next(
        question
        for question in gold.questions
        if question.question_id == dataset.questions[0].question_id
    )
    if field_name == "as_of_date":
        replacement = source.as_of_date + timedelta(days=1)
    elif field_name == "decision_cutoff_at":
        replacement = source.decision_cutoff_at + timedelta(seconds=1)
    elif field_name == "answerable":
        replacement = not source.answerable
    setattr(source, field_name, replacement)

    with pytest.raises(RuntimeError, match=rf"q001:{field_name}"):
        _validate_gold_provenance(dataset, gold)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("dataset_id", "other-source"),
        ("dataset_version", "999.0.0"),
    ],
)
def test_source_gold_identity_is_strictly_validated(field_name, replacement):
    dataset, gold = _datasets()
    setattr(gold, field_name, replacement)

    with pytest.raises(RuntimeError, match="source_dataset"):
        _validate_gold_provenance(dataset, gold)


@pytest.mark.asyncio
async def test_cleanup_closes_all_resources_even_when_one_close_fails():
    events: list[str] = []

    class AsyncResource:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        async def aclose(self):
            events.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    class SyncResource:
        def close(self):
            events.append("qdrant")

    class WrappedClient:
        def __init__(self):
            self._client = AsyncResource("embedding-client")

    resources = _RuntimeResources()
    resources.add("engine", AsyncResource("engine"))
    resources.add("qdrant", SyncResource())
    resources.add("embedding", WrappedClient())
    resources.add("reranker", AsyncResource("reranker", fail=True))
    resources.add("chat", AsyncResource("chat"))

    failures = await _close_runtime_resources(resources)

    assert events == ["chat", "reranker", "embedding-client", "qdrant", "engine"]
    assert failures == ["reranker:RuntimeError"]


def test_unknown_git_status_stays_unknown_instead_of_clean(monkeypatch):
    monkeypatch.setattr(
        "scripts.generate_answer_predictions._git_value",
        lambda *args: None,
    )

    assert _git_metadata() == (None, None)


@pytest.mark.asyncio
async def test_generate_finishes_and_checkpoints_raw_qa_before_loading_gold(
    monkeypatch,
    tmp_path,
):
    import scripts.generate_answer_predictions as runner

    cutoff = datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC)
    dataset = AnswerEvalDataset.model_validate(
        {
            "dataset_id": "answer-projection",
            "dataset_version": "1.0.0",
            "source_dataset_id": "source-gold",
            "source_dataset_version": "2.0.0",
            "source_split": "test",
            "questions": [
                {
                    "question_id": "q001",
                    "case_key": "golden_case_001",
                    "question": "first question",
                    "intent": "UNANSWERABLE",
                    "as_of_date": date(2026, 6, 30),
                    "decision_cutoff_at": cutoff,
                    "answerable": False,
                    "expected": {"acceptable_refusal_reason_codes": ["INSUFFICIENT_EVIDENCE"]},
                    "expected_refusal_reason": "missing evidence",
                },
                {
                    "question_id": "q002",
                    "case_key": "golden_case_001",
                    "question": "second question",
                    "intent": "UNANSWERABLE",
                    "as_of_date": date(2026, 6, 30),
                    "decision_cutoff_at": cutoff,
                    "answerable": False,
                    "expected": {"acceptable_refusal_reason_codes": ["INSUFFICIENT_EVIDENCE"]},
                    "expected_refusal_reason": "missing evidence",
                },
            ],
        }
    )
    query_dataset = AnswerQueryDataset.model_validate(
        {
            "dataset_id": "query-projection",
            "dataset_version": "1.0.0",
            "answer_eval_dataset_id": dataset.dataset_id,
            "answer_eval_dataset_version": dataset.dataset_version,
            "source_dataset_id": dataset.source_dataset_id,
            "source_dataset_version": dataset.source_dataset_version,
            "source_split": dataset.source_split,
            "ordering": "sha256(question_id)",
            "frozen": True,
            "questions": [
                {
                    "question_id": question.question_id,
                    "case_key": question.case_key,
                    "question": question.question,
                    "as_of_date": question.as_of_date,
                    "decision_cutoff_at": question.decision_cutoff_at,
                }
                for question in dataset.questions
            ],
        }
    )
    assert not hasattr(query_dataset.questions[0], "answerable")
    assert not hasattr(query_dataset.questions[0], "expected")
    gold = GoldDataset.model_validate(
        {
            "dataset_id": "source-gold",
            "dataset_version": "2.0.0",
            "anchors": [],
            "questions": [
                {
                    "question_id": question.question_id,
                    "case_key": question.case_key,
                    "split": "test",
                    "question": question.question,
                    "intent": question.intent,
                    "as_of_date": question.as_of_date,
                    "decision_cutoff_at": question.decision_cutoff_at,
                    "required_evidence_sets": [],
                    "answerable": False,
                    "expected_refusal_reason": "missing evidence",
                }
                for question in dataset.questions
            ],
        }
    )
    query_bytes = query_dataset.model_dump_json().encode()
    dataset_bytes = dataset.model_dump_json().encode()
    gold_bytes = gold.model_dump_json().encode()
    output = tmp_path / "predictions.json"
    query_path = tmp_path / "queries.json"
    dataset_path = tmp_path / "answer.json"
    gold_path = tmp_path / "gold.json"
    read_order: list[str] = []
    qa_calls: list[str] = []

    def read_frozen(path):
        if path == query_path:
            assert read_order == []
            read_order.append("queries")
            payload = query_bytes
        elif path == dataset_path:
            read_order.append("answer-eval")
            checkpoint = json.loads(runner._raw_checkpoint_path(output).read_text(encoding="utf-8"))
            assert qa_calls == ["first question", "second question"]
            assert checkpoint["qa_phase_complete"] is True
            assert checkpoint["completed_questions"] == 2
            payload = dataset_bytes
        else:
            assert path == gold_path
            assert read_order == ["queries", "answer-eval"]
            read_order.append("source-gold")
            payload = gold_bytes
        return payload, hashlib.sha256(payload).hexdigest()

    class FakeConnection:
        async def run_sync(self, _operation):
            return None

    class FakeBegin:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, *_args):
            return None

    class FakeEngine:
        def begin(self):
            return FakeBegin()

        async def dispose(self):
            return None

    class FakeService:
        def __init__(self, **_kwargs):
            pass

        async def ask(self, **kwargs):
            qa_calls.append(kwargs["question"])
            assert len(kwargs["idempotency_key"]) <= 128
            if kwargs["question"] == "first question":
                raise RuntimeError("provider unavailable")
            return SimpleNamespace(
                answer_status="ABSTAINED",
                refusal_reason_code="INSUFFICIENT_EVIDENCE",
                **_response_provenance_kwargs(),
            )

    async def fake_seed(*_args):
        return None

    settings = _settings()
    settings.qa_prompt_version = "grounded_qa_v1"
    settings.rrf_k = 60
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner, "_prompt_fingerprint", lambda _settings: ("v1", "b" * 64))
    monkeypatch.setattr(runner, "_git_metadata", lambda: (None, None))
    monkeypatch.setattr(runner, "_read_bytes_and_sha256", read_frozen)
    monkeypatch.setattr(runner, "create_engine", FakeEngine)
    monkeypatch.setattr(runner, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(runner, "build_object_store", lambda _settings: object())
    monkeypatch.setattr(runner, "build_qdrant_client", lambda _settings: object())
    monkeypatch.setattr(runner, "build_embedding_provider", lambda _settings: object())
    monkeypatch.setattr(runner, "build_reranker", lambda _settings: None)
    monkeypatch.setattr(runner, "build_chat_provider", lambda _settings: None)
    monkeypatch.setattr(runner, "RetrievalOrchestrator", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "seed_environment", fake_seed)
    monkeypatch.setattr(runner, "QAService", FakeService)

    result = await runner.generate(
        argparse.Namespace(
            query_dataset=query_path,
            dataset=dataset_path,
            gold_dataset=gold_path,
            output=output,
            top_k=8,
            limit=0,
            allow_disabled_llm=False,
        )
    )

    assert read_order == ["queries", "answer-eval", "source-gold"]
    assert result.metadata["query_dataset_sha256"] == hashlib.sha256(query_bytes).hexdigest()
    assert (
        result.metadata["answer_eval_dataset_sha256"] == hashlib.sha256(dataset_bytes).hexdigest()
    )
    assert result.metadata["source_gold_sha256"] == hashlib.sha256(gold_bytes).hexdigest()
    assert [item.status for item in result.predictions] == [
        PredictionStatus.TECHNICAL_FAILURE,
        PredictionStatus.REFUSED,
    ]
    assert (
        json.loads(output.read_text(encoding="utf-8"))["predictions"][0]["error_type"]
        == "RuntimeError"
    )
