import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from creditlens.demo_acceptance import (
    DEMO_CASE_ID,
    AcceptanceFailure,
    _validate_demo_case_identity,
    _validate_grounded_answer,
    _validate_report_content,
    _validate_trace,
    blocking_claim_fingerprint,
    load_frozen_hitl_allowlist,
    run_http_acceptance,
)

_SECTION_ID = "50000000-0000-0000-0000-000000000001"
_DOCUMENT_VERSION_ID = "60000000-0000-0000-0000-000000000001"
_PARSE_RUN_ID = "70000000-0000-0000-0000-000000000001"
_EVIDENCE_ID = "80000000-0000-0000-0000-000000000001"
_OPPOSING_EVIDENCE_ID = "80000000-0000-0000-0000-000000000002"
_DISCLAIMER = "本报告为授信预审辅助草稿，不构成授信批准或拒绝决定。"


def _demo_case_payload() -> dict:
    return {
        "case_id": str(DEMO_CASE_ID),
        "case_number": "golden_case_001",
        "product_code": "working_capital",
        "requested_amount": "5000000.00",
        "currency": "CNY",
        "as_of_date": "2026-06-30",
        "status": "DRAFT",
    }


def _candidate():
    return {
        "section_id": _SECTION_ID,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "parse_run_id": _PARSE_RUN_ID,
        "heading_path": ["第六条"],
        "page": 1,
        "text": "合成检索证据。",
        "text_hash": "d" * 64,
    }


def _citation():
    return {
        "evidence_id": _EVIDENCE_ID,
        "evidence_type": "DOCUMENT_SPAN",
        "section_id": _SECTION_ID,
        "document_version_id": _DOCUMENT_VERSION_ID,
        "parse_run_id": _PARSE_RUN_ID,
        "page_number": 1,
        "content_hash": "d" * 64,
    }


def _report_locator(citation: dict | None = None) -> dict:
    locator = dict(citation or _citation())
    locator.pop("evidence_id", None)
    return locator


def _review_source_claim(claim_id: str):
    return {
        "claim_id": claim_id,
        "category": "FINANCIAL",
        "statement": "合成源结论。",
        "verdict": "SUPPORTED",
        "review_status": "AUDITED",
        "evidence": {
            "source_claim_id": None,
            "supporting_evidence_ids": [_EVIDENCE_ID],
            "opposing_evidence_ids": [],
            "calculation_ids": [],
            "supporting_locators": [_report_locator()],
            "opposing_locators": [],
        },
    }


def _review_conflict_claim(claim_id: str, source_claim_id: str, *, status: str) -> dict:
    return {
        "claim_id": claim_id,
        "category": "DATA_CONFLICT",
        "statement": "合成冲突证据需要人工复核。",
        "verdict": "PARTIALLY_SUPPORTED",
        "review_status": status,
        "evidence": {
            "source_claim_id": source_claim_id,
            "supporting_evidence_ids": [],
            "opposing_evidence_ids": [_OPPOSING_EVIDENCE_ID],
            "calculation_ids": [],
            "supporting_locators": [],
            "opposing_locators": [
                {
                    **_report_locator(),
                    "content_hash": "a" * 64,
                }
            ],
        },
    }


def _completed_review(run_id: str, claims: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "state_version": 8,
        "execution": {"degraded": False, "degraded_agents": []},
        "claims": claims,
    }


def _valid_report_content(run_id: str, review_claims: list[dict]) -> dict:
    entries = []
    references = []
    for claim in review_claims:
        evidence = claim["evidence"]
        supporting = evidence["supporting_locators"]
        opposing = evidence["opposing_locators"]
        entries.append(
            {
                "claim_id": claim["claim_id"],
                "category": claim["category"],
                "statement": claim["statement"],
                "verdict": claim["verdict"],
                "review_status": claim["review_status"],
                "evidence_refs": evidence["supporting_evidence_ids"],
                "opposing_evidence_refs": evidence["opposing_evidence_ids"],
                "calculation_ids": evidence["calculation_ids"],
                "evidence_locators": supporting,
                "opposing_evidence_locators": opposing,
                "source_claim_id": evidence["source_claim_id"],
            }
        )
        references.extend(
            {"claim_id": claim["claim_id"], "polarity": "SUPPORTING", **locator}
            for locator in supporting
        )
        references.extend(
            {"claim_id": claim["claim_id"], "polarity": "OPPOSING", **locator}
            for locator in opposing
        )
    return {
        "run_id": run_id,
        "as_of_date": "2026-06-30",
        "claims": entries,
        "excluded_claims": 0,
        "missing_materials": [],
        "references": references,
        "degraded": False,
        "degraded_agents": [],
        "disclaimer": _DISCLAIMER,
    }


def _deep_rag():
    return {
        "query_spec": {
            "original_query": "流贷第六条要求是什么？",
            "standalone_query": "流贷第六条要求是什么？",
            "product_code": "working_capital",
            "as_of_date": "2026-06-30",
            "query_variants": [
                {"variant_id": "dense-main", "text": "资产负债率 第六条", "route": "dense"},
                {"variant_id": "sparse-main", "text": "资产负债率 第六条", "route": "sparse"},
            ],
        },
        "retrieval_trace": {
            "routes": [
                {
                    "route": route,
                    "candidates_count": 1 if route != "EXACT" else 0,
                    "rejected_count": 0,
                    "rejection_reasons": {},
                }
                for route in ("DENSE", "SPARSE", "SUMMARY", "EXACT")
            ],
            "rrf_k": 60,
            "fusion": {
                "rrf_k": 60,
                "input_lists": ["DENSE:main", "SPARSE:main", "SUMMARY"],
                "fused_count": 1,
            },
            "final_count": 1,
            "rerank_applied": True,
            "rerank_degraded": False,
        },
        "packing": {
            "sections": [
                {
                    **_candidate(),
                    "page_start": 1,
                    "page_end": 1,
                    "tokens_est": 10,
                    "rank": 1,
                }
            ],
            "total_tokens_est": 10,
            "budget": 100,
        },
    }


def _grounded_qa_payload(answer_status: str) -> dict:
    payload = {
        "run_id": "10000000-0000-0000-0000-000000000011",
        "answer_status": answer_status,
        "answer": "",
        "claims": [],
        "candidates": [_candidate()],
        **_deep_rag(),
    }
    if answer_status == "ANSWERED":
        payload.update(
            {
                "answer": "合成答案。",
                "claims": [{"verdict": "SUPPORTED", "citations": [_citation()]}],
            }
        )
    elif answer_status == "NEEDS_REVIEW":
        payload["claims"] = [{"verdict": "PARTIALLY_SUPPORTED", "citations": [_citation()]}]
    elif answer_status == "ABSTAINED":
        payload.update(
            {
                "abstention_reason": "当前合成证据不足。",
                "refusal_reason_code": "INSUFFICIENT_EVIDENCE",
            }
        )
    return payload


def _trace(*, integrity_status="VALID", delivery_status="PENDING"):
    complete = delivery_status == "COMPLETE"
    pending = delivery_status == "PENDING"
    invocations = (
        [
            {
                "invocation_id": "90000000-0000-0000-0000-000000000001",
                "integrity": {"valid": True},
                "delivery": {"status": "DELIVERED" if complete else "PENDING"},
            }
        ]
        if complete or pending
        else []
    )
    return {
        "events": [{"sequence_no": 1, "event_type": "STATE_CHANGED", "payload": {}}],
        "invocations": invocations,
        "integrity": {
            "status": integrity_status,
            "valid": integrity_status == "VALID",
            "invalid_count": 0,
        },
        "delivery": {
            "status": delivery_status,
            "complete": complete,
            "total": len(invocations),
            "counts": {
                "PENDING": len(invocations) if pending else 0,
                "PROCESSING": 0,
                "DELIVERED": len(invocations) if complete else 0,
                "DEAD": 0,
                "MISSING": 0,
                "INVALID": 0,
            },
        },
    }


@pytest.mark.parametrize("answer_status", ["ANSWERED", "NEEDS_REVIEW", "ABSTAINED"])
def test_grounded_qa_status_matrix_accepts_only_well_formed_business_states(answer_status):
    _validate_grounded_answer(_grounded_qa_payload(answer_status))


@pytest.mark.parametrize(
    ("answer_status", "field", "value", "error_code"),
    [
        ("ANSWERED", "answer", "", "QA_ANSWER_STATE_INVALID"),
        (
            "ANSWERED",
            "refusal_reason_code",
            "INSUFFICIENT_EVIDENCE",
            "QA_ANSWER_STATE_INVALID",
        ),
        ("NEEDS_REVIEW", "answer", "不应展示直接答案", "QA_REVIEW_STATE_INVALID"),
        ("NEEDS_REVIEW", "claims", [], "QA_REVIEW_STATE_INVALID"),
        (
            "ABSTAINED",
            "claims",
            [{"verdict": "SUPPORTED", "citations": [_citation()]}],
            "QA_ABSTENTION_STATE_INVALID",
        ),
        ("ABSTAINED", "refusal_reason_code", None, "QA_ABSTENTION_STATE_INVALID"),
    ],
)
def test_grounded_qa_status_matrix_rejects_cross_state_fields(
    answer_status,
    field,
    value,
    error_code,
):
    payload = _grounded_qa_payload(answer_status)
    payload[field] = value
    with pytest.raises(AcceptanceFailure, match=error_code):
        _validate_grounded_answer(payload)


def test_grounded_qa_rejects_candidate_a_packing_c_citation_b():
    payload = _grounded_qa_payload("ANSWERED")
    packed = payload["packing"]["sections"][0]
    packed.update(
        {
            "section_id": "50000000-0000-0000-0000-000000000003",
            "document_version_id": "60000000-0000-0000-0000-000000000003",
            "parse_run_id": "70000000-0000-0000-0000-000000000003",
            "text_hash": "c" * 64,
        }
    )
    payload["claims"][0]["citations"] = [
        {
            **_citation(),
            "evidence_id": "80000000-0000-0000-0000-000000000002",
            "section_id": "50000000-0000-0000-0000-000000000002",
            "document_version_id": "60000000-0000-0000-0000-000000000002",
            "parse_run_id": "70000000-0000-0000-0000-000000000002",
            "content_hash": "b" * 64,
        }
    ]

    with pytest.raises(AcceptanceFailure, match="QA_CITATION_NOT_PACKED"):
        _validate_grounded_answer(payload)


def test_grounded_qa_accepts_exact_citation_to_expanded_packed_section():
    """Packing may add an ACL-verified neighbor absent from the final display list."""

    payload = _grounded_qa_payload("ANSWERED")
    packed = payload["packing"]["sections"][0]
    packed.update(
        {
            "section_id": "50000000-0000-0000-0000-000000000003",
            "document_version_id": "60000000-0000-0000-0000-000000000003",
            "parse_run_id": "70000000-0000-0000-0000-000000000003",
            "text_hash": "c" * 64,
            "expanded": True,
        }
    )
    payload["claims"][0]["citations"] = [
        {
            **_citation(),
            "evidence_id": "80000000-0000-0000-0000-000000000003",
            "section_id": packed["section_id"],
            "document_version_id": packed["document_version_id"],
            "parse_run_id": packed["parse_run_id"],
            "content_hash": packed["text_hash"],
        }
    ]

    _validate_grounded_answer(payload)


@pytest.mark.parametrize("allow_empty", [True, False])
def test_trace_complete_state_is_cross_validated(allow_empty):
    _validate_trace(
        _trace(integrity_status="VALID", delivery_status="COMPLETE"),
        allow_empty=allow_empty,
    )


def test_trace_empty_state_is_allowed_only_for_qa():
    empty_trace = _trace(integrity_status="EMPTY", delivery_status="EMPTY")
    _validate_trace(empty_trace, allow_empty=True)
    with pytest.raises(AcceptanceFailure, match="TRACE_INTEGRITY_NOT_ACCEPTABLE"):
        _validate_trace(empty_trace, allow_empty=False)


@pytest.mark.parametrize(
    ("integrity_status", "delivery_status"),
    [("EMPTY", "COMPLETE"), ("VALID", "EMPTY")],
)
def test_trace_rejects_mixed_empty_and_complete_summaries(
    integrity_status,
    delivery_status,
):
    with pytest.raises(AcceptanceFailure, match="TRACE_EMPTY_STATE_INVALID"):
        _validate_trace(
            _trace(
                integrity_status=integrity_status,
                delivery_status=delivery_status,
            ),
            allow_empty=True,
        )


def test_blocking_claim_fingerprint_rejects_dangling_source_claim():
    claim = {
        "claim_id": "30000000-0000-0000-0000-000000000021",
        "category": "DATA_CONFLICT",
        "statement": "合成冲突证据需要人工复核。",
        "verdict": "PARTIALLY_SUPPORTED",
        "review_status": "PENDING",
        "evidence": {
            "source_claim_id": "40000000-0000-0000-0000-000000000021",
            "supporting_locators": [],
            "opposing_locators": [_citation()],
        },
    }
    with pytest.raises(AcceptanceFailure, match="HITL_SOURCE_CLAIM_NOT_UNIQUE"):
        blocking_claim_fingerprint(claim, [claim])


@pytest.mark.parametrize("missing_side", ["positive", "negative"])
def test_blocking_claim_fingerprint_requires_two_sided_evidence(missing_side):
    source_claim = _review_source_claim("40000000-0000-0000-0000-000000000022")
    claim = {
        "claim_id": "30000000-0000-0000-0000-000000000022",
        "category": "DATA_CONFLICT",
        "statement": "合成冲突证据需要人工复核。",
        "verdict": "PARTIALLY_SUPPORTED",
        "review_status": "PENDING",
        "evidence": {
            "source_claim_id": source_claim["claim_id"],
            "supporting_locators": [],
            "opposing_locators": [_citation()],
        },
    }
    if missing_side == "positive":
        source_claim["evidence"]["supporting_locators"] = []
    else:
        claim["evidence"]["opposing_locators"] = []

    with pytest.raises(AcceptanceFailure, match="HITL_TWO_SIDED_EVIDENCE_REQUIRED"):
        blocking_claim_fingerprint(claim, [source_claim, claim])


def test_blocking_claim_fingerprint_is_database_id_independent_but_content_bound():
    source_claim = _review_source_claim("40000000-0000-0000-0000-000000000023")
    conflict_claim = _review_conflict_claim(
        "30000000-0000-0000-0000-000000000023",
        source_claim["claim_id"],
        status="PENDING",
    )
    baseline = blocking_claim_fingerprint(
        conflict_claim,
        [source_claim, conflict_claim],
    )

    rebound_source = deepcopy(source_claim)
    rebound_conflict = deepcopy(conflict_claim)
    for claim in (rebound_source, rebound_conflict):
        for key in ("supporting_locators", "opposing_locators"):
            for locator in claim["evidence"][key]:
                locator["section_id"] = "51000000-0000-0000-0000-000000000023"
                locator["document_version_id"] = "61000000-0000-0000-0000-000000000023"
                locator["parse_run_id"] = "71000000-0000-0000-0000-000000000023"
    assert (
        blocking_claim_fingerprint(
            rebound_conflict,
            [rebound_source, rebound_conflict],
        )
        == baseline
    )

    rebound_conflict["evidence"]["opposing_locators"][0]["content_hash"] = "b" * 64
    assert (
        blocking_claim_fingerprint(
            rebound_conflict,
            [rebound_source, rebound_conflict],
        )
        != baseline
    )


def test_blocking_claim_fingerprint_validates_but_does_not_bind_local_calculation_hash():
    source_claim = _review_source_claim("40000000-0000-0000-0000-000000000024")
    source_claim["evidence"]["supporting_locators"] = [
        {
            "evidence_type": "CALCULATION",
            "section_id": None,
            "document_version_id": None,
            "parse_run_id": None,
            "page_number": None,
            "content_hash": "c" * 64,
        }
    ]
    conflict_claim = _review_conflict_claim(
        "30000000-0000-0000-0000-000000000024",
        source_claim["claim_id"],
        status="PENDING",
    )
    baseline = blocking_claim_fingerprint(
        conflict_claim,
        [source_claim, conflict_claim],
    )

    rebound_source = deepcopy(source_claim)
    rebound_source["evidence"]["supporting_locators"][0]["content_hash"] = "e" * 64
    assert (
        blocking_claim_fingerprint(
            conflict_claim,
            [rebound_source, conflict_claim],
        )
        == baseline
    )

    rebound_source["statement"] = "合成源结论发生变化。"
    assert (
        blocking_claim_fingerprint(
            conflict_claim,
            [rebound_source, conflict_claim],
        )
        != baseline
    )


def test_frozen_hitl_allowlist_binds_case_and_profile(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "creditlens.http-hitl-allowlist.v1",
                "case_id": str(DEMO_CASE_ID),
                "profile": "deterministic-offline",
                "blocking_claim_fingerprints": ["a" * 64],
            }
        ),
        encoding="utf-8",
    )
    frozen = load_frozen_hitl_allowlist(path)
    assert frozen is not None
    assert frozen.case_id == str(DEMO_CASE_ID)
    assert frozen.profile == "deterministic-offline"
    assert frozen.blocking_claim_fingerprints == frozenset({"a" * 64})
    with pytest.raises(AcceptanceFailure, match="HITL_ALLOWLIST_PROFILE_MISMATCH"):
        load_frozen_hitl_allowlist(path, expected_profile="configured-models")


def test_missing_frozen_hitl_allowlist_never_becomes_an_approval(tmp_path):
    assert load_frozen_hitl_allowlist(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    ("field", "tampered_value", "error_code"),
    [
        ("case_number", "golden_case_002", "DEMO_CASE_NUMBER_MISMATCH"),
        ("product_code", "factoring", "DEMO_CASE_PRODUCT_CODE_MISMATCH"),
        (
            "requested_amount",
            "5000000.01",
            "DEMO_CASE_REQUESTED_AMOUNT_MISMATCH",
        ),
        ("currency", "USD", "DEMO_CASE_CURRENCY_MISMATCH"),
        ("as_of_date", "2026-07-01", "DEMO_CASE_AS_OF_DATE_MISMATCH"),
        ("status", "APPROVED", "DEMO_CASE_STATUS_MISMATCH"),
    ],
)
async def test_http_acceptance_rejects_each_tampered_demo_case_root_field(
    field,
    tampered_value,
    error_code,
):
    payload = _demo_case_payload()
    payload[field] = tampered_value
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == f"/api/v1/cases/{DEMO_CASE_ID}":
            return httpx.Response(200, json=payload)
        return httpx.Response(500)

    with pytest.raises(AcceptanceFailure, match=error_code):
        await run_http_acceptance(
            "http://test",
            timeout_seconds=2,
            poll_seconds=0,
            transport=httpx.MockTransport(handler),
        )
    assert calls == ["/health/ready", f"/api/v1/cases/{DEMO_CASE_ID}"]


def test_demo_case_identity_accepts_exact_frozen_api_shape():
    _validate_demo_case_identity(_demo_case_payload(), expected_case_id=DEMO_CASE_ID)


def _report_validation_world():
    review_run_id = "20000000-0000-0000-0000-000000000099"
    source = _review_source_claim("40000000-0000-0000-0000-000000000099")
    conflict = _review_conflict_claim(
        "30000000-0000-0000-0000-000000000099",
        source["claim_id"],
        status="HUMAN_APPROVED",
    )
    review = _completed_review(review_run_id, [source, conflict])
    return review_run_id, review, _valid_report_content(review_run_id, review["claims"])


def test_report_content_rejects_synthetic_dict():
    review_run_id, review, _ = _report_validation_world()
    with pytest.raises(AcceptanceFailure, match="REPORT_CONTENT_SCHEMA_INVALID"):
        _validate_report_content(
            {"synthetic": True},
            review=review,
            review_run_id=review_run_id,
            case_payload=_demo_case_payload(),
            human_review_exercised=True,
        )


@pytest.mark.parametrize(
    ("target", "error_code"),
    [
        ("run_id", "REPORT_RUN_BINDING_INVALID"),
        ("case", "REPORT_CASE_BINDING_INVALID"),
        ("claim", "REPORT_CLAIM_BINDING_INVALID"),
        ("reference", "REPORT_REFERENCE_BINDING_INVALID"),
        ("review_status", "REPORT_REVIEW_STATE_INVALID"),
    ],
)
def test_report_content_rejects_run_case_claim_reference_and_review_tampering(
    target,
    error_code,
):
    review_run_id, review, content = _report_validation_world()
    review = deepcopy(review)
    content = deepcopy(content)
    if target == "run_id":
        content["run_id"] = "20000000-0000-0000-0000-000000000098"
    elif target == "case":
        content["as_of_date"] = "2026-06-29"
    elif target == "claim":
        content["claims"][0]["statement"] = "被篡改的报告结论。"
    elif target == "reference":
        content["references"][0]["content_hash"] = "e" * 64
    else:
        review["claims"][1]["review_status"] = "PENDING"

    with pytest.raises(AcceptanceFailure, match=error_code):
        _validate_report_content(
            content,
            review=review,
            review_run_id=review_run_id,
            case_payload=_demo_case_payload(),
            human_review_exercised=True,
        )


async def test_http_acceptance_exercises_question_hitl_report_and_trace():
    qa_run_id = "10000000-0000-0000-0000-000000000001"
    review_run_id = "20000000-0000-0000-0000-000000000001"
    claim_id = "30000000-0000-0000-0000-000000000001"
    state = {"decision": False, "review_trace_polls": 0}
    calls: list[tuple[str, str]] = []
    source_claim = _review_source_claim("40000000-0000-0000-0000-000000000001")
    blocking_claim = _review_conflict_claim(
        claim_id,
        source_claim["claim_id"],
        status="PENDING",
    )
    approved_claim = deepcopy(blocking_claim)
    approved_claim["review_status"] = "HUMAN_APPROVED"
    final_review = _completed_review(review_run_id, [source_claim, approved_claim])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == f"/api/v1/cases/{DEMO_CASE_ID}" and request.method == "GET":
            return httpx.Response(200, json=_demo_case_payload())
        if path.endswith("/questions"):
            return httpx.Response(
                200,
                json={
                    "run_id": qa_run_id,
                    "answer_status": "ANSWERED",
                    "answer": "synthetic answer",
                    "claims": [
                        {
                            "claim_id": claim_id,
                            "verdict": "SUPPORTED",
                            "citations": [_citation()],
                        }
                    ],
                    "candidates": [_candidate()],
                    **_deep_rag(),
                },
            )
        if path == f"/api/v1/runs/{qa_run_id}/trace":
            return httpx.Response(
                200, json=_trace(integrity_status="EMPTY", delivery_status="EMPTY")
            )
        if path == f"/api/v1/runs/{qa_run_id}":
            return httpx.Response(200, json={"run_id": qa_run_id, "status": "COMPLETED"})
        if path.endswith("/runs") and request.method == "POST":
            return httpx.Response(202, json={"run_id": review_run_id, "status": "RECEIVED"})
        if path == f"/api/v1/runs/{review_run_id}" and request.method == "GET":
            if state["decision"]:
                return httpx.Response(200, json=final_review)
            return httpx.Response(
                200,
                json={
                    "run_id": review_run_id,
                    "status": "HUMAN_REVIEW",
                    "state_version": 6,
                    "claims": [source_claim, blocking_claim],
                },
            )
        if path.endswith("/review-decisions"):
            body = json.loads(request.content)
            assert body["expected_state_version"] == 6
            assert body["target_claim_ids"] == [claim_id]
            state["decision"] = True
            return httpx.Response(200, json={"status": "COMPLETED"})
        if path.endswith("/report"):
            content = _valid_report_content(review_run_id, final_review["claims"])
            return httpx.Response(
                200,
                json={
                    "run_id": review_run_id,
                    "version_no": 1,
                    "status": "APPROVED_DRAFT",
                    "content_hash": hashlib.sha256(
                        json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "content": content,
                },
            )
        if path == f"/api/v1/runs/{review_run_id}/trace":
            state["review_trace_polls"] += 1
            if state["review_trace_polls"] == 1:
                return httpx.Response(
                    200,
                    json=_trace(integrity_status="VALID", delivery_status="PENDING"),
                )
            return httpx.Response(
                200, json=_trace(integrity_status="VALID", delivery_status="COMPLETE")
            )
        return httpx.Response(404, json={"detail": "not mocked"})

    report = await run_http_acceptance(
        "http://test",
        timeout_seconds=2,
        poll_seconds=0,
        transport=httpx.MockTransport(handler),
        expected_hitl_claim_fingerprints=frozenset(
            {blocking_claim_fingerprint(blocking_claim, [source_claim, blocking_claim])}
        ),
    )
    assert report.passed is True
    assert report.qa_answer_status == "ANSWERED"
    assert report.human_review_exercised is True
    assert report.approved_hitl_claim_fingerprints
    assert report.review_final_status == "COMPLETED"
    assert report.trace_integrity_status == "VALID"
    assert state["review_trace_polls"] == 2
    assert ("GET", f"/api/v1/runs/{review_run_id}/report") in calls
    assert ("GET", f"/api/v1/runs/{review_run_id}/trace") in calls


async def test_http_acceptance_fails_closed_on_degraded_qa_trace():
    qa_run_id = "10000000-0000-0000-0000-000000000002"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == f"/api/v1/cases/{DEMO_CASE_ID}":
            return httpx.Response(200, json=_demo_case_payload())
        if path.endswith("/questions"):
            return httpx.Response(
                200,
                json={
                    "run_id": qa_run_id,
                    "answer_status": "ABSTAINED",
                    "answer": "",
                    "abstention_reason": "合成证据不足。",
                    "refusal_reason_code": "INSUFFICIENT_EVIDENCE",
                    "claims": [],
                    "candidates": [_candidate()],
                    **_deep_rag(),
                },
            )
        if path == f"/api/v1/runs/{qa_run_id}/trace":
            return httpx.Response(200, json=_trace(integrity_status="DEGRADED"))
        if path == f"/api/v1/runs/{qa_run_id}":
            return httpx.Response(200, json={"run_id": qa_run_id, "status": "COMPLETED"})
        return httpx.Response(500)

    with pytest.raises(AcceptanceFailure, match="TRACE_INTEGRITY_NOT_ACCEPTABLE"):
        await run_http_acceptance(
            "http://test",
            timeout_seconds=2,
            poll_seconds=0,
            transport=httpx.MockTransport(handler),
        )


async def test_http_acceptance_rejects_answer_without_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == f"/api/v1/cases/{DEMO_CASE_ID}":
            return httpx.Response(200, json=_demo_case_payload())
        if path.endswith("/questions"):
            return httpx.Response(
                200,
                json={
                    "run_id": "10000000-0000-0000-0000-000000000003",
                    "answer_status": "ANSWERED",
                    "claims": [{"verdict": "SUPPORTED", "citations": []}],
                    "candidates": [_candidate()],
                },
            )
        return httpx.Response(500)

    with pytest.raises(AcceptanceFailure, match="QA_SUPPORTED_CLAIM_WITHOUT_CITATION"):
        await run_http_acceptance(
            "http://test",
            timeout_seconds=2,
            poll_seconds=0,
            transport=httpx.MockTransport(handler),
        )


async def test_http_acceptance_never_approves_unfrozen_blocking_claims():
    source_claim = _review_source_claim("40000000-0000-0000-0000-000000000009")
    claim = {
        "claim_id": "30000000-0000-0000-0000-000000000009",
        "category": "DATA_CONFLICT",
        "statement": "unexpected synthetic conflict",
        "verdict": "PARTIALLY_SUPPORTED",
        "review_status": "PENDING",
        "evidence": {
            "source_claim_id": "40000000-0000-0000-0000-000000000009",
            "supporting_locators": [],
            "opposing_locators": [
                {
                    "evidence_type": "DOCUMENT_SPAN",
                    "content_hash": "b" * 64,
                    "section_id": _SECTION_ID,
                    "document_version_id": _DOCUMENT_VERSION_ID,
                    "parse_run_id": _PARSE_RUN_ID,
                    "page_number": 1,
                }
            ],
        },
    }
    qa_run_id = "10000000-0000-0000-0000-000000000009"
    review_run_id = "20000000-0000-0000-0000-000000000009"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == f"/api/v1/cases/{DEMO_CASE_ID}":
            return httpx.Response(200, json=_demo_case_payload())
        if path.endswith("/questions"):
            return httpx.Response(
                200,
                json={
                    "run_id": qa_run_id,
                    "answer_status": "ABSTAINED",
                    "answer": "",
                    "abstention_reason": "合成证据不足。",
                    "refusal_reason_code": "INSUFFICIENT_EVIDENCE",
                    "claims": [],
                    "candidates": [_candidate()],
                    **_deep_rag(),
                },
            )
        if path == f"/api/v1/runs/{qa_run_id}":
            return httpx.Response(200, json={"status": "COMPLETED"})
        if path == f"/api/v1/runs/{qa_run_id}/trace":
            return httpx.Response(
                200, json=_trace(integrity_status="EMPTY", delivery_status="EMPTY")
            )
        if path.endswith("/runs") and request.method == "POST":
            return httpx.Response(202, json={"run_id": review_run_id})
        if path == f"/api/v1/runs/{review_run_id}":
            return httpx.Response(
                200,
                json={
                    "status": "HUMAN_REVIEW",
                    "state_version": 6,
                    "claims": [source_claim, claim],
                },
            )
        return httpx.Response(500)

    with pytest.raises(AcceptanceFailure, match="HITL_CLAIM_SET_MISMATCH"):
        await run_http_acceptance(
            "http://test",
            timeout_seconds=2,
            poll_seconds=0,
            transport=httpx.MockTransport(handler),
            expected_hitl_claim_fingerprints=frozenset({"c" * 64}),
        )
