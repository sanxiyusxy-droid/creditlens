from __future__ import annotations

from pathlib import Path

import pytest
import requests
from apps.demo.http_client import DemoHTTPError, get_binary, get_json, post_json
from apps.demo.presenter import present_retrieval, present_run_trace
from streamlit.testing.v1 import AppTest


def test_streamlit_demo_boots_fail_soft_when_api_is_offline(monkeypatch) -> None:
    def offline(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    app_path = Path(__file__).parents[2] / "apps" / "demo" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert any("Invocation Ledger" in item.value for item in app.subheader)


def test_present_retrieval_exposes_the_deep_rag_pipeline() -> None:
    payload = {
        "question": "流贷第六条的资负率要求是什么？",
        "as_of_date": "2026-06-30",
        "query_spec": {
            "original_query": "流贷第六条的资负率要求是什么？",
            "standalone_query": "流贷第六条的资负率要求是什么？",
            "intent": "POLICY_QA",
            "product_code": "working_capital",
            "as_of_date": "2026-06-30",
            "decision_cutoff_at": "2026-06-30T15:59:59Z",
            "immutable_numbers": [],
            "exact_terms": ["第六条", "流动资金贷款", "资产负债率"],
            "must_not_assume": ["缺失财务指标不得估算"],
            "rewrite_confidence": "HIGH",
            "subqueries": [
                {
                    "subquery_id": "main",
                    "question": "流贷第六条的资负率要求是什么？",
                    "sparse_terms": ["流动资金贷款", "资产负债率"],
                    "exact_terms": ["第六条", "流动资金贷款", "资产负债率"],
                }
            ],
            "query_variants": [
                {
                    "variant_id": "main_original_dense",
                    "subquery_id": "main",
                    "text": "流贷第六条的资负率要求是什么？",
                    "origin": "ORIGINAL",
                    "route": "dense",
                },
                {
                    "variant_id": "main_original_sparse",
                    "subquery_id": "main",
                    "text": "流贷第六条的资负率要求是什么？",
                    "origin": "ORIGINAL",
                    "route": "sparse",
                },
            ],
        },
        "retrieval_trace": {
            "routes": [
                {
                    "route": "DENSE",
                    "variant_id": "main_original_dense",
                    "candidates_count": 8,
                    "rejected_count": 1,
                    "rejection_reasons": {"SNAPSHOT_MISMATCH": 1},
                },
                {
                    "route": "SPARSE",
                    "variant_id": "main_original_sparse",
                    "candidates_count": 5,
                    "rejected_count": 0,
                    "rejection_reasons": {},
                },
                {
                    "route": "SUMMARY",
                    "variant_id": "",
                    "candidates_count": 2,
                    "rejected_count": 0,
                    "rejection_reasons": {},
                },
                {
                    "route": "EXACT",
                    "variant_id": "",
                    "candidates_count": 0,
                    "rejected_count": 0,
                    "rejection_reasons": {},
                },
            ],
            "fusion": {
                "rrf_k": 60,
                "input_lists": ["DENSE:main_original_dense", "SPARSE:main_original_sparse"],
                "fused_count": 11,
            },
            "final_count": 3,
            "rerank_applied": False,
            "rerank_degraded": True,
            "rerank_degraded_reason": "RERANKER_NOT_CONFIGURED",
            "query_spec_confidence": "HIGH",
        },
        "packing": {
            "sections": [
                {
                    "section_id": "section-1",
                    "heading_path": ["准入政策", "第六条"],
                    "page_start": 6,
                    "page_end": 6,
                    "tokens_est": 128,
                    "rank": 1,
                    "expanded": False,
                }
            ],
            "total_tokens_est": 128,
            "budget": 4096,
            "dropped": ["section-9"],
            "expanded_count": 0,
        },
    }

    view = present_retrieval(payload)

    assert view["available"] is True
    assert view["query"]["normalized_terms"] == ["流动资金贷款", "资产负债率"]
    assert [row["路由"] for row in view["route_rows"][:4]] == [
        "DENSE",
        "SPARSE",
        "SUMMARY",
        "EXACT",
    ]
    assert view["route_rows"][0]["候选数"] == 8
    assert view["route_rows"][3] == {
        "路由": "EXACT",
        "是否执行": True,
        "变体数": 1,
        "候选数": 0,
        "拒绝数": 0,
    }
    assert view["rejection_rows"][0]["拒绝原因"] == "SNAPSHOT_MISMATCH"
    assert view["fusion"] == {
        "rrf_k": 60,
        "input_lists": ["DENSE:main_original_dense", "SPARSE:main_original_sparse"],
        "fused_count": 11,
        "final_count": 3,
    }
    assert view["rerank"]["degraded"] is True
    assert view["rerank"]["reason"] == "RERANKER_NOT_CONFIGURED"
    assert view["packing"]["selected_count"] == 1
    assert view["packing"]["dropped"] == ["section-9"]


def test_present_retrieval_fails_soft_for_legacy_or_malformed_optional_fields() -> None:
    view = present_retrieval(
        {
            "question": "legacy question",
            "query_spec": "not-a-mapping",
            "retrieval_trace": {"routes": [None, "bad"], "fusion": None},
            "packing": ["not-a-mapping"],
            "candidates": None,
        }
    )

    assert view["query"]["original"] == "legacy question"
    assert view["variant_rows"] == []
    assert all(row["候选数"] == 0 for row in view["route_rows"][:4])
    assert view["fusion"]["final_count"] == 0
    assert len(view["warnings"]) == 2


@pytest.mark.parametrize(
    ("delivery", "integrity", "expected"),
    [
        (
            {"contract_version": "invocation_v2", "status": "EMPTY", "counts": {}},
            {"status": "EMPTY", "invalid_count": 0},
            "EMPTY",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "DEGRADED",
                "counts": {"MISSING": 1},
            },
            {"status": "VALID", "invalid_count": 0},
            "MISSING",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "DEGRADED",
                "counts": {"INVALID": 1},
            },
            {"status": "DEGRADED", "invalid_count": 1},
            "INVALID",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "PENDING",
                "counts": {"PROCESSING": 1},
            },
            {"status": "VALID", "invalid_count": 0},
            "PENDING",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "COMPLETE",
                "counts": {"DELIVERED": 2},
            },
            {"status": "VALID", "invalid_count": 0},
            "COMPLETE",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "DEGRADED",
                "counts": {"DEAD": 1},
            },
            {"status": "VALID", "invalid_count": 0},
            "DEGRADED",
        ),
        (
            {
                "contract_version": "invocation_v2",
                "status": "DEGRADED",
                "counts": {"DEAD": 1, "PENDING": 2},
            },
            {"status": "VALID", "invalid_count": 0},
            "DEGRADED",
        ),
    ],
)
def test_present_run_trace_derives_the_six_audit_states(
    delivery: dict, integrity: dict, expected: str
) -> None:
    assert present_run_trace({"delivery": delivery, "integrity": integrity})["state"] == expected


def test_present_run_trace_shows_model_tool_terminals_and_delivery() -> None:
    trace = {
        "integrity": {"status": "VALID", "valid": True, "invalid_count": 0},
        "delivery": {
            "contract_version": "invocation_v2",
            "status": "COMPLETE",
            "complete": True,
            "total": 2,
            "counts": {"DELIVERED": 2},
        },
        "invocations": [
            {
                "invocation_id": "model-1",
                "kind": "MODEL",
                "name": "grounded_answer",
                "status": "SUCCESS",
                "envelope": {"latency_ms": 42.5},
                "integrity": {"status": "VALID", "valid": True},
                "delivery": {"status": "DELIVERED", "attempts": 1},
            },
            {
                "invocation_id": "tool-1",
                "kind": "TOOL",
                "name": "search_policy",
                "status": "DENIED",
                "envelope": {"latency_ms": 2.0, "error_code": "POLICY_DENIED"},
                "integrity": {"status": "VALID", "valid": True},
                "delivery": {"status": "DELIVERED", "attempts": 2},
            },
        ],
        "events": [
            {
                "sequence_no": 1,
                "event_type": "AUTHORIZED",
                "payload": {},
                "occurred_at": "2026-08-29T00:00:00Z",
            }
        ],
    }

    view = present_run_trace(trace)

    assert view["state"] == "COMPLETE"
    assert [(row["类型"], row["终态"]) for row in view["invocation_rows"]] == [
        ("MODEL", "SUCCESS"),
        ("TOOL", "DENIED"),
    ]
    assert view["invocation_rows"][1]["投递次数"] == 2
    assert view["event_rows"][0]["事件"] == "AUTHORIZED"


def test_present_run_trace_distinguishes_legacy_from_empty_and_hides_bad_envelope() -> None:
    view = present_run_trace(
        {
            "delivery": {
                "contract_version": None,
                "status": "LEGACY_UNAVAILABLE",
                "complete": None,
                "counts": None,
            },
            "integrity": None,
            "invocations": [
                {
                    "invocation_id": "bad-1",
                    "envelope": None,
                    "integrity": {
                        "status": "DEGRADED",
                        "valid": False,
                        "error_code": "INVOCATION_INTEGRITY_FAILED",
                    },
                    "delivery": {"status": "INVALID"},
                },
                "malformed",
            ],
            "events": [None],
        }
    )

    assert view["state"] == "LEGACY_UNAVAILABLE"
    assert "不表示调用数为零" in view["warnings"][0]
    assert view["invocation_rows"][0]["名称"] == "—"
    assert view["invocation_rows"][0]["Delivery"] == "INVALID"
    assert len(view["event_rows"]) == 1


class _Response:
    def __init__(self, status_code=200, payload=None, content=b"png"):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_demo_http_boundary_returns_only_stable_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: _Response(503, {"secret": "must-not-surface"}),
    )
    with pytest.raises(DemoHTTPError) as unavailable:
        get_json("http://demo.invalid/qa")
    assert unavailable.value.code == "API_HTTP_ERROR"
    assert unavailable.value.status_code == 503
    assert "secret" not in str(unavailable.value)

    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response(200, ValueError("provider secret")),
    )
    with pytest.raises(DemoHTTPError) as invalid:
        post_json("http://demo.invalid/qa", payload={})
    assert invalid.value.code == "API_RESPONSE_INVALID"
    assert "provider secret" not in str(invalid.value)

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: _Response(content=b"ok"))
    assert get_binary("http://demo.invalid/evidence", params={}) == b"ok"
