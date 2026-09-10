"""Fail-soft presenters for the local Streamlit interview demo.

The API remains the source of truth.  These helpers only turn JSON-compatible
responses into small, stable table/metric structures and deliberately tolerate
legacy runs or partially missing optional fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ROUTE_ORDER = ("DENSE", "SPARSE", "SUMMARY", "EXACT")
DELIVERY_COUNT_ORDER = ("PENDING", "PROCESSING", "DELIVERED", "DEAD", "MISSING", "INVALID")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return max(0, int(value))
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def present_retrieval(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Present a Grounded-QA response without inventing unavailable RAG facts."""

    response = _mapping(payload)
    spec = _mapping(response.get("query_spec"))
    trace = _mapping(response.get("retrieval_trace"))
    packing = _mapping(response.get("packing"))
    channel_config = _mapping(response.get("channel_config"))

    subqueries = [_mapping(item) for item in _items(spec.get("subqueries"))]
    normalized_terms = _unique_strings(
        [term for subquery in subqueries for term in _items(subquery.get("sparse_terms"))]
    )
    normalization_rows = [
        {
            "子问题": _text(subquery.get("subquery_id")),
            "原问题": _text(subquery.get("question")),
            "归一/扩展术语": "、".join(_unique_strings(_items(subquery.get("sparse_terms"))))
            or "—",
            "精确词": "、".join(_unique_strings(_items(subquery.get("exact_terms")))) or "—",
        }
        for subquery in subqueries
    ]

    variant_rows = []
    for item in _items(spec.get("query_variants")):
        variant = _mapping(item)
        variant_rows.append(
            {
                "Variant": _text(variant.get("variant_id")),
                "子问题": _text(variant.get("subquery_id")),
                "路由": _text(variant.get("route")).upper(),
                "来源": _text(variant.get("origin")),
                "查询文本": _text(variant.get("text")),
            }
        )

    route_summary: dict[str, dict[str, Any]] = {
        route: {
            "路由": route,
            "是否执行": False,
            "变体数": 0,
            "候选数": 0,
            "拒绝数": 0,
        }
        for route in ROUTE_ORDER
    }
    rejection_rows: list[dict[str, Any]] = []
    for item in _items(trace.get("routes")):
        route_trace = _mapping(item)
        route = _text(route_trace.get("route"), "UNKNOWN").upper()
        summary = route_summary.setdefault(
            route,
            {
                "路由": route,
                "是否执行": False,
                "变体数": 0,
                "候选数": 0,
                "拒绝数": 0,
            },
        )
        summary["是否执行"] = True
        summary["变体数"] += 1
        summary["候选数"] += _count(route_trace.get("candidates_count"))
        summary["拒绝数"] += _count(route_trace.get("rejected_count"))
        reasons = _mapping(route_trace.get("rejection_reasons"))
        for reason, count in reasons.items():
            rejection_rows.append(
                {
                    "路由": route,
                    "Variant": _text(route_trace.get("variant_id")),
                    "拒绝原因": _text(reason),
                    "数量": _count(count),
                }
            )

    fusion = _mapping(trace.get("fusion"))
    rerank_degraded = bool(
        trace.get("rerank_degraded", channel_config.get("rerank_degraded", False))
    )
    rerank_reason = trace.get(
        "rerank_degraded_reason", channel_config.get("rerank_degraded_reason")
    )
    selected_sections = [_mapping(item) for item in _items(packing.get("sections"))]
    packed_rows = [
        {
            "Rank": _count(section.get("rank")),
            "章节": " > ".join(_unique_strings(_items(section.get("heading_path")))) or "—",
            "页码": (
                f"{_text(section.get('page_start'), '?')}–{_text(section.get('page_end'), '?')}"
            ),
            "Tokens(估算)": _count(section.get("tokens_est")),
            "相邻扩展": bool(section.get("expanded", False)),
            "Section": _text(section.get("section_id")),
        }
        for section in selected_sections
    ]

    warnings: list[str] = []
    if not spec:
        warnings.append("该响应没有 QuerySpec（可能是旧版本或幂等回放），其余结果仍可查看。")
    if not trace:
        warnings.append("该响应没有检索 Trace，不能还原各路召回与精排过程。")
    if not packing:
        warnings.append("该响应没有 Context Packing 明细。")

    extra_routes = [route for route in route_summary if route not in ROUTE_ORDER]
    return {
        "available": bool(spec or trace or packing),
        "query": {
            "original": _text(spec.get("original_query"), _text(response.get("question"))),
            "standalone": _text(spec.get("standalone_query")),
            "intent": _text(spec.get("intent")),
            "confidence": _text(
                spec.get("rewrite_confidence"), _text(trace.get("query_spec_confidence"))
            ),
            "product_code": _text(spec.get("product_code")),
            "as_of_date": _text(spec.get("as_of_date"), _text(response.get("as_of_date"))),
            "decision_cutoff_at": _text(spec.get("decision_cutoff_at")),
            "immutable_numbers": _unique_strings(_items(spec.get("immutable_numbers"))),
            "exact_terms": _unique_strings(_items(spec.get("exact_terms"))),
            "must_not_assume": _unique_strings(_items(spec.get("must_not_assume"))),
            "normalized_terms": normalized_terms,
        },
        "normalization_rows": normalization_rows,
        "variant_rows": variant_rows,
        "route_rows": [route_summary[route] for route in (*ROUTE_ORDER, *extra_routes)],
        "rejection_rows": rejection_rows,
        "fusion": {
            "rrf_k": _count(fusion.get("rrf_k", trace.get("rrf_k"))),
            "input_lists": _unique_strings(_items(fusion.get("input_lists"))),
            "fused_count": _count(fusion.get("fused_count", trace.get("fused_count"))),
            "final_count": _count(
                trace.get("final_count", len(_items(response.get("candidates"))))
            ),
        },
        "rerank": {
            "applied": bool(trace.get("rerank_applied", channel_config.get("rerank", False))),
            "degraded": rerank_degraded,
            "reason": _text(rerank_reason),
            "error_detail": _text(trace.get("rerank_error_detail")),
            "version": _text(trace.get("reranker_version")),
        },
        "packing": {
            "available": bool(packing),
            "selected_count": len(selected_sections),
            "total_tokens_est": _count(
                packing.get("total_tokens_est", channel_config.get("packing_tokens"))
            ),
            "budget": _count(packing.get("budget")),
            "expanded_count": _count(packing.get("expanded_count")),
            "dropped": _unique_strings(_items(packing.get("dropped"))),
            "rows": packed_rows,
        },
        "warnings": warnings,
    }


def _derive_trace_state(
    *,
    delivery_status: str,
    contract_version: str,
    counts: Mapping[str, int],
    invalid_count: int,
) -> str:
    """Derive a root-cause state while retaining the API aggregate separately."""

    if delivery_status == "LEGACY_UNAVAILABLE" or not contract_version:
        return "LEGACY_UNAVAILABLE"
    if invalid_count or counts.get("INVALID", 0):
        return "INVALID"
    if delivery_status == "EMPTY":
        return "EMPTY"
    if counts.get("MISSING", 0):
        return "MISSING"
    if delivery_status == "DEGRADED" or counts.get("DEAD", 0):
        return "DEGRADED"
    if delivery_status == "PENDING" or counts.get("PENDING", 0) or counts.get("PROCESSING", 0):
        return "PENDING"
    if delivery_status == "COMPLETE":
        return "COMPLETE"
    return "DEGRADED"


def present_run_trace(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Present persisted invocation-ledger and telemetry-delivery state fail-soft."""

    trace = _mapping(payload)
    integrity = _mapping(trace.get("integrity"))
    delivery = _mapping(trace.get("delivery"))
    raw_counts = _mapping(delivery.get("counts"))
    counts = {name: _count(raw_counts.get(name)) for name in DELIVERY_COUNT_ORDER}
    delivery_status = _text(delivery.get("status"), "LEGACY_UNAVAILABLE").upper()
    contract_version = _text(delivery.get("contract_version"), "")
    invalid_count = _count(integrity.get("invalid_count"))
    state = _derive_trace_state(
        delivery_status=delivery_status,
        contract_version=contract_version,
        counts=counts,
        invalid_count=invalid_count,
    )

    invocation_rows: list[dict[str, Any]] = []
    for item in _items(trace.get("invocations")):
        invocation = _mapping(item)
        envelope = _mapping(invocation.get("envelope"))
        item_integrity = _mapping(invocation.get("integrity"))
        item_delivery = _mapping(invocation.get("delivery"))
        invocation_rows.append(
            {
                "类型": _text(invocation.get("kind", envelope.get("kind"))),
                "名称": _text(invocation.get("name", envelope.get("name"))),
                "终态": _text(invocation.get("status", envelope.get("status"))),
                "耗时(ms)": envelope.get("latency_ms", "—"),
                "错误码": _text(envelope.get("error_code")),
                "账本完整性": _text(item_integrity.get("status")),
                "Delivery": _text(item_delivery.get("status")),
                "投递次数": _count(item_delivery.get("attempts")),
                "投递错误": _text(
                    item_delivery.get("last_error_code", item_delivery.get("error_code"))
                ),
                "Invocation ID": _text(invocation.get("invocation_id")),
            }
        )

    event_rows = []
    for item in _items(trace.get("events")):
        event = _mapping(item)
        event_rows.append(
            {
                "#": _count(event.get("sequence_no")),
                "事件": _text(event.get("event_type")),
                "内容": event.get("payload") if event.get("payload") is not None else {},
                "时间": _text(event.get("occurred_at")),
            }
        )

    warnings: list[str] = []
    if state == "LEGACY_UNAVAILABLE":
        warnings.append("旧 Run 没有 invocation_v2 账本；这表示不可用，不表示调用数为零。")
    elif state == "EMPTY":
        warnings.append("v2 Run 当前没有调用记录；可能确实未调用外部服务，也可能在首笔提交前失败。")
    if state == "MISSING":
        warnings.append("存在有效调用账本但缺少配套 Outbox，遥测链路不完整。")
    if state == "INVALID":
        warnings.append("持久化载荷、哈希、投影或绑定校验失败；API 已隐藏不可信 envelope。")
    if state == "DEGRADED":
        warnings.append("遥测投递存在 DEAD 或其他降级；调用业务结果与投递结果需分别判断。")

    return {
        "state": state,
        "summary": {
            "integrity_status": _text(integrity.get("status")),
            "integrity_valid": integrity.get("valid"),
            "invalid_count": invalid_count,
            "delivery_status": delivery_status,
            "delivery_complete": delivery.get("complete"),
            "contract_version": contract_version or "—",
            "total": _count(delivery.get("total", len(invocation_rows))),
        },
        "count_rows": [{"状态": name, "数量": counts[name]} for name in DELIVERY_COUNT_ORDER],
        "invocation_rows": invocation_rows,
        "event_rows": event_rows,
        "warnings": warnings,
    }
