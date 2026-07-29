"""QuerySpec 契约、规则化 Rewrite 与确定性 Validator（任务 18，文档 §8.3/§8.4）。

- Rewrite 不允许改变 case/borrower/product/amount/as_of_date/decision_cutoff_at；
- 原始 Query 至少保留一条 Dense 和一条 Sparse 支路；
- 数字、条款号、否定词必须保留；
- Validator 失败 => 丢弃 Rewrite，退回"原 Query + 安全词典扩展"。

MVP 使用确定性规则 Rewrite（术语词典 + 关键词扩展）；LLM Rewrite 后续通过
LLMProvider 插入，但其输出仍必须通过同一 Validator。
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from creditlens.retrieval.contracts import TrustedRequestContext

# 术语归一词典（保留原词，添加规范词）——版本化资产
TERM_NORMALIZATION = {
    "流贷": "流动资金贷款",
    "流资贷款": "流动资金贷款",
    "资负率": "资产负债率",
    "经营现金流": "经营活动现金流量净额",
    "两倍流动比": "流动比率",
}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零\d]+条")
_NEGATIONS = ["不得", "不能", "禁止", "不予", "不满足", "未", "没有", "无", "不"]


class ComparisonContext(BaseModel):
    label: str
    as_of_date: date
    decision_cutoff_at: datetime


class RetrievalQueryVariant(BaseModel):
    variant_id: str
    subquery_id: str
    text: str
    origin: Literal["ORIGINAL", "NORMALIZED", "SEMANTIC_REWRITE", "LEXICAL_EXPANSION", "COUNTER"]
    route: Literal["dense", "sparse", "summary", "exact"]


class RetrievalSubQuery(BaseModel):
    subquery_id: str
    question: str
    sparse_terms: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    route_hints: list[Literal["dense", "sparse", "summary", "exact", "sql"]] = Field(
        default_factory=lambda: ["dense", "sparse"]
    )
    required_evidence_types: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)


class QuerySpec(BaseModel):
    schema_version: str = "1.0"
    original_query: str
    standalone_query: str
    intent: Literal[
        "POLICY_QA",
        "POLICY_VERSION_COMPARE",
        "FINANCIAL_FACT",
        "FINANCIAL_TREND",
        "FULL_CREDIT_REVIEW",
        "MATERIAL_COMPLETENESS",
        "CROSS_SOURCE_CONFLICT",
        "UNANSWERABLE",
    ] = "POLICY_QA"
    product_code: str
    as_of_date: date
    decision_cutoff_at: datetime
    immutable_numbers: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    subqueries: list[RetrievalSubQuery] = Field(default_factory=list)
    query_variants: list[RetrievalQueryVariant] = Field(default_factory=list)
    comparison_contexts: list[ComparisonContext] = Field(default_factory=list)
    must_not_assume: list[str] = Field(default_factory=list)
    rewrite_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"

    MAX_SUBQUERIES: int = 5
    MAX_VARIANTS_PER_SUBQUERY: int = 3


@dataclass
class ValidationResult:
    ok: bool
    violations: list[str]


def extract_immutables(query: str) -> tuple[list[str], list[str]]:
    """抽取原问题中的数字与条款号（Rewrite 必须保留）。"""
    numbers = _NUMBER_RE.findall(query)
    articles = _ARTICLE_RE.findall(query)
    return numbers, articles


def build_query_spec(trusted: TrustedRequestContext, user_query: str) -> QuerySpec:
    """规则化 Query Understanding：术语归一 + 词法扩展；不改硬约束。"""
    numbers, articles = extract_immutables(user_query)

    normalized_terms = [
        normalized for raw, normalized in TERM_NORMALIZATION.items() if raw in user_query
    ]
    exact_terms = articles + normalized_terms

    subquery = RetrievalSubQuery(
        subquery_id="main",
        question=user_query,
        sparse_terms=normalized_terms,
        exact_terms=exact_terms,
        priority=5,
    )

    variants = [
        RetrievalQueryVariant(
            variant_id="main_original_dense",
            subquery_id="main",
            text=user_query,
            origin="ORIGINAL",
            route="dense",
        ),
        RetrievalQueryVariant(
            variant_id="main_original_sparse",
            subquery_id="main",
            text=user_query,
            origin="ORIGINAL",
            route="sparse",
        ),
    ]
    if normalized_terms:
        expansion = f"{user_query} {' '.join(normalized_terms)}"
        variants.append(
            RetrievalQueryVariant(
                variant_id="main_lexical_sparse",
                subquery_id="main",
                text=expansion,
                origin="LEXICAL_EXPANSION",
                route="sparse",
            )
        )

    return QuerySpec(
        original_query=user_query,
        standalone_query=user_query,
        product_code=trusted.product_code,
        as_of_date=trusted.as_of_date,
        decision_cutoff_at=trusted.decision_cutoff_at,
        immutable_numbers=numbers,
        exact_terms=exact_terms,
        subqueries=[subquery],
        query_variants=variants,
        must_not_assume=["缺失财务指标不得估算", "不得自动给出批准或拒贷结论"],
    )


def validate_query_spec(trusted: TrustedRequestContext, spec: QuerySpec) -> ValidationResult:
    """确定性 Rewrite Validator（文档 §8.4）。可选语义检查不能覆盖此处失败。"""
    violations: list[str] = []

    if spec.product_code != trusted.product_code:
        violations.append("PRODUCT_CODE_CHANGED")
    if spec.as_of_date != trusted.as_of_date:
        violations.append("AS_OF_DATE_CHANGED")
    if spec.decision_cutoff_at != trusted.decision_cutoff_at:
        violations.append("DECISION_CUTOFF_CHANGED")

    # 原始数字与条款号必须保留在 standalone 或至少一个变体中
    numbers, articles = extract_immutables(spec.original_query)
    searchable = spec.standalone_query + " " + " ".join(v.text for v in spec.query_variants)
    for number in numbers:
        if number not in searchable:
            violations.append(f"NUMBER_LOST:{number}")
    for article in articles:
        if article not in searchable:
            violations.append(f"ARTICLE_LOST:{article}")

    # 原始否定词必须保留
    for negation in _NEGATIONS:
        if negation in spec.original_query and negation not in spec.standalone_query:
            violations.append(f"NEGATION_LOST:{negation}")
            break

    if len(spec.subqueries) > spec.MAX_SUBQUERIES:
        violations.append("TOO_MANY_SUBQUERIES")

    subquery_ids = {s.subquery_id for s in spec.subqueries}
    variant_counts: dict[str, int] = {}
    original_routes: dict[str, set] = {}
    for variant in spec.query_variants:
        if variant.subquery_id not in subquery_ids:
            violations.append(f"VARIANT_ORPHAN:{variant.variant_id}")
        variant_counts[variant.subquery_id] = variant_counts.get(variant.subquery_id, 0) + 1
        if variant.origin == "ORIGINAL":
            original_routes.setdefault(variant.subquery_id, set()).add(variant.route)
    for subquery_id, count in variant_counts.items():
        if count > spec.MAX_VARIANTS_PER_SUBQUERY:
            violations.append(f"TOO_MANY_VARIANTS:{subquery_id}")
    for subquery_id in subquery_ids:
        routes = original_routes.get(subquery_id, set())
        if "dense" not in routes:
            violations.append(f"MISSING_ORIGINAL_DENSE:{subquery_id}")

    # 版本比较必须恰有两个 ComparisonContext；其他意图为空
    if spec.intent == "POLICY_VERSION_COMPARE":
        if len(spec.comparison_contexts) != 2:
            violations.append("COMPARISON_CONTEXTS_NOT_TWO")
    elif spec.comparison_contexts:
        violations.append("UNEXPECTED_COMPARISON_CONTEXTS")

    return ValidationResult(ok=not violations, violations=violations)


def safe_fallback(trusted: TrustedRequestContext, original_query: str) -> QuerySpec:
    """Validator 失败：丢弃 Rewrite，使用原 Query + 安全词典扩展。"""
    spec = build_query_spec(trusted, original_query)
    spec.rewrite_confidence = "LOW"
    return spec
