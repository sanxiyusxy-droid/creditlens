"""高级 RAG 单元测试：BM25 Sparse、RRF、QuerySpec Validator、Reranker。"""

import uuid
from datetime import UTC, date, datetime

from creditlens.retrieval.contracts import RetrievedCandidate, TrustedRequestContext
from creditlens.retrieval.fusion import rrf_fuse
from creditlens.retrieval.query_spec import (
    build_query_spec,
    safe_fallback,
    validate_query_spec,
)
from creditlens.retrieval.rerank import LexicalOverlapReranker
from creditlens.retrieval.sparse import Bm25SparseEncoder


def _candidate(section=None, doc=None, rank=1, text_hash=None, channel="DENSE"):
    return RetrievedCandidate(
        section_id=section or uuid.uuid4(),
        document_id=doc or uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        parse_run_id=uuid.uuid4(),
        page_start=1,
        page_end=1,
        heading_path=[],
        text="",
        text_hash=text_hash or uuid.uuid4().hex,
        channel=channel,
        rank=rank,
        raw_score=1.0 / rank,
    )


def _trusted() -> TrustedRequestContext:
    return TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        as_of_date=date(2026, 6, 30),
        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
    )


class TestBm25Sparse:
    def test_document_encoding_deterministic(self):
        encoder = Bm25SparseEncoder()
        a = encoder.encode_document("流动资金贷款的资产负债率不得高于百分之七十")
        b = encoder.encode_document("流动资金贷款的资产负债率不得高于百分之七十")
        assert a == b
        assert a[0], "应产生非空稀疏向量"

    def test_query_includes_exact_terms(self):
        encoder = Bm25SparseEncoder()
        base_indices, _ = encoder.encode_query("准入要求")
        with_term, _ = encoder.encode_query("准入要求", extra_terms=["流动资金贷款"])
        assert set(base_indices) < set(with_term)

    def test_tf_weighting(self):
        encoder = Bm25SparseEncoder()
        _, values = encoder.encode_document("贷款 贷款 贷款 企业")
        assert max(values) >= 3.0


class TestRrfFusion:
    def test_candidate_in_both_channels_ranks_higher(self):
        shared_section = uuid.uuid4()
        shared_a = _candidate(section=shared_section, rank=2, text_hash="h1")
        dense_only = _candidate(rank=1, text_hash="h2")
        shared_b = _candidate(section=shared_section, rank=1, text_hash="h1", channel="SPARSE")
        sparse_only = _candidate(rank=2, text_hash="h3", channel="SPARSE")

        fused = rrf_fuse({"DENSE": [dense_only, shared_a], "SPARSE": [shared_b, sparse_only]})
        assert fused[0].candidate.section_id == shared_section
        assert set(fused[0].channel_ranks) == {"DENSE", "SPARSE"}

    def test_dedup_by_text_hash(self):
        a = _candidate(rank=1, text_hash="same")
        b = _candidate(rank=2, text_hash="same")
        fused = rrf_fuse({"DENSE": [a, b]})
        assert len(fused) == 1

    def test_per_document_cap(self):
        doc = uuid.uuid4()
        candidates = [_candidate(doc=doc, rank=i) for i in range(1, 12)]
        fused = rrf_fuse({"DENSE": candidates}, max_candidates_per_document=8)
        assert len(fused) == 8


class TestQuerySpec:
    def test_build_preserves_hard_constraints(self):
        trusted = _trusted()
        spec = build_query_spec(trusted, "流贷申请 500 万元，第六条要求是什么？")
        assert spec.as_of_date == trusted.as_of_date
        assert "500" in spec.immutable_numbers
        assert "第六条" in spec.exact_terms
        assert "流动资金贷款" in spec.exact_terms  # 术语归一
        result = validate_query_spec(trusted, spec)
        assert result.ok, result.violations

    def test_validator_rejects_changed_date(self):
        trusted = _trusted()
        spec = build_query_spec(trusted, "准入要求")
        spec.as_of_date = date(2027, 1, 1)
        result = validate_query_spec(trusted, spec)
        assert "AS_OF_DATE_CHANGED" in result.violations

    def test_validator_rejects_lost_number(self):
        trusted = _trusted()
        spec = build_query_spec(trusted, "申请金额 500 万元是否超限")
        spec.standalone_query = "申请金额是否超限"
        spec.query_variants = [
            v.model_copy(update={"text": "申请金额是否超限"}) for v in spec.query_variants
        ]
        result = validate_query_spec(trusted, spec)
        assert any(v.startswith("NUMBER_LOST") for v in result.violations)

    def test_validator_requires_original_dense_variant(self):
        trusted = _trusted()
        spec = build_query_spec(trusted, "准入要求")
        spec.query_variants = [v for v in spec.query_variants if v.route != "dense"]
        result = validate_query_spec(trusted, spec)
        assert any(v.startswith("MISSING_ORIGINAL_DENSE") for v in result.violations)

    def test_safe_fallback_low_confidence(self):
        trusted = _trusted()
        spec = safe_fallback(trusted, "准入要求")
        assert spec.rewrite_confidence == "LOW"
        assert validate_query_spec(trusted, spec).ok


class TestReranker:
    async def test_lexical_overlap_ranks_relevant_higher(self):
        reranker = LexicalOverlapReranker()
        scores = await reranker.score(
            "资产负债率不得高于多少",
            ["借款人资产负债率不得高于百分之七十", "贷款利率按照定价管理办法执行"],
        )
        assert scores[0] > scores[1]
