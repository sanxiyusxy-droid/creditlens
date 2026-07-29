"""Weighted RRF 融合与去重（任务 15，文档 §8.9/§8.10）。

- 不直接相加 Cosine/BM25/Reranker 原始分数；
- 应用层实现 Weighted RRF，固定 k 与权重并可 Trace；
- 按 section_id/text_hash 去重；
- 单文档候选数上限，防止一个文档占满上下文。
"""

from collections import defaultdict
from dataclasses import dataclass, field

from creditlens.retrieval.contracts import RetrievedCandidate

# E0 融合基线使用等权；加权值是第二组实验起点（文档 §8.9）
DEFAULT_ROUTE_WEIGHTS = {
    "DENSE": 1.0,
    "SPARSE": 1.0,
    "SUMMARY": 1.0,
    "EXACT": 1.0,
}


@dataclass
class FusedCandidate:
    candidate: RetrievedCandidate
    fusion_score: float
    channel_ranks: dict[str, int] = field(default_factory=dict)


def rrf_fuse(
    ranked_lists: dict[str, list[RetrievedCandidate]],
    rrf_k: int = 60,
    route_weights: dict[str, float] | None = None,
    max_candidates_per_document: int = 8,
    limit: int = 80,
) -> list[FusedCandidate]:
    """ranked_lists: {channel_name: 已按 rank 排序且通过回表复核的候选}。

    score(d) = sum_r w_r / (k + rank_r(d))，按 section_id 聚合。
    """
    weights = route_weights or DEFAULT_ROUTE_WEIGHTS
    by_section: dict = {}
    scores: dict = defaultdict(float)
    channel_ranks: dict = defaultdict(dict)

    for channel, candidates in ranked_lists.items():
        weight = weights.get(channel, 1.0)
        seen_in_channel: set = set()
        for candidate in candidates:
            key = candidate.section_id
            if key in seen_in_channel:  # 同通道内先按 section 去重，保留最优 rank
                continue
            seen_in_channel.add(key)
            scores[key] += weight / (rrf_k + candidate.rank)
            channel_ranks[key][channel] = candidate.rank
            if key not in by_section:
                by_section[key] = candidate

    # text_hash 二次去重（不同 section 相同文本保留一个）
    fused = sorted(
        (
            FusedCandidate(
                candidate=by_section[key],
                fusion_score=scores[key],
                channel_ranks=dict(channel_ranks[key]),
            )
            for key in by_section
        ),
        key=lambda f: f.fusion_score,
        reverse=True,
    )
    seen_hashes: set[str] = set()
    doc_counts: dict = defaultdict(int)
    result: list[FusedCandidate] = []
    for item in fused:
        text_hash = item.candidate.text_hash
        doc_id = item.candidate.document_id
        if text_hash in seen_hashes:
            continue
        if doc_counts[doc_id] >= max_candidates_per_document:
            continue
        seen_hashes.add(text_hash)
        doc_counts[doc_id] += 1
        result.append(item)
        if len(result) >= limit:
            break
    return result
