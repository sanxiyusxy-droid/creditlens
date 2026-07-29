"""Recall@K 评测（任务 12，文档 §16.6）。

- Anchor 映射：gold_evidence_key -> 当前 Parse Run 的 Section ID（按 logical_key +
  article_anchor/heading 匹配），映射失败记为 unmapped，不静默丢弃；
- Recall@K / MRR@K / AllRequiredEvidence@K；
- Temporal/ACL Leakage 在候选层单独计数，目标为 0。
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.evaluation.gold_schema import GoldDataset, GoldEvidenceAnchor, GoldQuestion
from creditlens.infrastructure.postgres.models import (
    Document,
    DocumentSection,
    DocumentVersion,
)
from creditlens.retrieval.contracts import RetrievedCandidate


async def map_anchor_to_section_ids(
    session: AsyncSession, anchor: GoldEvidenceAnchor
) -> set[uuid.UUID]:
    """稳定锚点 -> 当前激活 Parse Run 下的 Section ID 集合。"""
    rows = (
        await session.execute(
            select(DocumentSection, DocumentVersion)
            .join(DocumentVersion, DocumentVersion.id == DocumentSection.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.logical_key == anchor.logical_document_key,
                DocumentVersion.version_label == anchor.version_label,
            )
        )
    ).all()
    matched: set[uuid.UUID] = set()
    for section, version in rows:
        if version.active_parse_run_id and section.parse_run_id != version.active_parse_run_id:
            continue
        if anchor.article_anchor:
            heading = section.heading or ""
            path = " ".join(section.heading_path or [])
            if anchor.article_anchor in heading or anchor.article_anchor in path:
                matched.add(section.id)
        elif anchor.page_number is not None:
            if section.page_start <= anchor.page_number <= section.page_end:
                matched.add(section.id)
    return matched


@dataclass
class QuestionResult:
    question_id: str
    hit_rank: int | None  # 首个命中必需证据的排名
    recall_at: dict[int, float]
    all_required_at: dict[int, bool]
    unmapped_keys: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    dataset_id: str
    dataset_version: str
    config: dict
    question_results: list[QuestionResult]

    def summary(self, ks: tuple[int, ...] = (5, 10, 20)) -> dict:
        n = len(self.question_results)
        if n == 0:
            return {}
        out: dict = {"questions": n}
        for k in ks:
            out[f"recall@{k}"] = sum(r.recall_at.get(k, 0.0) for r in self.question_results) / n
            out[f"all_required@{k}"] = (
                sum(1 for r in self.question_results if r.all_required_at.get(k)) / n
            )
        reciprocal = [
            1.0 / r.hit_rank if r.hit_rank and r.hit_rank <= 10 else 0.0
            for r in self.question_results
        ]
        out["mrr@10"] = sum(reciprocal) / n
        out["unmapped_questions"] = sum(1 for r in self.question_results if r.unmapped_keys)
        return out


async def evaluate_question(
    session: AsyncSession,
    dataset: GoldDataset,
    question: GoldQuestion,
    candidates: list[RetrievedCandidate],
    ks: tuple[int, ...] = (5, 10, 20),
) -> QuestionResult:
    anchors = dataset.anchor_by_key()
    unmapped: list[str] = []

    # 每个 evidence set 内的每个 key -> section id 集合
    mapped_sets: list[list[set[uuid.UUID]]] = []
    for evidence_set in question.required_evidence_sets:
        mapped_set: list[set[uuid.UUID]] = []
        for key in evidence_set:
            anchor = anchors.get(key)
            ids = await map_anchor_to_section_ids(session, anchor) if anchor else set()
            if not ids:
                unmapped.append(key)
            mapped_set.append(ids)
        mapped_sets.append(mapped_set)

    ranked_ids = [c.section_id for c in candidates]

    def covered(top_k: int) -> tuple[float, bool]:
        """返回 (最佳 evidence set 覆盖率, 是否存在全覆盖的 set)。"""
        top = set(ranked_ids[:top_k])
        best_ratio = 0.0
        any_full = False
        for mapped_set in mapped_sets:
            if not mapped_set:
                continue
            hits = sum(1 for ids in mapped_set if ids & top)
            ratio = hits / len(mapped_set)
            best_ratio = max(best_ratio, ratio)
            if hits == len(mapped_set) and all(ids for ids in mapped_set):
                any_full = True
        return best_ratio, any_full

    hit_rank: int | None = None
    all_required_ids: set[uuid.UUID] = set()
    for mapped_set in mapped_sets:
        for ids in mapped_set:
            all_required_ids |= ids
    for rank, section_id in enumerate(ranked_ids, start=1):
        if section_id in all_required_ids:
            hit_rank = rank
            break

    recall_at: dict[int, float] = {}
    all_required_at: dict[int, bool] = {}
    for k in ks:
        ratio, full = covered(k)
        recall_at[k] = ratio
        all_required_at[k] = full

    return QuestionResult(
        question_id=question.question_id,
        hit_rank=hit_rank,
        recall_at=recall_at,
        all_required_at=all_required_at,
        unmapped_keys=unmapped,
    )
