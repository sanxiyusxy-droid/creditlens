"""Recall@K 评测（任务 12，文档 §16.6）。

- Anchor 映射：gold_evidence_key -> 当前 Parse Run 的 Section ID（按 logical_key +
  article_anchor/heading 匹配），映射失败记为 unmapped，不静默丢弃；
- 映射必须限定在 GoldMappingScope（tenant + case_documents 绑定 + Snapshot 冻结
  parse_run）内，否则黄金证据可能命中跨租户/跨案件/非冻结解析版本的 Section，
  使 Recall 指标虚高（P0 加固）；
- Recall@K / MRR@K / AllRequiredEvidence@K；
- NDCG@K / Precision@K；
- Retrieved Evidence Precision / Recall（候选层证据命中率；无答案生成层，
  不宣称 Faithfulness / Citation Accuracy / Refusal Accuracy）；
- Temporal/ACL Leakage 在候选层单独计数，目标为 0。
- RefusalMetrics 仅预留给答案层；无 QA 生成层时不计入报告。
"""

import math
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.evaluation.gold_schema import GoldDataset, GoldEvidenceAnchor, GoldQuestion
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    Document,
    DocumentSection,
    DocumentVersion,
)
from creditlens.retrieval.contracts import RetrievedCandidate


@dataclass(frozen=True)
class GoldMappingScope:
    """Gold Anchor 映射的安全边界（与被评测检索链路使用同一冻结世界）。

    - tenant_id：租户隔离，避免跨租户同名 logical_key 命中；
    - case_id：只认 case_documents 绑定到本案件的 DocumentVersion；
    - allowed_parse_run_ids：只认本次 Snapshot 冻结的 Parse Run；它后来即使变为
      SUPERSEDED 仍属于该历史世界，同时拒绝新激活但未冻结的解析结果。
    """

    tenant_id: uuid.UUID
    case_id: uuid.UUID
    allowed_parse_run_ids: frozenset[uuid.UUID]


async def map_anchor_to_section_ids(
    session: AsyncSession, anchor: GoldEvidenceAnchor, scope: GoldMappingScope
) -> set[uuid.UUID]:
    """稳定锚点 -> Snapshot 冻结 Parse Run 下的 Section ID 集合（限定在 scope 内）。"""
    rows = (
        await session.scalars(
            select(DocumentSection)
            .join(DocumentVersion, DocumentVersion.id == DocumentSection.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(CaseDocument, CaseDocument.document_version_id == DocumentVersion.id)
            .where(
                Document.logical_key == anchor.logical_document_key,
                DocumentVersion.version_label == anchor.version_label,
                # 安全限定：租户 + 案件绑定（三张表同时校验，避免单点绕过）
                Document.tenant_id == scope.tenant_id,
                DocumentVersion.tenant_id == scope.tenant_id,
                DocumentSection.tenant_id == scope.tenant_id,
                CaseDocument.case_id == scope.case_id,
                # 只认本次 Snapshot 冻结的解析版本。旧 Snapshot 必须继续访问
                # 已变为 SUPERSEDED 的 ParseRun，不能再跟随 DocumentVersion.active_parse_run_id。
                DocumentSection.parse_run_id.in_(scope.allowed_parse_run_ids),
            )
        )
    ).all()
    matched: set[uuid.UUID] = set()
    for section in rows:
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
    ndcg_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    retrieved_evidence_precision: float = 0.0
    retrieved_evidence_recall: float = 0.0


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
            out[f"ndcg@{k}"] = sum(r.ndcg_at.get(k, 0.0) for r in self.question_results) / n
            out[f"precision@{k}"] = (
                sum(r.precision_at.get(k, 0.0) for r in self.question_results) / n
            )
        reciprocal = [
            1.0 / r.hit_rank if r.hit_rank and r.hit_rank <= 10 else 0.0
            for r in self.question_results
        ]
        out["mrr@10"] = sum(reciprocal) / n
        out["unmapped_questions"] = sum(1 for r in self.question_results if r.unmapped_keys)
        out["retrieved_evidence_precision"] = (
            sum(r.retrieved_evidence_precision for r in self.question_results) / n
        )
        out["retrieved_evidence_recall"] = (
            sum(r.retrieved_evidence_recall for r in self.question_results) / n
        )
        return out


async def evaluate_question(
    session: AsyncSession,
    dataset: GoldDataset,
    question: GoldQuestion,
    candidates: list[RetrievedCandidate],
    scope: GoldMappingScope,
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
            ids = await map_anchor_to_section_ids(session, anchor, scope) if anchor else set()
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
    ndcg_at: dict[int, float] = {}
    precision_at: dict[int, float] = {}
    for k in ks:
        ratio, full = covered(k)
        recall_at[k] = ratio
        all_required_at[k] = full
        # Precision@K：top-k 中有多少是 gold
        top_k_ids = set(ranked_ids[:k])
        gold_hits = len(top_k_ids & all_required_ids) if all_required_ids else 0
        precision_at[k] = gold_hits / k if k > 0 else 0.0
        # NDCG@K：graded relevance（必需证据=2，相关=1）
        ndcg_at[k] = _compute_ndcg(ranked_ids[:k], all_required_ids, k)

    # Retrieved Evidence Precision/Recall（候选层证据命中，基于全部候选）
    all_candidate_ids = set(ranked_ids)
    retrieved_evidence_precision = (
        len(all_candidate_ids & all_required_ids) / len(all_candidate_ids)
        if all_candidate_ids
        else 0.0
    )
    retrieved_evidence_recall = (
        len(all_candidate_ids & all_required_ids) / len(all_required_ids)
        if all_required_ids
        else 0.0
    )

    return QuestionResult(
        question_id=question.question_id,
        hit_rank=hit_rank,
        recall_at=recall_at,
        all_required_at=all_required_at,
        unmapped_keys=unmapped,
        ndcg_at=ndcg_at,
        precision_at=precision_at,
        retrieved_evidence_precision=retrieved_evidence_precision,
        retrieved_evidence_recall=retrieved_evidence_recall,
    )


def _compute_ndcg(ranked_ids: list[uuid.UUID], gold_ids: set[uuid.UUID], k: int) -> float:
    """NDCG@K：gold 命中 = relevance 2，未命中 = 0。"""
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for i, sid in enumerate(ranked_ids[:k]):
        if sid in gold_ids:
            dcg += 2.0 / math.log2(i + 2)  # i+2 因为位置从 1 开始
    # 理想排序：所有 gold 排最前
    ideal_count = min(len(gold_ids), k)
    idcg = sum(2.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


# ====================== Refusal 指标 ======================


@dataclass
class RefusalMetrics:
    """拒答相关指标。"""

    total_unanswerable: int = 0
    correct_refusals: int = 0
    total_answerable: int = 0
    false_refusals: int = 0

    @property
    def refusal_accuracy(self) -> float:
        """拒答题正确拒答率。"""
        return self.correct_refusals / self.total_unanswerable if self.total_unanswerable else 0.0

    @property
    def false_refusal_rate(self) -> float:
        """可答题被误拒率。"""
        return self.false_refusals / self.total_answerable if self.total_answerable else 0.0

    def summary(self) -> dict:
        return {
            "total_unanswerable": self.total_unanswerable,
            "correct_refusals": self.correct_refusals,
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "total_answerable": self.total_answerable,
            "false_refusals": self.false_refusals,
            "false_refusal_rate": round(self.false_refusal_rate, 4),
        }
