"""Context Packing（文档 §8.13，v1.1 统一编排器）。

将融合+精排后的完整候选池打包为有限 Token 预算的上下文：
- Token Budget：保守上界（1 字符 = 1 token，显式保守策略，不依赖具体 tokenizer）；
- 来源配额：单文档不超过 budget * max_per_document_ratio；
- 去重：text_hash 级 + heading_path 完全相同取最高分；
- 丢弃超预算候选后继续用后续候选补位（不中断）；
- 相邻段落扩展：命中 Section 的 ordinal 前后 1 条同章 Section，
  必须重新通过 ACL/Snapshot/质量/时点回表复核后才可补入（v1.1 P0）。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.infrastructure.postgres.models import (
    Document,
    DocumentSection,
    DocumentVersion,
)
from creditlens.retrieval.contracts import RetrievedCandidate, TrustedRequestContext
from creditlens.retrieval.dense import verify_candidate

if TYPE_CHECKING:
    from creditlens.application.snapshot_service import SnapshotContext


class PackedSection(BaseModel):
    """打包后的单条上下文段落。"""

    section_id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    parse_run_id: uuid.UUID
    heading_path: list[str]
    text: str
    text_hash: str
    page_start: int
    page_end: int
    tokens_est: int
    rank: int  # 融合/精排后的原始排名
    expanded: bool = False  # 是否为相邻段落扩展


class PackedContext(BaseModel):
    """Context Packing 输出。"""

    sections: list[PackedSection]
    total_tokens_est: int
    budget: int
    dropped: list[str] = Field(default_factory=list)  # 被丢弃的 section_id（超预算/配额）
    expanded_count: int = 0


def estimate_tokens(text: str) -> int:
    """Token 保守上界：1 字符 = 1 token。

    未接入真实 tokenizer 前的显式保守策略：对中文（常见 BPE 约 1 字 ≈ 1 token）
    与英文（约 4 字符 ≈ 1 token）都不低估，确保打包结果不超真实 budget。"""
    return max(1, len(text))


async def pack_context(
    session: AsyncSession,
    candidates: list[RetrievedCandidate],
    trusted: TrustedRequestContext | None = None,
    snapshot: SnapshotContext | None = None,
    token_budget: int = 4096,
    max_per_document_ratio: float = 0.6,
    expand_adjacent: bool = True,
) -> PackedContext:
    """将排序后的完整候选池打包为有限预算上下文。

    参数：
        candidates: 已按融合/精排排序的完整候选池（非 final_limit 截断后），
            超预算丢弃后继续用后续候选补位。
        trusted/snapshot: 相邻段落扩展时重新执行 ACL/Snapshot/质量/时点复核。
        token_budget: 总 Token 预算。
        max_per_document_ratio: 单文档最大占比。
        expand_adjacent: 是否扩展相邻段落。
    """
    doc_budget = int(token_budget * max_per_document_ratio)
    doc_usage: dict[uuid.UUID, int] = {}
    packed: list[PackedSection] = []
    dropped: list[str] = []
    seen_hashes: set[str] = set()
    seen_headings: set[tuple] = set()
    total_tokens = 0

    for candidate in candidates:
        # text_hash 去重
        if candidate.text_hash in seen_hashes:
            continue
        # heading_path 完全相同去重（保留最高分即最早出现的）
        heading_key = tuple(candidate.heading_path)
        if heading_key and heading_key in seen_headings:
            continue

        tokens = estimate_tokens(candidate.text)
        doc_id = candidate.document_id

        # 总预算检查
        if total_tokens + tokens > token_budget:
            dropped.append(str(candidate.section_id))
            continue
        # 单文档配额检查
        if doc_usage.get(doc_id, 0) + tokens > doc_budget:
            dropped.append(str(candidate.section_id))
            continue

        seen_hashes.add(candidate.text_hash)
        if heading_key:
            seen_headings.add(heading_key)
        doc_usage[doc_id] = doc_usage.get(doc_id, 0) + tokens
        total_tokens += tokens

        packed.append(
            PackedSection(
                section_id=candidate.section_id,
                document_id=candidate.document_id,
                document_version_id=candidate.document_version_id,
                parse_run_id=candidate.parse_run_id,
                heading_path=candidate.heading_path,
                text=candidate.text,
                text_hash=candidate.text_hash,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                tokens_est=tokens,
                rank=candidate.rank,
                expanded=False,
            )
        )

    # 相邻段落扩展（在预算允许时补入 ordinal 前后 1 条同章 Section）。
    # v1.1 P0：扩展段落是“新候选”，必须重新执行 ACL/Snapshot/质量/时点复核；
    # 并修复 document_id/document_version_id 混用（从 DocumentVersion 回表取真实值）。
    expanded_count = 0
    if expand_adjacent and packed and trusted is not None:
        expand_candidates: list[tuple[int, DocumentSection]] = []
        packed_ids = {p.section_id for p in packed}

        for section in packed:
            db_section = await session.get(DocumentSection, section.section_id)
            if db_section is None or db_section.parent_section_id is None:
                continue
            siblings = (
                await session.scalars(
                    select(DocumentSection)
                    .where(
                        DocumentSection.parent_section_id == db_section.parent_section_id,
                        DocumentSection.section_type.in_(["ARTICLE", "PARAGRAPH"]),
                    )
                    .order_by(DocumentSection.ordinal)
                )
            ).all()
            ordinal = db_section.ordinal
            for sibling in siblings:
                if sibling.id in packed_ids or sibling.id == db_section.id:
                    continue
                if abs(sibling.ordinal - ordinal) <= 1:
                    expand_candidates.append((section.rank, sibling))
                    packed_ids.add(sibling.id)

        # 按原始排名优先级插入扩展段落
        for rank, sibling in sorted(expand_candidates, key=lambda x: x[0]):
            tokens = estimate_tokens(sibling.text)
            # 复核前置：先过预算/配额，再回表（避免无谓查询）
            if total_tokens + tokens > token_budget:
                continue
            version = await session.get(DocumentVersion, sibling.document_version_id)
            if version is None:
                continue
            document = await session.get(Document, version.document_id)
            if document is not None and doc_usage.get(document.id, 0) + tokens > doc_budget:
                continue
            candidate = RetrievedCandidate(
                section_id=sibling.id,
                document_id=document.id if document else version.document_id,
                document_version_id=version.id,
                parse_run_id=sibling.parse_run_id,
                page_start=sibling.page_start,
                page_end=sibling.page_end,
                heading_path=sibling.heading_path or [],
                text="",
                text_hash=sibling.text_hash,
                channel="PACKING_ADJACENT",
                rank=rank,
                raw_score=0.0,
            )
            payload = {
                "document_id": str(document.id) if document else None,
                "document_type": document.document_type if document else None,
            }
            reason = await verify_candidate(session, trusted, candidate, payload, snapshot)
            if reason is not None:
                dropped.append(str(sibling.id))
                continue
            if sibling.text_hash in seen_hashes:
                continue
            total_tokens += tokens
            doc_id = document.id if document else version.document_id
            doc_usage[doc_id] = doc_usage.get(doc_id, 0) + tokens
            seen_hashes.add(sibling.text_hash)
            expanded_count += 1
            packed.append(
                PackedSection(
                    section_id=sibling.id,
                    document_id=doc_id,
                    document_version_id=version.id,
                    parse_run_id=sibling.parse_run_id,
                    heading_path=sibling.heading_path or [],
                    text=sibling.text,
                    text_hash=sibling.text_hash,
                    page_start=sibling.page_start,
                    page_end=sibling.page_end,
                    tokens_est=tokens,
                    rank=rank,
                    expanded=True,
                )
            )

    return PackedContext(
        sections=packed,
        total_tokens_est=total_tokens,
        budget=token_budget,
        dropped=dropped,
        expanded_count=expanded_count,
    )
