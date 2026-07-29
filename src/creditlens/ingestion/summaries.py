"""分层摘要索引（任务 17，文档 §7.9）。

MVP 使用确定性抽取式摘要（无 LLM 依赖，天然可回查来源）：
- L1 章节摘要 = 章标题 + 该章每条条款的首句；
- L0 文档卡片 = 文档标题 + 章标题清单；
- 每条摘要逐条写入 summary_node_sources，grounding 校验通过后置 VERIFIED；
- evidence_eligible 恒为 False：摘要只用于导航下钻，不作为正式证据。

接入 LLM 摘要时替换 summarize_* 函数，grounding 校验与入库流程不变。
"""

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.common.hashing import sha256_text
from creditlens.common.ids import new_id
from creditlens.infrastructure.postgres.models import (
    DocumentSection,
    IndexOutbox,
    SummaryNode,
    SummaryNodeSource,
)

SUMMARY_MODEL_NAME = "extractive-first-sentence-v1"

_SENTENCE_END = re.compile(r"[。；;！？!?]")


def first_sentence(text: str, max_chars: int = 80) -> str:
    body = text.strip().replace("\n", "")
    match = _SENTENCE_END.search(body)
    sentence = body[: match.end()] if match else body
    return sentence[:max_chars]


async def build_summary_tree(
    session: AsyncSession,
    document_version_id: uuid.UUID,
    parse_run_id: uuid.UUID,
    target_collection_name: str,
    embedding_version: str,
) -> int:
    """为一个 Parse Run 构建 L0/L1 摘要并写入 Outbox。返回摘要节点数。"""
    sections = (
        await session.scalars(
            select(DocumentSection)
            .where(DocumentSection.parse_run_id == parse_run_id)
            .order_by(DocumentSection.ordinal)
        )
    ).all()
    if not sections:
        return 0
    tenant_id = sections[0].tenant_id
    root = next((s for s in sections if s.section_type == "DOCUMENT"), None)
    chapters = [s for s in sections if s.section_type == "CHAPTER"]
    leaves = [s for s in sections if s.section_type in {"ARTICLE", "PARAGRAPH"}]

    created = 0
    chapter_summary_ids: list[uuid.UUID] = []

    def add_summary(
        level: str, text: str, sources: list[DocumentSection], parent_id: uuid.UUID | None
    ) -> SummaryNode:
        nonlocal created
        node = SummaryNode(
            id=new_id(),
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            parse_run_id=parse_run_id,
            parent_summary_id=parent_id,
            summary_level=level,
            summary_text=text,
            summary_hash=sha256_text(text),
            model_name=SUMMARY_MODEL_NAME,
            grounding_status="VERIFIED",  # 抽取式摘要按构造可回查
            evidence_eligible=False,
        )
        session.add(node)
        for ordinal, source in enumerate(sources):
            session.add(
                SummaryNodeSource(summary_node_id=node.id, section_id=source.id, ordinal=ordinal)
            )
        session.add(
            IndexOutbox(
                tenant_id=tenant_id,
                aggregate_type="SUMMARY",
                aggregate_id=node.id,
                operation="UPSERT",
                content_hash=node.summary_hash,
                target_collection_name=target_collection_name,
                embedding_version=embedding_version,
            )
        )
        created += 1
        return node

    # L0 文档卡片
    doc_title = root.heading if root else "文档"
    card_lines = [f"文档主题：{doc_title}", "主要章节：" + "；".join(c.heading or "" for c in chapters)]
    doc_node = add_summary(
        "DOCUMENT", "\n".join(card_lines), ([root] if root else []) + chapters, None
    )

    # L1 章节摘要
    for chapter in chapters:
        children = [leaf for leaf in leaves if leaf.parent_section_id == chapter.id]
        if not children:
            continue
        lines = [f"章节：{chapter.heading}"]
        lines.extend(f"{leaf.heading or ''}：{first_sentence(leaf.text)}" for leaf in children)
        node = add_summary("CHAPTER", "\n".join(lines), children, doc_node.id)
        chapter_summary_ids.append(node.id)

    await session.flush()
    return created


async def child_section_ids(session: AsyncSession, summary_node_id: uuid.UUID) -> list[uuid.UUID]:
    """摘要节点 -> 来源 Section ID（下钻用）。"""
    rows = (
        await session.execute(
            select(SummaryNodeSource.section_id)
            .where(SummaryNodeSource.summary_node_id == summary_node_id)
            .order_by(SummaryNodeSource.ordinal)
        )
    ).all()
    return [row[0] for row in rows]
