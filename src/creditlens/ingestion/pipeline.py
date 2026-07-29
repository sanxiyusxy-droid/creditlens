"""文档入库管线（任务 6/7/9 编排，文档 §7.12）。

流程：
1. 幂等创建 ParseRun（同 document_version + parser + config_hash 不重复解析）；
2. 解析 PDF -> 质量信号；
3. 结构切分 -> SectionDraft；
4. 同一事务：写 document_sections + index_outbox(PENDING) + parse_run=INDEX_PENDING；
5. Qdrant 写入由 Index Worker 消费 Outbox；全部完成后 Activation 原子切换
   document_versions.active_parse_run_id。
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import ObjectStorePort
from creditlens.common.clock import utc_now
from creditlens.common.errors import DataQualityBlockedError, DocumentParseFailedError
from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.parsers.pymupdf_parser import (
    PARSER_NAME,
    PARSER_VERSION,
    PyMuPdfParser,
)
from creditlens.infrastructure.postgres.models import (
    Document,
    DocumentSection,
    DocumentVersion,
    IndexOutbox,
    ParseRun,
)
from creditlens.ingestion.chunking.structure import build_sections


@dataclass
class IngestResult:
    parse_run_id: uuid.UUID
    section_count: int
    outbox_count: int
    reused: bool


class IngestionPipeline:
    def __init__(
        self,
        object_store: ObjectStorePort,
        target_collection_name: str,
        embedding_version: str,
        sparse_encoder_version: str | None = None,
        ingestion_config_hash: str | None = None,
        summary_collection_name: str | None = None,
    ):
        self._store = object_store
        self._parser = PyMuPdfParser()
        self._collection = target_collection_name
        self._summary_collection = summary_collection_name
        self._embedding_version = embedding_version
        self._sparse_encoder_version = sparse_encoder_version
        self._config_hash = ingestion_config_hash or sha256_text(
            f"{PARSER_NAME}:{PARSER_VERSION}:structure-chunker-v1"
        )

    async def ingest(self, session: AsyncSession, document_version_id: uuid.UUID) -> IngestResult:
        version = await session.get(DocumentVersion, document_version_id)
        if version is None:
            raise DocumentParseFailedError("document version 不存在")
        document = await session.get(Document, version.document_id)
        assert document is not None

        # 幂等：同一 version + parser + config 已成功/进行中的 ParseRun 直接复用
        existing = await session.scalar(
            select(ParseRun).where(
                ParseRun.document_version_id == document_version_id,
                ParseRun.parser_name == PARSER_NAME,
                ParseRun.parser_version == PARSER_VERSION,
                ParseRun.config_hash == self._config_hash,
                ParseRun.status.in_(
                    ["INDEX_PENDING", "SUCCEEDED", "SUCCEEDED_WITH_WARNINGS", "RUNNING"]
                ),
            )
        )
        if existing is not None:
            return IngestResult(existing.id, 0, 0, reused=True)

        max_gen = (
            await session.scalar(
                select(ParseRun.generation_no)
                .where(ParseRun.document_version_id == document_version_id)
                .order_by(ParseRun.generation_no.desc())
                .limit(1)
            )
            or 0
        )
        parse_run = ParseRun(
            tenant_id=version.tenant_id,
            document_version_id=document_version_id,
            generation_no=max_gen + 1,
            status="RUNNING",
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            config_hash=self._config_hash,
            started_at=utc_now(),
        )
        session.add(parse_run)
        await session.flush()

        data = self._store.get(version.object_uri)
        try:
            parsed = self._parser.parse(data)
        except Exception as exc:  # 解析失败保留原始对象与失败记录
            parse_run.status = "FAILED"
            parse_run.error_code = "DOCUMENT_PARSE_FAILED"
            parse_run.finished_at = utc_now()
            raise DocumentParseFailedError(str(exc)) from exc

        if not parsed.pages:
            parse_run.status = "FAILED"
            parse_run.error_code = "DATA_QUALITY_BLOCKED"
            parse_run.finished_at = utc_now()
            raise DataQualityBlockedError("解析结果无页面")

        version.page_count = parsed.metadata.get("page_count")
        drafts = build_sections(parsed, document.title, document.document_type)
        leaf_types = {"ARTICLE", "PARAGRAPH"}

        outbox_count = 0
        for draft in drafts:
            session.add(
                DocumentSection(
                    id=draft.id,
                    tenant_id=version.tenant_id,
                    document_version_id=version.id,
                    parse_run_id=parse_run.id,
                    parent_section_id=draft.parent_id,
                    section_type=draft.section_type,
                    ordinal=draft.ordinal,
                    heading=draft.heading,
                    heading_path=draft.heading_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    bbox=draft.bbox,
                    text=draft.text,
                    text_hash=draft.text_hash,
                    token_count=draft.token_count,
                    previous_section_id=draft.previous_id,
                    next_section_id=draft.next_id,
                )
            )
            # 只有叶子文本进入检索索引；结构节点仅用于层级与扩展
            if draft.section_type in leaf_types:
                session.add(
                    IndexOutbox(
                        tenant_id=version.tenant_id,
                        aggregate_type="SECTION",
                        aggregate_id=draft.id,
                        operation="UPSERT",
                        content_hash=draft.text_hash,
                        target_collection_name=self._collection,
                        embedding_version=self._embedding_version,
                        sparse_encoder_version=self._sparse_encoder_version,
                    )
                )
                outbox_count += 1

        warnings = []
        if parsed.quality_signals.get("needs_ocr"):
            warnings.append("native_text_coverage_below_threshold")
        parse_run.warnings = warnings
        parse_run.status = "INDEX_PENDING"
        parse_run.finished_at = utc_now()
        version.processing_status = "INDEXING"
        await session.flush()

        # 分层摘要索引（任务 17）：摘要失败不阻塞原文索引
        if self._summary_collection:
            from creditlens.ingestion.summaries import build_summary_tree

            summary_count = await build_summary_tree(
                session,
                document_version_id=version.id,
                parse_run_id=parse_run.id,
                target_collection_name=self._summary_collection,
                embedding_version=self._embedding_version,
            )
            outbox_count += summary_count

        return IngestResult(parse_run.id, len(drafts), outbox_count, reused=False)


async def activate_parse_run_if_complete(
    session: AsyncSession, parse_run_id: uuid.UUID
) -> bool:
    """Activation Guard（文档 §14.2）：ParseRun 解析成功且所有 Outbox 完成后，
    原子更新 document_versions.active_parse_run_id；旧 ParseRun 转 SUPERSEDED。"""
    parse_run = await session.get(ParseRun, parse_run_id)
    if parse_run is None or parse_run.status not in {
        "INDEX_PENDING",
        "SUCCEEDED",
        "SUCCEEDED_WITH_WARNINGS",
    }:
        return False

    pending = await session.scalar(
        select(IndexOutbox.id)
        .join(DocumentSection, DocumentSection.id == IndexOutbox.aggregate_id)
        .where(
            DocumentSection.parse_run_id == parse_run_id,
            IndexOutbox.status.notin_(["COMPLETED", "CANCELLED"]),
        )
        .limit(1)
    )
    if pending is not None:
        return False

    version = await session.get(DocumentVersion, parse_run.document_version_id)
    assert version is not None
    if version.active_parse_run_id and version.active_parse_run_id != parse_run_id:
        old = await session.get(ParseRun, version.active_parse_run_id)
        if old is not None:
            old.activation_status = "SUPERSEDED"
    parse_run.status = "SUCCEEDED_WITH_WARNINGS" if parse_run.warnings else "SUCCEEDED"
    parse_run.activation_status = "ACTIVE"
    version.active_parse_run_id = parse_run_id
    version.processing_status = "READY"
    await session.flush()
    return True
