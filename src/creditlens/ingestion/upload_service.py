"""文件上传与文档版本创建（任务 5）。

规则（文档 §6.4/§6.7/§7.2）：
- 原文件先落对象存储，路径由服务端构造：{tenant_id}/{document_id}/{version_id}/original.pdf；
- 服务端计算 SHA-256；同租户相同 content_hash 复用原始对象，但业务绑定单独保存；
- MIME 与文件头校验，不只看扩展名；
- 政策文件要求 valid_from（无法提供时由调用方显式确认后传入）。
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import ObjectStorePort
from creditlens.common.clock import utc_now
from creditlens.common.errors import DataQualityBlockedError, UploadIntegrityMismatchError
from creditlens.common.hashing import sha256_bytes
from creditlens.common.ids import new_id
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    Document,
    DocumentVersion,
)

_MAGIC_BYTES = {
    "application/pdf": b"%PDF",
}


@dataclass
class UploadCommand:
    tenant_id: uuid.UUID
    case_id: uuid.UUID | None
    logical_key: str
    title: str
    document_type: str  # REGULATION|INTERNAL_POLICY|ANNUAL_REPORT|...
    document_role: str  # BORROWER_PROVIDED|BANK_POLICY|...
    filename: str
    mime_type: str
    data: bytes
    version_label: str = "v1"
    valid_from: date | None = None
    valid_to: date | None = None
    source_available_at: datetime | None = None
    confidentiality: str = "INTERNAL"
    expected_sha256: str | None = None


@dataclass
class UploadResult:
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    content_hash: str
    object_uri: str
    deduplicated: bool


class UploadService:
    def __init__(self, object_store: ObjectStorePort, raw_bucket: str, max_file_size_mb: int = 100):
        self._store = object_store
        self._raw_bucket = raw_bucket
        self._max_bytes = max_file_size_mb * 1024 * 1024

    async def upload(self, session: AsyncSession, cmd: UploadCommand) -> UploadResult:
        self._validate(cmd)
        content_hash = sha256_bytes(cmd.data)
        if cmd.expected_sha256 and cmd.expected_sha256.lower() != content_hash:
            raise UploadIntegrityMismatchError(
                "上传内容 SHA-256 与声明不一致",
                {"expected": cmd.expected_sha256, "actual": content_hash},
            )

        document = await self._get_or_create_document(session, cmd)

        # 同租户内容去重：复用原始对象，不复用业务绑定
        existing = await session.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.tenant_id == cmd.tenant_id,
                DocumentVersion.content_hash == content_hash,
            )
            .limit(1)
        )
        deduplicated = existing is not None

        version_id = new_id()
        if deduplicated:
            object_uri = existing.object_uri
        else:
            key = f"{cmd.tenant_id}/{document.id}/{version_id}/original.pdf"
            object_uri = self._store.put(self._raw_bucket, key, cmd.data, cmd.mime_type)

        source_available_at = cmd.source_available_at or utc_now()
        if source_available_at.tzinfo is None:
            raise DataQualityBlockedError("source_available_at 必须携带时区")
        source_available_at = source_available_at.astimezone(UTC)

        version = DocumentVersion(
            id=version_id,
            tenant_id=cmd.tenant_id,
            document_id=document.id,
            version_label=cmd.version_label,
            valid_from=cmd.valid_from,
            valid_to=cmd.valid_to,
            source_available_at=source_available_at,
            object_uri=object_uri,
            source_filename=cmd.filename,
            mime_type=cmd.mime_type,
            file_size=len(cmd.data),
            content_hash=content_hash,
            processing_status="STORED",
        )
        session.add(version)
        await session.flush()  # PostgreSQL 外键：先落 version 再绑定 case_documents

        if cmd.case_id is not None:
            session.add(
                CaseDocument(
                    case_id=cmd.case_id,
                    document_version_id=version_id,
                    document_role=cmd.document_role,
                )
            )
        await session.flush()
        return UploadResult(
            document_id=document.id,
            document_version_id=version_id,
            content_hash=content_hash,
            object_uri=object_uri,
            deduplicated=deduplicated,
        )

    def _validate(self, cmd: UploadCommand) -> None:
        if not cmd.data:
            raise DataQualityBlockedError("空文件")
        if len(cmd.data) > self._max_bytes:
            raise DataQualityBlockedError("文件超过大小限制")
        magic = _MAGIC_BYTES.get(cmd.mime_type)
        if magic is not None and not cmd.data.startswith(magic):
            raise UploadIntegrityMismatchError(
                "文件头与声明的 MIME 不一致", {"mime_type": cmd.mime_type}
            )
        if cmd.document_type in {"REGULATION", "INTERNAL_POLICY"} and cmd.valid_from is None:
            raise DataQualityBlockedError(
                "政策/法规文件必须提供 valid_from；无法抽取时先人工确认", {"field": "valid_from"}
            )

    async def _get_or_create_document(self, session: AsyncSession, cmd: UploadCommand) -> Document:
        document = await session.scalar(
            select(Document).where(
                Document.tenant_id == cmd.tenant_id,
                Document.logical_key == cmd.logical_key,
            )
        )
        if document is None:
            document = Document(
                tenant_id=cmd.tenant_id,
                logical_key=cmd.logical_key,
                title=cmd.title,
                document_type=cmd.document_type,
                confidentiality=cmd.confidentiality,
            )
            session.add(document)
            await session.flush()
        return document
