"""Evidence Preview（任务 13）：EvidenceRef -> 原始 PDF 页渲染。

v0.9（P0-2）授权收口：
- 必须携带服务端构造的 TrustedRequestContext（含案件与 Membership 语义）；
- 证据所属 DocumentVersion 必须绑定在该案件（case_documents EXISTS）——
  同租户跨案件访问被拒绝；
- parse_run_id 必须与 Section 实际解析批次一致（防伪造 locator）；
- 打开证据时执行当前权限校验（Snapshot 不冻结已撤销的访问权）。
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.application.ports import ObjectStorePort
from creditlens.common.errors import AclDeniedError, CreditLensError
from creditlens.infrastructure.parsers.pymupdf_parser import PyMuPdfParser
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    DocumentSection,
    DocumentVersion,
)
from creditlens.retrieval.contracts import (
    EvidenceRef,
    RetrievedCandidate,
    TrustedRequestContext,
)


class EvidenceNotFoundError(CreditLensError):
    error_code = "EVIDENCE_NOT_FOUND"


def candidate_to_evidence_ref(candidate: RetrievedCandidate) -> EvidenceRef:
    return EvidenceRef(
        section_id=candidate.section_id,
        document_version_id=candidate.document_version_id,
        parse_run_id=candidate.parse_run_id,
        page_number=candidate.page_start,
        heading_path=candidate.heading_path,
        text_hash=candidate.text_hash,
    )


class EvidencePreviewService:
    def __init__(self, object_store: ObjectStorePort):
        self._store = object_store
        self._parser = PyMuPdfParser()

    async def render_page(
        self,
        session: AsyncSession,
        trusted: TrustedRequestContext,
        ref: EvidenceRef,
        zoom: float = 2.0,
    ) -> bytes:
        """校验定位契约与案件授权后渲染原始 PDF 页 PNG。"""
        section = await session.get(DocumentSection, ref.section_id)
        if section is None:
            raise EvidenceNotFoundError("section 不存在")
        if str(section.tenant_id) != str(trusted.tenant_id):
            raise AclDeniedError("跨租户证据访问被拒绝")
        if section.text_hash != ref.text_hash:
            raise EvidenceNotFoundError("引用文本哈希不一致（STALE_REFERENCE）")
        if section.parse_run_id != ref.parse_run_id:
            raise EvidenceNotFoundError("parse_run_id 与解析批次不一致")
        if not section.page_start <= ref.page_number <= section.page_end:
            raise EvidenceNotFoundError("页码不在 Section 页范围内")

        version = await session.get(DocumentVersion, ref.document_version_id)
        if version is None or version.id != section.document_version_id:
            raise EvidenceNotFoundError("document version 不匹配")

        # P0-2：案件级授权——文档必须绑定在受权案件，杜绝同租户跨案件读取
        if trusted.case_id is None:
            raise AclDeniedError("证据预览必须携带案件上下文")
        bound = await session.scalar(
            select(CaseDocument.case_id).where(
                CaseDocument.case_id == trusted.case_id,
                CaseDocument.document_version_id == version.id,
            )
        )
        if bound is None:
            raise AclDeniedError("该证据不属于当前案件（跨案件访问被拒绝）")

        data = self._store.get(version.object_uri)
        return self._parser.render_page_png(data, ref.page_number, zoom=zoom)
