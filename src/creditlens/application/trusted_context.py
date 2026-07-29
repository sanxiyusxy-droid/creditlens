"""服务端 TrustedRequestContext 构建器（v0.2，文档 §8.2/§12.2）。

- Context 只能由服务端从已验证身份 + Case + 数据库派生，调用方/模型不得自造；
- allowed_document_ids 来自 case_documents 绑定，空授权 = 默认拒绝；
- 提供 user_id 时要求存在未撤销的 Case Membership（ABAC：Tenant + Membership）。
"""

import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.common.clock import utc_now
from creditlens.common.errors import AclDeniedError, CaseNotFoundError
from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    CaseMembership,
    CreditCase,
    DocumentVersion,
)
from creditlens.retrieval.contracts import TrustedRequestContext


async def build_trusted_context(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    case_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
    purpose: str = "credit_review",
) -> TrustedRequestContext:
    """从数据库派生可信上下文。任何一步失败即拒绝，不降级放行。"""
    case = await session.get(CreditCase, case_id)
    if case is None:
        raise CaseNotFoundError("案件不存在")
    if str(case.tenant_id) != str(tenant_id):
        # 跨租户访问按不存在处理，不泄露案件存在性
        raise CaseNotFoundError("案件不存在")

    if user_id is not None:
        membership = await session.scalar(
            select(CaseMembership).where(
                CaseMembership.case_id == case_id,
                CaseMembership.user_id == user_id,
                CaseMembership.revoked_at.is_(None),
            )
        )
        if membership is None:
            raise AclDeniedError("用户不具备该案件的有效 Membership")

    # allowed_document_ids：案件绑定的文档（document 级，供 Qdrant payload 过滤）
    rows = (
        await session.execute(
            select(DocumentVersion.document_id)
            .join(CaseDocument, CaseDocument.document_version_id == DocumentVersion.id)
            .where(CaseDocument.case_id == case_id)
            .distinct()
        )
    ).all()
    allowed_document_ids = [row[0] for row in rows]

    cutoff = case.decision_cutoff_at
    if cutoff.tzinfo is None:
        from datetime import UTC

        cutoff = cutoff.replace(tzinfo=UTC)

    return TrustedRequestContext(
        request_id=uuid.uuid4(),
        tenant_id=case.tenant_id,
        user_id=user_id,
        purpose=purpose,
        case_id=case.id,
        borrower_entity_id=case.borrower_entity_id,
        product_code=case.product_code,
        as_of_date=case.as_of_date,
        decision_cutoff_at=cutoff,
        allowed_document_ids=allowed_document_ids,
    )


def acl_scope_hash(trusted: TrustedRequestContext) -> str:
    """用于 Snapshot 审计与缓存隔离；不冻结访问权本身（文档 §6.4）。"""
    payload = {
        "tenant_id": str(trusted.tenant_id),
        "case_id": str(trusted.case_id),
        "allowed_document_ids": sorted(str(d) for d in trusted.allowed_document_ids),
        "generated_date": utc_now().date().isoformat(),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))
