"""Case Snapshot 冻结服务（v0.2，文档 §6.4）。

- 创建 Run 时在同一事务中冻结：案件绑定的 DocumentVersion + 其当时的
  active ParseRun + Alias 解析出的物理 Collection；
- 在线检索只能读取 Snapshot 冻结的 Parse Run 与物理 Collection，
  已启动 Run 绝不跟随 Alias；
- 旧 Snapshot 可继续访问 SUPERSEDED 的 Parse Run；
- snapshot_hash 对成员排序后的 Canonical JSON 计算。
"""

import json
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.common.errors import CreditLensError
from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    CaseSnapshot,
    CreditCase,
    DocumentVersion,
    FinancialFact,
    SnapshotDocument,
    SnapshotFact,
    SnapshotIndex,
)
from creditlens.retrieval.contracts import TrustedRequestContext


class CaseNotReadyError(CreditLensError):
    error_code = "CASE_NOT_READY"


@dataclass
class SnapshotContext:
    """检索/工具执行使用的冻结输入世界。"""

    snapshot_id: uuid.UUID
    allowed_parse_run_ids: list[uuid.UUID] = field(default_factory=list)
    allowed_fact_ids: list[uuid.UUID] = field(default_factory=list)
    chunks_collection: str = ""
    summaries_collection: str = ""


async def freeze_snapshot(
    session: AsyncSession,
    trusted: TrustedRequestContext,
    chunks_collection: str,
    summaries_collection: str | None = None,
    acl_hash: str = "",
) -> SnapshotContext:
    """冻结案件当前输入世界。物理 Collection 名由调用方在冻结时解析 Alias 得到。

    要求案件至少有一份已激活解析的材料；否则 CASE_NOT_READY。
    """
    case = await session.get(CreditCase, trusted.case_id)
    if case is None:
        raise CaseNotReadyError("案件不存在")

    rows = (
        await session.execute(
            select(DocumentVersion)
            .join(CaseDocument, CaseDocument.document_version_id == DocumentVersion.id)
            .where(CaseDocument.case_id == case.id)
        )
    ).all()
    members: list[tuple[uuid.UUID, uuid.UUID]] = []
    for (version,) in rows:
        if version.active_parse_run_id is None:
            continue  # 未完成解析的材料不进入本次 Snapshot
        members.append((version.id, version.active_parse_run_id))
    if not members:
        raise CaseNotReadyError("案件没有任何已激活解析的材料")

    snapshot = CaseSnapshot(
        tenant_id=case.tenant_id,
        case_id=case.id,
        case_version=case.version,
        as_of_date=trusted.as_of_date,
        decision_cutoff_at=trusted.decision_cutoff_at,
        borrower_entity_id=case.borrower_entity_id,
        acl_scope_hash=acl_hash,
    )
    session.add(snapshot)
    await session.flush()

    for version_id, parse_run_id in members:
        session.add(
            SnapshotDocument(
                snapshot_id=snapshot.id,
                document_version_id=version_id,
                parse_run_id=parse_run_id,
            )
        )
    session.add(
        SnapshotIndex(
            snapshot_id=snapshot.id,
            index_family="CHUNKS",
            physical_collection_name=chunks_collection,
        )
    )
    if summaries_collection:
        session.add(
            SnapshotIndex(
                snapshot_id=snapshot.id,
                index_family="SUMMARIES",
                physical_collection_name=summaries_collection,
            )
        )

    # P0-1：冻结财务事实——借款人在本案件范围内、审查截止前可获得、
    # 未被拒绝且未被重述替代的 Fact；计算工具只允许读取该集合
    from datetime import UTC

    from sqlalchemy import or_

    superseded = select(FinancialFact.supersedes_fact_id).where(
        FinancialFact.supersedes_fact_id.is_not(None)
    )
    fact_rows = (
        await session.execute(
            select(FinancialFact.id).where(
                FinancialFact.tenant_id == case.tenant_id,
                FinancialFact.entity_id == case.borrower_entity_id,
                or_(FinancialFact.case_id == case.id, FinancialFact.case_id.is_(None)),
                FinancialFact.source_available_at <= trusted.decision_cutoff_at.astimezone(UTC),
                FinancialFact.verification_status != "REJECTED",
                FinancialFact.id.not_in(superseded),
            )
        )
    ).all()
    fact_ids = [row[0] for row in fact_rows]
    for fact_id in fact_ids:
        session.add(SnapshotFact(snapshot_id=snapshot.id, fact_id=fact_id))

    canonical = {
        "case_version": case.version,
        "members": sorted(f"{v}:{p}" for v, p in members),
        "facts": sorted(str(f) for f in fact_ids),
        "collections": sorted(c for c in [chunks_collection, summaries_collection or ""] if c),
        "acl_scope_hash": acl_hash,
        "as_of_date": trusted.as_of_date.isoformat(),
        "decision_cutoff_at": trusted.decision_cutoff_at.isoformat(),
    }
    snapshot.snapshot_hash = sha256_text(json.dumps(canonical, sort_keys=True))
    await session.flush()

    return SnapshotContext(
        snapshot_id=snapshot.id,
        allowed_parse_run_ids=[p for _, p in members],
        allowed_fact_ids=fact_ids,
        chunks_collection=chunks_collection,
        summaries_collection=summaries_collection or "",
    )


async def load_snapshot_context(session: AsyncSession, snapshot_id: uuid.UUID) -> SnapshotContext:
    """从数据库还原冻结上下文（Run 恢复/续跑时使用）。"""
    members = (
        await session.execute(
            select(SnapshotDocument.parse_run_id).where(SnapshotDocument.snapshot_id == snapshot_id)
        )
    ).all()
    facts = (
        await session.execute(
            select(SnapshotFact.fact_id).where(SnapshotFact.snapshot_id == snapshot_id)
        )
    ).all()
    indexes = (
        await session.execute(
            select(SnapshotIndex.index_family, SnapshotIndex.physical_collection_name).where(
                SnapshotIndex.snapshot_id == snapshot_id
            )
        )
    ).all()
    by_family = dict(indexes)
    return SnapshotContext(
        snapshot_id=snapshot_id,
        allowed_parse_run_ids=[row[0] for row in members],
        allowed_fact_ids=[row[0] for row in facts],
        chunks_collection=by_family.get("CHUNKS", ""),
        summaries_collection=by_family.get("SUMMARIES", ""),
    )
