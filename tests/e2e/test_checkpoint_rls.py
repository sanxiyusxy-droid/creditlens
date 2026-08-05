"""Checkpoint + RLS 集成测试（真实 PostgreSQL + NOSUPERUSER NOBYPASSRLS 业务角色）。

验证：
- checkpoint_commit 真实调用：中途提交后 RLS 上下文被重新注入，后续写读仍可见；
  （若未重新注入 SET LOCAL，业务角色在 NOBYPASSRLS 下查询将返回 0 行）
- 跨租户隔离：以租户 B 上下文实际查询租户 A 的 documents，断言不可见；
  且 WITH CHECK 阻止以租户 B 会话写入租户 A 的行；
- session_scope 异常自动回滚。

前置（CI integration job 已准备）：
- infra/postgres/rls_policies.sql 已应用；
- DATABASE_URL 使用 NOSUPERUSER NOBYPASSRLS 的业务角色连接。
"""

import uuid

import pytest
from sqlalchemy import select

from tests.conftest import requires_integration

pytestmark = [
    pytest.mark.integration,
    requires_integration,
    # pg_engine 为 session 级异步夹具：测试必须同处 session 事件循环，
    # 否则 asyncpg 连接跨循环报 "another operation is in progress"
    pytest.mark.asyncio(loop_scope="session"),
]

TENANT_A = uuid.UUID("00000000-0000-0000-0000-000000000091")
TENANT_B = uuid.UUID("00000000-0000-0000-0000-000000000092")
USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000399")


async def _ensure_tenant(factory, tenant_id, name):
    """tenants 表无 RLS 策略（元数据表），任何角色可写。"""
    from creditlens.infrastructure.postgres.models import Tenant
    from creditlens.infrastructure.postgres.session import session_scope

    async with session_scope(factory, tenant_id=tenant_id, user_id=USER_ID) as session:
        if await session.get(Tenant, tenant_id) is None:
            session.add(Tenant(id=tenant_id, name=name))


async def test_checkpoint_commit_restores_rls(pg_engine):
    """真实调用 checkpoint_commit：中途提交后 RLS 上下文必须被重新注入。"""
    from creditlens.infrastructure.postgres.models import Document
    from creditlens.infrastructure.postgres.session import (
        checkpoint_commit,
        create_session_factory,
        session_scope,
    )

    factory = create_session_factory(pg_engine)
    await _ensure_tenant(factory, TENANT_A, "租户 A")

    doc_before = Document(
        tenant_id=TENANT_A,
        logical_key=f"chk-before-{uuid.uuid4().hex[:8]}",
        title="Checkpoint 前写入",
        document_type="INTERNAL_POLICY",
    )
    doc_after = Document(
        tenant_id=TENANT_A,
        logical_key=f"chk-after-{uuid.uuid4().hex[:8]}",
        title="Checkpoint 后写入",
        document_type="INTERNAL_POLICY",
    )

    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_ID) as session:
        # 阶段一：写入并真实执行阶段 Checkpoint 提交
        session.add(doc_before)
        await session.flush()
        await checkpoint_commit(session)

        # 阶段二：SET LOCAL 已随上一事务失效——checkpoint_commit 必须重新注入，
        # 否则业务角色（NOBYPASSRLS）在本事务读回阶段一数据将得到 0 行
        rows = (await session.scalars(select(Document).where(Document.id == doc_before.id))).all()
        assert len(rows) == 1, "checkpoint_commit 后 RLS 上下文未恢复：阶段一数据不可见"

        # 阶段二继续写入并通过 WITH CHECK（tenant_id = app_current_tenant()）
        session.add(doc_after)
        await session.flush()

    # 新事务复核：两次 Checkpoint 间的数据均已持久化且对同租户可见
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_ID) as session:
        assert await session.get(Document, doc_before.id) is not None
        assert await session.get(Document, doc_after.id) is not None


async def test_cross_tenant_isolation(pg_engine):
    """租户 B 会话实际访问租户 A 的 documents：RLS 下必须不可见且不可写。"""
    from creditlens.infrastructure.postgres.models import Document
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    await _ensure_tenant(factory, TENANT_A, "租户 A")
    await _ensure_tenant(factory, TENANT_B, "租户 B")

    # 租户 A 写入一条受 RLS 租户隔离保护的 document
    secret_doc_id = uuid.uuid4()
    async with session_scope(factory, tenant_id=TENANT_A, user_id=USER_ID) as session:
        session.add(
            Document(
                id=secret_doc_id,
                tenant_id=TENANT_A,
                logical_key=f"secret-{uuid.uuid4().hex[:8]}",
                title="租户 A 私有材料",
                document_type="ANNUAL_REPORT",
            )
        )

    # 租户 B 上下文：按主键直取、条件查询、计数三重断言不可见
    async with session_scope(factory, tenant_id=TENANT_B, user_id=USER_ID) as session:
        assert await session.get(Document, secret_doc_id) is None, "跨租户主键读取必须被 RLS 拒绝"
        rows = (await session.scalars(select(Document).where(Document.id == secret_doc_id))).all()
        assert rows == [], "跨租户条件查询必须返回 0 行"
        count = await session.scalar(select(Document.id).where(Document.tenant_id == TENANT_A))
        assert count is None, "租户 B 不得看到租户 A 的任何 document"

        # WITH CHECK：以租户 B 会话写入 tenant_id=A 的行必须被拒绝
        session.add(
            Document(
                tenant_id=TENANT_A,
                logical_key=f"cross-write-{uuid.uuid4().hex[:8]}",
                title="越权写入",
                document_type="INTERNAL_POLICY",
            )
        )
        with pytest.raises(Exception) as exc_info:
            await session.flush()
        # flush 失败后 session 处于待回滚状态：先回滚，避免退出 scope 时
        # commit 抛 PendingRollbackError
        await session.rollback()
    # 确认拒绝原因是 RLS 策略而非其他错误
    message = str(exc_info.value).lower()
    assert "row-level security" in message or "policy" in message


async def test_session_rollback_on_error(pg_engine):
    """session_scope 异常时应自动回滚。"""
    from creditlens.infrastructure.postgres.models import Tenant
    from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

    factory = create_session_factory(pg_engine)
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000098")

    with pytest.raises(ValueError):
        async with session_scope(factory, tenant_id=tenant_id, user_id=USER_ID) as session:
            session.add(Tenant(id=tenant_id, name="将被回滚的租户"))
            await session.flush()
            raise ValueError("模拟异常")

    # 验证数据未持久化
    async with session_scope(factory, tenant_id=tenant_id, user_id=USER_ID) as session:
        tenant = await session.get(Tenant, tenant_id)
        assert tenant is None
