"""异步数据库会话工厂。"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from creditlens.common.config import get_settings


def create_engine(database_url: str | None = None) -> AsyncEngine:
    url = database_url or get_settings().database_url
    return create_async_engine(url, echo=False)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def set_rls_context(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
) -> None:
    """在当前事务内设置可信 RLS 会话上下文（文档 §6.5）。

    - 仅 PostgreSQL 生效（SQLite 开发模式为 no-op）；
    - SET LOCAL 事务结束自动失效，连接归还池后不残留；
    - 值经 uuid.UUID 强转，杜绝注入；服务端从已验证 Token 传入，不接受客户端任意值。
    """
    if session.bind is None or session.bind.dialect.name != "postgresql":
        return
    tenant = uuid.UUID(str(tenant_id))
    await session.execute(text(f"SET LOCAL app.tenant_id = '{tenant}'"))
    if user_id is not None:
        user = uuid.UUID(str(user_id))
        await session.execute(text(f"SET LOCAL app.user_id = '{user}'"))


async def checkpoint_commit(session: AsyncSession) -> None:
    """阶段 Checkpoint 提交（P0-4）并恢复 RLS 上下文。

    SET LOCAL 随事务结束失效——commit 后必须重新注入，否则业务角色
    （NOBYPASSRLS）在下一阶段的 UPDATE 匹配 0 行（v1.0 演示实测踩坑）。
    """
    await session.commit()
    context = session.info.get("rls_context")
    if context is not None:
        await set_rls_context(session, context[0], context[1])


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
    tenant_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
) -> AsyncIterator[AsyncSession]:
    """事务边界：成功提交，异常回滚。

    传入 tenant_id/user_id 时在事务开始处注入 RLS 上下文；
    业务角色（NOBYPASSRLS）连接下，未注入上下文的查询将得到 0 行。
    上下文同时存入 session.info，供 checkpoint_commit 在中途提交后恢复。
    """
    async with factory() as session:
        try:
            if tenant_id is not None:
                session.info["rls_context"] = (tenant_id, user_id)
                await set_rls_context(session, tenant_id, user_id)
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
