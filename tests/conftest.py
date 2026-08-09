"""共享测试夹具：临时 SQLite + 内存 Qdrant + 临时对象存储。

集成测试（真实 PG + Qdrant）：
    设置 DATABASE_URL 和 QDRANT_URL 环境变量后运行：
    uv run pytest tests/e2e/ -m integration
"""

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

# HTTP tests deliberately exercise the local demo identity. Production defaults remain
# fail-closed; these explicit test-only values are installed before apps.api.main is imported.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("API_IDENTITY_MODE", "demo")
os.environ.setdefault("ALLOW_INSECURE_DEMO_IDENTITY", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from creditlens.infrastructure.postgres.models import Base  # noqa: E402
from creditlens.infrastructure.postgres.session import create_session_factory  # noqa: E402

# ====================== 单元测试 Fixtures ======================


@pytest.fixture
async def engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine):
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s
        await s.commit()


@pytest.fixture
def qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(location=":memory:")


@pytest.fixture
def object_store(tmp_path):
    from creditlens.infrastructure.objectstore import LocalFsObjectStore

    return LocalFsObjectStore(str(tmp_path / "objects"))


@pytest.fixture
def policy_pdf_bytes():
    """合成政策 PDF（内存生成，不依赖种子脚本产物）。"""
    import fitz

    text = (PROJECT_ROOT / "data" / "synthetic" / "policy_manufacturing_wc_v2026.txt").read_text(
        encoding="utf-8"
    )
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for raw_line in text.splitlines():
        for chunk in [raw_line[i : i + 42] for i in range(0, len(raw_line), 42)] or [""]:
            if y > 780:
                page = doc.new_page()
                y = 60
            page.insert_text((50, y), chunk, fontsize=11, fontname="china-s")
            y += 18
        y += 4
    data = doc.tobytes()
    doc.close()
    return data


# ====================== 集成测试 Fixtures ======================


def _has_pg() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _has_qdrant() -> bool:
    return bool(os.environ.get("QDRANT_URL"))


requires_integration = pytest.mark.skipif(
    not (_has_pg() and _has_qdrant()),
    reason="需要 DATABASE_URL + QDRANT_URL 环境变量（集成测试）",
)


@pytest.fixture(scope="session")
def pg_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        pytest.skip("DATABASE_URL 未设置")
    return url


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    url = os.environ.get("QDRANT_URL", "")
    if not url:
        pytest.skip("QDRANT_URL 未设置")
    return url


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_engine(pg_database_url):
    """真实 PostgreSQL 引擎（session 级复用；loop_scope=session 保证
    asyncpg 连接不跨事件循环，集成测试需同样声明 loop_scope="session"）。"""
    engine = create_async_engine(pg_database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
def real_qdrant(qdrant_url):
    """真实 Qdrant 客户端（session 级复用）。"""
    from qdrant_client import QdrantClient

    return QdrantClient(url=qdrant_url)


@pytest.fixture
async def pg_session(pg_engine):
    """每个测试独立的 PG session（自动回滚）。"""
    factory = create_session_factory(pg_engine)
    async with factory() as s:
        yield s
        await s.rollback()
