"""共享测试夹具：临时 SQLite + 内存 Qdrant + 临时对象存储。"""

import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from creditlens.infrastructure.postgres.models import Base  # noqa: E402
from creditlens.infrastructure.postgres.session import create_session_factory  # noqa: E402


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
