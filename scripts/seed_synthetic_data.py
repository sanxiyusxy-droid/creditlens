"""合成数据种子脚本：生成合成政策 PDF -> 上传 -> 入库 -> 索引 -> 激活。

用法（本地离线模式，SQLite + Qdrant 内存 + 本地对象存储）：
    uv run python scripts/seed_synthetic_data.py

产出：
- data/synthetic/policy_manufacturing_wc_v2026.pdf
- 本地数据库中的黄金案件 golden_case_001、政策 DocumentVersion 与 Sections
- Qdrant credit_chunks_v1 中的可检索向量
"""

import asyncio
import sys
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz

from creditlens.common.config import get_settings
from creditlens.infrastructure.llm.embedding import build_embedding_provider
from creditlens.infrastructure.objectstore import build_object_store
from creditlens.infrastructure.postgres.models import (
    Base,
    CreditCase,
    Entity,
    Tenant,
)
from creditlens.infrastructure.postgres.session import (
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import CollectionManager, build_qdrant_client
from creditlens.ingestion.index_worker import IndexWorker, count_pending
from creditlens.ingestion.pipeline import IngestionPipeline, activate_parse_run_if_complete
from creditlens.ingestion.upload_service import UploadCommand, UploadService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
POLICY_TXT = SYNTH_DIR / "policy_manufacturing_wc_v2026.txt"
POLICY_PDF = SYNTH_DIR / "policy_manufacturing_wc_v2026.pdf"

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
BORROWER_ID = uuid.UUID("00000000-0000-0000-0000-000000000101")
CASE_ID = uuid.UUID("00000000-0000-0000-0000-000000000201")
# 演示用户：MVP 无登录层，API 以此固定身份模拟"已验证 Token"（RLS Membership 需要）
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")

# 多文档语料（v0.4）：含政策新旧两版、监管文件与年报节选
CORPUS = [
    {
        "txt": "policy_manufacturing_wc_v2026.txt",
        "logical_key": "policy_manufacturing_wc",
        "title": "示例银行小微企业流动资金贷款管理办法（合成演示版）",
        "document_type": "INTERNAL_POLICY",
        "document_role": "BANK_POLICY",
        "version_label": "2026",
        "valid_from": date(2026, 1, 1),
        "valid_to": None,
        "source_available_at": datetime(2026, 1, 1, tzinfo=UTC),
    },
    {
        "txt": "policy_manufacturing_wc_v2024.txt",
        "logical_key": "policy_manufacturing_wc",
        "title": "示例银行小微企业流动资金贷款管理办法（2024版·合成演示版）",
        "document_type": "INTERNAL_POLICY",
        "document_role": "BANK_POLICY",
        "version_label": "2024",
        "valid_from": date(2024, 1, 1),
        "valid_to": date(2026, 1, 1),  # 左闭右开：2026-01-01 起由新版替代
        "source_available_at": datetime(2024, 1, 1, tzinfo=UTC),
    },
    {
        "txt": "regulation_wc_admin.txt",
        "logical_key": "regulation_wc_admin",
        "title": "流动资金贷款管理监管指引（合成演示版）",
        "document_type": "REGULATION",
        "document_role": "REGULATORY",
        "version_label": "2024",
        "valid_from": date(2024, 3, 1),
        "valid_to": None,
        "source_available_at": datetime(2024, 3, 1, tzinfo=UTC),
    },
    {
        "txt": "annual_report_2025.txt",
        "logical_key": "annual_report_2025",
        "title": "示例制造有限公司 2025 年度报告（节选·合成演示版）",
        "document_type": "ANNUAL_REPORT",
        "document_role": "BORROWER_PROVIDED",
        "version_label": "2025",
        "valid_from": None,
        "valid_to": None,
        "source_available_at": datetime(2026, 4, 30, tzinfo=UTC),  # 年报次年 4 月末披露
    },
]


def text_to_pdf(text_path: Path, pdf_path: Path) -> bytes:
    """把合成政策文本渲染为带 CJK 字体的多页 PDF。"""
    text = text_path.read_text(encoding="utf-8")
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for raw_line in text.splitlines():
        # 手工换行：每 42 个字符一行
        chunks = [raw_line[i : i + 42] for i in range(0, len(raw_line), 42)] or [""]
        for chunk in chunks:
            if y > 780:
                page = doc.new_page()
                y = 60
            page.insert_text((50, y), chunk, fontsize=11, fontname="china-s")
            y += 18
        y += 4
    data = doc.tobytes()
    pdf_path.write_bytes(data)
    doc.close()
    return data


async def seed_environment(factory, store, qdrant, settings) -> None:
    """可复用种子逻辑：供 CLI 与评测脚本（同进程内存 Qdrant）调用。

    Collection 名与 embedding_version 由 settings 按 Provider 派生：
    哈希兜底 -> credit_chunks_v1；真实模型 -> credit_chunks_v2（不混用向量空间）。
    """
    embedder = build_embedding_provider(settings)
    chunks_collection = settings.chunks_collection_name
    summaries_collection = settings.summaries_collection_name
    manager = CollectionManager(qdrant, dense_dim=embedder.dim)
    manager.ensure_collection(chunks_collection, settings.qdrant_chunks_alias)
    manager.ensure_collection(summaries_collection, settings.qdrant_summaries_alias)
    print(f"[1/4] 语料 PDF 生成（{len(CORPUS)} 份）")

    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        if await session.get(Tenant, TENANT_ID) is None:
            # PostgreSQL 强制外键：按依赖顺序逐级 flush（Tenant -> Entity -> Case）
            session.add(Tenant(id=TENANT_ID, name="示例银行（合成）"))
            await session.flush()
            session.add(
                Entity(
                    id=BORROWER_ID,
                    tenant_id=TENANT_ID,
                    entity_type="COMPANY",
                    canonical_name="示例制造有限公司",
                    unified_social_credit_code="SYNTHETIC-91310000000000001X",
                    industry_code="C",
                )
            )
            await session.flush()
            session.add(
                CreditCase(
                    id=CASE_ID,
                    tenant_id=TENANT_ID,
                    case_number="golden_case_001",
                    borrower_entity_id=BORROWER_ID,
                    product_code="working_capital",
                    requested_amount=Decimal("5000000.00"),
                    currency="CNY",
                    loan_purpose="采购原材料",
                    application_date=date(2026, 6, 30),
                    as_of_date=date(2026, 6, 30),
                    decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
                    industry_code="C",
                )
            )
            await session.flush()
            # 演示用户 + Case Membership（RLS 下无 Membership 即不可见案件）
            from creditlens.infrastructure.postgres.models import AppUser, CaseMembership

            session.add(
                AppUser(
                    id=DEMO_USER_ID,
                    tenant_id=TENANT_ID,
                    external_subject="demo-analyst",
                    display_name="演示授信审查员",
                )
            )
            await session.flush()
            session.add(
                CaseMembership(case_id=CASE_ID, user_id=DEMO_USER_ID, case_role="ANALYST")
            )
        print("[2/4] 黄金案件 golden_case_001 就绪")

        upload_service = UploadService(store, settings.minio_raw_bucket)
        pipeline = IngestionPipeline(
            store,
            target_collection_name=chunks_collection,
            embedding_version=settings.effective_embedding_version,
            sparse_encoder_version=settings.sparse_encoder_version,
            summary_collection_name=summaries_collection,
        )
        worker = IndexWorker(qdrant, embedder)

        for spec in CORPUS:
            # P0-5：seed 幂等——同 logical_key + version_label 已入库则跳过
            # （PDF 生成含时间戳导致 content_hash 不稳定，故按业务键判断）
            from sqlalchemy import select as sa_select

            from creditlens.infrastructure.postgres.models import Document, DocumentVersion

            existing = await session.scalar(
                sa_select(DocumentVersion.id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.tenant_id == TENANT_ID,
                    Document.logical_key == spec["logical_key"],
                    DocumentVersion.version_label == spec["version_label"],
                )
                .limit(1)
            )
            if existing is not None:
                print(f"[3/4] {spec['logical_key']}@{spec['version_label']}: 已存在，跳过")
                continue
            txt_path = SYNTH_DIR / spec["txt"]
            pdf_bytes = text_to_pdf(txt_path, txt_path.with_suffix(".pdf"))
            result = await upload_service.upload(
                session,
                UploadCommand(
                    tenant_id=TENANT_ID,
                    case_id=CASE_ID,
                    logical_key=spec["logical_key"],
                    title=spec["title"],
                    document_type=spec["document_type"],
                    document_role=spec["document_role"],
                    filename=spec["txt"].replace(".txt", ".pdf"),
                    mime_type="application/pdf",
                    data=pdf_bytes,
                    version_label=spec["version_label"],
                    valid_from=spec["valid_from"],
                    valid_to=spec["valid_to"],
                    source_available_at=spec["source_available_at"],
                ),
            )
            ingest = await pipeline.ingest(session, result.document_version_id)
            while await count_pending(session) > 0:
                stats = await worker.process_batch(session)
                if stats.processed == 0 and stats.failed == 0:
                    break
            activated = await activate_parse_run_if_complete(session, ingest.parse_run_id)
            print(
                f"[3/4] {spec['logical_key']}@{spec['version_label']}: "
                f"sections={ingest.section_count} outbox={ingest.outbox_count} "
                f"activated={activated} reused={ingest.reused}"
            )
        print("[4/4] 全部语料入库完成")
    print("种子数据完成。")


async def seed() -> None:
    settings = get_settings()
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    store = build_object_store(settings)
    qdrant = build_qdrant_client(settings)
    await seed_environment(factory, store, qdrant, settings)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
