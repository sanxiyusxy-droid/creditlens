"""合成数据种子脚本（多案件模式）：生成合成 PDF -> 上传 -> 入库 -> 索引 -> 激活。

用法（本地离线模式，SQLite + Qdrant 内存 + 本地对象存储）：
    uv run python scripts/seed_synthetic_data.py

产出（3 个黄金案件）：
- golden_case_001：示例制造有限公司 + 小微流贷政策（保留原版）
- golden_case_002：星辰微电子科技有限公司 + 科技型企业流贷政策
- golden_case_003：恒达精密机械有限公司 + 保理业务（跨产品）

每个案件独立 Entity + CreditCase + CaseDocument 绑定；政策文档可共享绑定多案件。
"""

import asyncio
import sys
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz
from sqlalchemy import select

from creditlens.common.config import get_settings
from creditlens.common.ids import deterministic_point_id
from creditlens.demo_manifest import (
    DEMO_ASSET_NAMESPACE,
    expected_demo_qdrant_points,
    load_demo_asset_manifest,
)
from creditlens.infrastructure.llm.embedding import build_embedding_provider
from creditlens.infrastructure.objectstore import build_object_store
from creditlens.infrastructure.postgres.models import (
    Base,
    CreditCase,
    DocumentSection,
    DocumentVersion,
    Entity,
    IndexOutbox,
    SummaryNode,
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


def _asset_identity(spec: dict) -> str:
    return f"{spec['logical_key']}@{spec['version_label']}"


def _frozen_asset(spec: dict) -> dict:
    manifest = load_demo_asset_manifest(PROJECT_ROOT)
    matches = [
        item
        for item in manifest["assets"]
        if item["logical_key"] == spec["logical_key"]
        and item["version_label"] == spec["version_label"]
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen demo asset identity is missing or ambiguous")
    return matches[0]


# ---- 案件 002：科技型企业 ----
BORROWER_ID_002 = uuid.UUID("00000000-0000-0000-0000-000000000102")
CASE_ID_002 = uuid.UUID("00000000-0000-0000-0000-000000000202")

# ---- 案件 003：保理业务 ----
BORROWER_ID_003 = uuid.UUID("00000000-0000-0000-0000-000000000103")
CASE_ID_003 = uuid.UUID("00000000-0000-0000-0000-000000000203")

# ====================== 语料定义 ======================
# 案件 001 语料（原版保留）
CORPUS_CASE_001 = [
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
        "valid_to": date(2026, 1, 1),
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
        "source_available_at": datetime(2026, 4, 30, tzinfo=UTC),
    },
]

# 案件 002 语料：科技型企业
CORPUS_CASE_002 = [
    {
        "txt": "policy_tech_wc_v2026.txt",
        "logical_key": "policy_tech_wc",
        "title": "示例银行科技型企业流动资金贷款管理办法（合成演示版）",
        "document_type": "INTERNAL_POLICY",
        "document_role": "BANK_POLICY",
        "version_label": "2026",
        "valid_from": date(2026, 1, 1),
        "valid_to": None,
        "source_available_at": datetime(2026, 1, 1, tzinfo=UTC),
    },
    {
        "txt": "policy_tech_wc_v2024.txt",
        "logical_key": "policy_tech_wc",
        "title": "示例银行科技型企业流动资金贷款管理办法（2024版·合成演示版）",
        "document_type": "INTERNAL_POLICY",
        "document_role": "BANK_POLICY",
        "version_label": "2024",
        "valid_from": date(2024, 1, 1),
        "valid_to": date(2026, 1, 1),
        "source_available_at": datetime(2024, 1, 1, tzinfo=UTC),
    },
    {
        "txt": "annual_report_tech_2025.txt",
        "logical_key": "annual_report_tech_2025",
        "title": "星辰微电子科技有限公司 2025 年度报告（合成演示版）",
        "document_type": "ANNUAL_REPORT",
        "document_role": "BORROWER_PROVIDED",
        "version_label": "2025",
        "valid_from": None,
        "valid_to": None,
        "source_available_at": datetime(2026, 4, 30, tzinfo=UTC),
    },
]

# 案件 003 语料：保理业务（共享监管 + 制造业年报）
CORPUS_CASE_003 = [
    {
        "txt": "regulation_factoring.txt",
        "logical_key": "regulation_factoring",
        "title": "商业保理业务监督管理指引（合成演示版）",
        "document_type": "REGULATION",
        "document_role": "REGULATORY",
        "version_label": "2024",
        "valid_from": date(2024, 6, 1),
        "valid_to": None,
        "source_available_at": datetime(2024, 6, 1, tzinfo=UTC),
    },
    # 共享：制造业政策（保理案件也需参照流贷管理办法）
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
    # 共享：制造业年报（同一借款人集团）
    {
        "txt": "annual_report_2025.txt",
        "logical_key": "annual_report_2025",
        "title": "示例制造有限公司 2025 年度报告（节选·合成演示版）",
        "document_type": "ANNUAL_REPORT",
        "document_role": "BORROWER_PROVIDED",
        "version_label": "2025",
        "valid_from": None,
        "valid_to": None,
        "source_available_at": datetime(2026, 4, 30, tzinfo=UTC),
    },
]

# 兼容旧引用
CORPUS = CORPUS_CASE_001


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
    # Remove MuPDF's per-write trailer UUID so committed input always yields
    # the object hash frozen in the demo asset manifest.
    data = doc.tobytes(no_new_id=True)
    pdf_path.write_bytes(data)
    doc.close()
    return data


async def _ingest_corpus(
    session, corpus: list[dict], case_id, upload_service, pipeline, worker, settings
) -> None:
    """对一组语料执行上传 -> 切分 -> 向量化 -> 激活。幂等：同 logical_key+version 跳过。"""
    from sqlalchemy import select as sa_select

    from creditlens.infrastructure.postgres.models import Document, DocumentVersion

    for spec in corpus:
        frozen = _frozen_asset(spec)
        existing = await session.scalar(
            sa_select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.tenant_id == TENANT_ID,
                Document.logical_key == spec["logical_key"],
                DocumentVersion.version_label == spec["version_label"],
            )
            .limit(1)
        )
        if existing is not None:
            if (
                existing.id != uuid.UUID(frozen["document_version_id"])
                or existing.document_id != uuid.UUID(frozen["document_id"])
                or existing.content_hash != frozen["object_sha256"]
            ):
                raise RuntimeError("existing synthetic asset conflicts with frozen manifest")
            # A document can be shared by more than one demo case.  Its Qdrant
            # payload includes the complete case/entity scope, so establish the
            # current case binding before asking the exact-payload repair path to
            # compare against the frozen final manifest.  Otherwise the first
            # pass for a shared asset can only reproduce the earlier case scope
            # and is guaranteed to fail its own post-write verification.
            await _bind_existing_documents(session, case_id, [spec])
            repaired = await _repair_existing_index(
                session,
                existing,
                worker,
                settings,
            )
            print(
                f"  {spec['logical_key']}@{spec['version_label']}: 已存在，索引核验/修复={repaired}"
            )
            continue
        txt_path = SYNTH_DIR / spec["txt"]
        pdf_bytes = text_to_pdf(txt_path, txt_path.with_suffix(".pdf"))
        result = await upload_service.upload(
            session,
            UploadCommand(
                tenant_id=TENANT_ID,
                case_id=case_id,
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
                expected_sha256=frozen["object_sha256"],
                document_id=uuid.UUID(frozen["document_id"]),
                document_version_id=uuid.UUID(frozen["document_version_id"]),
            ),
        )
        ingest = await pipeline.ingest(
            session,
            result.document_version_id,
            deterministic_identity_key=_asset_identity(spec),
        )
        while await count_pending(session) > 0:
            stats = await worker.process_batch(session)
            if stats.processed == 0 and stats.failed == 0:
                break
        activated = await activate_parse_run_if_complete(session, ingest.parse_run_id)
        print(
            f"  {spec['logical_key']}@{spec['version_label']}: "
            f"sections={ingest.section_count} outbox={ingest.outbox_count} "
            f"activated={activated} reused={ingest.reused}"
        )


async def _repair_existing_index(session, version, worker, settings) -> int:
    """Reconcile one existing active parse against the current embedding profile.

    The old seed path skipped an existing DocumentVersion wholesale, which made
    a missing point or a changed embedding version look ready forever.  This
    repair path re-enqueues only missing deterministic points and leaves the
    existing document/version/volume intact.
    """

    if version.active_parse_run_id is None:
        raise RuntimeError("existing synthetic document has no active parse run")
    parse_run_id = version.active_parse_run_id
    expected: list[tuple[str, uuid.UUID, str, str, str | None]] = []
    sections = (
        await session.scalars(
            select(DocumentSection).where(
                DocumentSection.parse_run_id == parse_run_id,
                DocumentSection.section_type.in_(["ARTICLE", "PARAGRAPH"]),
            )
        )
    ).all()
    expected.extend(
        (
            "SECTION",
            section.id,
            section.text_hash,
            settings.chunks_collection_name,
            settings.sparse_encoder_version,
        )
        for section in sections
    )
    summaries = (
        await session.scalars(
            select(SummaryNode).where(
                SummaryNode.parse_run_id == parse_run_id,
                SummaryNode.grounding_status == "VERIFIED",
            )
        )
    ).all()
    expected.extend(
        (
            "SUMMARY",
            node.id,
            node.summary_hash,
            settings.summaries_collection_name,
            None,
        )
        for node in summaries
    )
    if not sections or not summaries:
        raise RuntimeError("existing synthetic parse is incomplete")

    manifest_points = expected_demo_qdrant_points(load_demo_asset_manifest(PROJECT_ROOT), settings)
    repaired = 0
    now = datetime.now(UTC)
    lease_cutoff = now - timedelta(minutes=5)
    for aggregate_type, aggregate_id, content_hash, collection, sparse_version in expected:
        point_id = deterministic_point_id(
            aggregate_id,
            content_hash,
            settings.effective_embedding_version,
        )
        found = worker._qdrant.retrieve(
            collection,
            ids=[str(point_id)],
            with_payload=True,
        )
        expected_payload = manifest_points.get(collection, {}).get(str(point_id))
        point_is_exact = bool(
            expected_payload is not None
            and len(found) == 1
            and str(found[0].id) == str(point_id)
            and (found[0].payload or {}) == expected_payload
        )
        ledgers = (
            await session.scalars(
                select(IndexOutbox)
                .where(
                    IndexOutbox.aggregate_type == aggregate_type,
                    IndexOutbox.aggregate_id == aggregate_id,
                    IndexOutbox.operation == "UPSERT",
                    IndexOutbox.content_hash == content_hash,
                    IndexOutbox.target_collection_name == collection,
                    IndexOutbox.embedding_version == settings.effective_embedding_version,
                )
                .order_by(IndexOutbox.created_at.desc())
            )
        ).all()
        completed = next((row for row in ledgers if row.status == "COMPLETED"), None)
        if point_is_exact:
            if completed is None:
                adoptable = next(
                    (row for row in ledgers if row.status in {"PENDING", "PROCESSING"}),
                    None,
                )
                if adoptable is None:
                    adoptable = IndexOutbox(
                        tenant_id=version.tenant_id,
                        aggregate_type=aggregate_type,
                        aggregate_id=aggregate_id,
                        operation="UPSERT",
                        content_hash=content_hash,
                        target_collection_name=collection,
                        embedding_version=settings.effective_embedding_version,
                        sparse_encoder_version=sparse_version,
                    )
                    session.add(adoptable)
                adoptable.status = "COMPLETED"
                adoptable.attempts = max(adoptable.attempts or 0, 1)
                adoptable.locked_at = None
                adoptable.last_error = None
                adoptable.completed_at = now
                repaired += 1
            continue

        pending = next((row for row in ledgers if row.status == "PENDING"), None)
        processing = next((row for row in ledgers if row.status == "PROCESSING"), None)
        fresh_processing = False
        if processing is not None:
            locked_at = processing.locked_at
            if locked_at is not None and locked_at.tzinfo is None:
                locked_at = locked_at.replace(tzinfo=UTC)
            if locked_at is None or locked_at <= lease_cutoff:
                processing.status = "PENDING"
                processing.available_at = now
                processing.locked_at = None
                processing.last_error = None
                pending = processing
                repaired += 1
            else:
                fresh_processing = True
        if pending is None and not fresh_processing:
            pending = IndexOutbox(
                tenant_id=version.tenant_id,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                operation="UPSERT",
                content_hash=content_hash,
                target_collection_name=collection,
                embedding_version=settings.effective_embedding_version,
                sparse_encoder_version=sparse_version,
            )
            session.add(pending)
            repaired += 1
    await session.flush()
    while await count_pending(session) > 0:
        stats = await worker.process_batch(session)
        if stats.processed == 0 and stats.failed == 0:
            break
    expected_completed = 0
    for aggregate_type, aggregate_id, content_hash, collection, _sparse_version in expected:
        point_id = deterministic_point_id(
            aggregate_id,
            content_hash,
            settings.effective_embedding_version,
        )
        found = worker._qdrant.retrieve(collection, ids=[str(point_id)], with_payload=True)
        expected_payload = manifest_points.get(collection, {}).get(str(point_id))
        if (
            len(found) != 1
            or str(found[0].id) != str(point_id)
            or (found[0].payload or {}) != expected_payload
        ):
            raise RuntimeError("existing synthetic index repair is incomplete")
        completed = await session.scalar(
            select(IndexOutbox.id)
            .where(
                IndexOutbox.aggregate_type == aggregate_type,
                IndexOutbox.aggregate_id == aggregate_id,
                IndexOutbox.operation == "UPSERT",
                IndexOutbox.content_hash == content_hash,
                IndexOutbox.target_collection_name == collection,
                IndexOutbox.embedding_version == settings.effective_embedding_version,
                IndexOutbox.status == "COMPLETED",
            )
            .limit(1)
        )
        if completed is None:
            raise RuntimeError("existing synthetic index ledger repair is incomplete")
        expected_completed += 1
    if expected_completed != len(expected):
        raise RuntimeError("existing synthetic index reconciliation cardinality mismatch")
    return repaired


async def _bind_existing_documents(session, case_id, corpus: list[dict]) -> None:
    """将已入库的文档版本绑定到案件（共享文档场景）。

    WP4 修复：CaseDocument 为复合主键无 id 列；绑定时必须带 document_role；
    幂等：已绑定不重复插入。"""
    from sqlalchemy import select as sa_select

    from creditlens.infrastructure.postgres.models import (
        CaseDocument,
        Document,
        DocumentVersion,
    )

    for spec in corpus:
        version_id = await session.scalar(
            sa_select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.tenant_id == TENANT_ID,
                Document.logical_key == spec["logical_key"],
                DocumentVersion.version_label == spec["version_label"],
            )
            .limit(1)
        )
        if version_id is None:
            continue
        already_bound = await session.scalar(
            sa_select(CaseDocument.case_id).where(
                CaseDocument.case_id == case_id,
                CaseDocument.document_version_id == version_id,
            )
        )
        if already_bound is None:
            session.add(
                CaseDocument(
                    case_id=case_id,
                    document_version_id=version_id,
                    document_role=spec["document_role"],
                )
            )
    await session.flush()


async def seed_environment(factory, store, qdrant, settings) -> None:
    """可复用种子逻辑：供 CLI 与评测脚本（同进程内存 Qdrant）调用。

    三个黄金案件：
    - golden_case_001：示例制造有限公司 + 小微流贷
    - golden_case_002：星辰微电子科技有限公司 + 科技型企业流贷
    - golden_case_003：恒达精密机械有限公司 + 保理
    """
    embedder = build_embedding_provider(settings)
    chunks_collection = settings.chunks_collection_name
    summaries_collection = settings.summaries_collection_name
    manager = CollectionManager(qdrant, dense_dim=embedder.dim)
    manager.ensure_collection(chunks_collection, settings.qdrant_chunks_alias)
    manager.ensure_collection(summaries_collection, settings.qdrant_summaries_alias)

    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        # WP4：Tenant/User/Entity/Case/Membership 逐项幂等创建：
        # 空库与已有 v1 数据库均可补齐三案件，连跑两次不增长。
        from sqlalchemy import select as sa_select

        from creditlens.infrastructure.postgres.models import AppUser, CaseMembership

        if await session.get(Tenant, TENANT_ID) is None:
            session.add(Tenant(id=TENANT_ID, name="示例银行（合成）"))
        if await session.get(AppUser, DEMO_USER_ID) is None:
            session.add(
                AppUser(
                    id=DEMO_USER_ID,
                    tenant_id=TENANT_ID,
                    external_subject="demo-analyst",
                    display_name="演示授信审查员",
                )
            )
        await session.flush()

        case_specs = [
            {
                "borrower_id": BORROWER_ID,
                "case_id": CASE_ID,
                "case_number": "golden_case_001",
                "name": "示例制造有限公司",
                "uscc": "SYNTHETIC-91310000000000001X",
                "product_code": "working_capital",
                "amount": "5000000.00",
                "purpose": "采购原材料",
            },
            {
                "borrower_id": BORROWER_ID_002,
                "case_id": CASE_ID_002,
                "case_number": "golden_case_002",
                "name": "星辰微电子科技有限公司",
                "uscc": "SYNTHETIC-91440300000000002Y",
                "product_code": "tech_working_capital",
                "amount": "15000000.00",
                "purpose": "技术研发及日常经营周转",
            },
            {
                "borrower_id": BORROWER_ID_003,
                "case_id": CASE_ID_003,
                "case_number": "golden_case_003",
                "name": "恒达精密机械有限公司",
                "uscc": "SYNTHETIC-91320500000000003Z",
                "product_code": "factoring",
                "amount": "8000000.00",
                "purpose": "应收账款保理融资",
            },
        ]
        for spec in case_specs:
            if await session.get(Entity, spec["borrower_id"]) is None:
                session.add(
                    Entity(
                        id=spec["borrower_id"],
                        tenant_id=TENANT_ID,
                        entity_type="COMPANY",
                        canonical_name=spec["name"],
                        unified_social_credit_code=spec["uscc"],
                        industry_code="C",
                    )
                )
            if await session.get(CreditCase, spec["case_id"]) is None:
                session.add(
                    CreditCase(
                        id=spec["case_id"],
                        tenant_id=TENANT_ID,
                        case_number=spec["case_number"],
                        borrower_entity_id=spec["borrower_id"],
                        product_code=spec["product_code"],
                        requested_amount=Decimal(spec["amount"]),
                        currency="CNY",
                        loan_purpose=spec["purpose"],
                        application_date=date(2026, 6, 30),
                        as_of_date=date(2026, 6, 30),
                        decision_cutoff_at=datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC),
                        industry_code="C",
                    )
                )
            # WP3/WP4：审批动作仅 REVIEWER/OWNER 可执行，演示用户授予 REVIEWER
            has_reviewer = await session.scalar(
                sa_select(CaseMembership.case_role).where(
                    CaseMembership.case_id == spec["case_id"],
                    CaseMembership.user_id == DEMO_USER_ID,
                    CaseMembership.case_role == "REVIEWER",
                )
            )
            if has_reviewer is None:
                session.add(
                    CaseMembership(
                        case_id=spec["case_id"], user_id=DEMO_USER_ID, case_role="REVIEWER"
                    )
                )
        await session.flush()

        print("[1/5] 三个黄金案件就绪（幂等）")

        upload_service = UploadService(store, settings.minio_raw_bucket)
        pipeline = IngestionPipeline(
            store,
            target_collection_name=chunks_collection,
            embedding_version=settings.effective_embedding_version,
            sparse_encoder_version=settings.sparse_encoder_version,
            summary_collection_name=summaries_collection,
            deterministic_identity_namespace=DEMO_ASSET_NAMESPACE,
        )
        worker = IndexWorker(qdrant, embedder)

        # 案件 001 语料入库
        print("[2/5] 案件 001 语料入库")
        await _ingest_corpus(
            session, CORPUS_CASE_001, CASE_ID, upload_service, pipeline, worker, settings
        )
        # WP4：文档已存在时 upload 被跳过，绑定必须单独幂等补齐
        await _bind_existing_documents(session, CASE_ID, CORPUS_CASE_001)

        # 案件 002 语料入库
        print("[3/5] 案件 002 语料入库")
        await _ingest_corpus(
            session, CORPUS_CASE_002, CASE_ID_002, upload_service, pipeline, worker, settings
        )
        await _bind_existing_documents(session, CASE_ID_002, CORPUS_CASE_002)

        # 案件 003 语料入库（共享文档只绑定不重复入库）
        print("[4/5] 案件 003 语料入库")
        await _ingest_corpus(
            session, CORPUS_CASE_003, CASE_ID_003, upload_service, pipeline, worker, settings
        )
        # 确保共享文档也绑定到案件 003
        await _bind_existing_documents(session, CASE_ID_003, CORPUS_CASE_003)

        # Bindings participate in the section payload (entity scope).  Re-run
        # exact reconciliation after every shared binding exists so a crash or
        # an earlier single-case payload cannot survive as a false-ready point.
        unique_specs = {
            _asset_identity(spec): spec
            for spec in [*CORPUS_CASE_001, *CORPUS_CASE_002, *CORPUS_CASE_003]
        }
        for spec in unique_specs.values():
            frozen = _frozen_asset(spec)
            version = await session.get(DocumentVersion, uuid.UUID(frozen["document_version_id"]))
            if version is None:
                raise RuntimeError("frozen synthetic version missing after seed")
            print(f"  {_asset_identity(spec)}: final exact-index reconciliation")
            await _repair_existing_index(session, version, worker, settings)

        print("[5/5] 全部语料入库完成")
    print("种子数据完成（3 案件）。")


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
