"""CreditLens API 入口（MVP）。

启动：uv run uvicorn apps.api.main:app --reload
本地离线模式使用 SQLite / 内存 Qdrant / 本地对象存储；生产模式通过 .env 指向
PostgreSQL / Qdrant / MinIO。注意：内存 Qdrant 进程重启后需重跑种子脚本。
"""

import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sqlalchemy import select  # noqa: E402

from creditlens.common.config import get_settings  # noqa: E402
from creditlens.common.errors import CreditLensError  # noqa: E402
from creditlens.evidence.preview import EvidencePreviewService  # noqa: E402
from creditlens.infrastructure.llm.embedding import build_embedding_provider  # noqa: E402
from creditlens.infrastructure.objectstore import build_object_store  # noqa: E402
from creditlens.infrastructure.postgres.models import (  # noqa: E402
    Base,
    ClaimRecord,
    CreditCase,
    ReviewRun,
    RunEvent,
)
from creditlens.infrastructure.postgres.session import (  # noqa: E402
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client  # noqa: E402
from creditlens.retrieval.contracts import EvidenceRef  # noqa: E402
from creditlens.retrieval.hybrid import HybridRetriever  # noqa: E402

settings = get_settings()
engine = create_engine()
session_factory = create_session_factory(engine)
object_store = build_object_store(settings)
qdrant = build_qdrant_client(settings)
embedder = build_embedding_provider(settings)
retriever = HybridRetriever(qdrant, embedder)
preview_service = EvidencePreviewService(object_store)

# MVP 单租户：与种子脚本一致
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
# MVP 无登录层：以固定演示用户模拟已验证 Token（RLS Membership 需要）
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # P0-4：孤儿 Run 回收——进程内后台任务不持久，重启后残留的非终态 Run
    # 置 FAILED（ORPHANED），不假装仍在执行。持久任务队列（Celery）列偏差 D9。
    from datetime import timedelta

    from creditlens.common.clock import utc_now

    terminal = ["COMPLETED", "FAILED", "DENIED", "HUMAN_REVIEW", "REWORK",
                "NEED_MORE_INFO", "DATA_QUALITY_BLOCKED", "SUPERSEDED"]
    cutoff = utc_now() - timedelta(minutes=30)
    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        orphans = (
            await session.scalars(
                select(ReviewRun).where(
                    ReviewRun.status.not_in(terminal), ReviewRun.started_at < cutoff
                )
            )
        ).all()
        for run in orphans:
            run.status = "FAILED"
            run.model_manifest = {**(run.model_manifest or {}), "failure_reason": "ORPHANED"}
    yield
    await engine.dispose()


app = FastAPI(title="CreditLens API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(CreditLensError)
async def creditlens_error_handler(request, exc: CreditLensError):
    from fastapi.responses import JSONResponse

    status = 403 if exc.error_code in {"ACL_DENIED", "TOOL_CALL_DENIED"} else 422
    return JSONResponse(
        status_code=status,
        content={"error_code": exc.error_code, "message": exc.message, "details": exc.details},
    )


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    from sqlalchemy import text

    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: uuid.UUID):
    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        case = await session.get(CreditCase, case_id)
        if case is None:
            raise HTTPException(404, "CASE_NOT_FOUND")
        return {
            "case_id": str(case.id),
            "case_number": case.case_number,
            "product_code": case.product_code,
            "requested_amount": str(case.requested_amount),
            "currency": case.currency,
            "as_of_date": case.as_of_date.isoformat(),
            "status": case.status,
        }


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 8


@app.post("/api/v1/cases/{case_id}/questions")
async def ask_question(case_id: uuid.UUID, body: QuestionRequest):
    """简单问答（SIMPLE_QA）：同样创建 Run 并冻结 Snapshot，
    不绕过时点、物理索引与审计契约（文档 §14.2）。"""
    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context
    from creditlens.infrastructure.postgres.models import ReviewRun

    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        trusted = await build_trusted_context(
            session, tenant_id=DEFAULT_TENANT_ID, case_id=case_id, user_id=DEMO_USER_ID
        )
        snapshot = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=settings.chunks_collection_name,
            summaries_collection=settings.summaries_collection_name,
            acl_hash=acl_scope_hash(trusted),
        )
        run = ReviewRun(
            tenant_id=trusted.tenant_id,
            case_id=trusted.case_id,
            run_type="SIMPLE_QA",
            status="COMPLETED",
            as_of_date=trusted.as_of_date,
            decision_cutoff_at=trusted.decision_cutoff_at,
            input_snapshot_id=snapshot.snapshot_id,
        )
        session.add(run)
        await session.flush()

        result = await retriever.retrieve(
            session,
            trusted,
            body.question,
            snapshot.chunks_collection,
            final_limit=body.top_k,
            snapshot=snapshot,
        )
        return {
            "question": body.question,
            "run_id": str(run.id),
            "snapshot_id": str(snapshot.snapshot_id),
            "candidates": [
                {
                    "section_id": str(c.section_id),
                    "document_version_id": str(c.document_version_id),
                    "heading_path": c.heading_path,
                    "page": c.page_start,
                    "text": c.text,
                    "text_hash": c.text_hash,
                }
                for c in result.candidates
            ],
            "channel_config": result.channel_config,
        }


@app.get("/api/v1/evidence/preview")
async def evidence_preview(
    case_id: uuid.UUID,
    section_id: uuid.UUID,
    document_version_id: uuid.UUID,
    parse_run_id: uuid.UUID,
    page_number: int,
    text_hash: str,
):
    """EvidenceRef -> 原始 PDF 页 PNG。

    P0-2：必须携带 case_id；服务端先做 Membership + 案件绑定授权再渲染。"""
    from creditlens.application.trusted_context import build_trusted_context

    ref = EvidenceRef(
        section_id=section_id,
        document_version_id=document_version_id,
        parse_run_id=parse_run_id,
        page_number=page_number,
        heading_path=[],
        text_hash=text_hash,
    )
    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        trusted = await build_trusted_context(
            session, DEFAULT_TENANT_ID, case_id, user_id=DEMO_USER_ID
        )
        png = await preview_service.render_page(session, trusted, ref)
    return Response(content=png, media_type="image/png")


class RunRequest(BaseModel):
    run_type: str = "FULL_REVIEW"


async def _execute_review_background(run_id: uuid.UUID, case_id: uuid.UUID) -> None:
    """后台执行完整预审（独立 session；预建 Run 处于 RECEIVED）。"""
    from creditlens.agents.wiring import build_supervisor
    from creditlens.application.snapshot_service import load_snapshot_context
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.infrastructure.llm.chat import build_chat_provider

    try:
        async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
            run = await session.get(ReviewRun, run_id)
            trusted = await build_trusted_context(session, DEFAULT_TENANT_ID, case_id, user_id=DEMO_USER_ID)
            snapshot = await load_snapshot_context(session, run.input_snapshot_id)
            supervisor, _ = build_supervisor(
                session, qdrant, embedder, snapshot, chat=build_chat_provider(settings)
            )
            await supervisor.execute_full_review(
                session, trusted, snapshot, run=run, commit_each_stage=True
            )
    except Exception:
        # 失败不得假成功：Run 置 FAILED 并保留已写入的 Trace（文档 §13）
        async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
            run = await session.get(ReviewRun, run_id)
            if run is not None and run.status not in {"COMPLETED", "HUMAN_REVIEW"}:
                run.status = "FAILED"


@app.post("/api/v1/cases/{case_id}/runs", status_code=202)
async def start_full_review(case_id: uuid.UUID, body: RunRequest):
    """启动完整预审：同事务冻结 Snapshot + 预建 Run，后台执行 DAG（文档 §14.4）。"""
    import asyncio

    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context

    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        trusted = await build_trusted_context(session, DEFAULT_TENANT_ID, case_id, user_id=DEMO_USER_ID)
        snapshot = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=settings.chunks_collection_name,
            summaries_collection=settings.summaries_collection_name,
            acl_hash=acl_scope_hash(trusted),
        )
        run = ReviewRun(
            tenant_id=trusted.tenant_id,
            case_id=trusted.case_id,
            run_type=body.run_type,
            status="RECEIVED",
            as_of_date=trusted.as_of_date,
            decision_cutoff_at=trusted.decision_cutoff_at,
            input_snapshot_id=snapshot.snapshot_id,
        )
        session.add(run)
        await session.flush()
        run_id = run.id

    asyncio.get_running_loop().create_task(_execute_review_background(run_id, case_id))
    return {
        "run_id": str(run_id),
        "status": "RECEIVED",
        "status_url": f"/api/v1/runs/{run_id}",
        "events_url": f"/api/v1/runs/{run_id}/events",
    }


@app.get("/api/v1/runs/{run_id}/events")
async def run_events_sse(run_id: uuid.UUID, last_event_id: int = 0):
    """SSE 进度：事实源为 run_events，支持 Last-Event-ID 续传（文档 §14.7）。

    P0-2：先对 Run 所属案件做 Membership 授权，再放事件流。"""
    import asyncio
    import json as jsonlib

    from fastapi.responses import StreamingResponse

    from creditlens.application.trusted_context import build_trusted_context

    # 授权前置：无权案件按 RUN_NOT_FOUND 处理，不泄露存在性
    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        try:
            await build_trusted_context(
                session, DEFAULT_TENANT_ID, run.case_id, user_id=DEMO_USER_ID
            )
        except CreditLensError as exc:
            raise HTTPException(404, "RUN_NOT_FOUND") from exc

    terminal = {"COMPLETED", "FAILED", "DENIED", "HUMAN_REVIEW", "NEED_MORE_INFO"}

    async def stream():
        cursor = last_event_id
        while True:
            async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
                events = (
                    await session.scalars(
                        select(RunEvent)
                        .where(RunEvent.run_id == run_id, RunEvent.sequence_no > cursor)
                        .order_by(RunEvent.sequence_no)
                    )
                ).all()
                run = await session.get(ReviewRun, run_id)
            for event in events:
                cursor = event.sequence_no
                payload = jsonlib.dumps(
                    {"type": event.event_type, **event.payload_redacted}, ensure_ascii=False
                )
                yield f"id: {event.sequence_no}\nevent: {event.event_type}\ndata: {payload}\n\n"
            if run is None or run.status in terminal:
                yield f"event: DONE\ndata: {{\"status\": \"{run.status if run else 'UNKNOWN'}\"}}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")


class ReviewDecisionRequest(BaseModel):
    action: str  # APPROVE_CLAIM | REJECT_CLAIM | REQUEST_CHANGES | ...
    target_claim_ids: list[uuid.UUID] = []
    reason_code: str = ""
    reason: str = ""


@app.post("/api/v1/runs/{run_id}/review-decisions")
async def submit_review_decision(run_id: uuid.UUID, body: ReviewDecisionRequest):
    """人工复核决定（任务 26）：追加写，不覆盖 Agent Claim。"""
    from creditlens.agents.supervisor import resume_after_human_review
    from creditlens.infrastructure.postgres.models import HumanDecision

    allowed_actions = {
        "APPROVE_CLAIM", "REJECT_CLAIM", "REQUEST_CHANGES", "REQUEST_MORE_INFORMATION",
        "RERUN_TASK", "OVERRIDE_WITH_REASON", "SUBMIT_REPORT", "APPROVE_REPORT_DRAFT",
    }
    if body.action not in allowed_actions:
        raise HTTPException(422, "INVALID_ACTION")

    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        decision = HumanDecision(
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
            target_claim_ids=[str(c) for c in body.target_claim_ids],
            action=body.action,
            reason_code=body.reason_code,
            reason=body.reason,
        )
        status = await resume_after_human_review(session, run_id, decision)
        return {"run_id": str(run_id), "status": status, "action": body.action}


@app.get("/api/v1/runs/{run_id}")
async def get_run(run_id: uuid.UUID):
    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        claims = (
            await session.scalars(select(ClaimRecord).where(ClaimRecord.run_id == run_id))
        ).all()
        return {
            "run_id": str(run.id),
            "status": run.status,
            "state_version": run.state_version,
            "claims": [
                {
                    "claim_id": str(c.id),
                    "category": c.category,
                    "statement": c.statement,
                    "verdict": c.verdict,
                    "review_status": c.review_status,
                    "evidence": c.payload,
                }
                for c in claims
            ],
        }


@app.get("/api/v1/runs/{run_id}/trace")
async def get_run_trace(run_id: uuid.UUID):
    from creditlens.application.trusted_context import build_trusted_context

    async with session_scope(session_factory, tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        # P0-2：Trace 属案件数据，先授权 Membership
        try:
            await build_trusted_context(
                session, DEFAULT_TENANT_ID, run.case_id, user_id=DEMO_USER_ID
            )
        except CreditLensError as exc:
            raise HTTPException(404, "RUN_NOT_FOUND") from exc
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.sequence_no)
            )
        ).all()
        if not events:
            raise HTTPException(404, "RUN_NOT_FOUND")
        return {
            "run_id": str(run_id),
            "events": [
                {
                    "sequence_no": e.sequence_no,
                    "event_type": e.event_type,
                    "payload": e.payload_redacted,
                    "occurred_at": e.occurred_at.isoformat()
                    if isinstance(e.occurred_at, datetime)
                    else e.occurred_at,
                }
                for e in events
            ],
        }
