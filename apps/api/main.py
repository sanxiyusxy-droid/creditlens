"""CreditLens API 入口（MVP）。

启动：uv run uvicorn apps.api.main:app --reload
本地离线模式使用 SQLite / 内存 Qdrant / 本地对象存储；生产模式通过 .env 指向
PostgreSQL / Qdrant / MinIO。注意：内存 Qdrant 进程重启后需重跑种子脚本。
"""

import inspect
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from creditlens import __version__  # noqa: E402
from creditlens.application.qa_service import QAService, QAServiceError  # noqa: E402
from creditlens.common.config import get_settings  # noqa: E402
from creditlens.common.errors import CreditLensError  # noqa: E402
from creditlens.evidence.preview import EvidencePreviewService  # noqa: E402
from creditlens.infrastructure.llm.chat import build_chat_provider  # noqa: E402
from creditlens.infrastructure.llm.embedding import build_embedding_provider  # noqa: E402
from creditlens.infrastructure.objectstore import build_object_store  # noqa: E402
from creditlens.infrastructure.postgres.artifact_integrity import (  # noqa: E402
    ArtifactIntegrityError,
    validate_grounded_qa_artifact_and_claims,
)
from creditlens.infrastructure.postgres.models import (  # noqa: E402
    ArtifactRecord,
    Base,
    ClaimRecord,
    CreditCase,
    EvidenceRecord,
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
from creditlens.retrieval.orchestrator import RetrievalOrchestrator  # noqa: E402
from creditlens.retrieval.rerank import build_reranker  # noqa: E402

settings = get_settings()
engine = create_engine()
session_factory = create_session_factory(engine)
object_store = build_object_store(settings)
qdrant = build_qdrant_client(settings)
embedder = build_embedding_provider(settings)
reranker = build_reranker(settings)
chat_provider = build_chat_provider(settings)
orchestrator = RetrievalOrchestrator(
    qdrant=qdrant, embedder=embedder, reranker=reranker, rrf_k=settings.rrf_k
)
preview_service = EvidencePreviewService(object_store)
logger = logging.getLogger(__name__)

# MVP 单租户：与种子脚本一致
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
# MVP 无登录层：以固定演示用户模拟已验证 Token（RLS Membership 需要）
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")


@dataclass(frozen=True, slots=True)
class APIIdentity:
    """Identity that has been accepted by the server-side request gate."""

    tenant_id: uuid.UUID
    user_id: uuid.UUID


_REQUEST_IDENTITY: ContextVar[APIIdentity | None] = ContextVar(
    "creditlens_api_identity", default=None
)
_PUBLIC_API_PATHS = frozenset(
    {
        "/health/live",
        "/health/ready",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)

# SSE 在这些状态下不再有后台阶段可等待。REWORK/数据质量阻断等状态虽然可能
# 由后续人工操作创建新的阶段，但当前连接必须结束，避免客户端无限轮询。
SSE_STOP_STATUSES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "DENIED",
        "HUMAN_REVIEW",
        "REWORK",
        "NEED_MORE_INFO",
        "DATA_QUALITY_BLOCKED",
        "SUPERSEDED",
    }
)


def _assert_demo_identity_is_safe(current_settings) -> None:
    """仅在双重显式授权的本地环境允许固定演示身份。"""
    mode = str(current_settings.api_identity_mode).strip().lower()
    environment = str(current_settings.app_env).strip().lower()
    if mode != "demo":
        raise RuntimeError("API_IDENTITY_PROVIDER_NOT_CONFIGURED")
    if not current_settings.allow_insecure_demo_identity:
        raise RuntimeError("INSECURE_DEMO_IDENTITY_NOT_ALLOWED")
    if environment not in {"local", "development", "dev", "test"}:
        raise RuntimeError("DEMO_IDENTITY_FORBIDDEN_OUTSIDE_LOCAL_ENV")


def _resolve_api_identity(current_settings) -> APIIdentity:
    """Resolve the only identity mode implemented in v1.3, failing closed otherwise."""
    _assert_demo_identity_is_safe(current_settings)
    return APIIdentity(tenant_id=DEFAULT_TENANT_ID, user_id=DEMO_USER_ID)


def _current_api_identity() -> APIIdentity:
    """Return the request-gated identity, also protecting direct endpoint calls in tests."""
    identity = _REQUEST_IDENTITY.get()
    if identity is not None:
        return identity
    return _resolve_api_identity(settings)


async def _close_resources_best_effort(
    resources: list[tuple[str, object | None, tuple[str, ...]]],
) -> list[str]:
    """互不影响地关闭运行时资源，返回关闭失败的资源名。"""
    failures: list[str] = []
    for resource_name, resource, method_names in resources:
        if resource is None:
            continue
        close_method = next(
            (
                method
                for method_name in method_names
                if callable(method := getattr(resource, method_name, None))
            ),
            None,
        )
        if close_method is None:
            continue
        try:
            result = close_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            failures.append(resource_name)
            logger.exception("failed to close API resource: %s", resource_name)
    return failures


async def _mark_run_failed_with_trace(
    session, run_id: uuid.UUID, event_type: str, error_type: str
) -> bool:
    """原子地把仍在执行的 Run 标为失败，并补一条最小审计事件。

    调用方先锁定 Run 行，避免与同一 Run 的状态迁移并发覆盖；序号在该锁内以
    ``max(sequence_no) + 1`` 生成。事件载荷仅保存异常类别，避免将底层详情
    写入可被 API 读取的 Trace。
    """
    run = await session.scalar(select(ReviewRun).where(ReviewRun.id == run_id).with_for_update())
    if run is None or run.status in SSE_STOP_STATUSES:
        return False
    run.status = "FAILED"
    run.state_version += 1
    last_sequence = await session.scalar(
        select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run.id)
    )
    session.add(
        RunEvent(
            run_id=run.id,
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            sequence_no=(last_sequence or 0) + 1,
            event_type=event_type,
            payload_redacted={"error_type": error_type},
        )
    )
    await session.flush()
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _assert_demo_identity_is_safe(settings)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # 没有 lease/heartbeat 协议时，启动过程不根据 started_at 推断非终态 Run 已失活。
        # 真实后台 task 的异常仍由 _execute_review_background 当场收口为 FAILED。
        yield
    finally:
        await _close_resources_best_effort(
            [
                ("chat", chat_provider, ("aclose", "close")),
                ("embedding", embedder, ("aclose", "close")),
                ("reranker", reranker, ("aclose", "close")),
                ("qdrant", qdrant, ("aclose", "close")),
                ("engine", engine, ("dispose", "aclose", "close")),
            ]
        )


app = FastAPI(title="CreditLens API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def require_api_identity(request: Request, call_next):
    """Fail closed on every business request even when ASGI lifespan is disabled."""
    normalized_path = request.url.path.rstrip("/") or "/"
    if normalized_path in _PUBLIC_API_PATHS:
        return await call_next(request)

    try:
        identity = _resolve_api_identity(settings)
    except RuntimeError:
        # Do not expose whether the provider, environment, or demo opt-in is misconfigured.
        return JSONResponse(
            status_code=503,
            content={"detail": {"error_code": "API_IDENTITY_UNAVAILABLE"}},
        )

    token = _REQUEST_IDENTITY.set(identity)
    try:
        return await call_next(request)
    finally:
        _REQUEST_IDENTITY.reset(token)


@app.exception_handler(CreditLensError)
async def creditlens_error_handler(request, exc: CreditLensError):
    if exc.error_code in {"REVIEW_CONFLICT", "IDEMPOTENCY_CONFLICT"}:
        status = 409  # WP3：并发审批冲突
    elif exc.error_code in {"ACL_DENIED", "TOOL_CALL_DENIED", "ACTION_NOT_AUTHORIZED"}:
        status = 403
    else:
        status = 422
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

    async with session_scope(session_factory) as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: uuid.UUID):
    identity = _current_api_identity()
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
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
    idempotency_key: str = Field(min_length=8, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    # 演示时点切换（任务 30）：不传则用案件默认时点
    as_of_date: date | None = None
    decision_cutoff_at: datetime | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("decision_cutoff_at")
    @classmethod
    def validate_cutoff_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("decision_cutoff_at 必须包含时区")
        return value


@app.post("/api/v1/cases/{case_id}/questions")
async def ask_question(case_id: uuid.UUID, body: QuestionRequest):
    """可审计简单问答：Snapshot → 多路检索 → Grounded QA → 引用审计。"""
    identity = _current_api_identity()
    service = QAService(
        session_factory=session_factory,
        orchestrator=orchestrator,
        settings=settings,
        chat=chat_provider,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
    )
    try:
        result = await service.ask(
            case_id=case_id,
            question=body.question,
            top_k=body.top_k,
            as_of_date=body.as_of_date,
            decision_cutoff_at=body.decision_cutoff_at,
            idempotency_key=body.idempotency_key,
        )
    except QAServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "GROUNDED_QA_FAILED",
                "run_id": str(exc.run_id),
                "error_type": exc.error_type,
                "trace_url": f"/api/v1/runs/{exc.run_id}/trace",
            },
        ) from exc
    return result.model_dump(mode="json")


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

    identity = _current_api_identity()

    ref = EvidenceRef(
        section_id=section_id,
        document_version_id=document_version_id,
        parse_run_id=parse_run_id,
        page_number=page_number,
        heading_path=[],
        text_hash=text_hash,
    )
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        trusted = await build_trusted_context(
            session, identity.tenant_id, case_id, user_id=identity.user_id
        )
        png = await preview_service.render_page(session, trusted, ref)
    return Response(content=png, media_type="image/png")


class RunRequest(BaseModel):
    run_type: str = "FULL_REVIEW"


async def _execute_review_background(
    run_id: uuid.UUID, case_id: uuid.UUID, identity: APIIdentity
) -> None:
    """后台执行完整预审（独立 session；预建 Run 处于 RECEIVED）。"""
    from creditlens.agents.wiring import build_supervisor
    from creditlens.application.snapshot_service import load_snapshot_context
    from creditlens.application.trusted_context import build_trusted_context

    try:
        async with session_scope(
            session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
        ) as session:
            run = await session.get(ReviewRun, run_id)
            trusted = await build_trusted_context(
                session, identity.tenant_id, case_id, user_id=identity.user_id
            )
            snapshot = await load_snapshot_context(session, run.input_snapshot_id)
            supervisor, _ = build_supervisor(
                session,
                qdrant,
                embedder,
                snapshot,
                chat=chat_provider,
                reranker=reranker,
                rrf_k=settings.rrf_k,
            )
            await supervisor.execute_full_review(
                session, trusted, snapshot, run=run, commit_each_stage=True
            )
    except Exception as exc:
        # 失败不得假成功：Run 置 FAILED 并保留已写入的 Trace（文档 §13）；
        # 记录异常类型便于排查（v1.0 演示踩坑教训）
        async with session_scope(
            session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
        ) as session:
            await _mark_run_failed_with_trace(
                session, run_id, "BACKGROUND_FAILED", type(exc).__name__
            )


@app.post("/api/v1/cases/{case_id}/runs", status_code=202)
async def start_full_review(case_id: uuid.UUID, body: RunRequest):
    """启动完整预审：同事务冻结 Snapshot + 预建 Run，后台执行 DAG（文档 §14.4）。"""
    import asyncio

    from creditlens.application.snapshot_service import freeze_snapshot
    from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context

    identity = _current_api_identity()
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        trusted = await build_trusted_context(
            session, identity.tenant_id, case_id, user_id=identity.user_id
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
            run_type=body.run_type,
            status="RECEIVED",
            as_of_date=trusted.as_of_date,
            decision_cutoff_at=trusted.decision_cutoff_at,
            input_snapshot_id=snapshot.snapshot_id,
        )
        session.add(run)
        await session.flush()
        run_id = run.id

    asyncio.get_running_loop().create_task(_execute_review_background(run_id, case_id, identity))
    return {
        "run_id": str(run_id),
        "status": "RECEIVED",
        "status_url": f"/api/v1/runs/{run_id}",
        "events_url": f"/api/v1/runs/{run_id}/events",
    }


@app.get("/api/v1/runs/{run_id}/events")
async def run_events_sse(
    run_id: uuid.UUID,
    last_event_id: int = Query(default=0, ge=0),
    last_event_id_header: int | None = Header(default=None, alias="Last-Event-ID", ge=0),
):
    """SSE 进度：事实源为 run_events，支持 Last-Event-ID 续传（文档 §14.7）。

    P0-2：先对 Run 所属案件做 Membership 授权，再放事件流。"""
    import asyncio
    import json as jsonlib

    from fastapi.responses import StreamingResponse

    from creditlens.application.trusted_context import build_trusted_context

    identity = _current_api_identity()
    # 授权前置：无权案件按 RUN_NOT_FOUND 处理，不泄露存在性
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        try:
            await build_trusted_context(
                session, identity.tenant_id, run.case_id, user_id=identity.user_id
            )
        except CreditLensError as exc:
            raise HTTPException(404, "RUN_NOT_FOUND") from exc

    # EventSource 的标准续传头优先；保留 query 参数，兼容既有客户端和调试链接。
    initial_cursor = last_event_id_header if last_event_id_header is not None else last_event_id

    async def stream():
        cursor = initial_cursor
        while True:
            async with session_scope(
                session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
            ) as session:
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
            if run is None or run.status in SSE_STOP_STATUSES:
                yield f'event: DONE\ndata: {{"status": "{run.status if run else "UNKNOWN"}"}}\n\n'
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")


class ReviewDecisionRequest(BaseModel):
    # WP3：RERUN_TASK/OVERRIDE_WITH_REASON 未真正实现，暂不开放
    action: str  # APPROVE_CLAIM | REJECT_CLAIM | REQUEST_CHANGES | ...
    target_claim_ids: list[uuid.UUID] = []
    reason_code: str = ""
    reason: str = ""
    # P1：幂等键与乐观锁改为必填——可选时并发请求可绕过版本校验，
    # 缺失即 422（契约错误），不再静默按"不校验"处理
    idempotency_key: str
    expected_state_version: int


@app.post("/api/v1/runs/{run_id}/review-decisions")
async def submit_review_decision(run_id: uuid.UUID, body: ReviewDecisionRequest):
    """人工复核决定（任务 26）：追加写，不覆盖 Agent Claim。

    WP3：审批类动作仅案件 REVIEWER/OWNER 可执行；reviewer 由服务端注入，
    客户端不可自报；RERUN_TASK/OVERRIDE_WITH_REASON 未实现暂不开放。"""
    from creditlens.agents.supervisor import resume_after_human_review
    from creditlens.common.errors import ActionNotAuthorizedError
    from creditlens.infrastructure.postgres.models import CaseMembership, HumanDecision

    identity = _current_api_identity()
    allowed_actions = {
        "APPROVE_CLAIM",
        "REJECT_CLAIM",
        "REQUEST_CHANGES",
        "REQUEST_MORE_INFORMATION",
        "SUBMIT_REPORT",
        "APPROVE_REPORT_DRAFT",
    }
    if body.action not in allowed_actions:
        raise HTTPException(422, "INVALID_ACTION")

    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")

        # WP3：REVIEWER/OWNER 动作授权（Membership 案件角色）
        roles = (
            await session.scalars(
                select(CaseMembership.case_role).where(
                    CaseMembership.case_id == run.case_id,
                    CaseMembership.user_id == identity.user_id,
                    CaseMembership.revoked_at.is_(None),
                )
            )
        ).all()
        approval_actions = {
            "APPROVE_CLAIM",
            "REJECT_CLAIM",
            "SUBMIT_REPORT",
            "APPROVE_REPORT_DRAFT",
        }
        if body.action in approval_actions and not {"REVIEWER", "OWNER"} & set(roles):
            raise ActionNotAuthorizedError(
                f"动作 {body.action} 仅 REVIEWER/OWNER 可执行",
                {"action": body.action, "roles": list(roles)},
            )

        decision = HumanDecision(
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
            target_claim_ids=[str(c) for c in body.target_claim_ids],
            action=body.action,
            reason_code=body.reason_code,
            reason=body.reason,
            # WP3：reviewer 服务端注入，不接受客户端自报
            reviewer_id=identity.user_id,
            idempotency_key=body.idempotency_key,
            target_version=body.expected_state_version,
        )
        status = await resume_after_human_review(session, run_id, decision)
        return {"run_id": str(run_id), "status": status, "action": body.action}


@app.get("/api/v1/runs/{run_id}")
async def get_run(run_id: uuid.UUID):
    identity = _current_api_identity()
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        claims = (
            await session.scalars(select(ClaimRecord).where(ClaimRecord.run_id == run_id))
        ).all()
        evidence_rows = (
            await session.scalars(select(EvidenceRecord).where(EvidenceRecord.run_id == run_id))
        ).all()
        qa_artifacts = (
            await session.scalars(
                select(ArtifactRecord)
                .where(
                    ArtifactRecord.run_id == run_id,
                    ArtifactRecord.artifact_type == "GROUNDED_ANSWER",
                )
                .order_by(ArtifactRecord.created_at.desc())
            )
        ).all()
        if len(qa_artifacts) > 1:
            raise HTTPException(409, "ARTIFACT_INTEGRITY_FAILED")
        qa_artifact = qa_artifacts[0] if qa_artifacts else None
        verified_qa_payload = None
        if qa_artifact is not None:
            try:
                verified_qa_payload = validate_grounded_qa_artifact_and_claims(
                    run=run,
                    artifact=qa_artifact,
                    claims=claims,
                    expected_prompt_version=settings.qa_prompt_version,
                )
            except ArtifactIntegrityError as exc:
                raise HTTPException(409, "ARTIFACT_INTEGRITY_FAILED") from exc
        # evidence_id 是跨 Run 可复用的逻辑键；EvidenceRecord.id 则是本次 Run 的
        # 数据库行主键。因此必须按 evidence_key 映射，不能误用行 id。
        locators_by_key = {
            str(evidence.evidence_key): {
                "evidence_type": evidence.evidence_type,
                "section_id": str(evidence.section_id) if evidence.section_id else None,
                "document_version_id": str(evidence.document_version_id)
                if evidence.document_version_id
                else None,
                "parse_run_id": (evidence.locator or {}).get("parse_run_id"),
                "page_number": evidence.page_number,
                "content_hash": evidence.content_hash,
            }
            for evidence in evidence_rows
        }

        def evidence_locators(evidence_ids: list) -> list[dict]:
            return [
                locators_by_key[str(evidence_id)]
                for evidence_id in evidence_ids
                if str(evidence_id) in locators_by_key
            ]

        response = {
            "run_id": str(run.id),
            "status": run.status,
            "state_version": run.state_version,
            "execution": {
                "degraded": bool((run.model_manifest or {}).get("degraded", False)),
                "degraded_agents": list((run.model_manifest or {}).get("degraded_agents", [])),
            },
            "claims": [
                {
                    "claim_id": str(c.id),
                    "category": c.category,
                    "statement": c.statement,
                    "verdict": c.verdict,
                    "review_status": c.review_status,
                    "evidence": {
                        **(c.payload or {}),
                        "supporting_locators": evidence_locators(
                            (c.payload or {}).get("supporting_evidence_ids", [])
                        ),
                        "opposing_locators": evidence_locators(
                            (c.payload or {}).get("opposing_evidence_ids", [])
                        ),
                    },
                }
                for c in claims
            ],
        }
        if verified_qa_payload is not None:
            payload = verified_qa_payload
            response["grounded_answer"] = {
                "answer_status": payload.get("answer_status"),
                "answer": payload.get("direct_answer", ""),
                "missing_information": payload.get("missing_information", []),
                "conflicts": payload.get("conflicts", []),
                "abstention_reason": payload.get("abstention_reason"),
                "refusal_reason_code": payload.get("refusal_reason_code"),
                "prompt_version": payload.get("prompt_version"),
                "generation_mode": payload.get("generation_mode"),
            }
        return response


@app.get("/api/v1/runs/{run_id}/report")
async def get_run_report(run_id: uuid.UUID):
    """最新报告版本（P0-3 持久化产物；APPROVED_DRAFT ≠ 真实授信批准）。"""
    from creditlens.application.trusted_context import build_trusted_context
    from creditlens.infrastructure.postgres.models import ReportVersion

    identity = _current_api_identity()
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        try:
            await build_trusted_context(
                session, identity.tenant_id, run.case_id, user_id=identity.user_id
            )
        except CreditLensError as exc:
            raise HTTPException(404, "RUN_NOT_FOUND") from exc
        report = await session.scalar(
            select(ReportVersion)
            .where(ReportVersion.run_id == run_id)
            .order_by(ReportVersion.version_no.desc())
            .limit(1)
        )
        if report is None:
            raise HTTPException(404, "REPORT_NOT_READY")
        return {
            "run_id": str(run_id),
            "version_no": report.version_no,
            "status": report.status,
            "content_hash": report.content_hash,
            "created_at": report.created_at.isoformat(),
            "content": report.content_json,
        }


@app.get("/api/v1/runs/{run_id}/trace")
async def get_run_trace(run_id: uuid.UUID):
    from creditlens.application.trusted_context import build_trusted_context

    identity = _current_api_identity()
    async with session_scope(
        session_factory, tenant_id=identity.tenant_id, user_id=identity.user_id
    ) as session:
        run = await session.get(ReviewRun, run_id)
        if run is None:
            raise HTTPException(404, "RUN_NOT_FOUND")
        # P0-2：Trace 属案件数据，先授权 Membership
        try:
            await build_trusted_context(
                session, identity.tenant_id, run.case_id, user_id=identity.user_id
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
