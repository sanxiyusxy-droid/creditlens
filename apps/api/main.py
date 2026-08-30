"""CreditLens API 入口（MVP）。

启动：uv run uvicorn apps.api.main:app --reload
本地离线模式使用 SQLite / 内存 Qdrant / 本地对象存储；生产模式通过 .env 指向
PostgreSQL / Qdrant / MinIO。注意：内存 Qdrant 进程重启后需重跑种子脚本。
"""

import asyncio
import contextlib
import hmac
import inspect
import logging
import os
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
from sqlalchemy import func, select, text  # noqa: E402

from creditlens import __version__  # noqa: E402
from creditlens.application.qa_service import QAService, QAServiceError  # noqa: E402
from creditlens.common.clock import utc_now  # noqa: E402
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
    InvocationRecord,
    ReviewRun,
    RunEvent,
    TelemetryOutbox,
)
from creditlens.infrastructure.postgres.session import (  # noqa: E402
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client  # noqa: E402
from creditlens.observability.invocation import (  # noqa: E402
    InvocationEnvelope,
    hash_invocation_envelope,
    invocation_run_event_payload,
)
from creditlens.observability.outbox_worker import (  # noqa: E402
    LocalDirectoryTelemetryExporter,
    NoopTelemetryExporter,
    TelemetryOutboxWorker,
)
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
_READINESS_PROBE_TIMEOUT_SECONDS = 3.0

# MVP 单租户：与种子脚本一致
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
# MVP 无登录层：以固定演示用户模拟已验证 Token（RLS Membership 需要）
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")


@dataclass(frozen=True, slots=True)
class APIIdentity:
    """Identity that has been accepted by the server-side request gate."""

    tenant_id: uuid.UUID
    user_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class _InvocationIntegrity:
    valid: bool
    envelope: InvocationEnvelope | None = None
    error_code: str | None = None


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


def _resolve_local_telemetry_directory() -> Path:
    configured = Path(str(settings.telemetry_local_directory))
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    resolved = configured.resolve()
    allowed_root = (PROJECT_ROOT / "evaluation" / "reports" / "local").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        raise RuntimeError("TELEMETRY_LOCAL_DIRECTORY_FORBIDDEN") from None
    return resolved


def _build_api_telemetry_worker() -> TelemetryOutboxWorker | None:
    """Build only the explicitly enabled local tenant-shard worker.

    A normal RLS application factory must never be treated as a cross-tenant
    worker identity.  Production deployments run a separate exporter service
    with a dedicated service role or one worker per tenant shard.
    """

    # Minimal protocol tests and older embedded settings objects predate this
    # optional worker; absence must preserve the secure disabled default.
    if not getattr(settings, "telemetry_outbox_worker_enabled", False):
        return None
    if str(settings.app_env).strip().lower() not in {"local", "development", "dev", "test"}:
        raise RuntimeError("API_TELEMETRY_WORKER_FORBIDDEN")
    backend = str(settings.telemetry_exporter_backend).strip().lower()
    if backend == "noop":
        exporter = NoopTelemetryExporter()
    elif backend == "local_directory":
        resolved = _resolve_local_telemetry_directory()
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise RuntimeError("TELEMETRY_LOCAL_DIRECTORY_UNAVAILABLE") from None
        if not resolved.is_dir() or not os.access(resolved, os.W_OK):
            raise RuntimeError("TELEMETRY_LOCAL_DIRECTORY_UNAVAILABLE")
        exporter = LocalDirectoryTelemetryExporter(resolved)
    else:
        raise RuntimeError("TELEMETRY_EXPORTER_NOT_CONFIGURED")
    poll_seconds = float(settings.telemetry_export_poll_seconds)
    if poll_seconds <= 0:
        raise RuntimeError("TELEMETRY_EXPORT_POLL_SECONDS_INVALID")
    batch_size = int(settings.telemetry_export_batch_size)
    if not 1 <= batch_size <= 1_000:
        raise RuntimeError("TELEMETRY_EXPORT_BATCH_SIZE_INVALID")
    return TelemetryOutboxWorker(
        session_factory,
        exporter,
        max_attempts=settings.telemetry_export_max_attempts,
        lease_seconds=settings.telemetry_export_lease_seconds,
        base_backoff_seconds=settings.telemetry_export_base_backoff_seconds,
        max_backoff_seconds=settings.telemetry_export_max_backoff_seconds,
        tenant_id=DEFAULT_TENANT_ID,
        user_id=DEMO_USER_ID,
    )


async def _postgresql_readiness_probe() -> bool:
    async with session_scope(session_factory) as session:
        await session.execute(text("SELECT 1"))
    return True


async def _qdrant_readiness_probe() -> bool:
    get_collections = getattr(qdrant, "get_collections", None)
    if not callable(get_collections):
        return False
    result = await asyncio.to_thread(get_collections)
    return result is not None


async def _object_store_readiness_probe() -> bool:
    backend = str(getattr(settings, "object_store_backend", "")).strip().lower()
    if backend == "minio":
        client = getattr(object_store, "_client", None)
        bucket_exists = getattr(client, "bucket_exists", None)
        if not callable(bucket_exists):
            return False
        buckets = tuple(
            dict.fromkeys(
                str(getattr(settings, name, "")).strip()
                for name in (
                    "minio_raw_bucket",
                    "minio_derived_bucket",
                    "minio_rendered_bucket",
                )
            )
        )
        if not buckets or any(not bucket for bucket in buckets):
            return False

        def all_buckets_exist() -> bool:
            return all(bool(bucket_exists(bucket)) for bucket in buckets)

        return await asyncio.to_thread(all_buckets_exist)
    if backend == "local_fs":
        root = Path(str(getattr(settings, "local_object_root", ""))).resolve()
        return await asyncio.to_thread(root.is_dir)
    return False


async def _telemetry_readiness_probe(app_state) -> bool:
    if not getattr(settings, "telemetry_outbox_worker_enabled", False):
        return True
    if not getattr(app_state, "telemetry_worker_enabled", False):
        return False
    task = getattr(app_state, "telemetry_task", None)
    done = getattr(task, "done", None)
    if not callable(done) or done():
        return False
    if str(getattr(settings, "telemetry_exporter_backend", "")).strip().lower() != (
        "local_directory"
    ):
        return False
    worker = getattr(app_state, "telemetry_worker", None)
    exporter = getattr(worker, "_exporter", None)
    if not isinstance(exporter, LocalDirectoryTelemetryExporter):
        return False
    directory = exporter.directory.resolve()
    allowed_root = (PROJECT_ROOT / "evaluation" / "reports" / "local").resolve()
    try:
        directory.relative_to(allowed_root)
    except ValueError:
        return False
    return directory.is_dir() and os.access(directory, os.W_OK)


async def _bounded_readiness_probe(probe) -> bool:
    try:
        return bool(await asyncio.wait_for(probe, timeout=_READINESS_PROBE_TIMEOUT_SECONDS))
    except Exception:
        # Dependency errors may contain credentials, URLs, SQL or provider responses.
        # Readiness intentionally collapses all of them to a stable component code.
        return False


async def _run_telemetry_worker(
    worker: TelemetryOutboxWorker,
    stop: asyncio.Event,
) -> None:
    poll_seconds = float(settings.telemetry_export_poll_seconds)
    if poll_seconds <= 0:
        raise RuntimeError("TELEMETRY_EXPORT_POLL_SECONDS_INVALID")
    while not stop.is_set():
        try:
            await worker.process_batch(batch_size=settings.telemetry_export_batch_size)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Never log exporter/database exception text: it may contain a URL,
            # SQL fragment or provider detail.  The outbox owns bounded delivery
            # error codes; a loop-level failure is only classified here.
            logger.warning("telemetry worker cycle failed: %s", type(error).__name__)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_seconds)


async def _stop_telemetry_worker(
    task: asyncio.Task | None,
    stop: asyncio.Event | None,
) -> None:
    if task is None:
        return
    if stop is not None:
        stop.set()
    try:
        await asyncio.wait_for(task, timeout=5)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
    except Exception as error:
        # Consume an already-terminated worker failure so shutdown can still
        # close every resource. Never log exception text or provider details.
        logger.warning("telemetry worker stopped unexpectedly: %s", type(error).__name__)


async def _mark_background_run_failed(
    run_id: uuid.UUID,
    identity: APIIdentity,
    *,
    event_type: str,
    error_type: str,
) -> None:
    async with session_scope(
        session_factory,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
    ) as session:
        await _mark_run_failed_with_trace(session, run_id, event_type, error_type)


async def _run_bounded_cancellation_cleanup(cleanup, *, timeout_seconds: float) -> None:
    """Best-effort cleanup that cannot indefinitely delay caller cancellation."""

    cleanup_task = asyncio.create_task(cleanup)

    def consume_result(task: asyncio.Task) -> None:
        with contextlib.suppress(BaseException):
            task.result()

    cleanup_task.add_done_callback(consume_result)
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(
            asyncio.shield(cleanup_task),
            timeout=timeout_seconds,
        )


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
    run.completed_at = utc_now()
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


def _validate_persisted_invocation(
    record: InvocationRecord,
    outbox: TelemetryOutbox | None,
    run: ReviewRun,
) -> _InvocationIntegrity:
    """Validate the append-only projection before exposing it through Trace."""

    try:
        envelope = InvocationEnvelope.model_validate(record.payload_redacted)
        projected = invocation_run_event_payload(envelope)
        payload_hash_matches = hmac.compare_digest(
            hash_invocation_envelope(envelope),
            record.payload_sha256,
        )
        record_matches = (
            isinstance(record.payload_redacted, dict)
            and projected == record.payload_redacted
            and envelope.invocation_id == record.invocation_id
            and envelope.contract_version == record.contract_version
            and envelope.kind.value == record.kind
            and envelope.name == record.name
            and envelope.provider == record.provider
            and envelope.model == record.model
            and envelope.version == record.version
            and envelope.actor_role == record.actor_role
            and envelope.task_id == record.task_id
            and envelope.status.value == record.status
            and envelope.ended_at == record.ended_at
            and record.tenant_id == run.tenant_id
            and record.case_id == run.case_id
            and record.run_id == run.id
            and payload_hash_matches
        )
        outbox_matches = outbox is None or (
            outbox.tenant_id == record.tenant_id
            and outbox.case_id == record.case_id
            and outbox.run_id == record.run_id
            and outbox.invocation_id == record.invocation_id
            and outbox.topic == "INVOCATION_TERMINATED"
            and outbox.status in {"PENDING", "PROCESSING", "DELIVERED", "DEAD"}
        )
    except Exception:
        return _InvocationIntegrity(False, error_code="INVOCATION_INTEGRITY_FAILED")
    if not record_matches or not outbox_matches:
        return _InvocationIntegrity(False, error_code="INVOCATION_INTEGRITY_FAILED")
    return _InvocationIntegrity(True, envelope=envelope)


def _trace_invocation_response(
    record: InvocationRecord,
    outbox: TelemetryOutbox | None,
    integrity: _InvocationIntegrity,
) -> dict:
    """Project only a revalidated envelope; invalid storage never gets echoed."""

    if not integrity.valid or integrity.envelope is None:
        return {
            "invocation_id": str(record.invocation_id),
            "contract_version": None,
            "kind": None,
            "name": None,
            "status": None,
            "payload_sha256": None,
            "envelope": None,
            "integrity": {
                "status": "DEGRADED",
                "valid": False,
                "error_code": integrity.error_code or "INVOCATION_INTEGRITY_FAILED",
            },
            "delivery": {
                "status": "INVALID",
                "error_code": "INVOCATION_INTEGRITY_FAILED",
            },
        }

    envelope = integrity.envelope
    if outbox is None:
        delivery = {
            "status": "MISSING",
            "attempts": 0,
            "last_error_code": None,
            "available_at": None,
            "delivered_at": None,
            "dead_at": None,
        }
    else:
        delivery = {
            "status": outbox.status,
            "attempts": outbox.attempts,
            "last_error_code": outbox.last_error_code,
            "available_at": outbox.available_at.isoformat(),
            "delivered_at": (
                outbox.delivered_at.isoformat() if outbox.delivered_at is not None else None
            ),
            "dead_at": outbox.dead_at.isoformat() if outbox.dead_at is not None else None,
        }
    return {
        "invocation_id": str(envelope.invocation_id),
        "contract_version": envelope.contract_version,
        "kind": envelope.kind.value,
        "name": envelope.name,
        "status": envelope.status.value,
        "payload_sha256": record.payload_sha256,
        "envelope": invocation_run_event_payload(envelope),
        "integrity": {"status": "VALID", "valid": True, "error_code": None},
        "delivery": delivery,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _assert_demo_identity_is_safe(settings)
    app.state.runtime_started = False
    app.state.telemetry_worker_enabled = False
    app.state.telemetry_worker = None
    app.state.telemetry_task = None
    telemetry_worker: TelemetryOutboxWorker | None = None
    telemetry_task: asyncio.Task | None = None
    telemetry_stop: asyncio.Event | None = None
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        telemetry_worker = _build_api_telemetry_worker()
        if telemetry_worker is not None:
            telemetry_stop = asyncio.Event()
            telemetry_task = asyncio.create_task(
                _run_telemetry_worker(telemetry_worker, telemetry_stop),
                name="creditlens-telemetry-outbox",
            )
            # Give an immediately failing task one scheduling turn. Invalid
            # configuration and startup failures must fail lifespan before the
            # application reports readiness.
            await asyncio.sleep(0)
            if telemetry_task.done():
                with contextlib.suppress(BaseException):
                    telemetry_task.result()
                raise RuntimeError("TELEMETRY_WORKER_START_FAILED")
        app.state.telemetry_worker_enabled = telemetry_task is not None
        app.state.telemetry_worker = telemetry_worker
        app.state.telemetry_task = telemetry_task
        app.state.runtime_started = True
        # 没有 lease/heartbeat 协议时，启动过程不根据 started_at 推断非终态 Run 已失活。
        # 真实后台 task 的异常仍由 _execute_review_background 当场收口为 FAILED。
        yield
    finally:
        app.state.runtime_started = False
        try:
            await _stop_telemetry_worker(telemetry_task, telemetry_stop)
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
app.state.runtime_started = False
app.state.telemetry_worker_enabled = False
app.state.telemetry_worker = None
app.state.telemetry_task = None


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
async def health_ready(request: Request):
    if not getattr(request.app.state, "runtime_started", False):
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error_code": "RUNTIME_NOT_READY",
                "unavailable": ["runtime"],
            },
        )

    object_store_component = (
        "minio"
        if str(getattr(settings, "object_store_backend", "")).strip().lower() == "minio"
        else "object_store"
    )
    checks = [
        ("postgresql", _postgresql_readiness_probe()),
        ("qdrant", _qdrant_readiness_probe()),
        (object_store_component, _object_store_readiness_probe()),
    ]
    if getattr(settings, "telemetry_outbox_worker_enabled", False):
        checks.append(("telemetry", _telemetry_readiness_probe(request.app.state)))
    results = await asyncio.gather(*(_bounded_readiness_probe(probe) for _, probe in checks))
    unavailable = [name for (name, _), ready in zip(checks, results, strict=True) if not ready]
    if unavailable:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error_code": "RUNTIME_NOT_READY",
                "unavailable": unavailable,
            },
        )
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
                invocation_fingerprint_secret=settings.invocation_fingerprint_secret,
                invocation_fingerprint_key_version=settings.invocation_fingerprint_key_version,
                invocation_session_factory=session_factory,
                invocation_cancel_persist_timeout_seconds=(
                    settings.invocation_cancel_persist_timeout_seconds
                ),
            )
            await supervisor.execute_full_review(
                session, trusted, snapshot, run=run, commit_each_stage=True
            )
    except asyncio.CancelledError:
        # Preserve cancellation control flow, but do not leave a checkpointed
        # run in EXECUTING. The invocation itself was persisted through the
        # Supervisor's independent cancellation writer.
        await _run_bounded_cancellation_cleanup(
            _mark_background_run_failed(
                run_id,
                identity,
                event_type="BACKGROUND_CANCELLED",
                error_type="FULL_REVIEW_CANCELLED",
            ),
            timeout_seconds=settings.invocation_cancel_persist_timeout_seconds,
        )
        raise
    except Exception as exc:
        # 失败不得假成功：Run 置 FAILED 并保留已写入的 Trace（文档 §13）；
        # 记录异常类型便于排查（v1.0 演示踩坑教训）
        await _mark_background_run_failed(
            run_id,
            identity,
            event_type="BACKGROUND_FAILED",
            error_type=getattr(exc, "error_code", type(exc).__name__),
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
        invocations = (
            await session.scalars(
                select(InvocationRecord)
                .where(InvocationRecord.run_id == run_id)
                .order_by(InvocationRecord.ended_at, InvocationRecord.invocation_id)
            )
        ).all()
        delivery_rows = (
            await session.scalars(
                select(TelemetryOutbox)
                .where(TelemetryOutbox.run_id == run_id)
                .order_by(TelemetryOutbox.created_at, TelemetryOutbox.id)
            )
        ).all()
        if not events and not invocations and not delivery_rows:
            raise HTTPException(404, "RUN_NOT_FOUND")
        delivery_by_invocation = {row.invocation_id: row for row in delivery_rows}
        counts = {
            status: 0
            for status in (
                "PENDING",
                "PROCESSING",
                "DELIVERED",
                "DEAD",
                "MISSING",
                "INVALID",
            )
        }
        integrity_by_invocation: dict[uuid.UUID, _InvocationIntegrity] = {}
        for invocation in invocations:
            delivery = delivery_by_invocation.get(invocation.invocation_id)
            integrity = _validate_persisted_invocation(invocation, delivery, run)
            integrity_by_invocation[invocation.invocation_id] = integrity
            if not integrity.valid:
                counts["INVALID"] += 1
            else:
                counts[delivery.status if delivery is not None else "MISSING"] += 1

        invocation_ids = {record.invocation_id for record in invocations}
        orphan_delivery_count = sum(
            row.invocation_id not in invocation_ids for row in delivery_rows
        )
        integrity_invalid_count = counts["INVALID"] + orphan_delivery_count

        is_v2_run = (run.model_manifest or {}).get("invocation_contract_version") == "invocation_v2"
        if integrity_invalid_count:
            delivery_status = "DEGRADED"
            delivery_complete = False
        elif not invocations and not is_v2_run:
            delivery_status = "LEGACY_UNAVAILABLE"
            delivery_complete: bool | None = None
        elif not invocations:
            delivery_status = "EMPTY"
            delivery_complete = False
        elif counts["DEAD"] or counts["MISSING"]:
            delivery_status = "DEGRADED"
            delivery_complete = False
        elif counts["PENDING"] or counts["PROCESSING"]:
            delivery_status = "PENDING"
            delivery_complete = False
        else:
            delivery_status = "COMPLETE"
            delivery_complete = True
        if not invocations and not delivery_rows and not is_v2_run:
            integrity_status = "LEGACY_UNAVAILABLE"
            integrity_valid: bool | None = None
        elif integrity_invalid_count:
            integrity_status = "DEGRADED"
            integrity_valid = False
        elif not invocations:
            integrity_status = "EMPTY"
            integrity_valid = False
        else:
            integrity_status = "VALID"
            integrity_valid = True
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
            "invocations": [
                _trace_invocation_response(
                    record,
                    delivery_by_invocation.get(record.invocation_id),
                    integrity_by_invocation[record.invocation_id],
                )
                for record in invocations
            ],
            "integrity": {
                "status": integrity_status,
                "valid": integrity_valid,
                "invalid_count": integrity_invalid_count,
            },
            "delivery": {
                "contract_version": (
                    "invocation_v2" if is_v2_run or invocations or delivery_rows else None
                ),
                "status": delivery_status,
                "complete": delivery_complete,
                "total": len(invocations),
                "counts": counts,
            },
        }
