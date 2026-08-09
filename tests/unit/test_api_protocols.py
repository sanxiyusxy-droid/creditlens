"""API 协议边界：SSE 续传/停止语义与错误码映射。"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from creditlens.common.errors import IdempotencyConflictError
from creditlens.infrastructure.postgres.models import (
    AppUser,
    ArtifactRecord,
    Base,
    CaseMembership,
    ClaimRecord,
    CreditCase,
    Entity,
    EvidenceRecord,
    ReviewRun,
    RunEvent,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory


def test_fixed_demo_identity_is_fail_closed_outside_local_environment():
    from apps.api import main as api_main

    from creditlens.common.config import Settings

    api_main._assert_demo_identity_is_safe(
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        )
    )
    assert Settings.model_fields["api_identity_mode"].default == "required"
    assert Settings.model_fields["allow_insecure_demo_identity"].default is False
    with pytest.raises(RuntimeError, match="INSECURE_DEMO_IDENTITY_NOT_ALLOWED"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="demo",
                allow_insecure_demo_identity=False,
                app_env="test",
            )
        )
    with pytest.raises(RuntimeError, match="DEMO_IDENTITY_FORBIDDEN"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="demo",
                allow_insecure_demo_identity=True,
                app_env="production",
            )
        )
    with pytest.raises(RuntimeError, match="IDENTITY_PROVIDER_NOT_CONFIGURED"):
        api_main._assert_demo_identity_is_safe(
            SimpleNamespace(
                api_identity_mode="required",
                allow_insecure_demo_identity=True,
                app_env="test",
            )
        )


@pytest.mark.parametrize(
    "unsafe_settings",
    [
        SimpleNamespace(
            api_identity_mode="required",
            allow_insecure_demo_identity=False,
            app_env="local",
        ),
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=False,
            app_env="test",
        ),
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="production",
        ),
    ],
    ids=["required-provider-missing", "demo-not-opted-in", "demo-in-production"],
)
async def test_request_identity_gate_fails_closed_without_lifespan(monkeypatch, unsafe_settings):
    """ASGITransport skips lifespan, so the request gate must independently reject access."""
    from apps.api import main as api_main

    monkeypatch.setattr(api_main, "settings", unsafe_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/cases/{uuid.uuid4()}/questions",
            json={"question": "is the case complete?", "idempotency_key": "identity-gate-001"},
        )
        live_response = await client.get("/health/live")

    assert response.status_code == 503
    assert response.json() == {"detail": {"error_code": "API_IDENTITY_UNAVAILABLE"}}
    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}


async def test_explicit_safe_demo_identity_works_without_lifespan(monkeypatch):
    """The local/test demo opt-in remains usable and supplies identity to the endpoint."""
    from apps.api import main as api_main

    captured: dict = {}

    class Result:
        @staticmethod
        def model_dump(*, mode: str):
            assert mode == "json"
            return {"accepted": True}

    class QAService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def ask(self, **_kwargs):
            return Result()

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        ),
    )
    monkeypatch.setattr(api_main, "QAService", QAService)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/cases/{uuid.uuid4()}/questions",
            json={"question": "is the case complete?", "idempotency_key": "identity-gate-002"},
        )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert captured["tenant_id"] == api_main.DEFAULT_TENANT_ID
    assert captured["user_id"] == api_main.DEMO_USER_ID


async def test_lifespan_does_not_infer_stale_runs_at_startup(monkeypatch):
    """没有 lease/heartbeat 时，启动只建表，不扫描或改写旧的非终态 Run。"""
    from apps.api import main as api_main

    calls: list[str] = []

    class Connection:
        async def run_sync(self, _operation):
            calls.append("create_all")

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    class Engine:
        def begin(self):
            return BeginContext()

        async def dispose(self):
            calls.append("engine_disposed")

    def fail_if_session_is_opened(*_args, **_kwargs):
        raise AssertionError("lifespan must not scan ReviewRun without a lease protocol")

    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            api_identity_mode="demo",
            allow_insecure_demo_identity=True,
            app_env="test",
        ),
    )
    monkeypatch.setattr(api_main, "engine", Engine())
    monkeypatch.setattr(api_main, "session_scope", fail_if_session_is_opened)
    monkeypatch.setattr(api_main, "chat_provider", None)
    monkeypatch.setattr(api_main, "embedder", object())
    monkeypatch.setattr(api_main, "reranker", None)
    monkeypatch.setattr(api_main, "qdrant", object())

    async with api_main.lifespan(SimpleNamespace()):
        calls.append("serving")

    assert calls == ["create_all", "serving", "engine_disposed"]


async def test_shutdown_closes_every_resource_when_one_close_fails():
    from apps.api import main as api_main

    closed: list[str] = []

    class AsyncResource:
        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail

        async def aclose(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    class SyncResource:
        def close(self):
            closed.append("qdrant")

    class Engine:
        async def dispose(self):
            closed.append("engine")

    failures = await api_main._close_resources_best_effort(
        [
            ("chat", AsyncResource("chat", fail=True), ("aclose", "close")),
            ("embedding", AsyncResource("embedding"), ("aclose", "close")),
            ("reranker", AsyncResource("reranker"), ("aclose", "close")),
            ("qdrant", SyncResource(), ("aclose", "close")),
            ("engine", Engine(), ("dispose",)),
        ]
    )

    assert failures == ["chat"]
    assert closed == ["chat", "embedding", "reranker", "qdrant", "engine"]


async def test_http_embedding_and_reranker_expose_connection_pool_close():
    from creditlens.infrastructure.llm.embedding import OpenAICompatEmbedding
    from creditlens.retrieval.rerank import HttpCrossEncoderReranker

    closed: list[str] = []

    class Client:
        def __init__(self, name: str):
            self.name = name

        async def aclose(self):
            closed.append(self.name)

    embedding = object.__new__(OpenAICompatEmbedding)
    embedding._client = Client("embedding")
    reranker = object.__new__(HttpCrossEncoderReranker)
    reranker._client = Client("reranker")

    await embedding.aclose()
    await reranker.aclose()

    assert closed == ["embedding", "reranker"]


@pytest.fixture
async def protocol_client(tmp_path, monkeypatch):
    """只构造 SSE 所需的最小案件/Membership 世界。"""
    from apps.api import main as api_main

    tenant_id = api_main.DEFAULT_TENANT_ID
    user_id = api_main.DEMO_USER_ID
    entity_id = uuid.uuid4()
    case_id = uuid.uuid4()
    run_id = uuid.uuid4()
    now = datetime(2026, 5, 1, tzinfo=UTC)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api_protocols.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    monkeypatch.setattr(api_main, "session_factory", factory)

    async with factory() as session:
        session.add_all(
            [
                Tenant(id=tenant_id, name="API protocol"),
                AppUser(
                    id=user_id,
                    tenant_id=tenant_id,
                    external_subject="api-protocol-user",
                    display_name="API protocol user",
                ),
                Entity(
                    id=entity_id,
                    tenant_id=tenant_id,
                    entity_type="COMPANY",
                    canonical_name="协议测试企业",
                ),
                CreditCase(
                    id=case_id,
                    tenant_id=tenant_id,
                    case_number="API-PROTOCOL",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1"),
                    application_date=date(2026, 5, 1),
                    as_of_date=date(2026, 5, 1),
                    decision_cutoff_at=now,
                ),
                CaseMembership(case_id=case_id, user_id=user_id, case_role="REVIEWER"),
                ReviewRun(
                    id=run_id,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    status="REWORK",
                    as_of_date=date(2026, 5, 1),
                    decision_cutoff_at=now,
                ),
                RunEvent(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    sequence_no=1,
                    event_type="STATE_CHANGED",
                    payload_redacted={"to": "REWORK"},
                ),
            ]
        )
        await session.commit()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
    ) as client:
        yield client, factory, run_id, case_id
    await engine.dispose()


async def test_sse_last_event_id_header_overrides_legacy_query(protocol_client):
    """标准 Last-Event-ID 头优先于兼容用 query 参数。"""
    client, _, run_id, _ = protocol_client
    response = await client.get(
        f"/api/v1/runs/{run_id}/events?last_event_id=0",
        headers={"Last-Event-ID": "1"},
    )

    assert response.status_code == 200
    assert "id: 1" not in response.text
    assert 'event: DONE\ndata: {"status": "REWORK"}' in response.text


async def test_sse_rework_stops_stream_and_query_cursor_remains_supported(protocol_client):
    """REWORK 不会让前端长连接悬挂，旧 query 续传仍会返回遗漏事件。"""
    client, _, run_id, _ = protocol_client
    response = await client.get(f"/api/v1/runs/{run_id}/events?last_event_id=0")

    assert response.status_code == 200
    assert "id: 1" in response.text
    assert 'event: DONE\ndata: {"status": "REWORK"}' in response.text


async def test_sse_rejects_negative_resume_cursor(protocol_client):
    """负游标不是合法的 RunEvent sequence，query/header 均应在入口返回 422。"""
    client, _, run_id, _ = protocol_client
    query_response = await client.get(f"/api/v1/runs/{run_id}/events?last_event_id=-1")
    header_response = await client.get(
        f"/api/v1/runs/{run_id}/events", headers={"Last-Event-ID": "-1"}
    )

    assert query_response.status_code == 422
    assert header_response.status_code == 422


async def test_idempotency_conflict_maps_to_409():
    from apps.api import main as api_main

    response = await api_main.creditlens_error_handler(
        None, IdempotencyConflictError("同一幂等键的请求内容不一致")
    )

    assert response.status_code == 409


async def test_failure_transition_appends_minimal_trace_and_bumps_version(protocol_client):
    """后台失败路径不应留下无 Trace 的 FAILED Run。"""
    from apps.api import main as api_main

    _, factory, _, case_id = protocol_client
    async with factory() as session:
        run = ReviewRun(
            tenant_id=api_main.DEFAULT_TENANT_ID,
            case_id=case_id,
            status="RUNNING",
            as_of_date=date(2026, 5, 1),
            decision_cutoff_at=datetime(2026, 5, 1, tzinfo=UTC),
            state_version=7,
        )
        session.add(run)
        await session.flush()
        session.add(
            RunEvent(
                run_id=run.id,
                tenant_id=run.tenant_id,
                case_id=run.case_id,
                sequence_no=4,
                event_type="STAGE_STARTED",
                payload_redacted={},
            )
        )
        await session.flush()

        changed = await api_main._mark_run_failed_with_trace(
            session, run.id, "BACKGROUND_FAILED", "RuntimeError"
        )
        assert changed is True
        await session.commit()

    async with factory() as session:
        persisted = await session.get(ReviewRun, run.id)
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.sequence_no)
            )
        ).all()
    assert persisted.status == "FAILED"
    assert persisted.state_version == 8
    assert events[-1].sequence_no == 5
    assert events[-1].event_type == "BACKGROUND_FAILED"
    assert events[-1].payload_redacted == {"error_type": "RuntimeError"}


async def test_get_run_exposes_supporting_and_opposing_locators_by_evidence_key(protocol_client):
    """HITL 返回稳定 evidence_key 对应的本 Run 文档定位，不能泄露其他 Run。"""
    from creditlens.agents.contracts import (
        AgentClaim,
        AnswerStatus,
        GroundedAnswerArtifact,
        RefusalReasonCode,
    )
    from creditlens.infrastructure.postgres.artifact_integrity import (
        canonical_artifact_payload_hash,
    )

    client, factory, run_id, _ = protocol_client
    supporting_key = uuid.uuid4()
    opposing_key = uuid.uuid4()
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        run.run_type = "SIMPLE_QA"
        run.model_manifest = {
            "workflow": "grounded_qa_v1",
            "prompt_version": "grounded_qa_v1",
            "model_invocation_ids": [],
        }
        source_claim = AgentClaim(
            category="DATA_CONFLICT",
            statement="conflicting evidence requires review",
            verdict="PARTIALLY_SUPPORTED",
            supporting_evidence_ids=[supporting_key],
            opposing_evidence_ids=[opposing_key],
            as_of_date=run.as_of_date,
        )
        grounded = GroundedAnswerArtifact(
            run_id=run.id,
            task_id="grounded_qa",
            producer="grounded_qa",
            lifecycle_status="VERIFIED",
            execution_status="INSUFFICIENT_EVIDENCE",
            claims=[source_claim],
            answer_status=AnswerStatus.ABSTAINED,
            abstention_reason="Conflicting evidence requires review.",
            refusal_reason_code=RefusalReasonCode.MISSING_BANK_STATEMENTS,
            prompt_version="grounded_qa_v1",
        )
        persisted_payload = grounded.model_dump(mode="json", exclude={"output_hash"})
        persisted_payload["generation_mode"] = "llm"
        artifact = ArtifactRecord(
            id=grounded.artifact_id,
            tenant_id=run.tenant_id,
            run_id=run.id,
            task_id="grounded_qa",
            artifact_type="GROUNDED_ANSWER",
            producer="grounded_qa",
            lifecycle_status="VERIFIED",
            execution_status=grounded.execution_status,
            payload=persisted_payload,
            output_hash=canonical_artifact_payload_hash(persisted_payload),
        )
        session.add(artifact)
        await session.flush()
        session.add_all(
            [
                RunEvent(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    case_id=run.case_id,
                    sequence_no=2,
                    event_type="EVIDENCE_READY",
                    payload_redacted={},
                ),
                EvidenceRecord(
                    evidence_key=supporting_key,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type="DOCUMENT_SPAN",
                    source_id=uuid.uuid4(),
                    document_version_id=uuid.uuid4(),
                    section_id=uuid.uuid4(),
                    page_number=3,
                    locator={"parse_run_id": str(uuid.uuid4())},
                    content_hash="a" * 64,
                ),
                EvidenceRecord(
                    evidence_key=opposing_key,
                    tenant_id=run.tenant_id,
                    run_id=run.id,
                    evidence_type="DOCUMENT_SPAN",
                    source_id=uuid.uuid4(),
                    document_version_id=uuid.uuid4(),
                    section_id=uuid.uuid4(),
                    page_number=4,
                    locator={"parse_run_id": str(uuid.uuid4())},
                    content_hash="b" * 64,
                ),
            ]
        )
        session.add(
            ClaimRecord(
                id=source_claim.claim_id,
                tenant_id=run.tenant_id,
                run_id=run.id,
                artifact_id=artifact.id,
                category=source_claim.category,
                statement=source_claim.statement,
                verdict=source_claim.verdict,
                as_of_date=run.as_of_date,
                review_status="PENDING",
                payload={
                    "supporting_evidence_ids": [str(supporting_key)],
                    "opposing_evidence_ids": [str(opposing_key)],
                    "calculation_ids": [],
                    "answer_status": "ABSTAINED",
                },
            )
        )
        await session.commit()

    response = await client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    evidence = response.json()["claims"][0]["evidence"]
    assert evidence["supporting_locators"][0]["page_number"] == 3
    assert evidence["opposing_locators"][0]["page_number"] == 4
    assert evidence["supporting_locators"][0]["parse_run_id"]
    assert response.json()["grounded_answer"]["refusal_reason_code"] == "MISSING_BANK_STATEMENTS"

    # Simulate storage corruption outside the column-restricted app role.
    async with factory() as session:
        claim = await session.get(ClaimRecord, source_claim.claim_id)
        claim.statement = "tampered claim"
        await session.commit()
    corrupted = await client.get(f"/api/v1/runs/{run_id}")
    assert corrupted.status_code == 409
    assert corrupted.json() == {"detail": "ARTIFACT_INTEGRITY_FAILED"}
