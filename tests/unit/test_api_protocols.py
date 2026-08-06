"""API 协议边界：SSE 续传/停止语义与错误码映射。"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

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
    client, factory, run_id, _ = protocol_client
    supporting_key = uuid.uuid4()
    opposing_key = uuid.uuid4()
    async with factory() as session:
        run = await session.get(ReviewRun, run_id)
        artifact = ArtifactRecord(
            tenant_id=run.tenant_id,
            run_id=run.id,
            task_id="protocol",
            artifact_type="test",
            producer="test",
            payload={},
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
                tenant_id=run.tenant_id,
                run_id=run.id,
                artifact_id=artifact.id,
                category="DATA_CONFLICT",
                statement="正反证据待复核",
                verdict="PARTIALLY_SUPPORTED",
                as_of_date=run.as_of_date,
                review_status="PENDING",
                payload={
                    "supporting_evidence_ids": [str(supporting_key)],
                    "opposing_evidence_ids": [str(opposing_key)],
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
