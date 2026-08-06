"""完整预审演示脚本（Multi-Agent 固定 DAG 端到端）。

用法：
    uv run python scripts/run_full_review_demo.py

流程：种子环境（政策 PDF + 案件 + 财务事实）-> Supervisor 固定 DAG
（Policy || Financial -> Challenger -> Auditor -> HITL/Reporting）
-> 打印 Claims、审计结果与 Trace。
"""

import asyncio
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sqlalchemy import select

from creditlens.agents.wiring import build_supervisor
from creditlens.application.snapshot_service import freeze_snapshot
from creditlens.application.trusted_context import acl_scope_hash, build_trusted_context
from creditlens.common.config import get_settings
from creditlens.infrastructure.llm.chat import build_chat_provider
from creditlens.infrastructure.llm.embedding import build_embedding_provider
from creditlens.infrastructure.objectstore import build_object_store
from creditlens.infrastructure.postgres.models import (
    Base,
    FinancialFact,
    HumanDecision,
    ReviewRun,
    RunEvent,
)
from creditlens.infrastructure.postgres.session import (
    create_engine,
    create_session_factory,
    session_scope,
)
from creditlens.infrastructure.qdrant.collections import build_qdrant_client
from seed_synthetic_data import BORROWER_ID, CASE_ID, DEMO_USER_ID, TENANT_ID, seed_environment

PERIOD_END = date(2025, 12, 31)
SYNTHETIC_FACTS = {
    "total_assets": "100000000",
    "total_liabilities": "65000000",
    "current_assets": "42000000",
    "current_liabilities": "30000000",
}


async def main() -> None:
    settings = get_settings()
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    store = build_object_store(settings)
    qdrant = build_qdrant_client(settings)
    await seed_environment(factory, store, qdrant, settings)

    async with session_scope(factory, tenant_id=TENANT_ID, user_id=DEMO_USER_ID) as session:
        existing = await session.scalar(
            select(FinancialFact.id).where(FinancialFact.case_id == CASE_ID).limit(1)
        )
        if existing is None:
            for metric, value in SYNTHETIC_FACTS.items():
                session.add(
                    FinancialFact(
                        tenant_id=TENANT_ID,
                        case_id=CASE_ID,
                        entity_id=BORROWER_ID,
                        metric_code=metric,
                        period_end=PERIOD_END,
                        period_type="INSTANT",
                        value=Decimal(value),
                        canonical_value=Decimal(value),
                        currency="CNY",
                        consolidation_scope="CONSOLIDATED",
                        extraction_method="SYNTHETIC",
                        # P0-1：可获得时间必须早于审查截止（年报 2026-04-30 披露口径）
                        source_available_at=datetime(2026, 4, 30, tzinfo=UTC),
                    )
                )
        print("[demo] 合成财务事实就绪")

        # v0.2：服务端派生可信上下文 + 冻结 Snapshot，Run 只读冻结输入世界
        trusted = await build_trusted_context(session, TENANT_ID, CASE_ID)
        snapshot = await freeze_snapshot(
            session,
            trusted,
            chunks_collection=settings.chunks_collection_name,
            summaries_collection=settings.summaries_collection_name,
            acl_hash=acl_scope_hash(trusted),
        )
        print(f"[demo] Snapshot 冻结: {snapshot.snapshot_id}")

        supervisor, gateway = build_supervisor(
            session,
            qdrant,
            build_embedding_provider(settings),
            snapshot,
            chat=build_chat_provider(settings),  # LLM 概括 statement；不可用自动降级
        )
        outcome = await supervisor.execute_full_review(session, trusted, snapshot)

        print(f"\n===== Run {outcome.run_id} 状态: {outcome.status} =====")
        for artifact in outcome.artifacts:
            print(f"\n--- {artifact.producer} ({artifact.execution_status}) ---")
            for claim in artifact.claims:
                marks = (
                    f"支持证据 {len(claim.supporting_evidence_ids)} 条"
                    f"，反证 {len(claim.opposing_evidence_ids)} 条"
                )
                print(f"  [{claim.category}/{claim.verdict}] {claim.statement}（{marks}）")

        if outcome.audit:
            print("\n===== 审计结果 =====")
            print(f"  accepted={len(outcome.audit.accepted_claim_ids)}")
            print(f"  rejected={len(outcome.audit.rejected_claim_ids)}")
            print(f"  human_review={len(outcome.audit.needs_human_review_claim_ids)}")
            print(f"  replay_failures={outcome.audit.replay_failures}")

        if outcome.status == "HUMAN_REVIEW":
            print("\n[demo] 模拟人工复核批准冲突 Claim…")
            run = await session.get(ReviewRun, outcome.run_id)
            if run is None:
                raise RuntimeError("演示 Run 不存在")
            decision = HumanDecision(
                tenant_id=TENANT_ID,
                case_id=CASE_ID,
                run_id=outcome.run_id,
                target_claim_ids=[str(c) for c in outcome.audit.needs_human_review_claim_ids],
                action="APPROVE_CLAIM",
                reason_code="DEMO_REVIEWED",
                reason="演示：正反证据已人工核对",
                # 与 API 一样显式满足 HITL 幂等/乐观锁契约，避免演示脚本走旧兼容分支。
                reviewer_id=DEMO_USER_ID,
                idempotency_key=f"demo-approve-{outcome.run_id}",
                target_version=run.state_version,
            )
            status = await supervisor.resume_after_human_review(session, outcome.run_id, decision)
            print(f"[demo] 人工复核后状态: {status}")

        events = (
            await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == outcome.run_id)
                .order_by(RunEvent.sequence_no)
            )
        ).all()
        print("\n===== Trace（run_events）=====")
        for event in events:
            print(f"  #{event.sequence_no:02d} {event.event_type} {event.payload_redacted}")

        tool_stats = {}
        for call in gateway.calls:
            key = f"{call.agent_role}:{call.tool_name}:{call.status}"
            tool_stats[key] = tool_stats.get(key, 0) + 1
        print("\n===== 工具调用统计 =====")
        for key, count in sorted(tool_stats.items()):
            print(f"  {key} x{count}")

    await engine.dispose()
    print("\n演示完成。")


if __name__ == "__main__":
    asyncio.run(main())
