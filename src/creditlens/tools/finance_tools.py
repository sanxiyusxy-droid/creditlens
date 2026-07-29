"""类型化财务工具（文档 §9.1）：不使用任意 Text-to-SQL。

v0.9（P0-1）：财务事实读取强制执行——
- 案件范围：fact.case_id == 本案件 或 公共事实（case_id IS NULL）；
- 时点：source_available_at <= decision_cutoff_at（防历史时点泄漏）；
- 质量：排除 verification_status=REJECTED 与已被重述/替代的事实；
- Snapshot：提供 allowed_fact_ids 时只读冻结集合（历史 Run 结果不随底层 Fact 变化）。
"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.formulas.engine import (
    CalculationArtifact,
    CalculationInput,
    FormulaRegistry,
    compute_metric,
)
from creditlens.infrastructure.postgres.models import FinancialFact


async def get_financial_facts(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    entity_id: uuid.UUID,
    metric_codes: list[str],
    case_id: uuid.UUID | None,
    decision_cutoff_at: datetime,
    period_end: date | None = None,
    allowed_fact_ids: list[uuid.UUID] | None = None,
) -> list[FinancialFact]:
    superseded = select(FinancialFact.supersedes_fact_id).where(
        FinancialFact.supersedes_fact_id.is_not(None)
    )
    stmt = select(FinancialFact).where(
        FinancialFact.tenant_id == tenant_id,
        FinancialFact.entity_id == entity_id,
        FinancialFact.metric_code.in_(metric_codes),
        # 案件范围：本案件绑定的事实 + 公共事实（如公开披露 XBRL）
        or_(FinancialFact.case_id == case_id, FinancialFact.case_id.is_(None)),
        # 时点边界：审查截止后才可获得的事实不可用（文档 §6.2）
        FinancialFact.source_available_at <= decision_cutoff_at.astimezone(UTC),
        # 质量：拒绝与被重述/替代的事实不参与计算
        FinancialFact.verification_status != "REJECTED",
        FinancialFact.id.not_in(superseded),
    )
    if period_end is not None:
        stmt = stmt.where(FinancialFact.period_end == period_end)
    if allowed_fact_ids is not None:
        # Snapshot 冻结：只读冻结集合（P0-1）
        stmt = stmt.where(FinancialFact.id.in_(allowed_fact_ids))
    stmt = stmt.order_by(FinancialFact.period_end.desc())
    return list((await session.scalars(stmt)).all())


async def compute_metric_for_entity(
    session: AsyncSession,
    registry: FormulaRegistry,
    tenant_id: uuid.UUID,
    entity_id: uuid.UUID,
    metric_code: str,
    formula_version: str,
    period_end: date,
    case_id: uuid.UUID | None,
    decision_cutoff_at: datetime,
    allowed_fact_ids: list[uuid.UUID] | None = None,
) -> CalculationArtifact:
    """从 financial_facts 取输入 -> 公式引擎计算 -> CalculationArtifact。
    缺输入返回 MISSING_INPUT，不估算。"""
    definition = registry.get(metric_code, formula_version)
    if definition is None:
        raise ValueError(f"公式未注册: {metric_code}@{formula_version}")

    facts = await get_financial_facts(
        session,
        tenant_id,
        entity_id,
        definition.required_inputs,
        case_id=case_id,
        decision_cutoff_at=decision_cutoff_at,
        period_end=period_end,
        allowed_fact_ids=allowed_fact_ids,
    )
    inputs: dict[str, CalculationInput] = {}
    for fact in facts:
        if fact.metric_code in inputs:
            continue  # 多来源冲突留给质量规则；此处取最先（可扩展来源等级）
        inputs[fact.metric_code] = CalculationInput(
            fact_id=fact.id,
            metric_code=fact.metric_code,
            raw_value=Decimal(str(fact.value)),
            canonical_value=Decimal(str(fact.canonical_value)),
            unit=fact.currency or "",
            period_start=fact.period_start,
            period_end=fact.period_end,
            consolidation_scope=fact.consolidation_scope,
        )
    return compute_metric(definition, inputs)
