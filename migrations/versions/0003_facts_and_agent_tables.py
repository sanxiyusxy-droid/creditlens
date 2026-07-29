"""迁移 0003：financial_facts/metric_definitions + review_runs/artifacts/claims/
evidence/human_decisions/run_events（任务 19/20 与 Multi-Agent 阶段）。

Revision ID: 0003_facts_and_agent_tables
Revises: 0002_summary_nodes
Create Date: 2026-07-28
"""

from alembic import op

from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    Base,
    ClaimRecord,
    EvidenceRecord,
    FinancialFact,
    FinancialMetricDefinition,
    HumanDecision,
    ReviewRun,
    RunEvent,
)

revision = "0003_facts_and_agent_tables"
down_revision = "0002_summary_nodes"
branch_labels = None
depends_on = None

_TABLES = [
    FinancialMetricDefinition.__table__,
    FinancialFact.__table__,
    ReviewRun.__table__,
    ArtifactRecord.__table__,
    ClaimRecord.__table__,
    EvidenceRecord.__table__,
    HumanDecision.__table__,
    RunEvent.__table__,
]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_TABLES, checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_TABLES)
