"""迁移 0002：summary_nodes 与 summary_node_sources（任务 17）。

Revision ID: 0002_summary_nodes
Revises: 0001_phase1_baseline
Create Date: 2026-07-28
"""

from alembic import op

from creditlens.infrastructure.postgres.models import Base, SummaryNode, SummaryNodeSource

revision = "0002_summary_nodes"
down_revision = "0001_phase1_baseline"
branch_labels = None
depends_on = None

_TABLES = [SummaryNode.__table__, SummaryNodeSource.__table__]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_TABLES, checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), tables=_TABLES)
