"""迁移 0004：case_snapshots / snapshot_documents / snapshot_indexes（v0.2）。

Revision ID: 0004_case_snapshots
Revises: 0003_facts_and_agent_tables
Create Date: 2026-07-28
"""

from alembic import op

from creditlens.infrastructure.postgres.models import (
    Base,
    CaseSnapshot,
    SnapshotDocument,
    SnapshotIndex,
)

revision = "0004_case_snapshots"
down_revision = "0003_facts_and_agent_tables"
branch_labels = None
depends_on = None

_TABLES = [CaseSnapshot.__table__, SnapshotDocument.__table__, SnapshotIndex.__table__]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_TABLES, checkfirst=True)
    # review_runs 增加冻结引用（创建后不可更新）。
    # 注意：0001-0003 基线走 create_all，新库上 review_runs 已含该列，需存在性检查。
    import sqlalchemy as sa

    from creditlens.infrastructure.postgres.models import GUID

    inspector = sa.inspect(op.get_bind())
    existing_columns = {col["name"] for col in inspector.get_columns("review_runs")}
    if "input_snapshot_id" not in existing_columns:
        op.add_column(
            "review_runs",
            sa.Column("input_snapshot_id", GUID, nullable=True),
        )


def downgrade() -> None:
    op.drop_column("review_runs", "input_snapshot_id")
    Base.metadata.drop_all(bind=op.get_bind(), tables=_TABLES)
