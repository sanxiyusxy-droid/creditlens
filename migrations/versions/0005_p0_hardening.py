"""迁移 0005：P0 修复（v0.9 审核意见）。

- snapshot_facts：财务事实冻结（P0-1）
- report_versions：报告版本持久化（P0-3）
- run_events 补 tenant_id/case_id 并回填（P0-2），供案件级授权与 RLS

Revision ID: 0005_p0_hardening
Revises: 0004_case_snapshots
Create Date: 2026-07-28
"""

import sqlalchemy as sa
from alembic import op

from creditlens.infrastructure.postgres.models import (
    GUID,
    Base,
    ReportVersion,
    SnapshotFact,
)

revision = "0005_p0_hardening"
down_revision = "0004_case_snapshots"
branch_labels = None
depends_on = None

_TABLES = [SnapshotFact.__table__, ReportVersion.__table__]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=_TABLES, checkfirst=True)

    inspector = sa.inspect(op.get_bind())
    existing = {col["name"] for col in inspector.get_columns("run_events")}
    if "tenant_id" not in existing:
        op.add_column("run_events", sa.Column("tenant_id", GUID, nullable=True))
    if "case_id" not in existing:
        op.add_column("run_events", sa.Column("case_id", GUID, nullable=True))
    # 回填既有事件的租户/案件维度
    op.execute(
        """
        UPDATE run_events e
        SET tenant_id = r.tenant_id, case_id = r.case_id
        FROM review_runs r
        WHERE e.run_id = r.id AND e.tenant_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("run_events", "case_id")
    op.drop_column("run_events", "tenant_id")
    Base.metadata.drop_all(bind=op.get_bind(), tables=_TABLES)
