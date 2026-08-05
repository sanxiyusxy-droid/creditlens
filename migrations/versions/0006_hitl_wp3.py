"""迁移 0006：WP3 HITL 收口（幂等键/乐观锁/服务端 reviewer）。

- human_decisions.target_version：乐观锁 expected_state_version（可空，兼容旧调用）
- human_decisions.reviewer_id：服务端注入的审批人（不接受客户端声明）
- human_decisions.idempotency_key：幂等键
- uq_human_decisions_run_idem：同一 Run 内幂等键唯一（数据库级防并发重复提交）

幂等：列与约束均先检查后补齐，可安全作用于 v1.0 已有库。

Revision ID: 0006_hitl_wp3
Revises: 0005_p0_hardening
Create Date: 2026-07-28
"""

import sqlalchemy as sa
from alembic import op

from creditlens.infrastructure.postgres.models import GUID

revision = "0006_hitl_wp3"
down_revision = "0005_p0_hardening"
branch_labels = None
depends_on = None

_COLUMNS = [
    ("target_version", sa.Column("target_version", sa.Integer, nullable=True)),
    ("reviewer_id", sa.Column("reviewer_id", GUID, nullable=True)),
    ("idempotency_key", sa.Column("idempotency_key", sa.String(128), nullable=True)),
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("human_decisions")}
    for name, column in _COLUMNS:
        if name not in existing:
            op.add_column("human_decisions", column)

    constraint_names = {c["name"] for c in inspector.get_unique_constraints("human_decisions")}
    if "uq_human_decisions_run_idem" not in constraint_names:
        op.create_unique_constraint(
            "uq_human_decisions_run_idem", "human_decisions", ["run_id", "idempotency_key"]
        )


def downgrade() -> None:
    op.drop_constraint("uq_human_decisions_run_idem", "human_decisions", type_="unique")
    for name, _ in reversed(_COLUMNS):
        op.drop_column("human_decisions", name)
