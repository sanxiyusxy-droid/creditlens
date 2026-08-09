"""Add tenant-scoped idempotency metadata for SIMPLE_QA requests.

Revision ID: 0008_qa_request_idempotency
Revises: 0007_evidence_run_key
Create Date: 2026-08-07
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_qa_request_idempotency"
down_revision = "0007_evidence_run_key"
branch_labels = None
depends_on = None

_CONSTRAINT = "uq_review_runs_tenant_request_idem"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("review_runs")}
    if "request_idempotency_key" not in columns:
        op.add_column(
            "review_runs",
            sa.Column("request_idempotency_key", sa.String(128), nullable=True),
        )
    if "request_hash" not in columns:
        op.add_column(
            "review_runs",
            sa.Column("request_hash", sa.String(64), nullable=False, server_default=""),
        )

    constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("review_runs")
    }
    if _CONSTRAINT not in constraints:
        op.create_unique_constraint(
            _CONSTRAINT,
            "review_runs",
            ["tenant_id", "request_idempotency_key"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("review_runs")
    }
    if _CONSTRAINT in constraints:
        op.drop_constraint(_CONSTRAINT, "review_runs", type_="unique")

    columns = {column["name"] for column in inspector.get_columns("review_runs")}
    if "request_hash" in columns:
        op.drop_column("review_runs", "request_hash")
    if "request_idempotency_key" in columns:
        op.drop_column("review_runs", "request_idempotency_key")
