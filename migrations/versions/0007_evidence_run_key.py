"""Give evidence rows a run-scoped stable logical key.

Revision ID: 0007_evidence_run_key
Revises: 0006_hitl_wp3
Create Date: 2026-08-06
"""

import sqlalchemy as sa
from alembic import op

from creditlens.infrastructure.postgres.models import GUID

revision = "0007_evidence_run_key"
down_revision = "0006_hitl_wp3"
branch_labels = None
depends_on = None

_CONSTRAINT = "uq_evidence_run_key"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"]: column for column in inspector.get_columns("evidence")}

    if "evidence_key" not in columns:
        op.add_column("evidence", sa.Column("evidence_key", GUID(), nullable=True))

    # Existing row ids were the stable evidence ids. Preserve that identity as
    # the new logical key before making the column mandatory.
    op.execute(sa.text("UPDATE evidence SET evidence_key = id WHERE evidence_key IS NULL"))
    op.alter_column(
        "evidence",
        "evidence_key",
        existing_type=GUID(),
        nullable=False,
    )

    constraint_names = {
        constraint["name"] for constraint in inspector.get_unique_constraints("evidence")
    }
    if _CONSTRAINT not in constraint_names:
        op.create_unique_constraint(
            _CONSTRAINT,
            "evidence",
            ["run_id", "evidence_key"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    constraint_names = {
        constraint["name"] for constraint in inspector.get_unique_constraints("evidence")
    }
    if _CONSTRAINT in constraint_names:
        op.drop_constraint(_CONSTRAINT, "evidence", type_="unique")

    columns = {column["name"] for column in inspector.get_columns("evidence")}
    if "evidence_key" in columns:
        op.drop_column("evidence", "evidence_key")
