"""Add append-only invocation facts and their durable telemetry outbox.

Revision ID: 0009_invocation_telemetry
Revises: 0008_qa_request_idempotency
Create Date: 2026-08-21
"""

import sqlalchemy as sa
from alembic import op

from creditlens.infrastructure.postgres.models import GUID

revision = "0009_invocation_telemetry"
down_revision = "0008_qa_request_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the v1.5 durable invocation telemetry tables.

    The baseline migration builds current ORM metadata for a brand-new
    database, so the existence guards are required both for fresh installs and
    for upgrades from the v1.4 head.
    """

    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())

    if "invocation_records" not in tables:
        op.create_table(
            "invocation_records",
            sa.Column("invocation_id", GUID(), nullable=False),
            sa.Column("tenant_id", GUID(), nullable=False),
            sa.Column("case_id", GUID(), nullable=False),
            sa.Column("run_id", GUID(), nullable=False),
            sa.Column(
                "contract_version",
                sa.String(length=32),
                nullable=False,
                server_default="invocation_v2",
            ),
            sa.Column("kind", sa.String(length=16), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("provider", sa.String(length=64), nullable=True),
            sa.Column("model", sa.String(length=128), nullable=True),
            sa.Column("version", sa.String(length=128), nullable=True),
            sa.Column("actor_role", sa.String(length=64), nullable=True),
            sa.Column("task_id", sa.String(length=128), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("payload_redacted", sa.JSON(), nullable=False),
            sa.Column("payload_sha256", sa.String(length=64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.CheckConstraint(
                "kind IN ('MODEL', 'TOOL')",
                name="ck_invocation_records_kind",
            ),
            sa.CheckConstraint(
                "status IN ('SUCCESS', 'FAILED', 'DENIED', 'CANCELLED')",
                name="ck_invocation_records_status",
            ),
            sa.CheckConstraint(
                "length(payload_sha256) = 64",
                name="ck_invocation_records_payload_sha256",
            ),
            sa.ForeignKeyConstraint(
                ["tenant_id"],
                ["tenants.id"],
                name="fk_invocation_records_tenant_id_tenants",
            ),
            sa.ForeignKeyConstraint(
                ["case_id"],
                ["credit_cases.id"],
                name="fk_invocation_records_case_id_credit_cases",
            ),
            sa.ForeignKeyConstraint(
                ["run_id"],
                ["review_runs.id"],
                name="fk_invocation_records_run_id_review_runs",
            ),
            sa.PrimaryKeyConstraint("invocation_id", name="pk_invocation_records"),
        )
        op.create_index(
            "ix_invocation_records_run_ended",
            "invocation_records",
            ["run_id", "ended_at", "invocation_id"],
            unique=False,
        )

    if "telemetry_outbox" not in tables:
        op.create_table(
            "telemetry_outbox",
            sa.Column("id", GUID(), nullable=False),
            sa.Column("tenant_id", GUID(), nullable=False),
            sa.Column("case_id", GUID(), nullable=False),
            sa.Column("run_id", GUID(), nullable=False),
            sa.Column("invocation_id", GUID(), nullable=False),
            sa.Column(
                "topic",
                sa.String(length=64),
                nullable=False,
                server_default="INVOCATION_TERMINATED",
            ),
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="PENDING",
            ),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "available_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error_code", sa.String(length=64), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("dead_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "topic = 'INVOCATION_TERMINATED'",
                name="ck_telemetry_outbox_topic",
            ),
            sa.CheckConstraint(
                "status IN ('PENDING', 'PROCESSING', 'DELIVERED', 'DEAD')",
                name="ck_telemetry_outbox_status",
            ),
            sa.CheckConstraint("attempts >= 0", name="ck_telemetry_outbox_attempts"),
            sa.CheckConstraint(
                "last_error_code IS NULL OR "
                "(length(last_error_code) BETWEEN 1 AND 64 "
                "AND last_error_code = upper(last_error_code) "
                "AND last_error_code NOT LIKE '% %')",
                name="ck_telemetry_outbox_error_code",
            ),
            sa.CheckConstraint(
                "(status = 'PENDING' AND locked_at IS NULL AND locked_until IS NULL "
                "AND delivered_at IS NULL AND dead_at IS NULL) OR "
                "(status = 'PROCESSING' AND attempts >= 1 AND locked_at IS NOT NULL "
                "AND locked_until IS NOT NULL AND locked_until > locked_at "
                "AND delivered_at IS NULL AND dead_at IS NULL) OR "
                "(status = 'DELIVERED' AND attempts >= 1 AND locked_at IS NULL "
                "AND locked_until IS NULL AND delivered_at IS NOT NULL AND dead_at IS NULL) OR "
                "(status = 'DEAD' AND attempts >= 1 AND locked_at IS NULL "
                "AND locked_until IS NULL AND delivered_at IS NULL AND dead_at IS NOT NULL)",
                name="ck_telemetry_outbox_lifecycle",
            ),
            sa.ForeignKeyConstraint(
                ["tenant_id"],
                ["tenants.id"],
                name="fk_telemetry_outbox_tenant_id_tenants",
            ),
            sa.ForeignKeyConstraint(
                ["case_id"],
                ["credit_cases.id"],
                name="fk_telemetry_outbox_case_id_credit_cases",
            ),
            sa.ForeignKeyConstraint(
                ["run_id"],
                ["review_runs.id"],
                name="fk_telemetry_outbox_run_id_review_runs",
            ),
            sa.ForeignKeyConstraint(
                ["invocation_id"],
                ["invocation_records.invocation_id"],
                name="fk_telemetry_outbox_invocation_id_invocation_records",
            ),
            sa.PrimaryKeyConstraint("id", name="pk_telemetry_outbox"),
            sa.UniqueConstraint(
                "invocation_id",
                name="uq_telemetry_outbox_invocation_id",
            ),
        )
        op.create_index(
            "ix_telemetry_outbox_status_available",
            "telemetry_outbox",
            ["status", "available_at"],
            unique=False,
        )
        op.create_index(
            "ix_telemetry_outbox_status_locked_until",
            "telemetry_outbox",
            ["status", "locked_until"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "telemetry_outbox" in tables:
        op.drop_table("telemetry_outbox")
    if "invocation_records" in tables:
        op.drop_table("invocation_records")
