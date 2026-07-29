"""阶段一基线：tenants/entities/credit_cases + documents/document_versions/parse_runs
+ document_sections + upload_sessions + index_outbox + search_index_versions。

Revision ID: 0001_phase1_baseline
Revises:
Create Date: 2026-07-28

说明：第一阶段基线直接以 ORM 元数据建表（模型即 Schema 基线，见文档 §6.4
"不要求第一天一次性创建全部表"）。后续对数据契约的修改必须写显式 Alembic 操作。
"""

from alembic import op

from creditlens.infrastructure.postgres.models import Base

revision = "0001_phase1_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
