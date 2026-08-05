"""核心表 ORM 模型（文档 §6.4 实现基线，按阶段拆分创建）。

约束遵循：
- 金额/比率 NUMERIC，禁止 FLOAT；
- 统一 UTC TIMESTAMPTZ；
- content_hash CHAR(64)；
- 文件字节变化 => 新 DocumentVersion；解析算法变化 => 新 ParseRun；
- Section 绑定 parse_run_id。

为兼容本地 SQLite 测试：UUID 使用 GUID TypeDecorator；数组/JSONB 使用 JSON。
"""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    types,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from creditlens.common.clock import utc_now
from creditlens.common.ids import new_id


class GUID(types.TypeDecorator):
    """跨方言 UUID：PostgreSQL 原生 UUID，其他方言 CHAR(36)。"""

    impl = types.CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(types.CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None or isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class UTCDateTime(types.TypeDecorator):
    """统一存 UTC；SQLite 读回时补 tzinfo。"""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            from datetime import UTC

            value = value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------- 任务 3


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    data_isolation_mode: Mapped[str] = mapped_column(String(32), default="SHARED_COLLECTION")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    external_subject: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        Index("ix_entities_tenant_uscc", "tenant_id", "unified_social_credit_code"),
        Index("ix_entities_tenant_name", "tenant_id", "canonical_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    entity_type: Mapped[str] = mapped_column(String(32))  # COMPANY|PERSON|BANK|REGULATOR|PRODUCT
    canonical_name: Mapped[str] = mapped_column(String(512))
    unified_social_credit_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    registration_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    industry_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    region_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    entity_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("entities.id"))
    alias: Mapped[str] = mapped_column(String(512))
    alias_type: Mapped[str] = mapped_column(String(32))  # SHORT_NAME|FORMER_NAME|OCR_VARIANT|MANUAL
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("1"))
    verified_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class CreditCase(Base):
    __tablename__ = "credit_cases"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_number: Mapped[str] = mapped_column(String(64))
    borrower_entity_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("entities.id"))
    product_code: Mapped[str] = mapped_column(String(64))
    requested_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    currency: Mapped[str] = mapped_column(String(3), default="CNY")
    loan_purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    application_date: Mapped[date] = mapped_column(Date)
    as_of_date: Mapped[date] = mapped_column(Date)
    decision_cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime)
    industry_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    region_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT")
    current_report_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now)
    version: Mapped[int] = mapped_column(Integer, default=1)  # 乐观锁


class CaseMembership(Base):
    __tablename__ = "case_memberships"

    case_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("credit_cases.id"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("app_users.id"), primary_key=True)
    case_role: Mapped[str] = mapped_column(
        String(32), primary_key=True
    )  # OWNER|ANALYST|REVIEWER|VIEWER
    granted_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    granted_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


# ---------------------------------------------------------------- 任务 4


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "logical_key", name="uq_documents_logical_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    logical_key: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(Text)
    issuer_entity_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    document_type: Mapped[str] = mapped_column(
        String(32)
    )  # REGULATION|INTERNAL_POLICY|ANNUAL_REPORT|...
    jurisdiction: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidentiality: Mapped[str] = mapped_column(String(32), default="INTERNAL")
    canonical_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (Index("ix_docver_tenant_hash", "tenant_id", "content_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    document_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("documents.id"))
    version_label: Mapped[str] = mapped_column(String(64))
    published_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)  # [from, to) 左闭右开
    source_available_at: Mapped[datetime] = mapped_column(UTCDateTime)
    supersedes_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    object_uri: Mapped[str] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(String(128))
    file_size: Mapped[int] = mapped_column(BigInteger)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    active_parse_run_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(32), default="RECEIVED")
    quality_status: Mapped[str] = mapped_column(String(16), default="PASS")  # PASS|WARN|BLOCKED
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    system_ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class ParseRun(Base):
    __tablename__ = "parse_runs"
    __table_args__ = (
        UniqueConstraint("document_version_id", "generation_no", name="uq_parse_runs_generation"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    document_version_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("document_versions.id"))
    generation_no: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    # QUEUED|RUNNING|VALIDATING|INDEX_PENDING|SUCCEEDED|SUCCEEDED_WITH_WARNINGS|FAILED|CANCELLED
    activation_status: Mapped[str] = mapped_column(String(32), default="CANDIDATE")
    # CANDIDATE|ACTIVE|SUPERSEDED|REVOKED|TOMBSTONED
    parser_name: Mapped[str] = mapped_column(String(64))
    parser_version: Mapped[str] = mapped_column(String(64))
    parser_image_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_hash: Mapped[str] = mapped_column(String(64))
    ocr_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ocr_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    quality_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class CaseDocument(Base):
    __tablename__ = "case_documents"

    case_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("credit_cases.id"), primary_key=True
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("document_versions.id"), primary_key=True
    )
    document_role: Mapped[str] = mapped_column(String(32))
    # BORROWER_PROVIDED|BANK_POLICY|REGULATORY|PUBLIC_DISCLOSURE|DERIVED
    is_required: Mapped[bool] = mapped_column(Boolean, default=False)
    bound_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


# ---------------------------------------------------------------- 任务 5


class UploadSession(Base):
    __tablename__ = "upload_sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("credit_cases.id"))
    object_key: Mapped[str] = mapped_column(Text)
    expected_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    expected_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    presigned_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="CREATED")
    # CREATED|UPLOADED|FINALIZED|EXPIRED|CANCELLED
    created_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    finalized_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


# ---------------------------------------------------------------- 任务 7


class DocumentSection(Base):
    __tablename__ = "document_sections"
    __table_args__ = (
        Index("ix_sections_tenant_docver_ordinal", "tenant_id", "document_version_id", "ordinal"),
        Index("ix_sections_parent_ordinal", "parent_section_id", "ordinal"),
        Index("ix_sections_text_hash", "text_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    document_version_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("document_versions.id"))
    parse_run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("parse_runs.id"))
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    section_type: Mapped[str] = mapped_column(String(32))
    # DOCUMENT|CHAPTER|SECTION|ARTICLE|PARAGRAPH|TABLE|TABLE_ROW|NOTE
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading_path: Mapped[list] = mapped_column(JSON, default=list)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    previous_section_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    next_section_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    extraction_method: Mapped[str] = mapped_column(
        String(16), default="NATIVE"
    )  # NATIVE|OCR|MANUAL
    extraction_confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("1"))
    quality_status: Mapped[str] = mapped_column(String(16), default="PASS")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


# ---------------------------------------------------------------- 任务 17


class SummaryNode(Base):
    __tablename__ = "summary_nodes"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    document_version_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("document_versions.id"))
    parse_run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("parse_runs.id"))
    parent_summary_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    summary_level: Mapped[str] = mapped_column(String(16))  # DOCUMENT|CHAPTER|SECTION
    summary_text: Mapped[str] = mapped_column(Text)
    summary_hash: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    model_revision: Mapped[str] = mapped_column(String(128), default="")
    prompt_version: Mapped[str] = mapped_column(String(64), default="")
    grounding_status: Mapped[str] = mapped_column(String(16), default="PENDING")
    # PENDING|VERIFIED|REJECTED
    evidence_eligible: Mapped[bool] = mapped_column(Boolean, default=False)  # 摘要永不作为正式证据
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class SummaryNodeSource(Base):
    __tablename__ = "summary_node_sources"
    __table_args__ = (
        UniqueConstraint("summary_node_id", "ordinal", name="uq_summary_source_ordinal"),
    )

    summary_node_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("summary_nodes.id"), primary_key=True
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("document_sections.id"), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)


# ---------------------------------------------------------------- 任务 19/20（阶段三）


class FinancialMetricDefinition(Base):
    __tablename__ = "financial_metric_definitions"

    metric_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    formula_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(64), default="")
    value_type: Mapped[str] = mapped_column(String(16))  # AMOUNT|RATIO|DAYS|COUNT|PERCENT
    canonical_unit: Mapped[str] = mapped_column(String(32), default="")
    formula_expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    required_inputs: Mapped[list] = mapped_column(JSON, default=list)
    period_rule: Mapped[str] = mapped_column(String(32), default="same_instant")
    rounding_rule: Mapped[dict] = mapped_column(JSON, default=dict)
    zero_policy: Mapped[str] = mapped_column(String(16), default="ERROR")
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class FinancialFact(Base):
    __tablename__ = "financial_facts"
    __table_args__ = (
        Index(
            "ix_facts_entity_metric_period", "tenant_id", "entity_id", "metric_code", "period_end"
        ),
        Index("ix_facts_case_metric", "case_id", "metric_code", "period_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("entities.id"))
    metric_code: Mapped[str] = mapped_column(String(64))
    metric_formula_version: Mapped[str] = mapped_column(String(64), default="raw")
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date] = mapped_column(Date)
    period_type: Mapped[str] = mapped_column(String(16), default="YEAR")
    # INSTANT|MONTH|QUARTER|HALF_YEAR|YEAR|TTM
    value: Mapped[Decimal] = mapped_column(Numeric(30, 8))
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    unit_scale: Mapped[Decimal] = mapped_column(Numeric, default=Decimal("1"))
    canonical_value: Mapped[Decimal] = mapped_column(Numeric(30, 8))
    source_document_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    source_parse_run_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    source_section_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_locator: Mapped[dict] = mapped_column(JSON, default=dict)
    consolidation_scope: Mapped[str] = mapped_column(String(16), default="UNKNOWN")
    # CONSOLIDATED|PARENT_ONLY|UNKNOWN
    is_restated: Mapped[bool] = mapped_column(Boolean, default=False)
    supersedes_fact_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String(16), default="SYNTHETIC")
    # XBRL|TABLE|OCR|MANUAL|SYNTHETIC
    extraction_confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("1"))
    verification_status: Mapped[str] = mapped_column(String(16), default="UNVERIFIED")
    source_available_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    system_ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class ReviewRun(Base):
    __tablename__ = "review_runs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("credit_cases.id"))
    run_type: Mapped[str] = mapped_column(String(32), default="FULL_REVIEW")
    # SIMPLE_QA|FULL_REVIEW|INCREMENTAL_REVIEW|EVALUATION
    status: Mapped[str] = mapped_column(String(32), default="RECEIVED")
    as_of_date: Mapped[date] = mapped_column(Date)
    decision_cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime)
    input_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, nullable=True
    )  # 创建后不可更新
    plan_version: Mapped[int] = mapped_column(Integer, default=1)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    model_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieval_config: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    task_id: Mapped[str] = mapped_column(String(64))
    artifact_type: Mapped[str] = mapped_column(String(64))
    contract_version: Mapped[str] = mapped_column(String(16), default="1.0")
    producer: Mapped[str] = mapped_column(String(64))
    lifecycle_status: Mapped[str] = mapped_column(String(16), default="CREATED")
    # CREATED|VALIDATED|VERIFIED|ACCEPTED|REJECTED|STALE
    execution_status: Mapped[str] = mapped_column(String(32), default="SUCCESS")
    # SUCCESS|PARTIAL|INSUFFICIENT_EVIDENCE|FAILED
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    input_hash: Mapped[str] = mapped_column(String(64), default="")
    output_hash: Mapped[str] = mapped_column(String(64), default="")
    supersedes_artifact_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class ClaimRecord(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    artifact_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("artifacts.id"))
    category: Mapped[str] = mapped_column(String(32))
    statement: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str] = mapped_column(String(32))
    # SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE
    severity: Mapped[str] = mapped_column(String(16), default="INFO")
    confidence_level: Mapped[str] = mapped_column(String(16), default="MEDIUM")
    as_of_date: Mapped[date] = mapped_column(Date)
    uncertainty_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="PENDING")
    # PENDING|AUDITED|NEEDS_REWORK|HUMAN_APPROVED|HUMAN_REJECTED
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # evidence/calculation id 列表
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class EvidenceRecord(Base):
    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    evidence_type: Mapped[str] = mapped_column(String(32))
    # DOCUMENT_SPAN|TABLE_CELL|SQL_FACT|CALCULATION|POLICY_RULE
    source_id: Mapped[uuid.UUID] = mapped_column(GUID)
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    section_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    locator: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64))
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_available_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class HumanDecision(Base):
    __tablename__ = "human_decisions"
    # WP3：同一 Run 内幂等键唯一（数据库级防并发重复提交）
    __table_args__ = (
        UniqueConstraint("run_id", "idempotency_key", name="uq_human_decisions_run_idem"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("credit_cases.id"))
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    target_claim_ids: Mapped[list] = mapped_column(JSON, default=list)
    # WP3：乐观锁 expected_state_version；None 表示不校验（兼容旧调用）
    target_version: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    action: Mapped[str] = mapped_column(String(32))
    # APPROVE_CLAIM|REJECT_CLAIM|REQUEST_CHANGES|REQUEST_MORE_INFORMATION|
    # RERUN_TASK|OVERRIDE_WITH_REASON|SUBMIT_REPORT|APPROVE_REPORT_DRAFT
    reason_code: Mapped[str] = mapped_column(String(64), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence_no", name="uq_run_events_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    # P0-2（v0.9）：事件表补租户/案件维度，SSE/Trace 才能做案件级授权与 RLS
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    case_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    sequence_no: Mapped[int] = mapped_column(BigInteger)
    event_type: Mapped[str] = mapped_column(String(64))
    payload_redacted: Mapped[dict] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class ReportVersion(Base):
    """预审报告版本（P0-3）：COMPLETED 前必须成功持久化（文档 §6.4）。

    WP3：VERIFIED_DRAFT = 自动链路审计通过的草稿；
    APPROVED_DRAFT = 人工批准后的草稿。两者都不表示真实授信审批通过。"""

    __tablename__ = "report_versions"
    __table_args__ = (UniqueConstraint("run_id", "version_no", name="uq_report_versions_run_no"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("credit_cases.id"))
    run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("review_runs.id"))
    version_no: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT")
    # DRAFT|VERIFIED_DRAFT|APPROVED_DRAFT|SUPERSEDED
    template_version: Mapped[str] = mapped_column(String(32), default="mvp-1")
    content_json: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


# ---------------------------------------------------------------- v0.2 Case Snapshot（文档 §6.4）


class CaseSnapshot(Base):
    __tablename__ = "case_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("credit_cases.id"))
    case_version: Mapped[int] = mapped_column(Integer, default=1)
    as_of_date: Mapped[date] = mapped_column(Date)
    decision_cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime)
    borrower_entity_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("entities.id"))
    acl_scope_hash: Mapped[str] = mapped_column(String(64), default="")
    snapshot_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class SnapshotDocument(Base):
    """冻结的 DocumentVersion + ParseRun 引用（不复制原文）。"""

    __tablename__ = "snapshot_documents"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("case_snapshots.id"), primary_key=True
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("document_versions.id"), primary_key=True
    )
    parse_run_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("parse_runs.id"))


class SnapshotIndex(Base):
    """冻结的物理 Collection；已启动 Run 绝不跟随 Alias（文档 §6.6）。"""

    __tablename__ = "snapshot_indexes"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("case_snapshots.id"), primary_key=True
    )
    index_family: Mapped[str] = mapped_column(String(16), primary_key=True)  # CHUNKS|SUMMARIES
    index_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    physical_collection_name: Mapped[str] = mapped_column(String(128))


class SnapshotFact(Base):
    """冻结的财务事实引用（P0-1，文档 §6.4 snapshot_facts）：
    计算工具只允许读取 Snapshot 内的 Fact，底层 Fact 变化不影响历史 Run。"""

    __tablename__ = "snapshot_facts"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("case_snapshots.id"), primary_key=True
    )
    fact_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("financial_facts.id"), primary_key=True
    )


# ---------------------------------------------------------------- 任务 8/9


class SearchIndexVersion(Base):
    __tablename__ = "search_index_versions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    index_family: Mapped[str] = mapped_column(String(16))  # CHUNKS|SUMMARIES
    collection_name: Mapped[str] = mapped_column(String(128))
    alias_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="BUILDING")
    # BUILDING|VALIDATING|ACTIVE|RETIRED|FAILED
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_revision: Mapped[str] = mapped_column(String(128))
    sparse_encoder: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_hash: Mapped[str] = mapped_column(String(64))
    expected_point_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    actual_point_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class IndexOutbox(Base):
    __tablename__ = "index_outbox"
    __table_args__ = (Index("ix_outbox_status_available", "status", "available_at"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=new_id)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"))
    aggregate_type: Mapped[str] = mapped_column(String(32))  # SECTION|SUMMARY
    aggregate_id: Mapped[uuid.UUID] = mapped_column(GUID)
    operation: Mapped[str] = mapped_column(String(16), default="UPSERT")  # UPSERT|TOMBSTONE
    content_hash: Mapped[str] = mapped_column(String(64))
    index_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    target_collection_name: Mapped[str] = mapped_column(String(128))
    embedding_version: Mapped[str] = mapped_column(String(128))
    sparse_encoder_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    # PENDING|PROCESSING|COMPLETED|FAILED|CANCELLED
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)  # 仅任务调度时间
    locked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
