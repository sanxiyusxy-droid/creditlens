"""检索契约（任务 10/13 使用；后续任务 18 扩展为完整 QuerySpec）。"""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field


class TrustedRequestContext(BaseModel):
    """可信请求上下文（文档 §8.2）：由服务端从 Token/Case/DB 产生，模型不得创建或修改。"""

    request_id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID | None = None
    role_codes: list[str] = Field(default_factory=list)
    purpose: str = "credit_review"
    case_id: uuid.UUID | None = None
    borrower_entity_id: uuid.UUID | None = None
    product_code: str = "working_capital"
    as_of_date: date
    decision_cutoff_at: datetime
    allowed_document_ids: list[uuid.UUID] = Field(default_factory=list)
    acl_tags: list[str] = Field(default_factory=list)


class RetrievedCandidate(BaseModel):
    """单条候选：Qdrant 命中 + PostgreSQL 回表结果。"""

    section_id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    parse_run_id: uuid.UUID
    page_start: int
    page_end: int
    heading_path: list[str]
    text: str
    text_hash: str
    channel: str  # DENSE|SPARSE|SUMMARY|EXACT
    rank: int
    raw_score: float
    rejection_reason: str | None = None  # 回表失败原因；None 表示通过


class EvidenceRef(BaseModel):
    """证据定位（任务 13）：任何候选可回到原始 PDF 页。"""

    section_id: uuid.UUID
    document_version_id: uuid.UUID
    parse_run_id: uuid.UUID
    page_number: int
    heading_path: list[str]
    text_hash: str


class RetrievalResult(BaseModel):
    query: str
    candidates: list[RetrievedCandidate]
    rejected: list[RetrievedCandidate]
    channel_config: dict
