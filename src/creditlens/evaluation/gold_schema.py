"""评测数据集 Schema（任务 11，文档 §16.2）。

`required_evidence_sets` 保存稳定 gold_evidence_key，不保存解析器生成的 Section UUID；
每次 Parse Run 通过 logical_document_key + heading anchor 映射到当前 Section ID，
这样才能公平比较切块器与解析器版本。
"""

from datetime import date, datetime

from pydantic import BaseModel, Field


class GoldEvidenceAnchor(BaseModel):
    gold_evidence_key: str
    logical_document_key: str
    version_label: str
    locator_type: str = "ARTICLE"  # PAGE_TEXT|ARTICLE|TABLE_CELL|FACT
    page_number: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    article_anchor: str | None = None  # 例如 "第十五条"
    canonical_text_hash: str | None = None


class GoldQuestion(BaseModel):
    question_id: str
    case_key: str  # 黄金案件稳定键（评测资产不绑定运行时 UUID）
    question: str
    intent: str
    as_of_date: date
    decision_cutoff_at: datetime

    # 允许多种正确证据组合：外层 OR，内层 AND
    required_evidence_sets: list[list[str]]
    opposing_evidence_keys: list[str] = Field(default_factory=list)

    answerable: bool = True
    expected_refusal_reason: str | None = None

    tags: list[str] = Field(default_factory=list)
    difficulty: str = "normal"
    annotator_notes: str = ""
    split: str = "test"  # dev（调参）/ test（冻结）


class GoldDataset(BaseModel):
    dataset_id: str
    dataset_version: str
    anchors: list[GoldEvidenceAnchor]
    questions: list[GoldQuestion]

    def anchor_by_key(self) -> dict[str, GoldEvidenceAnchor]:
        return {a.gold_evidence_key: a for a in self.anchors}
