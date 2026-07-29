"""Parser Adapter（任务 6，文档 §7.3）。

MVP 主线：PyMuPDF 原生文本层解析。Docling/PaddleOCR 作为后续 Adapter 接入，
接口保持一致，避免核心业务绑定单一解析器。
"""

from typing import Protocol

from pydantic import BaseModel, Field


class LayoutBlock(BaseModel):
    page_number: int  # 1-based
    bbox: list[float]  # [x0, y0, x1, y1]
    text: str
    reading_order: int
    extraction_method: str = "NATIVE"  # NATIVE|OCR


class ParsedPage(BaseModel):
    page_number: int
    width: float
    height: float
    text: str
    char_count: int
    has_text_layer: bool


class ParsedDocument(BaseModel):
    pages: list[ParsedPage]
    blocks: list[LayoutBlock]
    metadata: dict = Field(default_factory=dict)
    quality_signals: dict = Field(default_factory=dict)
    parser_manifest: dict = Field(default_factory=dict)


class DocumentParser(Protocol):
    def supports(self, mime_type: str, document_type: str) -> bool: ...

    def parse(self, data: bytes) -> ParsedDocument: ...

    def render_page_png(self, data: bytes, page_number: int, zoom: float = 2.0) -> bytes: ...
