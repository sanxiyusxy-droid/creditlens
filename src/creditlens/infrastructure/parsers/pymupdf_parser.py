"""PyMuPDF 解析器：页级文本、Block 坐标、文本层覆盖率与页面 PNG 渲染。"""

import fitz  # PyMuPDF

from creditlens.infrastructure.parsers import (
    LayoutBlock,
    ParsedDocument,
    ParsedPage,
)

PARSER_NAME = "pymupdf"
PARSER_VERSION = fitz.__doc__.split()[1] if fitz.__doc__ else "unknown"

# 文档 §7.3：原生文本覆盖率阈值（可配置，此处为默认值）
NATIVE_TEXT_COVERAGE_MIN = 0.65


class PyMuPdfParser:
    def supports(self, mime_type: str, document_type: str) -> bool:
        return mime_type == "application/pdf"

    def parse(self, data: bytes) -> ParsedDocument:
        pages: list[ParsedPage] = []
        blocks: list[LayoutBlock] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                text = page.get_text("text")
                raw_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,type)
                order = 0
                for raw in raw_blocks:
                    block_text = (raw[4] or "").strip()
                    if not block_text or raw[6] != 0:  # type 0 = 文本块
                        continue
                    blocks.append(
                        LayoutBlock(
                            page_number=page_number,
                            bbox=[raw[0], raw[1], raw[2], raw[3]],
                            text=block_text,
                            reading_order=order,
                        )
                    )
                    order += 1
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=page.rect.width,
                        height=page.rect.height,
                        text=text,
                        char_count=len(text.strip()),
                        has_text_layer=len(text.strip()) > 0,
                    )
                )

        text_pages = sum(1 for p in pages if p.has_text_layer)
        coverage = text_pages / len(pages) if pages else 0.0
        return ParsedDocument(
            pages=pages,
            blocks=blocks,
            metadata={"page_count": len(pages)},
            quality_signals={
                "native_text_coverage": coverage,
                "needs_ocr": coverage < NATIVE_TEXT_COVERAGE_MIN,
            },
            parser_manifest={
                "parser_name": PARSER_NAME,
                "parser_version": PARSER_VERSION,
            },
        )

    def render_page_png(self, data: bytes, page_number: int, zoom: float = 2.0) -> bytes:
        """渲染指定页为 PNG（1-based），用于 Evidence Preview（任务 13）。"""
        with fitz.open(stream=data, filetype="pdf") as doc:
            if not 1 <= page_number <= len(doc):
                raise ValueError(f"page {page_number} out of range 1..{len(doc)}")
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
