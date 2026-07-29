"""结构感知切分（任务 7，文档 §7.5）。

政策/法规：
- "第X章" 为 CHAPTER 节点，"第X条" 为 ARTICLE Leaf，一条完整条款优先作为一个 Leaf；
- 每个 Leaf 携带完整 heading_path；
- 相邻条款通过 previous/next 连接。

年报等一般文档：按自然段聚合为 PARAGRAPH，目标 token 数可配置。

结构完整性高于固定长度：政策条款即使略长也不截断限定条件。
"""

import re
import uuid
from dataclasses import dataclass, field

from creditlens.common.hashing import sha256_text
from creditlens.common.ids import new_id
from creditlens.infrastructure.parsers import ParsedDocument

_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零\d]+章\s*\S*")
_ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零\d]+条")
_HEADING_NUM_RE = re.compile(r"^[一二三四五六七八九十]+、|^\d+(\.\d+)*\s+\S+")

# 初始建议（文档 §7.5），块大小是评测参数
PARAGRAPH_TARGET_TOKENS = 450
PARAGRAPH_MAX_TOKENS = 800
POLICY_ARTICLE_MAX_TOKENS = 1000


def rough_token_count(text: str) -> int:
    """粗略 token 估算：CJK 每字符 1 token，其他按 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4) if text else 0


@dataclass
class SectionDraft:
    """入库前的 Section 草稿，字段对应 document_sections。"""

    id: uuid.UUID
    section_type: str  # DOCUMENT|CHAPTER|ARTICLE|PARAGRAPH
    ordinal: int
    heading: str | None
    heading_path: list[str]
    page_start: int
    page_end: int
    text: str
    text_hash: str
    token_count: int
    parent_id: uuid.UUID | None = None
    previous_id: uuid.UUID | None = None
    next_id: uuid.UUID | None = None
    bbox: dict | None = field(default=None)


@dataclass
class _Line:
    text: str
    page: int


def _collect_lines(parsed: ParsedDocument) -> list[_Line]:
    lines: list[_Line] = []
    for block in sorted(parsed.blocks, key=lambda b: (b.page_number, b.reading_order)):
        for raw_line in block.text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append(_Line(text=stripped, page=block.page_number))
    return lines


def build_sections(parsed: ParsedDocument, document_title: str, document_type: str) -> list[SectionDraft]:
    if document_type in {"REGULATION", "INTERNAL_POLICY"}:
        return _build_policy_sections(parsed, document_title)
    return _build_paragraph_sections(parsed, document_title)


def _build_policy_sections(parsed: ParsedDocument, title: str) -> list[SectionDraft]:
    """按 章 -> 条 恢复层级；条内文本（含款、项）保持在同一 Leaf。"""
    lines = _collect_lines(parsed)
    sections: list[SectionDraft] = []
    ordinal = 0

    root = SectionDraft(
        id=new_id(),
        section_type="DOCUMENT",
        ordinal=ordinal,
        heading=title,
        heading_path=[title],
        page_start=1,
        page_end=parsed.metadata.get("page_count", 1),
        text=title,
        text_hash=sha256_text(title),
        token_count=rough_token_count(title),
    )
    sections.append(root)
    ordinal += 1

    current_chapter: SectionDraft | None = None
    article_lines: list[_Line] = []
    article_heading: str | None = None
    prev_leaf: SectionDraft | None = None

    def flush_article() -> None:
        nonlocal ordinal, prev_leaf, article_lines, article_heading
        if not article_lines:
            return
        text = "\n".join(line.text for line in article_lines)
        parent = current_chapter or root
        heading_path = [*parent.heading_path, article_heading] if article_heading else parent.heading_path
        leaf = SectionDraft(
            id=new_id(),
            section_type="ARTICLE",
            ordinal=ordinal,
            heading=article_heading,
            heading_path=heading_path,
            page_start=article_lines[0].page,
            page_end=article_lines[-1].page,
            text=text,
            text_hash=sha256_text(text),
            token_count=rough_token_count(text),
            parent_id=parent.id,
        )
        if prev_leaf is not None:
            prev_leaf.next_id = leaf.id
            leaf.previous_id = prev_leaf.id
        sections.append(leaf)
        prev_leaf = leaf
        ordinal += 1
        article_lines = []
        article_heading = None

    for line in lines:
        if _CHAPTER_RE.match(line.text):
            flush_article()
            current_chapter = SectionDraft(
                id=new_id(),
                section_type="CHAPTER",
                ordinal=ordinal,
                heading=line.text,
                heading_path=[title, line.text],
                page_start=line.page,
                page_end=line.page,
                text=line.text,
                text_hash=sha256_text(line.text),
                token_count=rough_token_count(line.text),
                parent_id=root.id,
            )
            sections.append(current_chapter)
            ordinal += 1
        elif _ARTICLE_RE.match(line.text):
            flush_article()
            article_heading = _ARTICLE_RE.match(line.text).group(0)  # type: ignore[union-attr]
            article_lines = [line]
        elif article_lines:
            article_lines.append(line)
            if current_chapter is not None:
                current_chapter.page_end = max(current_chapter.page_end, line.page)
        # 章前导言、标题行等不产生 Leaf
    flush_article()
    return sections


def _build_paragraph_sections(parsed: ParsedDocument, title: str) -> list[SectionDraft]:
    """一般文档：识别数字/中文序号标题，其余按目标 token 聚合段落。"""
    lines = _collect_lines(parsed)
    sections: list[SectionDraft] = []
    ordinal = 0

    root = SectionDraft(
        id=new_id(),
        section_type="DOCUMENT",
        ordinal=ordinal,
        heading=title,
        heading_path=[title],
        page_start=1,
        page_end=parsed.metadata.get("page_count", 1),
        text=title,
        text_hash=sha256_text(title),
        token_count=rough_token_count(title),
    )
    sections.append(root)
    ordinal += 1

    current_heading: str | None = None
    buffer: list[_Line] = []
    buffer_tokens = 0
    prev_leaf: SectionDraft | None = None

    def flush() -> None:
        nonlocal ordinal, prev_leaf, buffer, buffer_tokens
        if not buffer:
            return
        text = "\n".join(line.text for line in buffer)
        heading_path = [title, current_heading] if current_heading else [title]
        leaf = SectionDraft(
            id=new_id(),
            section_type="PARAGRAPH",
            ordinal=ordinal,
            heading=current_heading,
            heading_path=heading_path,
            page_start=buffer[0].page,
            page_end=buffer[-1].page,
            text=text,
            text_hash=sha256_text(text),
            token_count=rough_token_count(text),
            parent_id=root.id,
        )
        if prev_leaf is not None:
            prev_leaf.next_id = leaf.id
            leaf.previous_id = prev_leaf.id
        sections.append(leaf)
        prev_leaf = leaf
        ordinal += 1
        buffer = []
        buffer_tokens = 0

    for line in lines:
        if _HEADING_NUM_RE.match(line.text) and rough_token_count(line.text) < 40:
            flush()
            current_heading = line.text
            continue
        line_tokens = rough_token_count(line.text)
        if buffer_tokens + line_tokens > PARAGRAPH_MAX_TOKENS:
            flush()
        buffer.append(line)
        buffer_tokens += line_tokens
        if buffer_tokens >= PARAGRAPH_TARGET_TOKENS:
            flush()
    flush()
    return sections
