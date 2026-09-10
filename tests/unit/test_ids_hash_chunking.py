"""单元测试：ID 派生、哈希、结构切分。"""

import uuid

import pytest

from creditlens.common.hashing import sha256_bytes, sha256_canonical_utf8_text, sha256_text
from creditlens.common.ids import deterministic_point_id


def test_deterministic_point_id_stable():
    section_id = uuid.UUID("00000000-0000-0000-0000-00000000abcd")
    a = deterministic_point_id(section_id, "hash1", "embed-v1")
    b = deterministic_point_id(section_id, "hash1", "embed-v1")
    assert a == b


def test_deterministic_point_id_changes_with_inputs():
    section_id = uuid.uuid4()
    base = deterministic_point_id(section_id, "hash1", "embed-v1")
    assert base != deterministic_point_id(section_id, "hash2", "embed-v1")
    assert base != deterministic_point_id(section_id, "hash1", "embed-v2")


def test_sha256():
    assert len(sha256_bytes(b"abc")) == 64
    assert sha256_text("abc") == sha256_bytes(b"abc")


def test_canonical_utf8_text_hash_is_platform_independent_and_content_sensitive():
    lf = "第一行\n第二行\n".encode()
    crlf = "第一行\r\n第二行\r\n".encode()
    cr = "第一行\r第二行\r".encode()

    assert sha256_canonical_utf8_text(lf) == sha256_canonical_utf8_text(crlf)
    assert sha256_canonical_utf8_text(lf) == sha256_canonical_utf8_text(cr)
    assert sha256_canonical_utf8_text(lf) != sha256_canonical_utf8_text("第一行\n第三行\n".encode())


def test_canonical_utf8_text_hash_rejects_invalid_utf8():
    with pytest.raises(UnicodeDecodeError):
        sha256_canonical_utf8_text(b"\xff")


class TestPolicyChunking:
    def _parse(self, policy_pdf_bytes):
        from creditlens.infrastructure.parsers.pymupdf_parser import PyMuPdfParser

        return PyMuPdfParser().parse(policy_pdf_bytes)

    def test_articles_extracted(self, policy_pdf_bytes):
        from creditlens.ingestion.chunking.structure import build_sections

        parsed = self._parse(policy_pdf_bytes)
        sections = build_sections(parsed, "合成政策", "INTERNAL_POLICY")
        articles = [s for s in sections if s.section_type == "ARTICLE"]
        chapters = [s for s in sections if s.section_type == "CHAPTER"]
        assert len(articles) == 19, "19 条条款应各为一个 Leaf"
        assert len(chapters) == 6

    def test_heading_path_and_links(self, policy_pdf_bytes):
        from creditlens.ingestion.chunking.structure import build_sections

        parsed = self._parse(policy_pdf_bytes)
        sections = build_sections(parsed, "合成政策", "INTERNAL_POLICY")
        articles = [s for s in sections if s.section_type == "ARTICLE"]
        art6 = next(a for a in articles if a.heading == "第六条")
        assert "第二章 准入条件" in " ".join(art6.heading_path)
        assert "资产负债率" in art6.text
        # 条款间 prev/next 链接
        art5 = next(a for a in articles if a.heading == "第五条")
        assert art5.next_id == art6.id
        assert art6.previous_id == art5.id

    def test_exception_not_split_from_rule(self, policy_pdf_bytes):
        """例外条件（第八条）保持完整，不被截断。"""
        from creditlens.ingestion.chunking.structure import build_sections

        parsed = self._parse(policy_pdf_bytes)
        sections = build_sections(parsed, "合成政策", "INTERNAL_POLICY")
        art8 = next(s for s in sections if s.heading == "第八条")
        assert "百分之一百三十" in art8.text
        assert "不得突破第五条" in art8.text
