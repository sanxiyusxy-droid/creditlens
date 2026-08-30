"""内容哈希：SHA-256 十六进制，64 字符，与 DB `CHAR(64)` 字段对应。"""

import hashlib
from collections.abc import AsyncIterator


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_canonical_utf8_text(data: bytes) -> str:
    """Hash UTF-8 text after canonicalizing CRLF and CR line endings to LF."""

    text = data.decode("utf-8")
    canonical_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_text(canonical_text)


async def sha256_stream(chunks: AsyncIterator[bytes]) -> tuple[str, int]:
    """流式哈希，返回 (hex_digest, total_size)；用于上传 finalize 校验。"""
    hasher = hashlib.sha256()
    size = 0
    async for chunk in chunks:
        hasher.update(chunk)
        size += len(chunk)
    return hasher.hexdigest(), size
