"""时钟：数据库统一存 UTC（文档 §6.2）。"""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)
