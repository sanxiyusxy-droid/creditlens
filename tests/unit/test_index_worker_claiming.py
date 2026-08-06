"""Index Outbox 多 Worker 领取查询的方言契约。"""

from datetime import UTC, datetime

from sqlalchemy.dialects import postgresql, sqlite

from creditlens.ingestion.index_worker import IndexWorker


def test_postgresql_claim_uses_for_update_skip_locked_but_sqlite_does_not():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    postgres_stmt = IndexWorker._claim_pending_entries_stmt(now, 8, "postgresql")
    sqlite_stmt = IndexWorker._claim_pending_entries_stmt(now, 8, "sqlite")

    postgres_sql = str(postgres_stmt.compile(dialect=postgresql.dialect()))
    sqlite_sql = str(sqlite_stmt.compile(dialect=sqlite.dialect()))

    assert "FOR UPDATE SKIP LOCKED" in postgres_sql
    assert "FOR UPDATE" not in sqlite_sql
