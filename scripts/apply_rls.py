"""在演示库启用 RLS 并创建业务角色（v0.6 运行时接线）。

用法：
    $env:APP_DB_PASSWORD="..."; uv run python scripts/apply_rls.py

完成后 API/Worker 应改用业务角色连接（.env.local）：
    DATABASE_URL=postgresql+asyncpg://creditlens_app:<密码>@localhost:5432/creditlens

说明：
- 策略来自 infra/postgres/rls_policies.sql（v0.5 已在独立库 6/6 验证）；
- creditlens_app 为 NOSUPERUSER NOBYPASSRLS：未设置 SET LOCAL 上下文的查询得 0 行；
- 迁移仍由超级账号执行（迁移账号与业务账号分离，文档 §6.5）。
"""

import os
import sys
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://creditlens:creditlens@localhost:5432/creditlens"


def _load_password() -> str:
    """优先环境变量；其次 .env.local 的 APP_DB_PASSWORD；
    最后从 .env.local 中 creditlens_app 的 DATABASE_URL 解析（不打印值）。"""
    password = os.environ.get("APP_DB_PASSWORD", "")
    if password:
        return password
    env_local = PROJECT_ROOT / ".env.local"
    if env_local.exists():
        import re

        for line in env_local.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("APP_DB_PASSWORD="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        for line in env_local.read_text(encoding="utf-8").splitlines():
            match = re.match(r"DATABASE_URL=.*//creditlens_app:([^@]+)@", line.strip())
            if match:
                return match.group(1)
    return ""


def main() -> None:
    password = _load_password()
    if not password:
        print("请在环境变量或 .env.local 中提供 APP_DB_PASSWORD（不写入仓库）")
        sys.exit(1)

    sql = (PROJECT_ROOT / "infra" / "postgres" / "rls_policies.sql").read_text(encoding="utf-8")
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
        print("[1/3] RLS 策略已应用到演示库")
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'creditlens_app'")
        if cur.fetchone() is None:
            cur.execute(
                "CREATE ROLE creditlens_app LOGIN PASSWORD %s "
                "NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE",
                (password,),
            )
            print("[2/3] 业务角色 creditlens_app 已创建（NOBYPASSRLS）")
        else:
            cur.execute("ALTER ROLE creditlens_app WITH PASSWORD %s", (password,))
            print("[2/3] 业务角色已存在，密码已更新")
        cur.execute("GRANT USAGE ON SCHEMA public TO creditlens_app")
        cur.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO creditlens_app"
        )
        # 审计/证据链表对业务角色只允许查询与追加；Claim/Run 仍需更新状态。
        cur.execute(
            "REVOKE UPDATE, DELETE ON run_events, human_decisions, report_versions, "
            "evidence, artifacts FROM creditlens_app"
        )
        cur.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public"
            " GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO creditlens_app"
        )
        cur.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO creditlens_app")
        cur.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public"
            " GRANT USAGE, SELECT ON SEQUENCES TO creditlens_app"
        )
        print("[3/3] 授权完成。请把 API 的 DATABASE_URL 切换到 creditlens_app。")
    conn.close()


if __name__ == "__main__":
    main()
