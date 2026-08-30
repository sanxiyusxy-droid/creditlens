"""在演示库启用 RLS 并创建业务角色（v0.6 运行时接线）。

用法：
    $env:APP_DB_PASSWORD="..."; uv run python scripts/apply_rls.py

全新演示库（Seed 前由管理身份创建授权根；普通 Seed 不得自授）：
    $env:APP_DB_PASSWORD="..."; uv run python scripts/apply_rls.py --bootstrap-demo-principals

完成后 API/Worker 应改用业务角色连接（.env.local）：
    DATABASE_URL=postgresql+asyncpg://creditlens_app:<密码>@localhost:5432/creditlens

说明：
- 策略来自 infra/postgres/rls_policies.sql（v0.5 已在独立库 6/6 验证）；
- creditlens_app 为 NOSUPERUSER NOBYPASSRLS：未设置 SET LOCAL 上下文的查询得 0 行；
- 迁移仍由超级账号执行（迁移账号与业务账号分离，文档 §6.5）。
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql as psycopg_sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://creditlens:creditlens@localhost:5432/creditlens"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="应用 RLS 与 creditlens_app 最小权限")
    parser.add_argument(
        "--bootstrap-demo-principals",
        action="store_true",
        help="以当前管理连接幂等创建三个演示案件及 Membership，供随后业务 Seed 使用",
    )
    return parser.parse_args()


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
    args = _parse_args()
    password = _load_password()
    if not password:
        print("请在环境变量或 .env.local 中提供 APP_DB_PASSWORD（不写入仓库）")
        sys.exit(1)

    sql = (PROJECT_ROOT / "infra" / "postgres" / "rls_policies.sql").read_text(encoding="utf-8")
    role_grants_sql = (PROJECT_ROOT / "infra" / "postgres" / "runtime_role_grants.sql").read_text(
        encoding="utf-8"
    )
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
        print("[1/3] RLS 策略已应用到演示库")
        if args.bootstrap_demo_principals:
            bootstrap_sql = (
                PROJECT_ROOT / "infra" / "postgres" / "ci_seed_principals.sql"
            ).read_text(encoding="utf-8")
            cur.execute(bootstrap_sql)
            print("[1/3] 演示 Principal/Case/Membership 已由管理身份幂等创建")
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'creditlens_app'")
        if cur.fetchone() is None:
            cur.execute(
                "CREATE ROLE creditlens_app LOGIN PASSWORD %s "
                "NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOINHERIT",
                (password,),
            )
            print("[2/3] 业务角色 creditlens_app 已创建（NOBYPASSRLS）")
        else:
            cur.execute(
                "ALTER ROLE creditlens_app WITH PASSWORD %s LOGIN "
                "NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOINHERIT",
                (password,),
            )
            print("[2/3] 业务角色已存在，密码与安全属性已收敛")
        cur.execute(
            "SELECT parent.rolname FROM pg_auth_members AS membership "
            "JOIN pg_roles AS child ON child.oid = membership.member "
            "JOIN pg_roles AS parent ON parent.oid = membership.roleid "
            "WHERE child.rolname = 'creditlens_app'"
        )
        if cur.fetchall():
            raise RuntimeError("creditlens_app 存在未授权 role membership；拒绝继续授权")
        cur.execute("SELECT current_database()")
        database_name = cur.fetchone()[0]
        cur.execute(
            psycopg_sql.SQL("REVOKE CREATE ON DATABASE {} FROM creditlens_app").format(
                psycopg_sql.Identifier(database_name)
            )
        )
        cur.execute(role_grants_sql)
        print("[3/3] 授权完成。请把 API 的 DATABASE_URL 切换到 creditlens_app。")
    conn.close()


if __name__ == "__main__":
    main()
