"""RLS 策略验证脚本（在独立验证库执行，不触碰演示库）。

用法（需 Compose PostgreSQL 运行中）：
    uv run python scripts/verify_rls.py

流程：
1. 重建独立库 creditlens_rls_test；
2. alembic upgrade head + 应用 infra/postgres/rls_policies.sql；
3. 以表 Owner 身份（FORCE RLS 下同样受策略约束）验证：
   - 未设置 Session Context -> 0 行；
   - 正确 tenant + 有 Membership -> 可见；
   - 正确 tenant + 无 Membership -> credit_cases 不可见；
   - 错误 tenant -> 全部不可见；
   - 跨租户写入被 WITH CHECK 拒绝。
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADMIN_DSN = "postgresql://creditlens:creditlens@localhost:5432/postgres"
TEST_DB = "creditlens_rls_test"
TEST_DSN = f"postgresql://creditlens:creditlens@localhost:5432/{TEST_DB}"
# 关键：验证必须用非特权业务角色——超级用户无条件绕过 RLS（文档 §6.5
# "管理迁移账号和业务运行账号分离"的直接原因）
VERIFY_ROLE = "creditlens_rls_verifier"
APP_DSN = f"postgresql://{VERIFY_ROLE}:app-test-only@localhost:5432/{TEST_DB}"

TENANT_A, TENANT_B = str(uuid.uuid4()), str(uuid.uuid4())
USER_1, USER_2 = str(uuid.uuid4()), str(uuid.uuid4())
CASE_A = str(uuid.uuid4())
ENTITY_A = str(uuid.uuid4())
RUN_A = str(uuid.uuid4())
ARTIFACT_A = str(uuid.uuid4())
CLAIM_A = str(uuid.uuid4())

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def recreate_test_db() -> None:
    conn = psycopg2.connect(ADMIN_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    conn.close()


def migrate_and_apply_rls() -> None:
    env = dict(
        os.environ,
        DATABASE_URL=f"postgresql+asyncpg://creditlens:creditlens@localhost:5432/{TEST_DB}",
    )
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    sql = (PROJECT_ROOT / "infra" / "postgres" / "rls_policies.sql").read_text(encoding="utf-8")
    conn = psycopg2.connect(TEST_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
        # 非特权业务角色（NOBYPASSRLS）：验证场景全部经由它执行
        cur.execute(
            """
            DO $$ BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'creditlens_rls_verifier') THEN
                CREATE ROLE creditlens_rls_verifier LOGIN PASSWORD 'app-test-only'
                  NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
              END IF;
            END $$;
            ALTER ROLE creditlens_rls_verifier WITH PASSWORD 'app-test-only'
              NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
            """
        )
        cur.execute("GRANT USAGE ON SCHEMA public TO creditlens_rls_verifier")
        cur.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
            "TO creditlens_rls_verifier"
        )
        cur.execute(
            "REVOKE UPDATE, DELETE ON run_events, human_decisions, report_versions, "
            "evidence, artifacts FROM creditlens_rls_verifier"
        )
        cur.execute("REVOKE UPDATE, DELETE ON claims FROM creditlens_rls_verifier")
        cur.execute("GRANT UPDATE (review_status) ON claims TO creditlens_rls_verifier")
        cur.execute(
            "REVOKE INSERT, UPDATE, DELETE ON tenants, app_users, "
            "financial_metric_definitions, search_index_versions, alembic_version "
            "FROM creditlens_rls_verifier"
        )
        cur.execute(
            "REVOKE INSERT, UPDATE, DELETE ON case_memberships FROM creditlens_rls_verifier"
        )
        cur.execute("REVOKE INSERT, DELETE ON credit_cases FROM creditlens_rls_verifier")
    conn.close()


def seed_fixture(conn) -> None:
    """管理身份写入授权根；普通业务数据仍由后续业务角色验证 RLS。"""
    with conn.cursor() as cur:
        # tenant A 上下文写入 A 侧数据
        cur.execute("BEGIN")
        cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_A,))
        cur.execute("SET LOCAL app.user_id = %s", (USER_1,))
        cur.execute(
            "INSERT INTO tenants (id, name, status, data_isolation_mode, created_at)"
            " VALUES (%s, 'RLS租户A', 'ACTIVE', 'SHARED_COLLECTION', now())",
            (TENANT_A,),
        )
        cur.execute(
            "INSERT INTO entities (id, tenant_id, entity_type, canonical_name, attributes, created_at)"
            " VALUES (%s, %s, 'COMPANY', 'RLS测试企业', '{}', now())",
            (ENTITY_A, TENANT_A),
        )
        cur.execute(
            "INSERT INTO app_users (id, tenant_id, external_subject, display_name, status, created_at)"
            " VALUES (%s, %s, 'rls-user-1', '有权用户', 'ACTIVE', now()),"
            "        (%s, %s, 'rls-user-2', '无权用户', 'ACTIVE', now())",
            (USER_1, TENANT_A, USER_2, TENANT_A),
        )
        cur.execute(
            "INSERT INTO credit_cases (id, tenant_id, case_number, borrower_entity_id,"
            " product_code, requested_amount, currency, application_date, as_of_date,"
            " decision_cutoff_at, status, created_at, updated_at, version)"
            " VALUES (%s, %s, 'rls-001', %s, 'working_capital', 1000000, 'CNY',"
            " '2026-06-30', '2026-06-30', now(), 'DRAFT', now(), now(), 1)",
            (CASE_A, TENANT_A, ENTITY_A),
        )
        cur.execute(
            "INSERT INTO case_memberships (case_id, user_id, case_role, granted_at)"
            " VALUES (%s, %s, 'ANALYST', now())",
            (CASE_A, USER_1),
        )
        cur.execute(
            "INSERT INTO review_runs "
            "(id, tenant_id, case_id, run_type, status, as_of_date, decision_cutoff_at, "
            "plan_version, state_version, model_manifest, retrieval_config, request_hash, "
            "started_at) "
            "VALUES (%s, %s, %s, 'FULL_REVIEW', 'HUMAN_REVIEW', '2026-06-30', now(), "
            "1, 1, '{}', '{}', '', now())",
            (RUN_A, TENANT_A, CASE_A),
        )
        cur.execute(
            "INSERT INTO artifacts "
            "(id, tenant_id, run_id, task_id, artifact_type, contract_version, producer, "
            "lifecycle_status, execution_status, payload, input_hash, output_hash, created_at) "
            "VALUES (%s, %s, %s, 'rls-check', 'rls-check', '1.0', 'rls-check', "
            "'VALIDATED', 'SUCCESS', '{}', '', '', now())",
            (ARTIFACT_A, TENANT_A, RUN_A),
        )
        cur.execute(
            "INSERT INTO claims "
            "(id, tenant_id, run_id, artifact_id, category, statement, verdict, severity, "
            "confidence_level, as_of_date, review_status, payload, created_at) "
            "VALUES (%s, %s, %s, %s, 'ELIGIBILITY', 'immutable statement', "
            "'SUPPORTED', 'INFO', 'MEDIUM', '2026-06-30', 'PENDING', '{}', now())",
            (CLAIM_A, TENANT_A, RUN_A, ARTIFACT_A),
        )
        cur.execute("COMMIT")
        # tenant B 仅建租户记录
        cur.execute("BEGIN")
        cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_B,))
        cur.execute(
            "INSERT INTO tenants (id, name, status, data_isolation_mode, created_at)"
            " VALUES (%s, 'RLS租户B', 'ACTIVE', 'SHARED_COLLECTION', now())",
            (TENANT_B,),
        )
        cur.execute("COMMIT")


def count_cases(conn, tenant: str | None, user: str | None) -> int:
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        if tenant:
            cur.execute("SET LOCAL app.tenant_id = %s", (tenant,))
        if user:
            cur.execute("SET LOCAL app.user_id = %s", (user,))
        cur.execute("SELECT count(*) FROM credit_cases")
        n = cur.fetchone()[0]
        cur.execute("COMMIT")
        return n


def visible_identity_ids(conn, tenant: str, user: str) -> tuple[list[str], list[str]]:
    """返回当前会话可见的 Tenant/User 身份根主键。"""
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute("SET LOCAL app.tenant_id = %s", (tenant,))
        cur.execute("SET LOCAL app.user_id = %s", (user,))
        cur.execute("SELECT id::text FROM tenants ORDER BY id")
        tenant_ids = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT id::text FROM app_users ORDER BY id")
        user_ids = [row[0] for row in cur.fetchall()]
        cur.execute("COMMIT")
        return tenant_ids, user_ids


def main() -> None:
    print("[1/3] 重建验证库并应用迁移 + RLS 策略…")
    recreate_test_db()
    migrate_and_apply_rls()

    conn = psycopg2.connect(TEST_DSN)
    print("[2/3] 写入多租户夹具（超级用户仅做种子；验证走业务角色）…")
    seed_fixture(conn)
    conn.close()

    # 验证场景全部使用非特权业务角色（NOBYPASSRLS）
    conn = psycopg2.connect(APP_DSN)

    print("[3/3] 验证场景：")
    check("未设置 Session Context 时不可见任何案件", count_cases(conn, None, None) == 0)
    check("正确租户 + 有 Membership 可见案件", count_cases(conn, TENANT_A, USER_1) == 1)
    check("正确租户 + 无 Membership 不可见案件", count_cases(conn, TENANT_A, USER_2) == 0)
    check("错误租户（B）不可见 A 的案件", count_cases(conn, TENANT_B, USER_1) == 0)

    own_tenants, own_users = visible_identity_ids(conn, TENANT_A, USER_1)
    check("业务身份只读取当前 Tenant", own_tenants == [TENANT_A])
    check("业务身份只读取当前 User 自身", own_users == [USER_1])
    cross_tenants, cross_users = visible_identity_ids(conn, TENANT_B, USER_1)
    check("切换 Tenant 后不可读取原 Tenant", cross_tenants == [TENANT_B])
    check("Tenant/User 上下文不匹配时 User fail-closed", cross_users == [])

    with conn.cursor() as cur:
        readonly_tables = (
            "tenants",
            "app_users",
            "financial_metric_definitions",
            "search_index_versions",
            "alembic_version",
        )
        for table in readonly_tables:
            qualified = f"public.{table}"
            cur.execute(
                "SELECT has_table_privilege(current_user, %s, 'SELECT')",
                (qualified,),
            )
            can_select = cur.fetchone()[0]
            can_mutate = False
            for privilege in ("INSERT", "UPDATE", "DELETE"):
                cur.execute(
                    "SELECT has_table_privilege(current_user, %s, %s)",
                    (qualified, privilege),
                )
                can_mutate = can_mutate or cur.fetchone()[0]
            check(f"{table} 仅授予只读权限", can_select and not can_mutate)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT "
            "has_table_privilege(current_user, 'public.claims', 'SELECT'), "
            "has_table_privilege(current_user, 'public.claims', 'INSERT'), "
            "has_table_privilege(current_user, 'public.claims', 'UPDATE'), "
            "has_table_privilege(current_user, 'public.claims', 'DELETE'), "
            "has_column_privilege(current_user, 'public.claims', 'review_status', 'UPDATE'), "
            "has_column_privilege(current_user, 'public.claims', 'statement', 'UPDATE'), "
            "has_column_privilege(current_user, 'public.claims', 'verdict', 'UPDATE'), "
            "has_column_privilege(current_user, 'public.claims', 'payload', 'UPDATE')"
        )
        claim_privileges = cur.fetchone()
    check(
        "Claim only grants column-level UPDATE(review_status)",
        claim_privileges == (True, True, False, False, True, False, False, False),
    )

    with conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_A,))
        cur.execute("SET LOCAL app.user_id = %s", (USER_1,))
        cur.execute("UPDATE claims SET review_status = 'AUDITED' WHERE id = %s", (CLAIM_A,))
        updated = cur.rowcount
        cur.execute("COMMIT")
    check("business role can update Claim.review_status", updated == 1)

    for column, value in (
        ("statement", "tampered"),
        ("verdict", "CONTRADICTED"),
        ("payload", "{}"),
    ):
        denied = False
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_A,))
                cur.execute("SET LOCAL app.user_id = %s", (USER_1,))
                cur.execute(f"UPDATE claims SET {column} = %s WHERE id = %s", (value, CLAIM_A))
                cur.execute("COMMIT")
        except psycopg2.errors.InsufficientPrivilege:
            conn.rollback()
            denied = True
        check(f"business role cannot update Claim.{column}", denied)

    # 业务角色不得撤销 Membership；授权变更只能由管理身份执行。
    with conn.cursor() as cur:
        denied = False
        try:
            cur.execute(
                "UPDATE case_memberships SET revoked_at = now() "
                "WHERE case_id = %s AND user_id = %s",
                (CASE_A, USER_1),
            )
            conn.commit()
        except psycopg2.errors.InsufficientPrivilege:
            conn.rollback()
            denied = True
    check("业务角色不能撤销 Membership", denied)

    admin_conn = psycopg2.connect(TEST_DSN)
    with admin_conn.cursor() as cur:
        cur.execute(
            "UPDATE case_memberships SET revoked_at = now() WHERE case_id = %s AND user_id = %s",
            (CASE_A, USER_1),
        )
        admin_conn.commit()
    admin_conn.close()
    check("Membership 撤销后立即不可见", count_cases(conn, TENANT_A, USER_1) == 0)

    # 跨租户写入被 WITH CHECK 拒绝
    denied = False
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute("SET LOCAL app.tenant_id = %s", (TENANT_B,))
            cur.execute("SET LOCAL app.user_id = %s", (USER_2,))
            cur.execute(
                "INSERT INTO documents (id, tenant_id, logical_key, title, document_type,"
                " confidentiality, created_at)"
                " VALUES (%s, %s, 'x', 'x', 'OTHER', 'INTERNAL', now())",
                (str(uuid.uuid4()), TENANT_A),  # B 上下文试图写 A 的数据
            )
            cur.execute("COMMIT")
    except psycopg2.Error:
        denied = True
        conn.rollback()
    check("跨租户写入被 WITH CHECK 拒绝", denied)

    conn.close()
    failed = [r for r in results if not r[1]]
    print(
        f"\n结论: {len(results) - len(failed)}/{len(results)} 通过"
        + ("" if not failed else f"，失败: {[r[0] for r in failed]}")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
