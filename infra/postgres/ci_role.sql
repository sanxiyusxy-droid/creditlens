-- CI 初始化：创建测试用最小权限业务角色。
-- 以超级用户执行：psql -h pg -U postgres -d creditlens_test -f infra/postgres/ci_role.sql
-- 表创建（Alembic）与 RLS 策略应用之后，还需通过 runtime_role_grants.sql 授予 DML 权限。
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'creditlens_app') THEN
    CREATE ROLE creditlens_app LOGIN PASSWORD 'creditlens_app'
      NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
  END IF;
END $$;

-- 无论角色是否已存在，都收敛到与本地 bootstrap 相同的安全属性。PostgreSQL 默认
-- INHERIT；若不显式 NOINHERIT，干净 CI 会与 demo preflight 的运行时契约不一致。
ALTER ROLE creditlens_app WITH LOGIN NOSUPERUSER NOBYPASSRLS
  NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
