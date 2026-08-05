-- CI 初始化：创建测试用业务角色（NOSUPERUSER NOBYPASSRLS）。
-- 以超级用户执行：psql -h pg -U postgres -d creditlens_test -f infra/postgres/ci_role.sql
-- 表创建（Alembic）与 RLS 策略应用之后，还需向该角色授予 DML 权限（见 .gitlab-ci.yml）。
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'creditlens_app') THEN
    CREATE ROLE creditlens_app LOGIN PASSWORD 'creditlens_app';
  END IF;
END $$;

-- 强制属性：非超级用户且不绕过 RLS（超级用户/BYPASSRLS 会使 RLS 测试失去意义）
ALTER ROLE creditlens_app NOSUPERUSER NOBYPASSRLS LOGIN;
