-- CreditLens PostgreSQL Row-Level Security 基线（文档 §6.5）
-- 状态：已交付、待真实 PostgreSQL 环境联调验证（本地 SQLite 无法执行）。
-- 应用方式（Compose 环境）：
--   psql postgresql://creditlens:creditlens@localhost:5432/creditlens -f infra/postgres/rls_policies.sql
-- 前置：应用连接必须在事务内设置可信 Session Context（由服务端从已验证 Token 设置）：
--   SET LOCAL app.tenant_id = '<uuid>';
--   SET LOCAL app.user_id  = '<uuid>';
-- 注意：
--   * RLS 是纵深防御，不替代 API 与 Tool Gateway 鉴权；
--   * 连接池复用前必须重置 Session；
--   * 管理迁移账号与业务运行账号分离；FORCE 使表 Owner 也受策略约束；
--   * 服务写入路径（ingestion/index worker）使用独立服务账号并同样设置 app.tenant_id。

-- ============ 辅助函数 ============
CREATE OR REPLACE FUNCTION app_current_tenant() RETURNS uuid AS $$
  SELECT NULLIF(current_setting('app.tenant_id', true), '')::uuid
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION app_current_user() RETURNS uuid AS $$
  SELECT NULLIF(current_setting('app.user_id', true), '')::uuid
$$ LANGUAGE sql STABLE;

-- 用户对案件是否有未撤销 Membership
CREATE OR REPLACE FUNCTION app_has_case_access(target_case uuid) RETURNS boolean AS $$
  SELECT EXISTS (
    SELECT 1 FROM case_memberships cm
    WHERE cm.case_id = target_case
      AND cm.user_id = app_current_user()
      AND cm.revoked_at IS NULL
  )
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- ============ 租户级隔离（无案件维度的表） ============
-- documents / document_versions / document_sections：租户隔离 + 写入校验
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'documents', 'document_versions', 'parse_runs', 'document_sections',
    'summary_nodes', 'entities', 'entity_aliases', 'financial_facts',
    'index_outbox', 'case_snapshots'
  ] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I
         USING (tenant_id = app_current_tenant())
         WITH CHECK (tenant_id = app_current_tenant())', t);
  END LOOP;
END $$;

-- ============ 案件级隔离（tenant + Case Membership，文档 §6.5 示例强化版） ============
ALTER TABLE credit_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_cases FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_tenant_isolation ON credit_cases;
CREATE POLICY case_tenant_isolation ON credit_cases
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(credit_cases.id)
  )
  WITH CHECK (tenant_id = app_current_tenant());

-- Evidence / Claims / Runs / Case Documents：同样的 Membership EXISTS 约束
-- （只按 tenant_id 的策略不算完成案件隔离）
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['review_runs', 'claims', 'evidence', 'artifacts', 'human_decisions'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS case_membership_isolation ON %I', t);
  END LOOP;
END $$;

CREATE POLICY case_membership_isolation ON review_runs
  USING (tenant_id = app_current_tenant() AND app_has_case_access(review_runs.case_id))
  WITH CHECK (tenant_id = app_current_tenant());

CREATE POLICY case_membership_isolation ON claims
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs r
                WHERE r.id = claims.run_id AND app_has_case_access(r.case_id))
  )
  WITH CHECK (tenant_id = app_current_tenant());

CREATE POLICY case_membership_isolation ON evidence
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs r
                WHERE r.id = evidence.run_id AND app_has_case_access(r.case_id))
  )
  WITH CHECK (tenant_id = app_current_tenant());

CREATE POLICY case_membership_isolation ON artifacts
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs r
                WHERE r.id = artifacts.run_id AND app_has_case_access(r.case_id))
  )
  WITH CHECK (tenant_id = app_current_tenant());

CREATE POLICY case_membership_isolation ON human_decisions
  USING (tenant_id = app_current_tenant() AND app_has_case_access(human_decisions.case_id))
  WITH CHECK (tenant_id = app_current_tenant());

-- run_events（P0-2，v0.9）：事件按案件隔离；tenant/case 为空的历史行不可见
ALTER TABLE run_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE run_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON run_events;
CREATE POLICY case_membership_isolation ON run_events
  USING (
    tenant_id = app_current_tenant()
    AND run_events.case_id IS NOT NULL
    AND app_has_case_access(run_events.case_id)
  )
  WITH CHECK (tenant_id = app_current_tenant());

-- report_versions（P0-3，v0.9）
ALTER TABLE report_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE report_versions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON report_versions;
CREATE POLICY case_membership_isolation ON report_versions
  USING (tenant_id = app_current_tenant() AND app_has_case_access(report_versions.case_id))
  WITH CHECK (tenant_id = app_current_tenant());

ALTER TABLE case_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON case_documents;
CREATE POLICY case_membership_isolation ON case_documents
  USING (app_has_case_access(case_documents.case_id))
  WITH CHECK (app_has_case_access(case_documents.case_id));

-- ============ 服务账号（ingestion/index worker）：仅租户隔离的写路径 ============
-- 建议单独创建 service 角色并授予绕过案件级策略的专用 permissive policy：
--   CREATE ROLE creditlens_service LOGIN PASSWORD '...';
--   CREATE POLICY service_tenant_write ON <table> FOR ALL TO creditlens_service
--     USING (tenant_id = app_current_tenant())
--     WITH CHECK (tenant_id = app_current_tenant());
-- 全局管理员的跨案件读取必须走单独、显式审计的策略，不通过业务账号临时关闭 RLS。
