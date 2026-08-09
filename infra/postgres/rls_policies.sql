-- CreditLens PostgreSQL Row-Level Security 基线（文档 §6.5）
-- 状态：已在真实 PostgreSQL + NOSUPERUSER NOBYPASSRLS 业务角色下通过集成验证。
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

-- Tenant 与 User 是身份授权根，不能沿用普通业务表的全表权限。
-- 业务连接只需读取当前 Tenant 元数据和当前登录 User 自身；身份创建、停用、
-- 租户切换等管理操作必须由独立管理身份执行。没有 DML policy，因此即使未来
-- 误授表级写权限也会由 FORCE RLS fail-closed。
ALTER TABLE public.tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tenants FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS current_tenant_metadata ON public.tenants;
CREATE POLICY current_tenant_metadata ON public.tenants
  FOR SELECT
  USING (id = public.app_current_tenant());

ALTER TABLE public.app_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.app_users FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS current_user_identity ON public.app_users;
CREATE POLICY current_user_identity ON public.app_users
  FOR SELECT
  USING (
    tenant_id = public.app_current_tenant()
    AND id = public.app_current_user()
  );

-- 全局指标定义、物理索引版本和 Alembic 版本不具备 tenant_id，不能用 RLS
-- 做行归属判断，只能作为只读运行时目录。若业务角色已经存在，则本基线本身也
-- 立即收回身份根与全局目录 DML；fresh install 中角色稍后创建时由授权脚本重复执行。
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'creditlens_app') THEN
    EXECUTE 'REVOKE INSERT, UPDATE, DELETE ON public.tenants, public.app_users, public.financial_metric_definitions, public.search_index_versions, public.alembic_version FROM creditlens_app';
  END IF;
END $$;

-- 用户对案件是否有未撤销 Membership。
-- SECURITY DEFINER 只用于读取授权根表；固定 search_path 且限定表/函数名，避免
-- 调用者通过临时 schema 或同名对象劫持函数解析。
CREATE OR REPLACE FUNCTION public.app_has_case_access(target_case uuid) RETURNS boolean AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.case_memberships AS cm
    WHERE cm.case_id = target_case
      AND cm.user_id = public.app_current_user()
      AND cm.revoked_at IS NULL
  )
$$ LANGUAGE sql STABLE SECURITY DEFINER
   SET search_path = pg_catalog, public;

-- Membership 是授权根。业务角色只需读取“自己的有效授权”来构建可信上下文；
-- 没有 INSERT/UPDATE/DELETE policy，即使未来误授表级 DML 也会由 RLS 默认拒绝。
-- 表本身没有 tenant_id，案件租户一致性由所有消费该授权的父表策略继续校验。
ALTER TABLE public.case_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.case_memberships FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS current_user_memberships ON public.case_memberships;
CREATE POLICY current_user_memberships ON public.case_memberships
  FOR SELECT
  USING (
    user_id = public.app_current_user()
    AND revoked_at IS NULL
  );

-- ============ 租户级隔离（无案件维度的表） ============
-- documents / document_versions / document_sections：租户隔离 + 写入校验
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'documents', 'document_versions', 'parse_runs', 'document_sections',
    'summary_nodes', 'entities', 'entity_aliases', 'financial_facts',
    'index_outbox'
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

-- summary_node_sources 没有 tenant_id，必须同时从摘要父节点和原文 Section
-- 继承租户；写入时还要求二者属于同一 DocumentVersion + ParseRun，阻止把
-- 另一租户/解析批次的 Section 挂到当前租户摘要。
ALTER TABLE summary_node_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE summary_node_sources FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS summary_source_parent_access ON summary_node_sources;
CREATE POLICY summary_source_parent_access ON summary_node_sources
  USING (
    EXISTS (
      SELECT 1
      FROM summary_nodes AS sn
      JOIN document_sections AS ds
        ON ds.id = summary_node_sources.section_id
       AND ds.tenant_id = sn.tenant_id
       AND ds.document_version_id = sn.document_version_id
       AND ds.parse_run_id = sn.parse_run_id
      WHERE sn.id = summary_node_sources.summary_node_id
        AND sn.tenant_id = app_current_tenant()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1
      FROM summary_nodes AS sn
      JOIN document_sections AS ds
        ON ds.id = summary_node_sources.section_id
       AND ds.tenant_id = sn.tenant_id
       AND ds.document_version_id = sn.document_version_id
       AND ds.parse_run_id = sn.parse_run_id
      WHERE sn.id = summary_node_sources.summary_node_id
        AND sn.tenant_id = app_current_tenant()
    )
  );

-- Snapshot 是案件级冻结世界，不能只按 tenant 隔离。父表必须同时校验案件
-- Membership；三个无 tenant_id/case_id 的子表则通过父 Snapshot 继承授权。
ALTER TABLE case_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_snapshots FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON case_snapshots;
DROP POLICY IF EXISTS case_membership_isolation ON case_snapshots;
CREATE POLICY case_membership_isolation ON case_snapshots
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(case_snapshots.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = case_snapshots.case_id
        AND c.tenant_id = case_snapshots.tenant_id
        AND c.borrower_entity_id = case_snapshots.borrower_entity_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(case_snapshots.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = case_snapshots.case_id
        AND c.tenant_id = case_snapshots.tenant_id
        AND c.borrower_entity_id = case_snapshots.borrower_entity_id
    )
  );

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['snapshot_documents', 'snapshot_indexes', 'snapshot_facts'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS snapshot_parent_access ON %I', t);
  END LOOP;
END $$;

-- 冻结文档只能来自该案件已绑定、同租户的 DocumentVersion；ParseRun 必须
-- 属于同一版本且在写入冻结点仍为该版本的 active ParseRun。
CREATE POLICY snapshot_parent_access ON snapshot_documents
  USING (
    EXISTS (
      SELECT 1 FROM case_snapshots AS s
      WHERE s.id = snapshot_documents.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1
      FROM case_snapshots AS s
      JOIN case_documents AS cd
        ON cd.case_id = s.case_id
       AND cd.document_version_id = snapshot_documents.document_version_id
      JOIN document_versions AS dv
        ON dv.id = snapshot_documents.document_version_id
       AND dv.tenant_id = s.tenant_id
      JOIN documents AS d
        ON d.id = dv.document_id
       AND d.tenant_id = s.tenant_id
      JOIN parse_runs AS pr
        ON pr.id = snapshot_documents.parse_run_id
       AND pr.document_version_id = dv.id
       AND pr.tenant_id = s.tenant_id
      WHERE s.id = snapshot_documents.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
        AND dv.active_parse_run_id = pr.id
    )
  );

-- search_index_versions 当前没有 tenant_id/case_id，数据库无法证明某个版本归属
-- 当前案件。故业务写入 fail-closed：只允许现有冻结流程使用的 NULL 引用；在模型
-- 增加租户归属外键前，不允许挂接任何不可证明归属的 index_version_id。
CREATE POLICY snapshot_parent_access ON snapshot_indexes
  USING (
    EXISTS (
      SELECT 1 FROM case_snapshots AS s
      WHERE s.id = snapshot_indexes.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
    )
  )
  WITH CHECK (
    snapshot_indexes.index_version_id IS NULL
    AND snapshot_indexes.index_family IN ('CHUNKS', 'SUMMARIES')
    AND btrim(snapshot_indexes.physical_collection_name) <> ''
    AND EXISTS (
      SELECT 1 FROM case_snapshots AS s
      WHERE s.id = snapshot_indexes.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
    )
  );

-- Fact 必须与 Snapshot 的租户、借款主体、案件范围和 decision cutoff 一致；
-- 被拒绝或在冻结时已经被重述替代的 Fact 不得挂接。
CREATE POLICY snapshot_parent_access ON snapshot_facts
  USING (
    EXISTS (
      SELECT 1 FROM case_snapshots AS s
      WHERE s.id = snapshot_facts.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1
      FROM case_snapshots AS s
      JOIN financial_facts AS f
        ON f.id = snapshot_facts.fact_id
       AND f.tenant_id = s.tenant_id
       AND f.entity_id = s.borrower_entity_id
       AND (f.case_id = s.case_id OR f.case_id IS NULL)
       AND f.source_available_at <= s.decision_cutoff_at
       AND f.verification_status <> 'REJECTED'
      WHERE s.id = snapshot_facts.snapshot_id
        AND s.tenant_id = app_current_tenant()
        AND app_has_case_access(s.case_id)
        AND NOT EXISTS (
          SELECT 1 FROM financial_facts AS replacement
          WHERE replacement.tenant_id = s.tenant_id
            AND replacement.supersedes_fact_id = f.id
        )
    )
  );

-- ============ 案件级隔离（tenant + Case Membership，文档 §6.5 示例强化版） ============
ALTER TABLE credit_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_cases FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_tenant_isolation ON credit_cases;
CREATE POLICY case_tenant_isolation ON credit_cases
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(credit_cases.id)
    AND EXISTS (
      SELECT 1 FROM entities AS borrower
      WHERE borrower.id = credit_cases.borrower_entity_id
        AND borrower.tenant_id = credit_cases.tenant_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(credit_cases.id)
    AND EXISTS (
      SELECT 1 FROM entities AS borrower
      WHERE borrower.id = credit_cases.borrower_entity_id
        AND borrower.tenant_id = credit_cases.tenant_id
    )
  );

-- Upload Session 是案件级资源，不能只凭 tenant_id 访问或写入。
ALTER TABLE upload_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE upload_sessions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON upload_sessions;
CREATE POLICY case_membership_isolation ON upload_sessions
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(upload_sessions.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = upload_sessions.case_id
        AND c.tenant_id = upload_sessions.tenant_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(upload_sessions.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = upload_sessions.case_id
        AND c.tenant_id = upload_sessions.tenant_id
    )
  );

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
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(review_runs.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = review_runs.case_id
        AND c.tenant_id = review_runs.tenant_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(review_runs.case_id)
    AND EXISTS (
      SELECT 1 FROM credit_cases AS c
      WHERE c.id = review_runs.case_id
        AND c.tenant_id = review_runs.tenant_id
    )
    AND (
      review_runs.input_snapshot_id IS NULL
      OR EXISTS (
        SELECT 1 FROM case_snapshots AS s
        WHERE s.id = review_runs.input_snapshot_id
          AND s.tenant_id = review_runs.tenant_id
          AND s.case_id = review_runs.case_id
      )
    )
  );

CREATE POLICY case_membership_isolation ON claims
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs AS r
                WHERE r.id = claims.run_id
                  AND r.tenant_id = claims.tenant_id
                  AND app_has_case_access(r.case_id))
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = claims.run_id
        AND r.tenant_id = claims.tenant_id
        AND app_has_case_access(r.case_id)
    )
    AND EXISTS (
      SELECT 1 FROM artifacts AS a
      WHERE a.id = claims.artifact_id
        AND a.run_id = claims.run_id
        AND a.tenant_id = claims.tenant_id
    )
  );

CREATE POLICY case_membership_isolation ON evidence
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs AS r
                WHERE r.id = evidence.run_id
                  AND r.tenant_id = evidence.tenant_id
                  AND app_has_case_access(r.case_id))
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = evidence.run_id
        AND r.tenant_id = evidence.tenant_id
        AND app_has_case_access(r.case_id)
    )
  );

CREATE POLICY case_membership_isolation ON artifacts
  USING (
    tenant_id = app_current_tenant()
    AND EXISTS (SELECT 1 FROM review_runs AS r
                WHERE r.id = artifacts.run_id
                  AND r.tenant_id = artifacts.tenant_id
                  AND app_has_case_access(r.case_id))
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = artifacts.run_id
        AND r.tenant_id = artifacts.tenant_id
        AND app_has_case_access(r.case_id)
    )
  );

CREATE POLICY case_membership_isolation ON human_decisions
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(human_decisions.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = human_decisions.run_id
        AND r.tenant_id = human_decisions.tenant_id
        AND r.case_id = human_decisions.case_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(human_decisions.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = human_decisions.run_id
        AND r.tenant_id = human_decisions.tenant_id
        AND r.case_id = human_decisions.case_id
    )
  );

-- run_events（P0-2，v0.9）：事件按案件隔离；tenant/case 为空的历史行不可见
ALTER TABLE run_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE run_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON run_events;
CREATE POLICY case_membership_isolation ON run_events
  USING (
    tenant_id = app_current_tenant()
    AND run_events.case_id IS NOT NULL
    AND app_has_case_access(run_events.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = run_events.run_id
        AND r.tenant_id = run_events.tenant_id
        AND r.case_id = run_events.case_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND run_events.case_id IS NOT NULL
    AND app_has_case_access(run_events.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = run_events.run_id
        AND r.tenant_id = run_events.tenant_id
        AND r.case_id = run_events.case_id
    )
  );

-- report_versions（P0-3，v0.9）
ALTER TABLE report_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE report_versions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON report_versions;
CREATE POLICY case_membership_isolation ON report_versions
  USING (
    tenant_id = app_current_tenant()
    AND app_has_case_access(report_versions.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = report_versions.run_id
        AND r.tenant_id = report_versions.tenant_id
        AND r.case_id = report_versions.case_id
    )
  )
  WITH CHECK (
    tenant_id = app_current_tenant()
    AND app_has_case_access(report_versions.case_id)
    AND EXISTS (
      SELECT 1 FROM review_runs AS r
      WHERE r.id = report_versions.run_id
        AND r.tenant_id = report_versions.tenant_id
        AND r.case_id = report_versions.case_id
    )
  );

ALTER TABLE case_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS case_membership_isolation ON case_documents;
CREATE POLICY case_membership_isolation ON case_documents
  USING (
    app_has_case_access(case_documents.case_id)
    AND EXISTS (
      SELECT 1
      FROM credit_cases AS c
      JOIN document_versions AS dv
        ON dv.id = case_documents.document_version_id
       AND dv.tenant_id = c.tenant_id
      JOIN documents AS d
        ON d.id = dv.document_id
       AND d.tenant_id = c.tenant_id
      WHERE c.id = case_documents.case_id
        AND c.tenant_id = app_current_tenant()
    )
  )
  WITH CHECK (
    app_has_case_access(case_documents.case_id)
    AND EXISTS (
      SELECT 1
      FROM credit_cases AS c
      JOIN document_versions AS dv
        ON dv.id = case_documents.document_version_id
       AND dv.tenant_id = c.tenant_id
      JOIN documents AS d
        ON d.id = dv.document_id
       AND d.tenant_id = c.tenant_id
      WHERE c.id = case_documents.case_id
        AND c.tenant_id = app_current_tenant()
    )
  );

-- ============ 服务账号（ingestion/index worker）：仅租户隔离的写路径 ============
-- 建议单独创建 service 角色并授予绕过案件级策略的专用 permissive policy：
--   CREATE ROLE creditlens_service LOGIN PASSWORD '...';
--   CREATE POLICY service_tenant_write ON <table> FOR ALL TO creditlens_service
--     USING (tenant_id = app_current_tenant())
--     WITH CHECK (tenant_id = app_current_tenant());
-- 全局管理员的跨案件读取必须走单独、显式审计的策略，不通过业务账号临时关闭 RLS。
