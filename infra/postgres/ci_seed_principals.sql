-- Integration-only authorization bootstrap.
--
-- The runtime role must never be able to grant its own Case Membership.  These
-- deterministic principals/cases are therefore created by the migration/admin
-- identity before creditlens_app receives DML privileges and runs the ordinary
-- synthetic-data seed.  The seed remains idempotent and performs all document,
-- ingestion and index writes through the RLS-constrained runtime role.

INSERT INTO tenants (id, name, status, data_isolation_mode, created_at)
VALUES
  (
    '00000000-0000-0000-0000-000000000001',
    '示例银行（合成）', 'ACTIVE', 'SHARED_COLLECTION', now()
  ),
  (
    '00000000-0000-0000-0000-0000000000c1',
    '并发测试租户', 'ACTIVE', 'SHARED_COLLECTION', now()
  )
ON CONFLICT (id) DO NOTHING;

INSERT INTO app_users (id, tenant_id, external_subject, display_name, status, created_at)
VALUES
  (
    '00000000-0000-0000-0000-000000000301',
    '00000000-0000-0000-0000-000000000001',
    'demo-analyst', '演示授信审查员', 'ACTIVE', now()
  ),
  (
    '00000000-0000-0000-0000-0000000000c4',
    '00000000-0000-0000-0000-0000000000c1',
    'concurrency-reviewer', '并发复核员', 'ACTIVE', now()
  )
ON CONFLICT (id) DO NOTHING;

INSERT INTO entities (
  id, tenant_id, entity_type, canonical_name, unified_social_credit_code,
  industry_code, attributes, created_at
)
VALUES
  (
    '00000000-0000-0000-0000-000000000101',
    '00000000-0000-0000-0000-000000000001',
    'COMPANY', '示例制造有限公司', 'SYNTHETIC-91310000000000001X',
    'C', '{}'::json, now()
  ),
  (
    '00000000-0000-0000-0000-000000000102',
    '00000000-0000-0000-0000-000000000001',
    'COMPANY', '星辰微电子科技有限公司', 'SYNTHETIC-91440300000000002Y',
    'C', '{}'::json, now()
  ),
  (
    '00000000-0000-0000-0000-000000000103',
    '00000000-0000-0000-0000-000000000001',
    'COMPANY', '恒达精密机械有限公司', 'SYNTHETIC-91320500000000003Z',
    'C', '{}'::json, now()
  ),
  (
    '00000000-0000-0000-0000-0000000000c3',
    '00000000-0000-0000-0000-0000000000c1',
    'COMPANY', '并发测试借款人', NULL,
    NULL, '{}'::json, now()
  )
ON CONFLICT (id) DO NOTHING;

INSERT INTO credit_cases (
  id, tenant_id, case_number, borrower_entity_id, product_code,
  requested_amount, currency, loan_purpose, application_date, as_of_date,
  decision_cutoff_at, industry_code, status, created_at, updated_at, version
)
VALUES
  (
    '00000000-0000-0000-0000-000000000201',
    '00000000-0000-0000-0000-000000000001',
    'golden_case_001', '00000000-0000-0000-0000-000000000101',
    'working_capital', 5000000.00, 'CNY', '采购原材料',
    DATE '2026-06-30', DATE '2026-06-30', TIMESTAMPTZ '2026-06-30 15:59:59+00',
    'C', 'DRAFT', now(), now(), 1
  ),
  (
    '00000000-0000-0000-0000-000000000202',
    '00000000-0000-0000-0000-000000000001',
    'golden_case_002', '00000000-0000-0000-0000-000000000102',
    'tech_working_capital', 15000000.00, 'CNY', '技术研发及日常经营周转',
    DATE '2026-06-30', DATE '2026-06-30', TIMESTAMPTZ '2026-06-30 15:59:59+00',
    'C', 'DRAFT', now(), now(), 1
  ),
  (
    '00000000-0000-0000-0000-000000000203',
    '00000000-0000-0000-0000-000000000001',
    'golden_case_003', '00000000-0000-0000-0000-000000000103',
    'factoring', 8000000.00, 'CNY', '应收账款保理融资',
    DATE '2026-06-30', DATE '2026-06-30', TIMESTAMPTZ '2026-06-30 15:59:59+00',
    'C', 'DRAFT', now(), now(), 1
  ),
  (
    '00000000-0000-0000-0000-0000000000c2',
    '00000000-0000-0000-0000-0000000000c1',
    'CONCURRENCY-001', '00000000-0000-0000-0000-0000000000c3',
    'SME_WORKING_CAPITAL', 5000000.00, 'CNY', NULL,
    DATE '2026-03-31', DATE '2026-03-31', TIMESTAMPTZ '2026-03-31 23:59:59+00',
    NULL, 'DRAFT', now(), now(), 1
  )
ON CONFLICT (id) DO NOTHING;

INSERT INTO case_memberships (
  case_id, user_id, case_role, granted_by, granted_at, revoked_at
)
VALUES
  (
    '00000000-0000-0000-0000-000000000201',
    '00000000-0000-0000-0000-000000000301',
    'REVIEWER', NULL, now(), NULL
  ),
  (
    '00000000-0000-0000-0000-000000000202',
    '00000000-0000-0000-0000-000000000301',
    'REVIEWER', NULL, now(), NULL
  ),
  (
    '00000000-0000-0000-0000-000000000203',
    '00000000-0000-0000-0000-000000000301',
    'REVIEWER', NULL, now(), NULL
  ),
  (
    '00000000-0000-0000-0000-0000000000c2',
    '00000000-0000-0000-0000-0000000000c4',
    'REVIEWER', NULL, now(), NULL
  )
ON CONFLICT (case_id, user_id, case_role) DO NOTHING;
