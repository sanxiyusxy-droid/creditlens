-- Least-privilege runtime grants shared by local bootstrap and every CI runner.
-- Execute as the database owner after migrations and RLS policies are applied.

GRANT USAGE ON SCHEMA public TO creditlens_app;
REVOKE CREATE ON SCHEMA public FROM creditlens_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO creditlens_app;

-- Frozen snapshot inputs are strictly append-only.
REVOKE UPDATE, DELETE ON
    case_snapshots,
    snapshot_documents,
    snapshot_indexes,
    snapshot_facts
FROM creditlens_app;

-- Audit and evidence facts are append-only.
REVOKE UPDATE, DELETE ON
    run_events,
    human_decisions,
    report_versions,
    evidence,
    artifacts,
    invocation_records
FROM creditlens_app;

-- Delivery workers may advance lifecycle fields only.
REVOKE UPDATE, DELETE ON telemetry_outbox FROM creditlens_app;
GRANT UPDATE (
    status,
    attempts,
    available_at,
    locked_at,
    locked_until,
    last_error_code,
    delivered_at,
    dead_at
) ON telemetry_outbox TO creditlens_app;

-- Review identity and temporal boundary fields are immutable.
REVOKE UPDATE, DELETE ON review_runs FROM creditlens_app;
GRANT UPDATE (
    status,
    state_version,
    model_manifest,
    completed_at
) ON review_runs TO creditlens_app;

-- Claim content is immutable; only the review projection may advance.
REVOKE UPDATE, DELETE ON claims FROM creditlens_app;
GRANT UPDATE (review_status) ON claims TO creditlens_app;

-- Identity roots and global definitions are admin-only.
REVOKE INSERT, UPDATE, DELETE ON
    tenants,
    app_users,
    financial_metric_definitions,
    search_index_versions,
    alembic_version
FROM creditlens_app;
REVOKE INSERT, UPDATE, DELETE ON case_memberships FROM creditlens_app;

-- Cases are created by an admin. Runtime code may change explicit workflow fields only.
REVOKE INSERT, UPDATE, DELETE ON credit_cases FROM creditlens_app;
GRANT UPDATE (
    loan_purpose,
    industry_code,
    region_code,
    status,
    current_report_id,
    updated_at,
    version
) ON credit_cases TO creditlens_app;

-- New tables and sequences default to fail-closed DML.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    REVOKE INSERT, UPDATE, DELETE ON TABLES FROM creditlens_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO creditlens_app;
REVOKE UPDATE ON ALL SEQUENCES IN SCHEMA public FROM creditlens_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO creditlens_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    REVOKE UPDATE ON SEQUENCES FROM creditlens_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO creditlens_app;
