"""Fail-closed bootstrap and preflight checks for the local synthetic demo.

This module deliberately separates privileged setup (Alembic/RLS, performed by
``scripts/start_demo.ps1``) from runtime preparation.  Runtime preparation uses
the same ``creditlens_app`` role as the API and therefore cannot grant itself
membership or bypass row-level security.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from alembic.config import Config
from alembic.script import ScriptDirectory
from minio import Minio
from qdrant_client import models as qm
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from creditlens.common.hashing import sha256_text
from creditlens.demo_manifest import (
    expected_demo_qdrant_points,
    load_demo_asset_manifest,
)
from creditlens.infrastructure.llm.embedding import resolve_embedding_dim
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    CreditCase,
    Document,
    DocumentSection,
    DocumentVersion,
    FinancialFact,
    IndexOutbox,
    ParseRun,
    SummaryNode,
    SummaryNodeSource,
)
from creditlens.infrastructure.postgres.session import session_scope
from creditlens.infrastructure.qdrant.collections import CollectionManager

DEMO_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000301")

_DEMO_CASES = (
    (
        uuid.UUID("00000000-0000-0000-0000-000000000201"),
        uuid.UUID("00000000-0000-0000-0000-000000000101"),
        {
            "total_assets": "100000000",
            "total_liabilities": "65000000",
            "current_assets": "42000000",
            "current_liabilities": "30000000",
        },
    ),
    (
        uuid.UUID("00000000-0000-0000-0000-000000000202"),
        uuid.UUID("00000000-0000-0000-0000-000000000102"),
        {
            "total_assets": "180000000",
            "total_liabilities": "99000000",
            "current_assets": "76000000",
            "current_liabilities": "48000000",
        },
    ),
    (
        uuid.UUID("00000000-0000-0000-0000-000000000203"),
        uuid.UUID("00000000-0000-0000-0000-000000000103"),
        {
            "total_assets": "130000000",
            "total_liabilities": "78000000",
            "current_assets": "51000000",
            "current_liabilities": "36000000",
        },
    ),
)
_PERIOD_END = date(2025, 12, 31)
_SOURCE_AVAILABLE_AT = datetime(2026, 4, 30, tzinfo=UTC)
_RLS_TABLES = (
    "tenants",
    "app_users",
    "entities",
    "entity_aliases",
    "case_memberships",
    "credit_cases",
    "documents",
    "document_versions",
    "parse_runs",
    "case_documents",
    "upload_sessions",
    "document_sections",
    "summary_nodes",
    "summary_node_sources",
    "financial_facts",
    "review_runs",
    "artifacts",
    "claims",
    "evidence",
    "human_decisions",
    "run_events",
    "invocation_records",
    "telemetry_outbox",
    "report_versions",
    "case_snapshots",
    "snapshot_documents",
    "snapshot_indexes",
    "snapshot_facts",
    "index_outbox",
)
_READ_ONLY_TABLES = {
    "tenants",
    "app_users",
    "case_memberships",
    "financial_metric_definitions",
    "search_index_versions",
    "alembic_version",
}
_FULL_DML_TABLES = {
    "entities",
    "entity_aliases",
    "documents",
    "document_versions",
    "parse_runs",
    "case_documents",
    "upload_sessions",
    "document_sections",
    "summary_nodes",
    "summary_node_sources",
    "financial_facts",
    "index_outbox",
}
_APPEND_ONLY_TABLES = {
    "case_snapshots",
    "snapshot_documents",
    "snapshot_indexes",
    "snapshot_facts",
    "review_runs",
    "artifacts",
    "claims",
    "evidence",
    "human_decisions",
    "run_events",
    "invocation_records",
    "telemetry_outbox",
    "report_versions",
}
_COLUMN_UPDATE_ALLOWLIST = {
    "credit_cases": {
        "loan_purpose",
        "industry_code",
        "region_code",
        "status",
        "current_report_id",
        "updated_at",
        "version",
    },
    "review_runs": {"status", "state_version", "model_manifest", "completed_at"},
    "claims": {"review_status"},
    "telemetry_outbox": {
        "status",
        "attempts",
        "available_at",
        "locked_at",
        "locked_until",
        "last_error_code",
        "delivered_at",
        "dead_at",
    },
}
_RUNTIME_TABLES = tuple(sorted(set(_RLS_TABLES) | _READ_ONLY_TABLES))
_RLS_SELECT_ONLY = {"tenants", "app_users", "case_memberships"}


class BootstrapInvariantError(RuntimeError):
    """A stable, non-sensitive bootstrap failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ComponentResult:
    name: str
    status: str
    code: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: str
    ready: bool
    profile: str
    components: tuple[ComponentResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "profile": self.profile,
            "components": [asdict(component) for component in self.components],
        }


def validate_demo_settings(
    settings: Any, *, allow_configured_models: bool = False
) -> dict[str, Any]:
    """Validate the local-only profile without returning endpoints or secrets."""

    environment = str(settings.app_env).strip().lower()
    if environment not in {"local", "development", "dev", "test"}:
        raise BootstrapInvariantError("DEMO_ENV_FORBIDDEN")
    if str(settings.api_identity_mode).strip().lower() != "demo":
        raise BootstrapInvariantError("DEMO_IDENTITY_REQUIRED")
    if not settings.allow_insecure_demo_identity:
        raise BootstrapInvariantError("DEMO_IDENTITY_OPT_IN_REQUIRED")
    if not str(settings.database_url).startswith("postgresql+asyncpg://"):
        raise BootstrapInvariantError("POSTGRES_RUNTIME_URL_REQUIRED")
    if str(settings.qdrant_url).strip() in {"", ":memory:"}:
        raise BootstrapInvariantError("EXTERNAL_QDRANT_REQUIRED")
    if str(settings.object_store_backend).strip().lower() != "minio":
        raise BootstrapInvariantError("MINIO_BACKEND_REQUIRED")
    if not all(
        str(value).strip()
        for value in (
            settings.minio_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
            settings.minio_raw_bucket,
            settings.minio_derived_bucket,
            settings.minio_rendered_bucket,
        )
    ):
        raise BootstrapInvariantError("MINIO_CONFIGURATION_INCOMPLETE")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    try:
        database_url = make_url(str(settings.database_url))
        # asyncpg accepts query parameters such as ``?host=remote`` that
        # override the visible authority host. A local-only guard must reject
        # every connection-target override before constructing a provider.
        forbidden_target_overrides = {
            "host",
            "hostaddr",
            "port",
            "service",
            "servicefile",
            "dsn",
        }
        if forbidden_target_overrides.intersection(
            str(key).strip().lower() for key in database_url.query
        ):
            raise BootstrapInvariantError("DEMO_DATABASE_TARGET_OVERRIDE_FORBIDDEN")
        database_host = database_url.host
        database_user = database_url.username
        database_name = database_url.database
        qdrant_host = urlparse(str(settings.qdrant_url)).hostname
        minio_host = urlparse(f"//{settings.minio_endpoint}").hostname
    except BootstrapInvariantError:
        raise
    except (ArgumentError, TypeError, ValueError):
        raise BootstrapInvariantError("DEMO_INFRASTRUCTURE_URL_INVALID") from None
    if any(host not in local_hosts for host in (database_host, qdrant_host, minio_host)):
        raise BootstrapInvariantError("DEMO_INFRASTRUCTURE_NOT_LOCAL")
    if database_user != "creditlens_app":
        raise BootstrapInvariantError("DEMO_RUNTIME_DATABASE_ROLE_REQUIRED")
    if database_name != "creditlens":
        raise BootstrapInvariantError("DEMO_RUNTIME_DATABASE_NAME_REQUIRED")

    llm_provider = str(settings.llm_provider).strip().lower()
    embedding_provider = str(settings.embedding_provider).strip().lower()
    rerank_provider = str(settings.rerank_provider).strip().lower()
    if llm_provider not in {"disabled", "openai_compatible"}:
        raise BootstrapInvariantError("LLM_PROVIDER_UNSUPPORTED")
    if embedding_provider not in {"hash_fallback", "openai_compatible"}:
        raise BootstrapInvariantError("EMBEDDING_PROVIDER_UNSUPPORTED")
    if rerank_provider not in {"disabled", "lexical_fallback", "http"}:
        raise BootstrapInvariantError("RERANK_PROVIDER_UNSUPPORTED")
    if not allow_configured_models:
        if llm_provider != "disabled" or embedding_provider != "hash_fallback":
            raise BootstrapInvariantError("EXTERNAL_MODEL_PROFILE_NOT_AUTHORIZED")
        if rerank_provider not in {"disabled", "lexical_fallback"}:
            raise BootstrapInvariantError("EXTERNAL_RERANK_PROFILE_NOT_AUTHORIZED")
    if llm_provider == "disabled" and not settings.qa_allow_extractive_fallback:
        raise BootstrapInvariantError("DEMO_QA_FALLBACK_REQUIRED")
    if llm_provider == "openai_compatible" and not all(
        str(getattr(settings, name, "")).strip()
        for name in ("llm_api_base", "llm_api_key", "llm_model")
    ):
        raise BootstrapInvariantError("LLM_CONFIGURATION_INCOMPLETE")
    if embedding_provider == "hash_fallback" and (
        not str(getattr(settings, "embedding_version", "")).strip()
        or int(getattr(settings, "embedding_dim", 0)) <= 0
    ):
        raise BootstrapInvariantError("HASH_EMBEDDING_CONFIGURATION_INCOMPLETE")
    if embedding_provider == "openai_compatible" and (
        not all(
            str(getattr(settings, name, "")).strip()
            for name in ("embedding_api_base", "embedding_api_key", "embedding_model")
        )
        or int(getattr(settings, "embedding_dim", 0)) <= 0
    ):
        raise BootstrapInvariantError("EMBEDDING_CONFIGURATION_INCOMPLETE")
    if rerank_provider == "http" and (
        not str(getattr(settings, "rerank_api_base", "")).strip()
        or not str(getattr(settings, "rerank_model", "")).strip()
        or not (
            str(getattr(settings, "rerank_api_key", "")).strip()
            or str(getattr(settings, "embedding_api_key", "")).strip()
        )
    ):
        raise BootstrapInvariantError("RERANK_CONFIGURATION_INCOMPLETE")
    for provider, raw_url in (
        (llm_provider, getattr(settings, "llm_api_base", "")),
        (embedding_provider, getattr(settings, "embedding_api_base", "")),
        (rerank_provider, getattr(settings, "rerank_api_base", "")),
    ):
        if provider in {"openai_compatible", "http"}:
            parsed_provider_url = urlparse(str(raw_url))
            if (
                parsed_provider_url.scheme not in {"http", "https"}
                or not parsed_provider_url.hostname
                or parsed_provider_url.username is not None
                or parsed_provider_url.password is not None
                or parsed_provider_url.fragment
            ):
                raise BootstrapInvariantError("MODEL_PROVIDER_URL_INVALID")
            if (
                parsed_provider_url.scheme == "http"
                and parsed_provider_url.hostname not in local_hosts
            ):
                raise BootstrapInvariantError("MODEL_PROVIDER_TLS_REQUIRED")
    return {
        "environment": environment,
        "identity_mode": "demo",
        "database": "postgresql",
        "vector_store": "qdrant",
        "object_store": "minio",
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "rerank_provider": rerank_provider,
        "external_models_authorized": allow_configured_models,
    }


async def ensure_demo_financial_facts(session_factory) -> int:
    """Idempotently add synthetic facts through the RLS-constrained app role."""

    inserted = 0
    async with session_scope(
        session_factory,
        tenant_id=DEMO_TENANT_ID,
        user_id=DEMO_USER_ID,
    ) as session:
        # The table intentionally has no broad business uniqueness constraint.
        # Serialize only this fixed demo seed so concurrent one-click launches
        # cannot both pass the existence check and append duplicate facts.
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            await session.execute(text("SELECT pg_advisory_xact_lock(1672193646)"))
        for case_id, entity_id, values in _DEMO_CASES:
            case = await session.get(CreditCase, case_id)
            if case is None or case.tenant_id != DEMO_TENANT_ID:
                raise BootstrapInvariantError("DEMO_CASE_NOT_VISIBLE")
            if case.borrower_entity_id != entity_id:
                raise BootstrapInvariantError("DEMO_CASE_ENTITY_MISMATCH")
            for metric_code, raw_value in values.items():
                existing = await session.scalar(
                    select(FinancialFact.id)
                    .where(
                        FinancialFact.tenant_id == DEMO_TENANT_ID,
                        FinancialFact.case_id == case_id,
                        FinancialFact.entity_id == entity_id,
                        FinancialFact.metric_code == metric_code,
                        FinancialFact.period_end == _PERIOD_END,
                        FinancialFact.extraction_method == "SYNTHETIC",
                    )
                    .limit(1)
                )
                if existing is not None:
                    continue
                value = Decimal(raw_value)
                session.add(
                    FinancialFact(
                        tenant_id=DEMO_TENANT_ID,
                        case_id=case_id,
                        entity_id=entity_id,
                        metric_code=metric_code,
                        period_end=_PERIOD_END,
                        period_type="INSTANT",
                        value=value,
                        canonical_value=value,
                        currency="CNY",
                        consolidation_scope="CONSOLIDATED",
                        extraction_method="SYNTHETIC",
                        extraction_confidence=Decimal("1"),
                        verification_status="VERIFIED",
                        source_locator={"dataset": "creditlens-synthetic-demo-v1"},
                        source_available_at=_SOURCE_AVAILABLE_AT,
                    )
                )
                inserted += 1
        await session.flush()
    return inserted


def _expected_alembic_heads(project_root) -> set[str]:
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    return set(ScriptDirectory.from_config(config).get_heads())


def _expected_table_privileges() -> set[tuple[str, str]]:
    expected = {(table, "SELECT") for table in _RUNTIME_TABLES}
    for table in _FULL_DML_TABLES:
        expected.update((table, privilege) for privilege in ("INSERT", "UPDATE", "DELETE"))
    for table in _APPEND_ONLY_TABLES:
        expected.add((table, "INSERT"))
    return expected


async def _inspect_runtime_database_gate(session, project_root, settings) -> dict[str, Any]:
    """Read-only identity, migration, RLS and exact-grant gate."""

    if session.bind is None or session.bind.dialect.name != "postgresql":
        raise BootstrapInvariantError("POSTGRES_REQUIRED")
    role_row = (
        await session.execute(
            text(
                "SELECT current_user AS current_role, session_user AS session_role, "
                "current_database() AS database_name, current_schema() AS schema_name, "
                "r.rolsuper, r.rolbypassrls, r.rolcreatedb, r.rolcreaterole, "
                "r.rolreplication, r.rolinherit, r.rolcanlogin, "
                "d.datdba = r.oid AS owns_database "
                "FROM pg_roles AS r JOIN pg_database AS d ON d.datname = current_database() "
                "WHERE r.rolname = current_user"
            )
        )
    ).one_or_none()
    expected_database = make_url(str(settings.database_url)).database
    if (
        role_row is None
        or role_row.current_role != "creditlens_app"
        or role_row.session_role != "creditlens_app"
        or role_row.database_name != expected_database
        or role_row.schema_name != "public"
    ):
        raise BootstrapInvariantError("RUNTIME_DATABASE_IDENTITY_MISMATCH")
    if (
        role_row.rolsuper
        or role_row.rolbypassrls
        or role_row.rolcreatedb
        or role_row.rolcreaterole
        or role_row.rolreplication
        or role_row.rolinherit
        or not role_row.rolcanlogin
        or role_row.owns_database
    ):
        raise BootstrapInvariantError("RUNTIME_ROLE_ATTRIBUTES_FORBIDDEN")
    memberships = (
        (
            await session.execute(
                text(
                    "SELECT parent.rolname FROM pg_auth_members AS membership "
                    "JOIN pg_roles AS child ON child.oid = membership.member "
                    "JOIN pg_roles AS parent ON parent.oid = membership.roleid "
                    "WHERE child.rolname = current_user"
                )
            )
        )
        .scalars()
        .all()
    )
    if memberships:
        raise BootstrapInvariantError("RUNTIME_ROLE_MEMBERSHIP_FORBIDDEN")

    namespace_privileges = (
        await session.execute(
            text(
                "SELECT "
                "has_schema_privilege(current_user, 'public', 'USAGE') AS schema_usage, "
                "has_schema_privilege(current_user, 'public', 'CREATE') AS schema_create, "
                "has_database_privilege(current_user, current_database(), 'CONNECT') "
                "AS database_connect, "
                "has_database_privilege(current_user, current_database(), 'CREATE') "
                "AS database_create"
            )
        )
    ).one()
    if (
        not namespace_privileges.schema_usage
        or namespace_privileges.schema_create
        or not namespace_privileges.database_connect
        or namespace_privileges.database_create
    ):
        raise BootstrapInvariantError("RUNTIME_NAMESPACE_PRIVILEGE_MATRIX_MISMATCH")

    current_heads = set(
        (await session.execute(text("SELECT version_num FROM alembic_version"))).scalars()
    )
    if current_heads != _expected_alembic_heads(project_root):
        raise BootstrapInvariantError("ALEMBIC_HEAD_MISMATCH")

    rls_rows = (
        await session.execute(
            text(
                "SELECT relname, relrowsecurity, relforcerowsecurity "
                "FROM pg_class WHERE relnamespace = 'public'::regnamespace "
                "AND relkind IN ('r', 'p') AND relrowsecurity"
            )
        )
    ).all()
    rls_by_table = {row.relname: (row.relrowsecurity, row.relforcerowsecurity) for row in rls_rows}
    if set(rls_by_table) != set(_RLS_TABLES) or not all(
        enabled and forced for enabled, forced in rls_by_table.values()
    ):
        raise BootstrapInvariantError("RLS_TABLE_ALLOWLIST_MISMATCH")

    policy_rows = (
        await session.execute(
            text(
                "SELECT tablename, policyname, cmd FROM pg_policies "
                "WHERE schemaname = 'public' "
                "AND tablename = ANY(CAST(:tables AS text[]))"
            ),
            {"tables": list(_RLS_TABLES)},
        )
    ).all()
    policy_names = {
        table: (
            "current_tenant_metadata"
            if table == "tenants"
            else "current_user_identity"
            if table == "app_users"
            else "current_user_memberships"
            if table == "case_memberships"
            else "tenant_isolation"
            if table
            in {
                "documents",
                "document_versions",
                "parse_runs",
                "document_sections",
                "summary_nodes",
                "entities",
                "entity_aliases",
                "financial_facts",
                "index_outbox",
            }
            else "summary_source_parent_access"
            if table == "summary_node_sources"
            else "snapshot_parent_access"
            if table in {"snapshot_documents", "snapshot_indexes", "snapshot_facts"}
            else "case_tenant_isolation"
            if table == "credit_cases"
            else "case_membership_isolation"
        )
        for table in _RLS_TABLES
    }
    expected_policies: set[tuple[str, str, str]] = set()
    for table in _RLS_TABLES:
        if table == "case_snapshots":
            expected_policies.update(
                {
                    (table, "case_snapshot_select", "SELECT"),
                    (table, "case_snapshot_insert", "INSERT"),
                }
            )
        elif table in {"snapshot_documents", "snapshot_indexes", "snapshot_facts"}:
            expected_policies.update(
                {
                    (table, "snapshot_parent_select", "SELECT"),
                    (table, "snapshot_parent_insert", "INSERT"),
                }
            )
        elif table == "credit_cases":
            expected_policies.update(
                {
                    (table, "case_tenant_select", "SELECT"),
                    (table, "case_tenant_update", "UPDATE"),
                }
            )
        else:
            expected_policies.add(
                (
                    table,
                    policy_names[table],
                    "SELECT" if table in _RLS_SELECT_ONLY else "ALL",
                )
            )
    actual_policies = {(row.tablename, row.policyname, row.cmd) for row in policy_rows}
    if actual_policies != expected_policies:
        raise BootstrapInvariantError("RLS_POLICY_ALLOWLIST_MISMATCH")

    table_privilege_rows = (
        await session.execute(
            text(
                "SELECT table_name, privilege_type, "
                "has_table_privilege(current_user, format('public.%I', table_name), privilege_type) "
                "AS allowed FROM unnest(CAST(:tables AS text[])) AS table_name "
                "CROSS JOIN unnest(CAST(:privileges AS text[])) AS privilege_type"
            ),
            {
                "tables": list(_RUNTIME_TABLES),
                "privileges": [
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "TRUNCATE",
                    "REFERENCES",
                    "TRIGGER",
                ],
            },
        )
    ).all()
    actual_table_privileges = {
        (row.table_name, row.privilege_type) for row in table_privilege_rows if row.allowed
    }
    if actual_table_privileges != _expected_table_privileges():
        raise BootstrapInvariantError("RUNTIME_TABLE_PRIVILEGE_MATRIX_MISMATCH")

    column_rows = (
        await session.execute(
            text(
                "SELECT columns.table_name, columns.column_name, privilege_type, "
                "has_column_privilege(current_user, "
                "format('public.%I', columns.table_name), columns.column_name, privilege_type) "
                "AS allowed FROM information_schema.columns AS columns "
                "CROSS JOIN unnest(CAST(:privileges AS text[])) AS privilege_type "
                "WHERE columns.table_schema = 'public' "
                "AND columns.table_name = ANY(CAST(:tables AS text[]))"
            ),
            {
                "tables": list(_RUNTIME_TABLES),
                "privileges": ["INSERT", "UPDATE", "REFERENCES"],
            },
        )
    ).all()
    expected_column_privileges: set[tuple[str, str, str]] = set()
    for row in column_rows:
        allowed = (
            (row.table_name in _FULL_DML_TABLES and row.privilege_type in {"INSERT", "UPDATE"})
            or (row.table_name in _APPEND_ONLY_TABLES and row.privilege_type == "INSERT")
            or (
                row.privilege_type == "UPDATE"
                and row.column_name in _COLUMN_UPDATE_ALLOWLIST.get(row.table_name, set())
            )
        )
        if allowed:
            expected_column_privileges.add((row.table_name, row.column_name, row.privilege_type))
    actual_column_privileges = {
        (row.table_name, row.column_name, row.privilege_type) for row in column_rows if row.allowed
    }
    if actual_column_privileges != expected_column_privileges:
        raise BootstrapInvariantError("RUNTIME_COLUMN_PRIVILEGE_MATRIX_MISMATCH")

    sequence_rows = (
        await session.execute(
            text(
                "SELECT relname, "
                "has_sequence_privilege(current_user, oid, 'USAGE') AS can_usage, "
                "has_sequence_privilege(current_user, oid, 'SELECT') AS can_select, "
                "has_sequence_privilege(current_user, oid, 'UPDATE') AS can_update "
                "FROM pg_class WHERE relnamespace = 'public'::regnamespace "
                "AND relkind = 'S'"
            )
        )
    ).all()
    if any(not row.can_usage or not row.can_select or row.can_update for row in sequence_rows):
        raise BootstrapInvariantError("RUNTIME_SEQUENCE_PRIVILEGE_MATRIX_MISMATCH")

    return {
        "runtime_role": role_row.current_role,
        "database_identity_verified": True,
        "alembic_heads": sorted(current_heads),
        "rls_forced_tables": len(rls_by_table),
        "rls_policy_count": len(actual_policies),
        "table_privilege_cells": len(actual_table_privileges),
        "column_privilege_cells": len(actual_column_privileges),
        "role_membership_count": 0,
        "sequence_privilege_count": len(sequence_rows),
    }


async def verify_runtime_database_gate(session_factory, project_root, settings) -> dict[str, Any]:
    """Public pre-write gate used by bootstrap before constructing writers."""

    async with session_factory() as session:
        return await _inspect_runtime_database_gate(session, project_root, settings)


async def inspect_postgres(session_factory, project_root, settings=None) -> dict[str, Any]:
    """Verify migration head, non-bypass runtime role, RLS and synthetic roots."""

    if settings is None:
        raise BootstrapInvariantError("RUNTIME_SETTINGS_REQUIRED")
    runtime_gate = await verify_runtime_database_gate(session_factory, project_root, settings)

    try:
        manifest = load_demo_asset_manifest(project_root)
    except ValueError as exc:
        raise BootstrapInvariantError(str(exc)) from None
    assets = manifest["assets"]
    expected_version_ids = {uuid.UUID(asset["document_version_id"]) for asset in assets}
    expected_parse_ids = {uuid.UUID(asset["parse_run_id"]) for asset in assets}
    expected_logical_keys = {asset["logical_key"] for asset in assets}
    expected_section_ids = {
        uuid.UUID(section["id"]) for asset in assets for section in asset["sections"]
    }
    expected_summary_ids = {
        uuid.UUID(summary["id"]) for asset in assets for summary in asset["summaries"]
    }
    case_ids = {case_id for case_id, _entity_id, _values in _DEMO_CASES}

    async with session_scope(
        session_factory,
        tenant_id=DEMO_TENANT_ID,
        user_id=DEMO_USER_ID,
    ) as session:
        cases = (await session.scalars(select(CreditCase).where(CreditCase.id.in_(case_ids)))).all()
        facts = (
            await session.scalars(
                select(FinancialFact).where(
                    FinancialFact.tenant_id == DEMO_TENANT_ID,
                    FinancialFact.case_id.in_(case_ids),
                    FinancialFact.period_end == _PERIOD_END,
                    FinancialFact.extraction_method == "SYNTHETIC",
                )
            )
        ).all()
        bindings = (
            await session.scalars(select(CaseDocument).where(CaseDocument.case_id.in_(case_ids)))
        ).all()
        document_rows = (
            await session.execute(
                select(Document, DocumentVersion)
                .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                .where(
                    Document.tenant_id == DEMO_TENANT_ID,
                    Document.logical_key.in_(expected_logical_keys),
                )
            )
        ).all()
        parses = (
            await session.scalars(
                select(ParseRun).where(ParseRun.document_version_id.in_(expected_version_ids))
            )
        ).all()
        sections = (
            await session.scalars(
                select(DocumentSection).where(DocumentSection.parse_run_id.in_(expected_parse_ids))
            )
        ).all()
        summaries = (
            await session.scalars(
                select(SummaryNode).where(SummaryNode.parse_run_id.in_(expected_parse_ids))
            )
        ).all()
        summary_sources = (
            await session.scalars(
                select(SummaryNodeSource).where(
                    SummaryNodeSource.summary_node_id.in_(expected_summary_ids)
                )
            )
        ).all()
        outbox = (
            await session.scalars(
                select(IndexOutbox).where(
                    IndexOutbox.aggregate_id.in_(expected_section_ids | expected_summary_ids),
                    IndexOutbox.embedding_version == settings.effective_embedding_version,
                    IndexOutbox.target_collection_name.in_(
                        [settings.chunks_collection_name, settings.summaries_collection_name]
                    ),
                )
            )
        ).all()

    expected_facts = {
        (case_id, metric_code): (entity_id, Decimal(raw_value))
        for case_id, entity_id, values in _DEMO_CASES
        for metric_code, raw_value in values.items()
    }
    actual_fact_keys = [(fact.case_id, fact.metric_code) for fact in facts]
    fact_counts = {
        str(case_id): sum(fact.case_id == case_id for fact in facts)
        for case_id, _, _ in _DEMO_CASES
    }
    expected_cases = {
        _DEMO_CASES[0][0]: (
            "golden_case_001",
            _DEMO_CASES[0][1],
            "working_capital",
            Decimal("5000000.00"),
            "采购原材料",
        ),
        _DEMO_CASES[1][0]: (
            "golden_case_002",
            _DEMO_CASES[1][1],
            "tech_working_capital",
            Decimal("15000000.00"),
            "技术研发及日常经营周转",
        ),
        _DEMO_CASES[2][0]: (
            "golden_case_003",
            _DEMO_CASES[2][1],
            "factoring",
            Decimal("8000000.00"),
            "应收账款保理融资",
        ),
    }
    if {case.id for case in cases} != set(expected_cases):
        raise BootstrapInvariantError("DEMO_CASE_SET_MISMATCH")
    for case in cases:
        case_number, borrower_id, product_code, amount, purpose = expected_cases[case.id]
        cutoff = case.decision_cutoff_at
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        if (
            case.tenant_id != DEMO_TENANT_ID
            or case.case_number != case_number
            or case.borrower_entity_id != borrower_id
            or case.product_code != product_code
            or case.requested_amount != amount
            or case.currency != "CNY"
            or case.loan_purpose != purpose
            or case.application_date != date(2026, 6, 30)
            or case.as_of_date != date(2026, 6, 30)
            or cutoff != datetime(2026, 6, 30, 15, 59, 59, tzinfo=UTC)
            or case.industry_code != "C"
        ):
            raise BootstrapInvariantError("DEMO_CASE_VALUE_MISMATCH")
    if len(actual_fact_keys) != len(expected_facts) or set(actual_fact_keys) != set(expected_facts):
        raise BootstrapInvariantError("DEMO_FINANCIAL_FACT_SET_MISMATCH")
    for fact in facts:
        entity_id, value = expected_facts[(fact.case_id, fact.metric_code)]
        available_at = fact.source_available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=UTC)
        if (
            fact.entity_id != entity_id
            or fact.value != value
            or fact.canonical_value != value
            or fact.currency != "CNY"
            or fact.verification_status != "VERIFIED"
            or fact.source_locator != {"dataset": "creditlens-synthetic-demo-v1"}
            or available_at != _SOURCE_AVAILABLE_AT
        ):
            raise BootstrapInvariantError("DEMO_FINANCIAL_FACT_VALUE_MISMATCH")

    expected_bindings = {
        (
            uuid.UUID(binding["case_id"]),
            uuid.UUID(asset["document_version_id"]),
            binding["document_role"],
            binding["is_required"],
        )
        for asset in assets
        for binding in asset["bindings"]
    }
    actual_bindings = {
        (row.case_id, row.document_version_id, row.document_role, row.is_required)
        for row in bindings
    }
    if actual_bindings != expected_bindings:
        raise BootstrapInvariantError("DEMO_DOCUMENT_BINDING_SET_MISMATCH")

    expected_assets = {(asset["logical_key"], asset["version_label"]): asset for asset in assets}
    actual_asset_keys = {
        (document.logical_key, version.version_label) for document, version in document_rows
    }
    if actual_asset_keys != set(expected_assets) or len(document_rows) != len(expected_assets):
        raise BootstrapInvariantError("DEMO_DOCUMENT_VERSION_SET_MISMATCH")
    for document, version in document_rows:
        asset = expected_assets[(document.logical_key, version.version_label)]
        source_at = version.source_available_at
        if source_at.tzinfo is None:
            source_at = source_at.replace(tzinfo=UTC)
        if (
            document.id != uuid.UUID(asset["document_id"])
            or document.title != asset["title"]
            or document.document_type != asset["document_type"]
            or document.confidentiality != asset["confidentiality"]
            or version.id != uuid.UUID(asset["document_version_id"])
            or version.tenant_id != DEMO_TENANT_ID
            or version.valid_from
            != (date.fromisoformat(asset["valid_from"]) if asset["valid_from"] else None)
            or version.valid_to
            != (date.fromisoformat(asset["valid_to"]) if asset["valid_to"] else None)
            or source_at.isoformat() != asset["source_available_at"]
            or version.object_uri != asset["object_uri"]
            or version.source_filename != asset["source_filename"]
            or version.mime_type != asset["mime_type"]
            or version.file_size != asset["object_size"]
            or version.page_count != asset["page_count"]
            or version.content_hash != asset["object_sha256"]
            or version.active_parse_run_id != uuid.UUID(asset["parse_run_id"])
            or version.processing_status != "READY"
            or not version.is_active
        ):
            raise BootstrapInvariantError("DEMO_DOCUMENT_VERSION_VALUE_MISMATCH")

    expected_parse_map = {uuid.UUID(asset["parse_run_id"]): asset for asset in assets}
    if {row.id for row in parses} != set(expected_parse_map) or len(parses) != len(assets):
        raise BootstrapInvariantError("DEMO_PARSE_RUN_SET_MISMATCH")
    for parse in parses:
        asset = expected_parse_map[parse.id]
        if (
            parse.document_version_id != uuid.UUID(asset["document_version_id"])
            or parse.generation_no != 1
            or parse.status not in {"SUCCEEDED", "SUCCEEDED_WITH_WARNINGS"}
            or parse.activation_status != "ACTIVE"
            or parse.parser_name != asset["parser_name"]
            or parse.parser_version != asset["parser_version"]
            or parse.config_hash != asset["parser_config_hash"]
        ):
            raise BootstrapInvariantError("DEMO_PARSE_RUN_VALUE_MISMATCH")

    expected_sections = {
        uuid.UUID(section["id"]): (asset, section)
        for asset in assets
        for section in asset["sections"]
    }
    if {row.id for row in sections} != set(expected_sections) or len(sections) != len(
        expected_sections
    ):
        raise BootstrapInvariantError("DEMO_SECTION_SET_MISMATCH")
    for section in sections:
        asset, expected = expected_sections[section.id]
        if (
            section.document_version_id != uuid.UUID(asset["document_version_id"])
            or section.parse_run_id != uuid.UUID(asset["parse_run_id"])
            or section.parent_section_id
            != (uuid.UUID(expected["parent_section_id"]) if expected["parent_section_id"] else None)
            or section.previous_section_id
            != (
                uuid.UUID(expected["previous_section_id"])
                if expected["previous_section_id"]
                else None
            )
            or section.next_section_id
            != (uuid.UUID(expected["next_section_id"]) if expected["next_section_id"] else None)
            or section.section_type != expected["section_type"]
            or section.ordinal != expected["ordinal"]
            or section.heading != expected["heading"]
            or section.heading_path != expected["heading_path"]
            or section.page_start != expected["page_start"]
            or section.page_end != expected["page_end"]
            or section.text_hash != expected["text_hash"]
            or sha256_text(section.text) != expected["text_hash"]
            or section.quality_status != "PASS"
        ):
            raise BootstrapInvariantError("DEMO_SECTION_VALUE_MISMATCH")

    expected_summaries = {
        uuid.UUID(summary["id"]): (asset, summary)
        for asset in assets
        for summary in asset["summaries"]
    }
    if {row.id for row in summaries} != set(expected_summaries) or len(summaries) != len(
        expected_summaries
    ):
        raise BootstrapInvariantError("DEMO_SUMMARY_SET_MISMATCH")
    for summary in summaries:
        asset, expected = expected_summaries[summary.id]
        if (
            summary.document_version_id != uuid.UUID(asset["document_version_id"])
            or summary.parse_run_id != uuid.UUID(asset["parse_run_id"])
            or summary.parent_summary_id
            != (uuid.UUID(expected["parent_summary_id"]) if expected["parent_summary_id"] else None)
            or summary.summary_level != expected["summary_level"]
            or summary.summary_hash != expected["summary_hash"]
            or sha256_text(summary.summary_text) != expected["summary_hash"]
            or summary.grounding_status != "VERIFIED"
            or summary.evidence_eligible
        ):
            raise BootstrapInvariantError("DEMO_SUMMARY_VALUE_MISMATCH")
    expected_sources = {
        (uuid.UUID(summary["id"]), uuid.UUID(section_id), ordinal)
        for asset in assets
        for summary in asset["summaries"]
        for ordinal, section_id in enumerate(summary["source_section_ids"])
    }
    actual_sources = {(row.summary_node_id, row.section_id, row.ordinal) for row in summary_sources}
    if actual_sources != expected_sources:
        raise BootstrapInvariantError("DEMO_SUMMARY_SOURCE_SET_MISMATCH")

    expected_outbox = {
        (
            "SECTION",
            uuid.UUID(section["id"]),
            section["text_hash"],
            settings.chunks_collection_name,
            settings.effective_embedding_version,
            settings.sparse_encoder_version,
        )
        for asset in assets
        for section in asset["sections"]
        if section["section_type"] in {"ARTICLE", "PARAGRAPH"}
    } | {
        (
            "SUMMARY",
            uuid.UUID(summary["id"]),
            summary["summary_hash"],
            settings.summaries_collection_name,
            settings.effective_embedding_version,
            None,
        )
        for asset in assets
        for summary in asset["summaries"]
    }
    actual_outbox = {
        (
            row.aggregate_type,
            row.aggregate_id,
            row.content_hash,
            row.target_collection_name,
            row.embedding_version,
            row.sparse_encoder_version,
        )
        for row in outbox
    }
    if actual_outbox != expected_outbox or any(
        row.status != "COMPLETED" or row.operation != "UPSERT" for row in outbox
    ):
        raise BootstrapInvariantError("CURRENT_EMBEDDING_OUTBOX_SET_MISMATCH")

    return {
        **runtime_gate,
        "visible_demo_cases": len(cases),
        "financial_fact_counts": fact_counts,
        "document_version_count": len(document_rows),
        "active_parse_count": len(parses),
        "section_count": len(sections),
        "summary_count": len(summaries),
        "current_embedding_ledger_identity_count": len(actual_outbox),
        "_asset_manifest": manifest,
    }


def _scroll_current_points(
    client,
    collection: str,
    settings: Any,
    document_version_ids: list[str],
) -> list[dict[str, Any]]:
    scroll_filter = qm.Filter(
        must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(DEMO_TENANT_ID))),
            qm.FieldCondition(
                key="embedding_version",
                match=qm.MatchValue(value=settings.effective_embedding_version),
            ),
            qm.FieldCondition(key="tombstoned", match=qm.MatchValue(value=False)),
            qm.FieldCondition(
                key="document_version_id",
                match=qm.MatchAny(any=document_version_ids),
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        rows.extend({"id": str(point.id), "payload": point.payload or {}} for point in points)
        if offset is None:
            return rows


def inspect_qdrant(
    client,
    settings: Any,
    asset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify aliases, dimensions and the frozen point ID/payload exact set."""

    if asset_manifest is None:
        raise BootstrapInvariantError("QDRANT_ASSET_MANIFEST_REQUIRED")
    expected_points = expected_demo_qdrant_points(asset_manifest, settings)
    document_version_ids = [asset["document_version_id"] for asset in asset_manifest["assets"]]

    expected_dimension = resolve_embedding_dim(settings)
    manager = CollectionManager(client, dense_dim=expected_dimension)
    expected = {
        settings.qdrant_chunks_alias: settings.chunks_collection_name,
        settings.qdrant_summaries_alias: settings.summaries_collection_name,
    }
    point_counts: dict[str, int] = {}
    statuses: dict[str, str] = {}
    for alias, collection in expected.items():
        if manager.resolve_alias(alias) != collection:
            raise BootstrapInvariantError("QDRANT_ALIAS_MISMATCH")
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        dense = (
            vectors.get("dense") if isinstance(vectors, dict) else getattr(vectors, "dense", None)
        )
        if dense is None or int(dense.size) != expected_dimension:
            raise BootstrapInvariantError("QDRANT_VECTOR_DIMENSION_MISMATCH")
        raw_status = getattr(info, "status", "unknown")
        status = str(getattr(raw_status, "value", raw_status)).lower()
        if status not in {"green", "ok"}:
            raise BootstrapInvariantError("QDRANT_COLLECTION_NOT_GREEN")
        points = _scroll_current_points(client, collection, settings, document_version_ids)
        actual_points = {row["id"]: row["payload"] for row in points}
        if len(actual_points) != len(points):
            raise BootstrapInvariantError("QDRANT_DUPLICATE_POINT_ID")
        if set(actual_points) != set(expected_points[collection]):
            raise BootstrapInvariantError("QDRANT_POINT_ID_SET_MISMATCH")
        if actual_points != expected_points[collection]:
            raise BootstrapInvariantError("QDRANT_POINT_PAYLOAD_SET_MISMATCH")
        count = len(points)
        point_counts[collection] = count
        statuses[collection] = status
    return {
        "aliases": expected,
        "collection_statuses": statuses,
        "point_counts": point_counts,
    }


def inspect_minio(
    settings: Any,
    asset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify buckets and every expected immutable raw object byte-for-byte."""

    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    buckets = (
        settings.minio_raw_bucket,
        settings.minio_derived_bucket,
        settings.minio_rendered_bucket,
    )
    missing = [bucket for bucket in buckets if not client.bucket_exists(bucket)]
    if missing:
        raise BootstrapInvariantError("MINIO_BUCKETS_INCOMPLETE")
    if asset_manifest is None:
        raise BootstrapInvariantError("MINIO_ASSET_MANIFEST_REQUIRED")
    if asset_manifest["raw_bucket"] != settings.minio_raw_bucket:
        raise BootstrapInvariantError("MINIO_MANIFEST_BUCKET_MISMATCH")
    verified_objects = 0
    for item in asset_manifest["assets"]:
        parsed = urlparse(item["object_uri"])
        if parsed.scheme != "s3" or parsed.netloc != settings.minio_raw_bucket:
            raise BootstrapInvariantError("MINIO_OBJECT_URI_INVALID")
        key = parsed.path.lstrip("/")
        try:
            response = client.get_object(parsed.netloc, key)
            try:
                payload = response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception:
            raise BootstrapInvariantError("MINIO_EXPECTED_OBJECT_MISSING") from None
        if (
            len(payload) != item["object_size"]
            or not hashlib.sha256(payload).hexdigest() == item["object_sha256"]
        ):
            raise BootstrapInvariantError("MINIO_OBJECT_INTEGRITY_MISMATCH")
        verified_objects += 1
    return {
        "required_buckets": sorted(buckets),
        "available_bucket_count": len(buckets),
        "verified_raw_objects": verified_objects,
    }


async def _capture(name: str, operation) -> ComponentResult:
    try:
        details = await operation
        return ComponentResult(name=name, status="PASS", code="READY", details=details)
    except BootstrapInvariantError as exc:
        return ComponentResult(name=name, status="FAIL", code=exc.code, details={})
    except Exception as exc:
        return ComponentResult(
            name=name,
            status="FAIL",
            code=f"{name.upper()}_CHECK_FAILED",
            details={"error_type": type(exc).__name__},
        )


async def run_preflight(
    *,
    settings: Any,
    session_factory,
    qdrant,
    project_root,
    allow_configured_models: bool = False,
) -> PreflightReport:
    """Return a safe machine-readable report; any failed component means not ready."""

    async def config_check() -> dict[str, Any]:
        return validate_demo_settings(
            settings,
            allow_configured_models=allow_configured_models,
        )

    configuration = await _capture("configuration", config_check())
    if configuration.status != "PASS":
        blocked = tuple(
            ComponentResult(
                name=name,
                status="SKIP",
                code="CONFIGURATION_BLOCKED",
                details={},
            )
            for name in ("postgres", "qdrant", "minio")
        )
        return PreflightReport(
            schema_version="creditlens.demo-preflight.v1",
            ready=False,
            profile="configured-models" if allow_configured_models else "deterministic-offline",
            components=(configuration, *blocked),
        )

    postgres = await _capture(
        "postgres",
        inspect_postgres(session_factory, project_root, settings),
    )
    asset_manifest = postgres.details.get("_asset_manifest") if postgres.status == "PASS" else None
    if postgres.status == "PASS":
        postgres = ComponentResult(
            name=postgres.name,
            status=postgres.status,
            code=postgres.code,
            details={
                key: value for key, value in postgres.details.items() if not key.startswith("_")
            },
        )
    qdrant_result, minio_result = await asyncio.gather(
        _capture(
            "qdrant",
            asyncio.to_thread(inspect_qdrant, qdrant, settings, asset_manifest),
        ),
        _capture(
            "minio",
            asyncio.to_thread(inspect_minio, settings, asset_manifest),
        ),
    )
    infrastructure = (postgres, qdrant_result, minio_result)
    components = (configuration, *infrastructure)
    return PreflightReport(
        schema_version="creditlens.demo-preflight.v1",
        ready=all(component.status == "PASS" for component in components),
        profile="configured-models" if allow_configured_models else "deterministic-offline",
        components=components,
    )
