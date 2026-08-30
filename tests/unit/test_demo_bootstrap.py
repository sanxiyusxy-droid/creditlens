import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.seed_synthetic_data import SYNTH_DIR, text_to_pdf
from sqlalchemy import func, select

from creditlens.demo_bootstrap import (
    DEMO_TENANT_ID,
    DEMO_USER_ID,
    BootstrapInvariantError,
    ensure_demo_financial_facts,
    inspect_qdrant,
    run_preflight,
    validate_demo_settings,
)
from creditlens.demo_manifest import (
    expected_demo_qdrant_points,
    load_demo_asset_manifest,
    stable_demo_id,
)
from creditlens.infrastructure.postgres.models import (
    CreditCase,
    Entity,
    FinancialFact,
    Tenant,
)
from creditlens.infrastructure.postgres.session import create_session_factory, session_scope

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings(**overrides):
    values = {
        "app_env": "local",
        "api_identity_mode": "demo",
        "allow_insecure_demo_identity": True,
        "database_url": "postgresql+asyncpg://creditlens_app:redacted@localhost/creditlens",
        "qdrant_url": "http://localhost:6333",
        "object_store_backend": "minio",
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "local-user",
        "minio_secret_key": "local-secret",
        "minio_raw_bucket": "creditlens-raw",
        "minio_derived_bucket": "creditlens-parsed",
        "minio_rendered_bucket": "creditlens-rendered",
        "minio_secure": False,
        "llm_provider": "disabled",
        "llm_api_base": "",
        "llm_api_key": "",
        "llm_model": "",
        "qa_allow_extractive_fallback": True,
        "embedding_provider": "hash_fallback",
        "embedding_version": "hash-embed-v1",
        "embedding_api_base": "",
        "embedding_api_key": "",
        "embedding_model": "",
        "rerank_provider": "lexical_fallback",
        "rerank_api_base": "",
        "rerank_api_key": "",
        "rerank_model": "",
        "embedding_dim": 4,
        "effective_embedding_version": "hash-embed-v1",
        "sparse_encoder_version": "bm25-jieba-v1",
        "qdrant_chunks_alias": "chunks-current",
        "qdrant_summaries_alias": "summaries-current",
        "chunks_collection_name": "chunks-v1",
        "summaries_collection_name": "summaries-v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_demo_settings_default_profile_is_local_and_non_networked():
    details = validate_demo_settings(_settings())
    assert details["environment"] == "local"
    assert details["external_models_authorized"] is False
    assert details["llm_provider"] == "disabled"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"app_env": "production"}, "DEMO_ENV_FORBIDDEN"),
        ({"api_identity_mode": "required"}, "DEMO_IDENTITY_REQUIRED"),
        ({"allow_insecure_demo_identity": False}, "DEMO_IDENTITY_OPT_IN_REQUIRED"),
        ({"database_url": "sqlite+aiosqlite:///demo.db"}, "POSTGRES_RUNTIME_URL_REQUIRED"),
        ({"qdrant_url": ":memory:"}, "EXTERNAL_QDRANT_REQUIRED"),
        ({"object_store_backend": "local_fs"}, "MINIO_BACKEND_REQUIRED"),
        (
            {"database_url": "postgresql+asyncpg://app:redacted@db.example/demo"},
            "DEMO_INFRASTRUCTURE_NOT_LOCAL",
        ),
        (
            {"database_url": ("postgresql+asyncpg://app:redacted@localhost/demo?host=db.example")},
            "DEMO_DATABASE_TARGET_OVERRIDE_FORBIDDEN",
        ),
        (
            {"database_url": "postgresql+asyncpg://app:redacted@localhost/demo?service=remote"},
            "DEMO_DATABASE_TARGET_OVERRIDE_FORBIDDEN",
        ),
        ({"qdrant_url": "https://qdrant.example"}, "DEMO_INFRASTRUCTURE_NOT_LOCAL"),
        ({"minio_endpoint": "minio.example:9000"}, "DEMO_INFRASTRUCTURE_NOT_LOCAL"),
        ({"qa_allow_extractive_fallback": False}, "DEMO_QA_FALLBACK_REQUIRED"),
        ({"llm_provider": "openai_compatible"}, "EXTERNAL_MODEL_PROFILE_NOT_AUTHORIZED"),
        ({"embedding_provider": "openai_compatible"}, "EXTERNAL_MODEL_PROFILE_NOT_AUTHORIZED"),
        ({"rerank_provider": "http"}, "EXTERNAL_RERANK_PROFILE_NOT_AUTHORIZED"),
    ],
)
def test_demo_settings_fail_closed(overrides, code):
    with pytest.raises(BootstrapInvariantError, match=code):
        validate_demo_settings(_settings(**overrides))


def test_configured_models_require_explicit_authorization():
    details = validate_demo_settings(
        _settings(
            llm_provider="openai_compatible",
            llm_api_base="https://models.example/v1",
            llm_api_key="synthetic-test-key",
            llm_model="chat-model",
            embedding_provider="openai_compatible",
            embedding_api_base="https://models.example/v1",
            embedding_api_key="synthetic-test-key",
            embedding_model="embed-model",
            rerank_provider="http",
            rerank_api_base="https://models.example/v1/rerank",
            rerank_api_key="synthetic-test-key",
            rerank_model="rerank-model",
        ),
        allow_configured_models=True,
    )
    assert details["external_models_authorized"] is True


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"database_url": "postgresql+asyncpg://admin:x@localhost/creditlens"},
            "DEMO_RUNTIME_DATABASE_ROLE_REQUIRED",
        ),
        (
            {"database_url": "postgresql+asyncpg://creditlens_app:x@localhost/other"},
            "DEMO_RUNTIME_DATABASE_NAME_REQUIRED",
        ),
        ({"llm_provider": "unknown"}, "LLM_PROVIDER_UNSUPPORTED"),
        (
            {
                "llm_provider": "openai_compatible",
                "llm_api_base": "https://models.example/v1",
            },
            "LLM_CONFIGURATION_INCOMPLETE",
        ),
        (
            {
                "llm_provider": "openai_compatible",
                "llm_api_base": "http://models.example/v1",
                "llm_api_key": "synthetic-test-key",
                "llm_model": "chat-model",
            },
            "MODEL_PROVIDER_TLS_REQUIRED",
        ),
        (
            {
                "rerank_provider": "http",
                "rerank_api_base": "https://user:password@models.example/rerank",
                "rerank_api_key": "synthetic-test-key",
                "rerank_model": "rerank-model",
            },
            "MODEL_PROVIDER_URL_INVALID",
        ),
    ],
)
def test_configured_profile_completeness_and_transport_fail_closed(overrides, code):
    with pytest.raises(BootstrapInvariantError, match=code):
        validate_demo_settings(
            _settings(**overrides),
            allow_configured_models=True,
        )


def test_frozen_asset_manifest_pins_source_objects_and_deterministic_identities(tmp_path):
    manifest = load_demo_asset_manifest(PROJECT_ROOT)
    assert len(manifest["assets"]) == 8
    assert sum(len(asset["bindings"]) for asset in manifest["assets"]) == 10
    for asset in manifest["assets"]:
        identity = f"{asset['logical_key']}@{asset['version_label']}"
        assert asset["document_id"] == str(stable_demo_id("document", asset["logical_key"]))
        assert asset["document_version_id"] == str(stable_demo_id("version", identity))
        assert asset["parse_run_id"] == str(stable_demo_id("parse", identity))
        source = SYNTH_DIR / asset["source_text_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == asset["source_text_sha256"]
        pdf_bytes = text_to_pdf(source, tmp_path / asset["source_filename"])
        assert len(pdf_bytes) == asset["object_size"]
        assert hashlib.sha256(pdf_bytes).hexdigest() == asset["object_sha256"]
        for section in asset["sections"]:
            expected = stable_demo_id(
                "section",
                f"{identity}:{section['ordinal']}:{section['section_type']}:{section['text_hash']}",
            )
            assert section["id"] == str(expected)
        for ordinal, summary in enumerate(asset["summaries"]):
            expected = stable_demo_id(
                "summary",
                f"{identity}:{ordinal}:{summary['summary_level']}:{summary['summary_hash']}",
            )
            assert summary["id"] == str(expected)


def test_bootstrap_runs_read_only_runtime_gate_before_any_seed_writer():
    source = (PROJECT_ROOT / "scripts" / "bootstrap_demo.py").read_text(encoding="utf-8")
    gate = source.index("await verify_runtime_database_gate")
    assert gate < source.index("qdrant = build_qdrant_client")
    assert gate < source.index("seed_environment(session_factory")
    assert gate < source.index("ensure_demo_financial_facts(session_factory)")


def test_apply_rls_converges_role_attributes_and_rejects_membership():
    source = (PROJECT_ROOT / "scripts" / "apply_rls.py").read_text(encoding="utf-8")
    for attribute in (
        "NOSUPERUSER",
        "NOBYPASSRLS",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOINHERIT",
    ):
        assert source.count(attribute) >= 2
    assert "pg_auth_members" in source
    assert "拒绝继续授权" in source


def test_runtime_privilege_contract_keeps_snapshots_append_only_and_case_roots_immutable():
    import creditlens.demo_bootstrap as bootstrap

    snapshot_tables = {
        "case_snapshots",
        "snapshot_documents",
        "snapshot_indexes",
        "snapshot_facts",
    }
    assert snapshot_tables <= bootstrap._APPEND_ONLY_TABLES
    assert snapshot_tables.isdisjoint(bootstrap._FULL_DML_TABLES)
    expected_table_privileges = bootstrap._expected_table_privileges()
    for table in snapshot_tables:
        assert (table, "SELECT") in expected_table_privileges
        assert (table, "INSERT") in expected_table_privileges
        assert (table, "UPDATE") not in expected_table_privileges
        assert (table, "DELETE") not in expected_table_privileges

    assert ("credit_cases", "UPDATE") not in expected_table_privileges
    assert bootstrap._COLUMN_UPDATE_ALLOWLIST["credit_cases"] == {
        "loan_purpose",
        "industry_code",
        "region_code",
        "status",
        "current_report_id",
        "updated_at",
        "version",
    }

    grant_source = (PROJECT_ROOT / "infra" / "postgres" / "runtime_role_grants.sql").read_text(
        encoding="utf-8"
    )
    compact_grants = " ".join(grant_source.split())
    assert (
        "REVOKE UPDATE, DELETE ON case_snapshots, snapshot_documents, "
        "snapshot_indexes, snapshot_facts FROM creditlens_app"
    ) in compact_grants
    assert "REVOKE INSERT, UPDATE, DELETE ON credit_cases FROM creditlens_app" in compact_grants
    assert (
        "GRANT UPDATE ( loan_purpose, industry_code, region_code, status, "
        "current_report_id, updated_at, version ) ON credit_cases TO creditlens_app"
    ) in compact_grants

    policy_source = (PROJECT_ROOT / "infra" / "postgres" / "rls_policies.sql").read_text(
        encoding="utf-8"
    )
    for policy in (
        "case_snapshot_select",
        "case_snapshot_insert",
        "snapshot_parent_select",
        "snapshot_parent_insert",
        "case_tenant_select",
        "case_tenant_update",
    ):
        assert f"CREATE POLICY {policy}" in policy_source

    freeze_source = (
        PROJECT_ROOT / "src" / "creditlens" / "application" / "snapshot_service.py"
    ).read_text(encoding="utf-8")
    snapshot_hash_assignment = freeze_source.index(
        "snapshot_hash=sha256_text(json.dumps(canonical, sort_keys=True))"
    )
    first_snapshot_flush = freeze_source.index("await session.flush()", snapshot_hash_assignment)
    assert snapshot_hash_assignment < first_snapshot_flush
    assert "snapshot.snapshot_hash =" not in freeze_source


def test_start_demo_runs_preflight_and_http_acceptance_without_reset():
    script = (PROJECT_ROOT / "scripts" / "start_demo.ps1").read_text(encoding="utf-8")
    assert "scripts/bootstrap_demo.py" in script
    assert "scripts/http_demo_acceptance.py" in script
    assert "$UseConfiguredModels" in script
    assert '$env:LLM_PROVIDER = "disabled"' in script
    assert "scripts/apply_rls.py --bootstrap-demo-principals" in script
    assert 'Get-LocalEnvValue "DATABASE_URL"' in script
    assert "[System.Uri]::UnescapeDataString($Matches[1])" in script
    assert "creditlens_app:([^@]+)@(localhost|127\\.0\\.0\\.1):5432/creditlens" in script
    lowered = script.lower()
    assert "docker compose down" not in lowered
    assert "drop database" not in lowered
    assert "reset --hard" not in lowered


def test_offline_integration_profiles_match_frozen_http_acceptance():
    sources = (
        PROJECT_ROOT / "scripts" / "run_integration.ps1",
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
        PROJECT_ROOT / ".gitlab-ci.yml",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert 'RERANK_PROVIDER = "lexical_fallback"' in text or (
            'RERANK_PROVIDER: "lexical_fallback"' in text
        )
        assert 'RERANK_PROVIDER = "disabled"' not in text
        assert 'RERANK_PROVIDER: "disabled"' not in text

    grant_path = "infra/postgres/runtime_role_grants.sql"
    assert grant_path in (PROJECT_ROOT / "scripts" / "run_integration.ps1").read_text(
        encoding="utf-8"
    )
    assert grant_path in (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert grant_path in (PROJECT_ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")


def test_start_demo_fails_closed_before_using_a_nonlocal_docker_daemon():
    script = (PROJECT_ROOT / "scripts" / "start_demo.ps1").read_text(encoding="utf-8")

    guard_call = "$DockerContextName = Assert-LocalDockerContext"
    assert guard_call in script
    assert script.index(guard_call) < script.index("$Uv = Resolve-Uv")
    assert "$env:DOCKER_HOST" in script
    assert "docker context show" in script
    assert "docker context inspect" in script
    assert "REMOTE_DOCKER_CONTEXT_FORBIDDEN" in script
    assert "npipe:////./pipe/docker_engine" in script
    assert "npipe:////./pipe/dockerdesktoplinuxengine" in script
    assert "tcp://" not in script[script.index("function Test-ApprovedLocalDockerEndpoint") :]


async def test_financial_fact_seed_is_idempotent(engine):
    factory = create_session_factory(engine)
    case_rows = (
        (
            uuid.UUID("00000000-0000-0000-0000-000000000201"),
            uuid.UUID("00000000-0000-0000-0000-000000000101"),
        ),
        (
            uuid.UUID("00000000-0000-0000-0000-000000000202"),
            uuid.UUID("00000000-0000-0000-0000-000000000102"),
        ),
        (
            uuid.UUID("00000000-0000-0000-0000-000000000203"),
            uuid.UUID("00000000-0000-0000-0000-000000000103"),
        ),
    )
    async with session_scope(factory) as session:
        session.add(Tenant(id=DEMO_TENANT_ID, name="synthetic bank"))
        for index, (case_id, entity_id) in enumerate(case_rows, start=1):
            session.add(
                Entity(
                    id=entity_id,
                    tenant_id=DEMO_TENANT_ID,
                    entity_type="COMPANY",
                    canonical_name=f"synthetic borrower {index}",
                )
            )
            session.add(
                CreditCase(
                    id=case_id,
                    tenant_id=DEMO_TENANT_ID,
                    case_number=f"golden_case_{index:03d}",
                    borrower_entity_id=entity_id,
                    product_code="working_capital",
                    requested_amount=Decimal("1000000"),
                    application_date=date(2026, 6, 30),
                    as_of_date=date(2026, 6, 30),
                    decision_cutoff_at=datetime(2026, 6, 30, tzinfo=UTC),
                )
            )

    assert await ensure_demo_financial_facts(factory) == 12
    assert await ensure_demo_financial_facts(factory) == 0
    async with session_scope(factory) as session:
        count = await session.scalar(select(func.count()).select_from(FinancialFact))
        rows = (await session.scalars(select(FinancialFact))).all()
    assert count == 12
    assert {row.verification_status for row in rows} == {"VERIFIED"}
    assert {row.extraction_method for row in rows} == {"SYNTHETIC"}


async def test_preflight_error_output_never_contains_exception_message(monkeypatch, tmp_path):
    import creditlens.demo_bootstrap as bootstrap

    async def broken_postgres(*_args):
        raise RuntimeError("postgresql://admin:top-secret@bank.example")

    def broken_qdrant(*_args):
        raise RuntimeError("qdrant-api-key=top-secret")

    def broken_minio(*_args):
        raise RuntimeError("minio-secret=top-secret")

    monkeypatch.setattr(bootstrap, "inspect_postgres", broken_postgres)
    monkeypatch.setattr(bootstrap, "inspect_qdrant", broken_qdrant)
    monkeypatch.setattr(bootstrap, "inspect_minio", broken_minio)
    report = await run_preflight(
        settings=_settings(),
        session_factory=object(),
        qdrant=object(),
        project_root=tmp_path,
    )
    serialized = json.dumps(report.to_dict())
    assert report.ready is False
    assert "top-secret" not in serialized
    assert {component.details.get("error_type") for component in report.components[1:]} == {
        "RuntimeError"
    }
    # The constant identity is exported for bootstrap consumers, but never accepted from clients.
    assert str(DEMO_USER_ID) == "00000000-0000-0000-0000-000000000301"


async def test_invalid_configuration_blocks_all_infrastructure_access(monkeypatch, tmp_path):
    import creditlens.demo_bootstrap as bootstrap

    def unexpected(*_args):
        raise AssertionError("infrastructure must not be contacted")

    monkeypatch.setattr(bootstrap, "inspect_postgres", unexpected)
    monkeypatch.setattr(bootstrap, "inspect_qdrant", unexpected)
    monkeypatch.setattr(bootstrap, "inspect_minio", unexpected)
    report = await run_preflight(
        settings=_settings(app_env="production"),
        session_factory=object(),
        qdrant=object(),
        project_root=tmp_path,
    )
    assert report.ready is False
    assert report.components[0].code == "DEMO_ENV_FORBIDDEN"
    assert {component.status for component in report.components[1:]} == {"SKIP"}


def test_qdrant_preflight_requires_frozen_point_identity_and_payload_exact_set():
    settings = _settings()
    manifest = load_demo_asset_manifest(PROJECT_ROOT)
    frozen_points = expected_demo_qdrant_points(manifest, settings)

    class FakeQdrant:
        drop_one = False
        tamper_payload = False

        def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(alias_name="chunks-current", collection_name="chunks-v1"),
                    SimpleNamespace(
                        alias_name="summaries-current",
                        collection_name="summaries-v1",
                    ),
                ]
            )

        def get_collection(self, _collection):
            return SimpleNamespace(
                status="green",
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=4)})
                ),
            )

        def scroll(self, *, collection_name, **_kwargs):
            rows = list(frozen_points[collection_name].items())
            if self.drop_one:
                rows = rows[1:]
            if self.tamper_payload and rows:
                point_id, payload = rows[0]
                rows[0] = (point_id, {**payload, "document_type": "TAMPERED"})
            return (
                [
                    SimpleNamespace(
                        id=point_id,
                        payload=payload,
                    )
                    for point_id, payload in rows
                ],
                None,
            )

    client = FakeQdrant()
    result = inspect_qdrant(client, settings, manifest)

    assert result["point_counts"] == {
        collection: len(points) for collection, points in frozen_points.items()
    }

    client.drop_one = True
    with pytest.raises(BootstrapInvariantError, match="QDRANT_POINT_ID_SET_MISMATCH"):
        inspect_qdrant(client, settings, manifest)

    client.drop_one = False
    client.tamper_payload = True
    with pytest.raises(BootstrapInvariantError, match="QDRANT_POINT_PAYLOAD_SET_MISMATCH"):
        inspect_qdrant(client, settings, manifest)


def test_qdrant_preflight_uses_resolved_dimension_for_known_online_model():
    settings = _settings(
        embedding_provider="openai_compatible",
        embedding_model="BAAI/bge-m3",
        embedding_api_base="https://embedding.example/v1",
        embedding_api_key="configured",
        effective_embedding_version="BAAI/bge-m3@api",
        chunks_collection_name="chunks-v2",
        summaries_collection_name="summaries-v2",
    )
    manifest = load_demo_asset_manifest(PROJECT_ROOT)
    frozen_points = expected_demo_qdrant_points(manifest, settings)

    class FakeQdrant:
        def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(alias_name="chunks-current", collection_name="chunks-v2"),
                    SimpleNamespace(
                        alias_name="summaries-current",
                        collection_name="summaries-v2",
                    ),
                ]
            )

        def get_collection(self, _collection):
            return SimpleNamespace(
                status="green",
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=1024)})
                ),
            )

        def scroll(self, *, collection_name, **_kwargs):
            return (
                [
                    SimpleNamespace(id=point_id, payload=payload)
                    for point_id, payload in frozen_points[collection_name].items()
                ],
                None,
            )

    result = inspect_qdrant(FakeQdrant(), settings, manifest)

    assert result["point_counts"] == {
        collection: len(points) for collection, points in frozen_points.items()
    }
