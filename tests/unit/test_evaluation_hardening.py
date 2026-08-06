"""评测冻结世界与 Manifest 溯源的回归测试。"""

import subprocess
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from scripts import run_evaluation

from creditlens.common.hashing import sha256_bytes
from creditlens.evaluation.gold_schema import GoldEvidenceAnchor
from creditlens.evaluation.recall import GoldMappingScope, map_anchor_to_section_ids
from creditlens.infrastructure.postgres.models import (
    CaseDocument,
    CreditCase,
    Document,
    DocumentSection,
    DocumentVersion,
    Entity,
    ParseRun,
    Tenant,
)


async def test_gold_mapping_keeps_historical_snapshot_after_new_parse_activates(session):
    """历史 Snapshot 必须继续映射已变为 SUPERSEDED 的 ParseRun。"""
    tenant_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    case_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    old_run_id = uuid.uuid4()
    new_run_id = uuid.uuid4()
    old_section_id = uuid.uuid4()

    session.add_all(
        [
            Tenant(id=tenant_id, name="历史快照租户"),
            Entity(
                id=entity_id,
                tenant_id=tenant_id,
                entity_type="COMPANY",
                canonical_name="历史快照借款人",
            ),
            CreditCase(
                id=case_id,
                tenant_id=tenant_id,
                case_number="SNAPSHOT-GOLD-001",
                borrower_entity_id=entity_id,
                product_code="WORKING_CAPITAL",
                requested_amount=Decimal("1000000.00"),
                application_date=date(2026, 1, 1),
                as_of_date=date(2026, 1, 1),
                decision_cutoff_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Document(
                id=document_id,
                tenant_id=tenant_id,
                logical_key="snapshot-policy",
                title="历史快照政策",
                document_type="INTERNAL_POLICY",
            ),
            DocumentVersion(
                id=version_id,
                tenant_id=tenant_id,
                document_id=document_id,
                version_label="2026",
                source_available_at=datetime(2026, 1, 1, tzinfo=UTC),
                object_uri="local://snapshot-policy.pdf",
                source_filename="snapshot-policy.pdf",
                mime_type="application/pdf",
                file_size=100,
                content_hash="1" * 64,
                active_parse_run_id=new_run_id,
            ),
            ParseRun(
                id=old_run_id,
                tenant_id=tenant_id,
                document_version_id=version_id,
                generation_no=1,
                status="SUCCEEDED",
                activation_status="SUPERSEDED",
                parser_name="test",
                parser_version="1",
                config_hash="2" * 64,
            ),
            ParseRun(
                id=new_run_id,
                tenant_id=tenant_id,
                document_version_id=version_id,
                generation_no=2,
                status="SUCCEEDED",
                activation_status="ACTIVE",
                parser_name="test",
                parser_version="2",
                config_hash="3" * 64,
            ),
            CaseDocument(
                case_id=case_id,
                document_version_id=version_id,
                document_role="BANK_POLICY",
            ),
            DocumentSection(
                id=old_section_id,
                tenant_id=tenant_id,
                document_version_id=version_id,
                parse_run_id=old_run_id,
                section_type="ARTICLE",
                ordinal=1,
                heading="第六条",
                heading_path=["第六条"],
                page_start=1,
                page_end=1,
                text="旧解析批次中的有效条款。",
                text_hash="4" * 64,
            ),
            DocumentSection(
                tenant_id=tenant_id,
                document_version_id=version_id,
                parse_run_id=new_run_id,
                section_type="ARTICLE",
                ordinal=1,
                heading="第六条",
                heading_path=["第六条"],
                page_start=1,
                page_end=1,
                text="新解析批次中的同一条款。",
                text_hash="5" * 64,
            ),
        ]
    )
    await session.flush()

    mapped = await map_anchor_to_section_ids(
        session,
        GoldEvidenceAnchor(
            gold_evidence_key="snapshot-policy:art06",
            logical_document_key="snapshot-policy",
            version_label="2026",
            article_anchor="第六条",
        ),
        GoldMappingScope(
            tenant_id=tenant_id,
            case_id=case_id,
            allowed_parse_run_ids=frozenset({old_run_id}),
        ),
    )

    assert mapped == {old_section_id}


@pytest.mark.parametrize(
    ("status", "dirty", "dirty_files", "untracked", "generated_reports"),
    [
        ("?? evaluation/reports/generated.json\n", False, 0, 1, 1),
        (
            "?? evaluation/reports/generated.json\n?? scratch-notes.md\n",
            True,
            1,
            2,
            1,
        ),
        (" M src/creditlens/example.py\n?? evaluation/reports/generated.json\n", True, 1, 1, 1),
    ],
)
def test_git_provenance_ignores_only_generated_reports(
    monkeypatch, status, dirty, dirty_files, untracked, generated_reports
):
    responses = {
        ("rev-parse", "HEAD"): "deadbeef\n",
        ("status", "--porcelain", "--untracked-files=all"): status,
        ("rev-parse", "--abbrev-ref", "HEAD"): "main\n",
        ("describe", "--tags", "--always", "--dirty"): "v-test\n",
    }

    def fake_run(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=responses[tuple(command[1:])])

    monkeypatch.setattr(subprocess, "run", fake_run)

    provenance = run_evaluation.git_provenance()

    assert provenance["git_dirty"] is dirty
    assert provenance["git_dirty_files"] == dirty_files
    assert provenance["git_untracked_files"] == untracked
    assert provenance["git_generated_report_files"] == generated_reports


def test_file_hashes_separate_seed_script_from_source_corpus(tmp_path, monkeypatch):
    (tmp_path / "config" / "formulas").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    corpus_dir = tmp_path / "data" / "synthetic"
    corpus_dir.mkdir(parents=True)
    (tmp_path / "uv.lock").write_bytes(b"lock")
    (tmp_path / "config" / "formulas" / "registry_v1.yaml").write_bytes(b"formula")
    seed_script = tmp_path / "scripts" / "seed_synthetic_data.py"
    seed_script.write_bytes(b"seed-v1")
    (corpus_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("beta", encoding="utf-8")
    (corpus_dir / "derived.pdf").write_bytes(b"non-semantic-pdf-container")
    monkeypatch.setattr(run_evaluation, "PROJECT_ROOT", tmp_path)

    before = run_evaluation.file_hashes()
    assert before["seed_script_sha256"] == sha256_bytes(b"seed-v1")
    assert before["seed_corpus_sha256"] != before["seed_script_sha256"]

    (corpus_dir / "derived.pdf").write_bytes(b"different-container-metadata")
    assert run_evaluation.file_hashes()["seed_corpus_sha256"] == before["seed_corpus_sha256"]

    (corpus_dir / "b.txt").write_text("beta changed", encoding="utf-8")
    after = run_evaluation.file_hashes()
    assert after["seed_corpus_sha256"] != before["seed_corpus_sha256"]
    assert after["seed_script_sha256"] == before["seed_script_sha256"]
