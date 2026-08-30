from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.seed_synthetic_data import TENANT_ID, _repair_existing_index
from sqlalchemy import select

from creditlens.demo_manifest import (
    expected_demo_qdrant_points,
    load_demo_asset_manifest,
)
from creditlens.infrastructure.postgres.models import (
    Document,
    DocumentSection,
    DocumentVersion,
    IndexOutbox,
    ParseRun,
    SummaryNode,
    Tenant,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings():
    return SimpleNamespace(
        chunks_collection_name="chunks-v1",
        summaries_collection_name="summaries-v1",
        effective_embedding_version="hash-embed-v1",
        sparse_encoder_version="bm25-jieba-v1",
    )


class _PointStore:
    def __init__(self):
        self.points: dict[tuple[str, str], SimpleNamespace] = {}

    def retrieve(self, collection, *, ids, with_payload):
        del with_payload
        return [
            self.points[(collection, point_id)]
            for point_id in ids
            if (collection, point_id) in self.points
        ]


class _CrashRecoveryWorker:
    def __init__(self, qdrant, expected_points):
        self._qdrant = qdrant
        self._expected_points = expected_points

    async def process_batch(self, session):
        rows = (
            await session.scalars(select(IndexOutbox).where(IndexOutbox.status == "PENDING"))
        ).all()
        processed = 0
        for row in rows:
            candidates = self._expected_points[row.target_collection_name]
            point_id, payload = next(
                (point_id, payload)
                for point_id, payload in candidates.items()
                if payload.get("section_id") == str(row.aggregate_id)
                or payload.get("summary_node_id") == str(row.aggregate_id)
            )
            self._qdrant.points[(row.target_collection_name, point_id)] = SimpleNamespace(
                id=point_id,
                payload=payload,
            )
            row.status = "COMPLETED"
            row.completed_at = datetime.now(UTC)
            processed += 1
        await session.flush()
        return SimpleNamespace(processed=processed, failed=0)


async def _world(session):
    settings = _settings()
    manifest = load_demo_asset_manifest(PROJECT_ROOT)
    asset = manifest["assets"][0]
    leaf = next(
        section
        for section in asset["sections"]
        if section["section_type"] in {"ARTICLE", "PARAGRAPH"}
    )
    summary = asset["summaries"][0]
    session.add(Tenant(id=TENANT_ID, name="synthetic"))
    session.add(
        Document(
            id=uuid.UUID(asset["document_id"]),
            tenant_id=TENANT_ID,
            logical_key=asset["logical_key"],
            title=asset["title"],
            document_type=asset["document_type"],
        )
    )
    version = DocumentVersion(
        id=uuid.UUID(asset["document_version_id"]),
        tenant_id=TENANT_ID,
        document_id=uuid.UUID(asset["document_id"]),
        version_label=asset["version_label"],
        source_available_at=datetime.fromisoformat(asset["source_available_at"]),
        object_uri=asset["object_uri"],
        source_filename=asset["source_filename"],
        mime_type=asset["mime_type"],
        file_size=asset["object_size"],
        content_hash=asset["object_sha256"],
        active_parse_run_id=uuid.UUID(asset["parse_run_id"]),
        processing_status="READY",
    )
    session.add(version)
    session.add(
        ParseRun(
            id=uuid.UUID(asset["parse_run_id"]),
            tenant_id=TENANT_ID,
            document_version_id=version.id,
            generation_no=1,
            status="SUCCEEDED",
            activation_status="ACTIVE",
            parser_name=asset["parser_name"],
            parser_version=asset["parser_version"],
            config_hash=asset["parser_config_hash"],
        )
    )
    session.add(
        DocumentSection(
            id=uuid.UUID(leaf["id"]),
            tenant_id=TENANT_ID,
            document_version_id=version.id,
            parse_run_id=uuid.UUID(asset["parse_run_id"]),
            section_type=leaf["section_type"],
            ordinal=leaf["ordinal"],
            heading=leaf["heading"],
            heading_path=leaf["heading_path"],
            page_start=leaf["page_start"],
            page_end=leaf["page_end"],
            text="fixture text",
            text_hash=leaf["text_hash"],
        )
    )
    session.add(
        SummaryNode(
            id=uuid.UUID(summary["id"]),
            tenant_id=TENANT_ID,
            document_version_id=version.id,
            parse_run_id=uuid.UUID(asset["parse_run_id"]),
            summary_level=summary["summary_level"],
            summary_text="fixture summary",
            summary_hash=summary["summary_hash"],
            model_name="extractive-first-sentence-v1",
            grounding_status="VERIFIED",
        )
    )
    await session.flush()
    return version, settings, expected_demo_qdrant_points(manifest, settings), leaf, summary


async def test_repair_adopts_verified_points_when_completed_ledger_was_lost(session):
    version, settings, expected_points, leaf, summary = await _world(session)
    qdrant = _PointStore()
    for collection, points in expected_points.items():
        for point_id, payload in points.items():
            if (
                payload.get("section_id") == leaf["id"]
                or payload.get("summary_node_id") == summary["id"]
            ):
                qdrant.points[(collection, point_id)] = SimpleNamespace(
                    id=point_id, payload=payload
                )
    worker = _CrashRecoveryWorker(qdrant, expected_points)

    assert await _repair_existing_index(session, version, worker, settings) == 2
    ledgers = (await session.scalars(select(IndexOutbox))).all()
    assert len(ledgers) == 2
    assert {row.status for row in ledgers} == {"COMPLETED"}
    assert all(row.attempts >= 1 and row.completed_at is not None for row in ledgers)


async def test_repair_requeues_expired_processing_leases_and_rebuilds_points(session):
    version, settings, expected_points, leaf, summary = await _world(session)
    stale = datetime.now(UTC) - timedelta(minutes=10)
    for aggregate_type, item, collection, sparse in (
        ("SECTION", leaf, settings.chunks_collection_name, settings.sparse_encoder_version),
        ("SUMMARY", summary, settings.summaries_collection_name, None),
    ):
        content_hash = item.get("text_hash") or item["summary_hash"]
        session.add(
            IndexOutbox(
                tenant_id=TENANT_ID,
                aggregate_type=aggregate_type,
                aggregate_id=uuid.UUID(item["id"]),
                operation="UPSERT",
                content_hash=content_hash,
                target_collection_name=collection,
                embedding_version=settings.effective_embedding_version,
                sparse_encoder_version=sparse,
                status="PROCESSING",
                attempts=1,
                locked_at=stale,
            )
        )
    await session.flush()
    qdrant = _PointStore()
    worker = _CrashRecoveryWorker(qdrant, expected_points)

    assert await _repair_existing_index(session, version, worker, settings) == 2
    ledgers = (await session.scalars(select(IndexOutbox))).all()
    assert {row.status for row in ledgers} == {"COMPLETED"}
    assert len(qdrant.points) == 2


async def test_repair_does_not_steal_a_fresh_processing_lease(session):
    version, settings, expected_points, leaf, _summary = await _world(session)
    session.add(
        IndexOutbox(
            tenant_id=TENANT_ID,
            aggregate_type="SECTION",
            aggregate_id=uuid.UUID(leaf["id"]),
            operation="UPSERT",
            content_hash=leaf["text_hash"],
            target_collection_name=settings.chunks_collection_name,
            embedding_version=settings.effective_embedding_version,
            sparse_encoder_version=settings.sparse_encoder_version,
            status="PROCESSING",
            attempts=1,
            locked_at=datetime.now(UTC),
        )
    )
    await session.flush()
    worker = _CrashRecoveryWorker(_PointStore(), expected_points)

    with pytest.raises(RuntimeError, match="index repair is incomplete"):
        await _repair_existing_index(session, version, worker, settings)
