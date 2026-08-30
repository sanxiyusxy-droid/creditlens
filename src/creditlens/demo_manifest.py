"""Frozen identity helpers for the local synthetic demo corpus.

The manifest is a release artifact, not a snapshot exported from the current
database.  Seed and preflight both consume it so a compromised/stale database
cannot define what "ready" means.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from creditlens.common.ids import deterministic_point_id

DEMO_ASSET_NAMESPACE = uuid.UUID("1b5a9d87-8fe3-5e45-8f50-ec2f70b6f8c1")
DEMO_ASSET_MANIFEST = Path("evaluation/datasets/demo_asset_manifest_v1.json")
DEMO_ASSET_SCHEMA = "creditlens.demo-assets.v1"


def stable_demo_id(kind: str, identity: str) -> uuid.UUID:
    """Return the release-stable UUID for one manifest-owned object."""

    return uuid.uuid5(DEMO_ASSET_NAMESPACE, f"{kind}:{identity}")


def load_demo_asset_manifest(project_root: Path) -> dict[str, Any]:
    """Load and minimally validate the committed manifest, fail closed."""

    path = project_root / DEMO_ASSET_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("DEMO_ASSET_MANIFEST_UNREADABLE") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != DEMO_ASSET_SCHEMA:
        raise ValueError("DEMO_ASSET_MANIFEST_SCHEMA_INVALID")
    assets = payload.get("assets")
    if not isinstance(assets, list) or len(assets) != 8:
        raise ValueError("DEMO_ASSET_MANIFEST_CARDINALITY_INVALID")
    keys = [(item.get("logical_key"), item.get("version_label")) for item in assets]
    if len(set(keys)) != len(keys):
        raise ValueError("DEMO_ASSET_MANIFEST_DUPLICATE_IDENTITY")
    return payload


def expected_demo_qdrant_points(
    manifest: dict[str, Any], settings: Any
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the exact point ID/payload set from the frozen release manifest.

    Only the explicitly selected runtime embedding version and physical
    collection names vary.  No expected identity is read from PostgreSQL.
    """

    result: dict[str, dict[str, dict[str, Any]]] = {
        settings.chunks_collection_name: {},
        settings.summaries_collection_name: {},
    }
    tenant_id = manifest["tenant_id"]
    embedding_version = settings.effective_embedding_version
    for asset in manifest["assets"]:
        entity_ids = sorted({binding["borrower_entity_id"] for binding in asset["bindings"]})
        for section in asset["sections"]:
            if section["section_type"] not in {"ARTICLE", "PARAGRAPH"}:
                continue
            point_id = str(
                deterministic_point_id(
                    uuid.UUID(section["id"]),
                    section["text_hash"],
                    embedding_version,
                )
            )
            result[settings.chunks_collection_name][point_id] = {
                "tenant_id": tenant_id,
                "point_type": "original_section",
                "document_id": asset["document_id"],
                "document_version_id": asset["document_version_id"],
                "parse_run_id": asset["parse_run_id"],
                "section_id": section["id"],
                "parent_section_id": section["parent_section_id"],
                "entity_ids": entity_ids,
                "document_type": asset["document_type"],
                "product_codes": ["working_capital"],
                "valid_from": asset["valid_from"],
                "valid_to": asset["valid_to"],
                "source_available_at": asset["source_available_at"],
                "tombstoned": False,
                "confidentiality": asset["confidentiality"],
                "acl_tags": ["credit_analyst", "risk_reviewer"],
                "quality_status": "PASS",
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "heading_path": section["heading_path"],
                "text_hash": section["text_hash"],
                "embedding_version": embedding_version,
            }
        for summary in asset["summaries"]:
            point_id = str(
                deterministic_point_id(
                    uuid.UUID(summary["id"]),
                    summary["summary_hash"],
                    embedding_version,
                )
            )
            result[settings.summaries_collection_name][point_id] = {
                "tenant_id": tenant_id,
                "point_type": "summary_node",
                "summary_node_id": summary["id"],
                "summary_level": summary["summary_level"],
                "document_id": asset["document_id"],
                "document_version_id": asset["document_version_id"],
                "parse_run_id": asset["parse_run_id"],
                "document_type": asset["document_type"],
                "valid_from": asset["valid_from"],
                "valid_to": asset["valid_to"],
                "source_available_at": asset["source_available_at"],
                "tombstoned": False,
                "confidentiality": asset["confidentiality"],
                "quality_status": "PASS",
                "text_hash": summary["summary_hash"],
                "embedding_version": embedding_version,
            }
    return result
