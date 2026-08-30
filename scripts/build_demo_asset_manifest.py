"""Explicit maintainer tool for refreshing the frozen synthetic asset manifest.

This is never called by bootstrap.  A manifest update is a reviewed release
change alongside the committed source texts and ingestion code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from creditlens.common.hashing import sha256_text  # noqa: E402
from creditlens.demo_manifest import (  # noqa: E402
    DEMO_ASSET_NAMESPACE,
    DEMO_ASSET_SCHEMA,
    stable_demo_id,
)
from creditlens.infrastructure.parsers.pymupdf_parser import (  # noqa: E402
    PARSER_NAME,
    PARSER_VERSION,
    PyMuPdfParser,
)
from creditlens.ingestion.chunking.structure import build_sections  # noqa: E402
from creditlens.ingestion.summaries import first_sentence  # noqa: E402
from seed_synthetic_data import (  # noqa: E402
    BORROWER_ID,
    BORROWER_ID_002,
    BORROWER_ID_003,
    CASE_ID,
    CASE_ID_002,
    CASE_ID_003,
    CORPUS_CASE_001,
    CORPUS_CASE_002,
    CORPUS_CASE_003,
    SYNTH_DIR,
    TENANT_ID,
    text_to_pdf,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation/datasets/demo_asset_manifest_v1.json",
    )
    return parser.parse_args()


def _iso(value):
    return value.isoformat() if value is not None else None


def _build() -> dict:
    bindings: dict[tuple[str, str], list[dict]] = {}
    specs: dict[tuple[str, str], dict] = {}
    document_titles: dict[str, str] = {}
    for case_id, borrower_id, corpus in (
        (CASE_ID, BORROWER_ID, CORPUS_CASE_001),
        (CASE_ID_002, BORROWER_ID_002, CORPUS_CASE_002),
        (CASE_ID_003, BORROWER_ID_003, CORPUS_CASE_003),
    ):
        for spec in corpus:
            key = (spec["logical_key"], spec["version_label"])
            specs[key] = spec
            # ``Document.title`` belongs to the logical document, not to an
            # individual version.  Seed ingests corpora in this same order and
            # keeps the title from the first version it sees (the current one).
            # Freeze that canonical title for every version so generated
            # heading paths match the production ingestion path exactly.
            document_titles.setdefault(spec["logical_key"], spec["title"])
            bindings.setdefault(key, []).append(
                {
                    "case_id": str(case_id),
                    "borrower_entity_id": str(borrower_id),
                    "document_role": spec["document_role"],
                    "is_required": False,
                }
            )

    assets = []
    parser_config_hash = sha256_text(f"{PARSER_NAME}:{PARSER_VERSION}:structure-chunker-v1")
    for (logical_key, version_label), spec in sorted(specs.items()):
        identity = f"{logical_key}@{version_label}"
        document_id = stable_demo_id("document", logical_key)
        version_id = stable_demo_id("version", identity)
        parse_id = stable_demo_id("parse", identity)
        text_path = SYNTH_DIR / spec["txt"]
        pdf_bytes = text_to_pdf(text_path, text_path.with_suffix(".pdf"))
        parsed = PyMuPdfParser().parse(pdf_bytes)
        canonical_title = document_titles[logical_key]
        drafts = build_sections(parsed, canonical_title, spec["document_type"])
        old_to_new = {
            draft.id: stable_demo_id(
                "section",
                f"{identity}:{draft.ordinal}:{draft.section_type}:{draft.text_hash}",
            )
            for draft in drafts
        }
        sections = []
        for draft in drafts:
            sections.append(
                {
                    "id": str(old_to_new[draft.id]),
                    "section_type": draft.section_type,
                    "ordinal": draft.ordinal,
                    "parent_section_id": (
                        str(old_to_new[draft.parent_id]) if draft.parent_id else None
                    ),
                    "previous_section_id": (
                        str(old_to_new[draft.previous_id]) if draft.previous_id else None
                    ),
                    "next_section_id": (str(old_to_new[draft.next_id]) if draft.next_id else None),
                    "heading": draft.heading,
                    "heading_path": draft.heading_path,
                    "page_start": draft.page_start,
                    "page_end": draft.page_end,
                    "text_hash": draft.text_hash,
                }
            )

        root = next((item for item in drafts if item.section_type == "DOCUMENT"), None)
        chapters = [item for item in drafts if item.section_type == "CHAPTER"]
        leaves = [item for item in drafts if item.section_type in {"ARTICLE", "PARAGRAPH"}]
        summary_specs: list[tuple[str, str, list, int | None]] = []
        card = "\n".join(
            [
                f"文档主题：{root.heading if root else '文档'}",
                "主要章节：" + "；".join(chapter.heading or "" for chapter in chapters),
            ]
        )
        summary_specs.append(("DOCUMENT", card, ([root] if root else []) + chapters, None))
        for chapter in chapters:
            children = [leaf for leaf in leaves if leaf.parent_id == chapter.id]
            if not children:
                continue
            summary = "\n".join(
                [f"章节：{chapter.heading}"]
                + [f"{leaf.heading or ''}：{first_sentence(leaf.text)}" for leaf in children]
            )
            summary_specs.append(("CHAPTER", summary, children, 0))
        summaries = []
        summary_ids = [
            stable_demo_id(
                "summary",
                f"{identity}:{index}:{level}:{sha256_text(summary)}",
            )
            for index, (level, summary, _sources, _parent) in enumerate(summary_specs)
        ]
        for index, (level, summary, sources, parent_index) in enumerate(summary_specs):
            summaries.append(
                {
                    "id": str(summary_ids[index]),
                    "summary_level": level,
                    "parent_summary_id": (
                        str(summary_ids[parent_index]) if parent_index is not None else None
                    ),
                    "summary_hash": sha256_text(summary),
                    "source_section_ids": [str(old_to_new[source.id]) for source in sources],
                }
            )
        assets.append(
            {
                "logical_key": logical_key,
                "version_label": version_label,
                "source_text_file": spec["txt"],
                "source_text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
                "title": canonical_title,
                "document_type": spec["document_type"],
                "confidentiality": "INTERNAL",
                "document_id": str(document_id),
                "document_version_id": str(version_id),
                "parse_run_id": str(parse_id),
                "parser_name": PARSER_NAME,
                "parser_version": PARSER_VERSION,
                "parser_config_hash": parser_config_hash,
                "source_filename": spec["txt"].replace(".txt", ".pdf"),
                "mime_type": "application/pdf",
                "valid_from": _iso(spec["valid_from"]),
                "valid_to": _iso(spec["valid_to"]),
                "source_available_at": _iso(spec["source_available_at"]),
                "object_uri": (
                    f"s3://creditlens-raw/{TENANT_ID}/{document_id}/{version_id}/original.pdf"
                ),
                "object_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "object_size": len(pdf_bytes),
                "page_count": parsed.metadata.get("page_count"),
                "bindings": sorted(
                    bindings[(logical_key, version_label)], key=lambda x: x["case_id"]
                ),
                "sections": sections,
                "summaries": summaries,
            }
        )
    return {
        "schema_version": DEMO_ASSET_SCHEMA,
        "identity_namespace": str(DEMO_ASSET_NAMESPACE),
        "tenant_id": str(TENANT_ID),
        "raw_bucket": "creditlens-raw",
        "assets": assets,
    }


if __name__ == "__main__":
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_build(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
