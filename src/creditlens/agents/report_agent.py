"""Report Agent（v1.1，文档 §10.2 扩展）。

从 Supervisor 内部 JSON 渲染拆为独立 Agent：
- 只读取 AUDITED / HUMAN_APPROVED Claim；
- 不允许重新检索或增加新事实；
- 输出结构化 ReportContent（claims、引用、公式、缺失材料、免责声明）；
- 支持 Markdown 渲染和版本 Diff；
- 持久化仍写 report_versions 表，content_hash 不变。
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.common.hashing import sha256_text
from creditlens.infrastructure.postgres.artifact_integrity import (
    validate_claim_records_against_artifacts,
)
from creditlens.infrastructure.postgres.models import (
    ArtifactRecord,
    ClaimRecord,
    EvidenceRecord,
    ReportVersion,
    ReviewRun,
)

AGENT_ROLE = "report_generator"

DISCLAIMER = "本报告为授信预审辅助草稿，不构成授信批准或拒绝决定。"


class ReportClaimEntry(BaseModel):
    """报告中单条 Claim 的结构化表示。"""

    claim_id: str
    category: str
    statement: str
    verdict: str
    review_status: str
    evidence_refs: list[str] = Field(default_factory=list)
    opposing_evidence_refs: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    # WP3：真实 Evidence locator（支持回原文）与反证来源追踪
    evidence_locators: list[dict] = Field(default_factory=list)
    opposing_evidence_locators: list[dict] = Field(default_factory=list)
    source_claim_id: str | None = None


class ReportContent(BaseModel):
    """Report Agent 输出结构。"""

    run_id: str
    as_of_date: str
    claims: list[ReportClaimEntry]
    excluded_claims: int = 0
    missing_materials: list[str] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    degraded: bool = False
    degraded_agents: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


@dataclass
class DiffEntry:
    """报告版本 Diff 条目。"""

    action: str  # ADDED | REMOVED | CHANGED
    claim_id: str
    detail: str = ""


class ReportAgent:
    """独立 Report Agent：只读已通过审计/人工批准的 Claim，生成结构化报告。"""

    async def generate(self, session: AsyncSession, run: ReviewRun) -> ReportContent:
        """从 ClaimRecord 生成报告内容（不检索、不新增事实）。"""
        claims = (
            await session.scalars(select(ClaimRecord).where(ClaimRecord.run_id == run.id))
        ).all()

        # WP3：构建 evidence_id -> 真实 locator 映射，报告可直接回原文
        # P1：优先取独立落库的 EvidenceRecord（含 parse_run_id，可证明引用的是
        # Snapshot 冻结的解析批次）；Artifact payload 作为兜底来源
        artifacts = (
            await session.scalars(select(ArtifactRecord).where(ArtifactRecord.run_id == run.id))
        ).all()
        # Claims are mutable only in their workflow review_status. Before any
        # report materialization, bind every other field back to the append-only
        # Artifact payload (and its hash when present).
        validate_claim_records_against_artifacts(
            run=run,
            artifacts=artifacts,
            claims=claims,
        )
        locator_by_evidence: dict[str, dict] = {}
        for artifact in artifacts:
            for evidence in (artifact.payload or {}).get("evidence", []):
                locator_by_evidence[str(evidence.get("evidence_id"))] = {
                    "evidence_type": evidence.get("evidence_type"),
                    "document_version_id": evidence.get("document_version_id"),
                    "section_id": evidence.get("section_id"),
                    "parse_run_id": evidence.get("parse_run_id"),
                    "page_number": evidence.get("page_number"),
                    "content_hash": evidence.get("content_hash"),
                }
        evidence_rows = (
            await session.scalars(select(EvidenceRecord).where(EvidenceRecord.run_id == run.id))
        ).all()
        for row in evidence_rows:
            locator_by_evidence[str(row.evidence_key)] = {
                "evidence_type": row.evidence_type,
                "document_version_id": str(row.document_version_id)
                if row.document_version_id
                else None,
                "section_id": str(row.section_id) if row.section_id else None,
                "parse_run_id": (row.locator or {}).get("parse_run_id"),
                "page_number": row.page_number,
                "content_hash": row.content_hash,
            }

        accepted = [c for c in claims if c.review_status in {"AUDITED", "HUMAN_APPROVED"}]
        insufficient = [c for c in claims if c.verdict == "INSUFFICIENT_EVIDENCE"]
        manifest = run.model_manifest or {}
        degraded_agents = list(
            dict.fromkeys(str(item) for item in manifest.get("degraded_agents", []))
        )
        degraded = bool(manifest.get("degraded") or degraded_agents)

        entries: list[ReportClaimEntry] = []
        references: list[dict] = []
        for c in accepted:
            payload = c.payload or {}
            evidence_ids = payload.get("supporting_evidence_ids", [])
            opposing_evidence_ids = payload.get("opposing_evidence_ids", [])
            calc_ids = payload.get("calculation_ids", [])
            locators = [
                locator_by_evidence[eid] for eid in evidence_ids if eid in locator_by_evidence
            ]
            opposing_locators = [
                locator_by_evidence[eid]
                for eid in opposing_evidence_ids
                if eid in locator_by_evidence
            ]
            references.extend(
                {"claim_id": str(c.id), "polarity": "SUPPORTING", **loc} for loc in locators
            )
            references.extend(
                {"claim_id": str(c.id), "polarity": "OPPOSING", **loc} for loc in opposing_locators
            )
            entries.append(
                ReportClaimEntry(
                    claim_id=str(c.id),
                    category=c.category,
                    statement=c.statement,
                    verdict=c.verdict,
                    review_status=c.review_status,
                    evidence_refs=evidence_ids,
                    opposing_evidence_refs=opposing_evidence_ids,
                    calculation_ids=calc_ids,
                    evidence_locators=locators,
                    opposing_evidence_locators=opposing_locators,
                    source_claim_id=payload.get("source_claim_id"),
                )
            )

        missing = [f"{c.statement}（{c.uncertainty_reason or '材料缺失'}）" for c in insufficient]

        return ReportContent(
            run_id=str(run.id),
            as_of_date=run.as_of_date.isoformat() if run.as_of_date else "",
            claims=entries,
            excluded_claims=len(claims) - len(accepted),
            missing_materials=missing,
            references=references,
            degraded=degraded,
            degraded_agents=degraded_agents,
            disclaimer=DISCLAIMER,
        )

    async def persist(
        self,
        session: AsyncSession,
        run: ReviewRun,
        content: ReportContent,
        status: str = "VERIFIED_DRAFT",
    ) -> ReportVersion:
        """持久化报告版本（content_hash 用于完整性校验，不等同于 WORM 存证）。

        WP3：状态区分——自动链路生成 VERIFIED_DRAFT；人工批准后
        由调用方显式传 APPROVED_DRAFT。两者都不代表真实授信批准。"""
        from sqlalchemy import func

        last_no = await session.scalar(
            select(func.max(ReportVersion.version_no)).where(ReportVersion.run_id == run.id)
        )
        content_dict = content.model_dump(mode="json")
        report = ReportVersion(
            tenant_id=run.tenant_id,
            case_id=run.case_id,
            run_id=run.id,
            version_no=(last_no or 0) + 1,
            status=status,
            content_json=content_dict,
            content_hash=sha256_text(json.dumps(content_dict, ensure_ascii=False, sort_keys=True)),
        )
        session.add(report)
        await session.flush()
        return report

    def render_markdown(self, content: ReportContent) -> str:
        """渲染 Markdown 格式报告。"""
        lines = [
            "# 授信预审报告（草稿）",
            "",
            f"- Run ID: {content.run_id}",
            f"- 审查时点: {content.as_of_date}",
            f"- 通过 Claim 数: {len(content.claims)}",
            f"- 排除 Claim 数: {content.excluded_claims}",
            f"- 覆盖状态: {'DEGRADED（部分审查覆盖缺失）' if content.degraded else 'COMPLETE'}",
            "",
            "## 审查结论",
            "",
        ]
        if content.degraded_agents:
            lines.insert(7, f"- 缺失/降级 Agent: {', '.join(content.degraded_agents)}")
        for entry in content.claims:
            lines.append(f"### [{entry.category}] {entry.verdict}")
            lines.append("")
            lines.append(entry.statement)
            lines.append("")
            lines.append(f"- 复核状态: {entry.review_status}")
            if entry.evidence_refs:
                lines.append(f"- 支持证据数: {len(entry.evidence_refs)}")
            if entry.opposing_evidence_refs:
                lines.append(f"- 反证数: {len(entry.opposing_evidence_refs)}")
            if entry.calculation_ids:
                lines.append(f"- 计算痕迹: {len(entry.calculation_ids)}")
            lines.append("")

        if content.missing_materials:
            lines.append("## 缺失材料")
            lines.append("")
            for item in content.missing_materials:
                lines.append(f"- {item}")
            lines.append("")

        lines.append("---")
        lines.append(f"*{content.disclaimer}*")
        return "\n".join(lines)

    @staticmethod
    def diff_reports(old: ReportContent, new: ReportContent) -> list[DiffEntry]:
        """版本 Diff：比较两次报告内容的 Claim 变化。"""
        old_map = {c.claim_id: c for c in old.claims}
        new_map = {c.claim_id: c for c in new.claims}
        diffs: list[DiffEntry] = []

        for claim_id, entry in new_map.items():
            if claim_id not in old_map:
                diffs.append(
                    DiffEntry(action="ADDED", claim_id=claim_id, detail=entry.statement[:80])
                )
            elif old_map[claim_id].statement != entry.statement:
                diffs.append(
                    DiffEntry(action="CHANGED", claim_id=claim_id, detail=entry.statement[:80])
                )

        for claim_id in old_map:
            if claim_id not in new_map:
                diffs.append(
                    DiffEntry(
                        action="REMOVED", claim_id=claim_id, detail=old_map[claim_id].statement[:80]
                    )
                )

        return diffs
