"""Risk Challenger（任务 24，文档 §8.16；v1.1 五维冲突判断）。

- 只围绕现有 Claim 和硬约束生成反向 Query，不引入新的企业、日期或案件；
- 每个高影响 Claim 最多一轮；
- 找到反证后保留双方 Evidence，不做多数投票；
- 找不到反证不等于 Claim 自动正确；
- v1.1 WP3：按“指标、期间、单位、口径、数值”五维判断冲突：
  真冲突（DATA_CONFLICT，PARTIALLY_SUPPORTED）进审计/人工裁决；
  补充材料（MISSING_MATERIAL，INSUFFICIENT_EVIDENCE）仅提示，
  不阻断流程、不送 HITL；source_claim_id 持久化可追踪。
"""

import re
import uuid

from creditlens.agents.contracts import AgentArtifact, AgentClaim
from creditlens.agents.policy_agent import _candidate_to_evidence, consume_evidence_sections
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.tools.gateway import ToolGateway

AGENT_ROLE = "challenger"

# 类别 -> 反向检索模板（确定性；LLM 版本仍须通过同一约束）
_COUNTER_TEMPLATES = {
    "ELIGIBILITY": "例外 特殊情形 不适用 另有规定",
    "FINANCIAL": "下降 恶化 不满足 低于 高于 预警",
    "CASH_FLOW": "现金流 下降 为负 回款延迟",
    "EXCEPTION": "禁止 不得突破 限制",
    "CONCENTRATION": "集中度 占比 上升 回款延迟",
}

_HIGH_IMPACT = {"HIGH", "CRITICAL"}

# 数字提取（用于矛盾判断）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
# WP3 五维：期间提取（YYYY年 / YYYY-MM-DD / YYYY-MM）与单位提取
_PERIOD_RE = re.compile(r"(20\d{2})\s*年(?:\s*(\d{1,2})\s*月)?|20\d{2}-\d{2}(?:-\d{2})?")
_UNIT_RE = re.compile(r"(万元|亿元|元|%|百分之|倍)")


def _extract_periods(text: str) -> set[str]:
    """提取文本中的期间标识（年/年月），用于期间一致性判断。"""
    periods: set[str] = set()
    for match in _PERIOD_RE.finditer(text):
        raw = match.group(0)
        if "年" in raw:
            year = match.group(1)
            month = match.group(2)
            periods.add(f"{year}-{int(month):02d}" if month else year)
        else:
            periods.add(raw[:7])  # YYYY-MM
    return periods


def _extract_units(text: str) -> set[str]:
    units = set(_UNIT_RE.findall(text))
    # “百分之”与 % 视为同一单位（口径归一）
    if "百分之" in units:
        units.discard("百分之")
        units.add("%")
    return units


def assess_conflict(claim_text: str, counter_text: str) -> tuple[bool, str]:
    """WP3 五维冲突判断（指标、期间、单位、口径、数值），确定性实现。

    返回 (is_conflict, reason)：
    - 指标维度：反证 Query 由同一 Claim 生成，指标已隐式锚定；
    - 期间维度：双方期间明确且不相交 -> 期间不同，非冲突（口径差异）；
    - 单位维度：双方单位明确且不一致 -> 单位/口径不同，非冲突；
    - 数值维度：双方都含数值且存在不同数值 -> 判定冲突；
    - 任一方无数值 -> 仅补充材料，不判冲突。
    """
    claim_numbers = set(_NUMBER_RE.findall(claim_text))
    counter_numbers = set(_NUMBER_RE.findall(counter_text))
    if not claim_numbers or not counter_numbers:
        return False, "任一方无数值，不构成数值冲突"
    if claim_numbers == counter_numbers:
        return False, "数值一致，无冲突"

    # 期间维度：期间不同 -> 口径差异而非冲突
    claim_periods = _extract_periods(claim_text)
    counter_periods = _extract_periods(counter_text)
    if claim_periods and counter_periods and not (claim_periods & counter_periods):
        return False, "期间不一致，属口径差异而非冲突"

    # 单位维度：单位不同 -> 不可直接比较，不判冲突
    claim_units = _extract_units(claim_text)
    counter_units = _extract_units(counter_text)
    if claim_units and counter_units and not (claim_units & counter_units):
        return False, "单位不一致，不可直接比较"

    return bool(claim_numbers - counter_numbers) and bool(
        counter_numbers - claim_numbers
    ), "同指标/期间/单位下数值不一致"


def _has_numeric_contradiction(claim_text: str, counter_text: str) -> bool:
    """向后兼容包装：仅返回是否冲突。"""
    return assess_conflict(claim_text, counter_text)[0]


class Challenger:
    def __init__(self, gateway: ToolGateway, challenge_all_supported: bool = True):
        self._gateway = gateway
        # MVP 案件小，对全部 SUPPORTED Claim 反证；规模化后只挑高影响
        self._challenge_all = challenge_all_supported

    async def run(
        self,
        run_id: uuid.UUID,
        task_id: str,
        trusted: TrustedRequestContext,
        provisional: list[AgentArtifact],
    ) -> AgentArtifact:
        artifact = AgentArtifact(run_id=run_id, task_id=task_id, producer=AGENT_ROLE)
        challenged = 0

        for source in provisional:
            evidence_by_id = {e.evidence_id: e for e in source.evidence}
            for claim in source.claims:
                if claim.verdict != "SUPPORTED":
                    continue
                if not self._challenge_all and claim.severity not in _HIGH_IMPACT:
                    continue
                template = _COUNTER_TEMPLATES.get(claim.category)
                if template is None:
                    continue
                counter_query = f"{claim.statement[:40]} {template}"
                result = await self._gateway.invoke(
                    AGENT_ROLE, "search_counter_evidence", trusted=trusted, query=counter_query
                )
                challenged += 1

                supporting_sections = {
                    evidence_by_id[eid].section_id
                    for eid in claim.supporting_evidence_ids
                    if eid in evidence_by_id
                }
                # WP2：消费 Packed Sections（含相邻复核与预算控制）
                packed = consume_evidence_sections(result, limit=3)
                counter = [c for c in packed if c.section_id not in supporting_sections]
                if not counter:
                    continue

                # WP3：五维冲突判断（指标/期间/单位/口径/数值）
                counter_text = " ".join(c.text[:200] for c in counter if c.text)
                is_conflict, conflict_reason = assess_conflict(claim.statement, counter_text)

                opposing_ids = []
                for candidate in counter:
                    ref = _candidate_to_evidence(candidate)
                    if all(e.evidence_id != ref.evidence_id for e in artifact.evidence):
                        artifact.evidence.append(ref)
                    opposing_ids.append(ref.evidence_id)

                if is_conflict:
                    artifact.claims.append(
                        AgentClaim(
                            category="DATA_CONFLICT",
                            statement=(
                                f"对既有结论「{claim.statement[:40]}…」发现数字不一致的反向材料，"
                                "支持与反对证据均已保留，待审计与人工裁决。"
                            ),
                            verdict="PARTIALLY_SUPPORTED",
                            severity="MEDIUM",
                            opposing_evidence_ids=opposing_ids,
                            as_of_date=trusted.as_of_date,
                            uncertainty_reason=f"五维冲突判断：{conflict_reason}",
                            recommended_follow_up=["由审查员核对正反证据的数字口径与时点"],
                            source_claim_id=claim.claim_id,
                        )
                    )
                else:
                    # WP3：非数字矛盾 = 补充材料，不是冲突：不阻断流程、不送 HITL，
                    # 仅作为审阅提示（INSUFFICIENT_EVIDENCE + uncertainty_reason）
                    artifact.claims.append(
                        AgentClaim(
                            category="MISSING_MATERIAL",
                            statement=(
                                f"对既有结论「{claim.statement[:40]}…」存在补充性材料，"
                                "未发现直接数字矛盾，建议审查员参阅。"
                            ),
                            verdict="INSUFFICIENT_EVIDENCE",
                            severity="LOW",
                            opposing_evidence_ids=opposing_ids,
                            as_of_date=trusted.as_of_date,
                            uncertainty_reason=f"补充材料（{conflict_reason}），仅供审阅参考",
                            source_claim_id=claim.claim_id,
                        )
                    )

        artifact.unresolved_issues.append({"challenged_claims": challenged})
        return artifact
