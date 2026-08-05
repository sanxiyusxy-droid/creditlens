"""公式引擎与 Contract 单元测试（任务 19/20）。"""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from creditlens.agents.contracts import (
    AgentArtifact,
    AgentClaim,
    AgentEvidenceRef,
    validate_artifact_contract,
)
from creditlens.formulas.engine import (
    CalculationInput,
    FormulaRegistry,
    compute_metric,
    replay_calculation,
)

AS_OF = date(2026, 6, 30)


def _input(metric_code: str, value: str, period_end=date(2025, 12, 31)) -> CalculationInput:
    return CalculationInput(
        fact_id=uuid.uuid4(),
        metric_code=metric_code,
        raw_value=Decimal(value),
        canonical_value=Decimal(value),
        period_end=period_end,
    )


class TestFormulaEngine:
    def setup_method(self):
        self.registry = FormulaRegistry()

    def test_debt_ratio_calculated_as_percent(self):
        definition = self.registry.get("debt_ratio", "1.0")
        calc = compute_metric(
            definition,
            {
                "total_liabilities": _input("total_liabilities", "6500"),
                "total_assets": _input("total_assets", "10000"),
            },
        )
        assert calc.status == "CALCULATED"
        assert calc.result == Decimal("65.00")

    def test_missing_input_never_estimated(self):
        definition = self.registry.get("debt_ratio", "1.0")
        calc = compute_metric(definition, {"total_assets": _input("total_assets", "10000")})
        assert calc.status == "MISSING_INPUT"
        assert calc.result is None

    def test_division_by_zero(self):
        definition = self.registry.get("current_ratio", "1.0")
        calc = compute_metric(
            definition,
            {
                "current_assets": _input("current_assets", "100"),
                "current_liabilities": _input("current_liabilities", "0"),
            },
        )
        assert calc.status == "DIVISION_BY_ZERO"

    def test_period_conflict(self):
        definition = self.registry.get("debt_ratio", "1.0")
        calc = compute_metric(
            definition,
            {
                "total_liabilities": _input("total_liabilities", "1", date(2025, 12, 31)),
                "total_assets": _input("total_assets", "2", date(2024, 12, 31)),
            },
        )
        assert calc.status == "PERIOD_CONFLICT"

    def test_average_expression(self):
        definition = self.registry.get("receivable_days", "1.0")
        calc = compute_metric(
            definition,
            {
                "accounts_receivable_begin": _input("accounts_receivable_begin", "4700"),
                "accounts_receivable_end": _input("accounts_receivable_end", "9100"),
                "revenue": _input("revenue", "50000"),
            },
        )
        assert calc.status == "CALCULATED"
        # average(4700, 9100)/50000*365 = 50.37
        assert calc.result == Decimal("50.37")

    def test_replay_consistency(self):
        definition = self.registry.get("debt_ratio", "1.0")
        inputs = {
            "total_liabilities": _input("total_liabilities", "6500"),
            "total_assets": _input("total_assets", "10000"),
        }
        original = compute_metric(definition, inputs)
        consistent, replayed = replay_calculation(self.registry, original)
        assert consistent
        assert replayed.result == original.result


def _evidence(evidence_type="DOCUMENT_SPAN") -> AgentEvidenceRef:
    return AgentEvidenceRef(
        evidence_id=uuid.uuid4(),
        evidence_type=evidence_type,
        source_id=uuid.uuid4(),
        content_hash="x" * 64,
        source_available_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestArtifactContract:
    def _artifact(self, claims, evidence=()):
        return AgentArtifact(
            run_id=uuid.uuid4(),
            task_id="t",
            producer="test",
            claims=claims,
            evidence=list(evidence),
        )

    def test_supported_without_evidence_rejected(self):
        claim = AgentClaim(
            category="ELIGIBILITY", statement="满足准入", verdict="SUPPORTED", as_of_date=AS_OF
        )
        result = validate_artifact_contract(self._artifact([claim]), AS_OF)
        assert any("SUPPORTED_WITHOUT_EVIDENCE" in v for v in result.violations)

    def test_unknown_evidence_id_rejected(self):
        claim = AgentClaim(
            category="ELIGIBILITY",
            statement="满足准入",
            verdict="SUPPORTED",
            supporting_evidence_ids=[uuid.uuid4()],  # 伪造 ID
            as_of_date=AS_OF,
        )
        result = validate_artifact_contract(self._artifact([claim]), AS_OF)
        assert any("UNKNOWN_EVIDENCE" in v for v in result.violations)

    def test_numeric_financial_claim_requires_calculation(self):
        evidence = _evidence()
        claim = AgentClaim(
            category="FINANCIAL",
            statement="资产负债率为 65%",
            verdict="SUPPORTED",
            supporting_evidence_ids=[evidence.evidence_id],
            as_of_date=AS_OF,
        )
        result = validate_artifact_contract(self._artifact([claim], [evidence]), AS_OF)
        assert any("NUMERIC_CLAIM_WITHOUT_FACT" in v for v in result.violations)

    def test_forbidden_determination(self):
        claim = AgentClaim(
            category="ELIGIBILITY",
            statement="建议拒贷",
            verdict="INSUFFICIENT_EVIDENCE",
            uncertainty_reason="x",
            as_of_date=AS_OF,
        )
        result = validate_artifact_contract(self._artifact([claim]), AS_OF)
        assert any("FORBIDDEN_DETERMINATION" in v for v in result.violations)

    def test_valid_artifact_passes(self):
        evidence = _evidence()
        claim = AgentClaim(
            category="ELIGIBILITY",
            statement="适用条款为第六条",
            verdict="SUPPORTED",
            supporting_evidence_ids=[evidence.evidence_id],
            as_of_date=AS_OF,
        )
        result = validate_artifact_contract(self._artifact([claim], [evidence]), AS_OF)
        assert result.ok, result.violations
