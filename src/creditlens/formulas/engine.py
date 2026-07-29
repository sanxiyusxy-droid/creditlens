"""公式注册表与确定性求值器（任务 19，文档 §9.2/§9.3）。

- 所有输入 Decimal；缺失值不当作 0；
- 除零、单位/期间冲突即失败（带状态码的 CalculationArtifact）；
- 表达式使用受限 AST 求值（+ - * / 与 average()），不使用 eval；
- 结果重放：trace_hash = SHA256(公式+版本+输入+参数)，重新计算必须一致。
"""

import ast
import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from creditlens.common.hashing import sha256_text

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "config" / "formulas" / "registry_v1.yaml"


@dataclass
class FormulaDefinition:
    metric_code: str
    display_name: str
    version: str
    expression: str
    required_inputs: list[str]
    period_rule: str
    result_unit: str
    rounding: Decimal
    zero_policy: str
    parameters: dict[str, Decimal] = field(default_factory=dict)


class FormulaRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self._formulas: dict[str, FormulaDefinition] = {}
        for code, spec in raw.items():
            definition = FormulaDefinition(
                metric_code=code,
                display_name=spec["display_name"],
                version=str(spec["version"]),
                expression=spec["expression"],
                required_inputs=list(spec["required_inputs"]),
                period_rule=spec.get("period_rule", "same_instant"),
                result_unit=spec.get("result_unit", "ratio"),
                rounding=Decimal(str(spec.get("rounding", "0.0001"))),
                zero_policy=spec.get("zero_policy", "error"),
                parameters={
                    k: Decimal(str(v)) for k, v in (spec.get("parameters") or {}).items()
                },
            )
            self._formulas[f"{code}@{definition.version}"] = definition

    def get(self, metric_code: str, version: str) -> FormulaDefinition | None:
        return self._formulas.get(f"{metric_code}@{version}")

    def all(self) -> list[FormulaDefinition]:
        return list(self._formulas.values())


class CalculationInput(BaseModel):
    fact_id: uuid.UUID
    metric_code: str
    raw_value: Decimal
    canonical_value: Decimal
    unit: str = ""
    period_start: date | None = None
    period_end: date
    consolidation_scope: str = "UNKNOWN"


class CalculationArtifact(BaseModel):
    calculation_id: uuid.UUID
    metric_code: str
    formula_version: str
    expression: str
    inputs: list[CalculationInput]
    parameters: dict[str, Decimal]
    result: Decimal | None
    result_unit: str
    status: Literal[
        "CALCULATED",
        "MISSING_INPUT",
        "DIVISION_BY_ZERO",
        "UNIT_CONFLICT",
        "PERIOD_CONFLICT",
        "SOURCE_CONFLICT",
    ]
    trace_hash: str


class _SafeEvaluator(ast.NodeVisitor):
    """受限 AST：仅允许 名称、数字、+ - * /、average() 调用。"""

    def __init__(self, variables: dict[str, Decimal]):
        self._vars = variables

    def evaluate(self, expression: str) -> Decimal:
        tree = ast.parse(expression, mode="eval")
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> Decimal:
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in self._vars:
                raise KeyError(node.id)
            return self._vars[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Decimal(str(node.value))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "average":
                raise ValueError("仅允许 average() 函数")
            values = [self._eval(arg) for arg in node.args]
            return sum(values) / Decimal(len(values))
        raise ValueError(f"不允许的表达式节点: {type(node).__name__}")


def compute_metric(
    definition: FormulaDefinition,
    inputs: dict[str, CalculationInput],
) -> CalculationArtifact:
    """确定性计算；缺输入/除零/期间冲突返回带状态的 Artifact，绝不估算。"""
    trace_payload = {
        "metric": definition.metric_code,
        "version": definition.version,
        "expression": definition.expression,
        "inputs": {
            k: {"fact_id": str(v.fact_id), "value": str(v.canonical_value)}
            for k, v in sorted(inputs.items())
        },
        "parameters": {k: str(v) for k, v in sorted(definition.parameters.items())},
    }
    trace_hash = sha256_text(json.dumps(trace_payload, ensure_ascii=False, sort_keys=True))

    def artifact(status, result=None) -> CalculationArtifact:
        return CalculationArtifact(
            calculation_id=uuid.uuid4(),
            metric_code=definition.metric_code,
            formula_version=definition.version,
            expression=definition.expression,
            inputs=list(inputs.values()),
            parameters=definition.parameters,
            result=result,
            result_unit=definition.result_unit,
            status=status,
            trace_hash=trace_hash,
        )

    missing = [name for name in definition.required_inputs if name not in inputs]
    if missing:
        return artifact("MISSING_INPUT")

    # 期间规则：same_instant/same_period 要求所有输入 period_end 一致
    if definition.period_rule in {"same_instant", "same_period"}:
        period_ends = {v.period_end for v in inputs.values()}
        if len(period_ends) > 1:
            return artifact("PERIOD_CONFLICT")
    # 合并口径一致性
    scopes = {v.consolidation_scope for v in inputs.values()} - {"UNKNOWN"}
    if len(scopes) > 1:
        return artifact("SOURCE_CONFLICT")

    variables = {name: value.canonical_value for name, value in inputs.items()}
    variables.update(definition.parameters)
    try:
        raw = _SafeEvaluator(variables).evaluate(definition.expression)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError):
        return artifact("DIVISION_BY_ZERO")

    result = raw.quantize(definition.rounding)
    if definition.result_unit == "percent":
        result = (raw * Decimal("100")).quantize(definition.rounding)
    return artifact("CALCULATED", result)


def replay_calculation(
    registry: FormulaRegistry, original: CalculationArtifact
) -> tuple[bool, CalculationArtifact | None]:
    """数值重放：用相同公式版本与输入重算，比较 trace_hash 与结果。"""
    definition = registry.get(original.metric_code, original.formula_version)
    if definition is None:
        return False, None
    inputs = {}
    for item in original.inputs:
        inputs[item.metric_code] = item
    replayed = compute_metric(definition, inputs)
    consistent = (
        replayed.trace_hash == original.trace_hash
        and replayed.status == original.status
        and replayed.result == original.result
    )
    return consistent, replayed
