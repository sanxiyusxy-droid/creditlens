"""Risk 阈值版本化配置（WP3）。

阈值不再硬编码在 Agent 逻辑中，而是作为版本化配置管理：
- 每次调整阈值必须升版本（risk-thresholds-v1 -> v2 ...）；
- Run 的 model_manifest 记录实际使用的配置版本，保证可追溯；
- 评测/审计按配置版本对齐口径。
"""

RISK_THRESHOLD_CONFIGS: dict[str, dict] = {
    "risk-thresholds-v1": {
        "debt_ratio": {"warn_above": 70.0, "unit": "%", "display": "资产负债率"},
        "current_ratio": {"warn_below": 1.0, "unit": "", "display": "流动比率"},
    },
}

# 当前默认版本（冻结配置时固定，写入 Manifest）
DEFAULT_RISK_THRESHOLD_VERSION = "risk-thresholds-v1"


def load_risk_thresholds(version: str | None = None) -> tuple[str, dict]:
    """返回 (version, thresholds)。未知版本显式报错，不静默回退。"""
    resolved = version or DEFAULT_RISK_THRESHOLD_VERSION
    if resolved not in RISK_THRESHOLD_CONFIGS:
        raise KeyError(f"未知风险阈值配置版本: {resolved}")
    return resolved, RISK_THRESHOLD_CONFIGS[resolved]
