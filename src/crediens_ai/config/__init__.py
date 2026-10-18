"""Configuration package for platform settings.

This package intentionally re-exports the legacy global configuration values
used throughout the platform while also exposing the newer config-driven
module registry for industry-specific orchestration.
"""

from .platform import DEFAULT_MODULE_CONFIGS, RiskModuleConfig, get_module_config

DEFAULT_INDUSTRIES = {
    "IT": {
        "risk_types": ["PROJECT", "CYBER", "OPERATIONAL"],
        "default_features": [
            "team_size",
            "bug_count",
            "pending_tasks",
            "test_coverage",
            "technical_debt",
        ],
    },
    "MANUFACTURING": {
        "risk_types": ["EQUIPMENT", "SUPPLY_CHAIN"],
        "default_features": [
            "machine_age",
            "vibration",
            "temperature",
            "maintenance_history",
            "operating_hours",
        ],
    },
    "BANKING": {
        "risk_types": ["CREDIT", "FRAUD"],
        "default_features": [
            "income",
            "loan_amount",
            "credit_history",
            "existing_debt",
        ],
    },
}

DEFAULT_POLICY_BANDS = {
    "LOW": (0.0, 0.20),
    "MEDIUM": (0.21, 0.50),
    "HIGH": (0.51, 1.00),
}

DEFAULT_MODEL_METADATA = {
    "project-risk-v1": {
        "model_family": "Late-fusion ensemble",
        "description": "Reference implementation for project risk scoring",
    },
    "operational-risk-v1": {
        "model_family": "Heuristic risk scorer",
        "description": "Reference implementation for IT operational risk scoring",
    },
    "equipment-risk-v1": {
        "model_family": "Gradient-boosted risk model",
        "description": "Reference implementation for manufacturing equipment risk",
    },
    "credit-risk-v1": {
        "model_family": "Late-fusion ensemble",
        "description": "Reference implementation for credit risk",
    },
}

__all__ = [
    "DEFAULT_INDUSTRIES",
    "DEFAULT_POLICY_BANDS",
    "DEFAULT_MODEL_METADATA",
    "RiskModuleConfig",
    "DEFAULT_MODULE_CONFIGS",
    "get_module_config",
]
