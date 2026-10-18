from __future__ import annotations

from typing import Any, Dict, List


class IndustryRegistry:
    """Registry for domain-specific risk modules and policies."""

    _MODULES: Dict[str, Dict[str, Any]] = {
        "IT": {
            "PROJECT": {
                "features": ["team_size", "bug_count", "critical_bugs", "pending_tasks", "test_coverage", "technical_debt"],
                "default_model": "project-risk-v1",
                "policy": {"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
            },
            "CYBER": {
                "features": ["failed_logins", "vulnerabilities", "critical_vulnerabilities", "patch_age", "mfa_coverage"],
                "default_model": "project-risk-v1",
                "policy": {"LOW": (0.0, 0.25), "MEDIUM": (0.25, 0.55), "HIGH": (0.55, 1.0)},
            },
            "OPERATIONAL": {
                "features": ["code_complexity", "bug_density", "open_issues", "sprint_velocity", "technical_debt", "test_coverage", "failed_deployments"],
                "default_model": "operational-risk-v1",
                "policy": {"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
            },
        },
        "MANUFACTURING": {
            "EQUIPMENT": {
                "features": ["machine_age", "vibration", "temperature", "maintenance_history", "operating_hours"],
                "default_model": "equipment-risk-v1",
                "policy": {"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
            },
            "SUPPLY_CHAIN": {
                "features": ["supplier_reliability", "delivery_delay", "inventory_level", "supplier_concentration"],
                "default_model": "equipment-risk-v1",
                "policy": {"LOW": (0.0, 0.25), "MEDIUM": (0.25, 0.55), "HIGH": (0.55, 1.0)},
            },
        },
        "BANKING": {
            "CREDIT": {
                "features": ["income", "loan_amount", "credit_history", "existing_debt", "employment_years"],
                "default_model": "credit-risk-v1",
                "policy": {"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
            },
            "FRAUD": {
                "features": ["transaction_amount", "anomaly_score", "velocity", "account_age"],
                "default_model": "credit-risk-v1",
                "policy": {"LOW": (0.0, 0.15), "MEDIUM": (0.15, 0.45), "HIGH": (0.45, 1.0)},
            },
        },
    }

    @classmethod
    def get_supported_industries(cls) -> List[str]:
        return sorted(cls._MODULES.keys())

    @classmethod
    def get_risk_types(cls, industry: str) -> List[str]:
        return sorted(cls._MODULES.get(industry, {}).keys())

    @classmethod
    def get_module(cls, industry: str, risk_type: str) -> Dict[str, Any]:
        return cls._MODULES.get(industry, {}).get(risk_type, {})

    @classmethod
    def is_supported(cls, industry: str, risk_type: str) -> bool:
        return bool(cls.get_module(industry, risk_type))
