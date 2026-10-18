from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RiskModuleConfig:
    industry: str
    risk_type: str
    model_version: str
    features: List[str]
    categorical_features: Dict[str, List[str]] = field(default_factory=dict)
    policy_bands: Dict[str, tuple[float, float]] = field(default_factory=lambda: {"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)})
    monitor_threshold: float = 0.4
    human_review_threshold: float = 0.55
    compliance_rules: List[str] = field(default_factory=lambda: [
        "explanation_available",
        "model_version_recorded",
        "audit_metadata_present",
        "fairness_status_available",
    ])


DEFAULT_MODULE_CONFIGS = {
    "IT": {
        "PROJECT": RiskModuleConfig(
            industry="IT",
            risk_type="PROJECT",
            model_version="project-risk-v1",
            features=["team_size", "bug_count", "critical_bugs", "pending_tasks", "test_coverage", "technical_debt"],
            categorical_features={"technical_debt": ["Low", "Medium", "High", "Critical"]},
            policy_bands={"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
        ),
        "CYBER": RiskModuleConfig(
            industry="IT",
            risk_type="CYBER",
            model_version="project-risk-v1",
            features=["failed_logins", "vulnerabilities", "critical_vulnerabilities", "patch_age", "mfa_coverage"],
            policy_bands={"LOW": (0.0, 0.25), "MEDIUM": (0.25, 0.55), "HIGH": (0.55, 1.0)},
        ),
        "OPERATIONAL": RiskModuleConfig(
            industry="IT",
            risk_type="OPERATIONAL",
            model_version="operational-risk-v1",
            features=["code_complexity", "bug_density", "open_issues", "sprint_velocity", "technical_debt", "test_coverage", "failed_deployments"],
            categorical_features={"technical_debt": ["Low", "Medium", "High", "Critical"]},
            policy_bands={"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
        ),
    },
    "MANUFACTURING": {
        "EQUIPMENT": RiskModuleConfig(
            industry="MANUFACTURING",
            risk_type="EQUIPMENT",
            model_version="equipment-risk-v1",
            features=["machine_age", "vibration", "temperature", "maintenance_history", "operating_hours"],
            categorical_features={"maintenance_history": ["Current", "Late", "Overdue"]},
            policy_bands={"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
        ),
        "SUPPLY_CHAIN": RiskModuleConfig(
            industry="MANUFACTURING",
            risk_type="SUPPLY_CHAIN",
            model_version="equipment-risk-v1",
            features=["supplier_reliability", "delivery_delay", "inventory_level", "supplier_concentration"],
            policy_bands={"LOW": (0.0, 0.25), "MEDIUM": (0.25, 0.55), "HIGH": (0.55, 1.0)},
        ),
    },
    "BANKING": {
        "CREDIT": RiskModuleConfig(
            industry="BANKING",
            risk_type="CREDIT",
            model_version="credit-risk-v1",
            features=["income", "loan_amount", "credit_history", "existing_debt", "employment_years"],
            categorical_features={"credit_history": ["Poor", "Fair", "Good", "Excellent"]},
            policy_bands={"LOW": (0.0, 0.2), "MEDIUM": (0.2, 0.5), "HIGH": (0.5, 1.0)},
        ),
        "FRAUD": RiskModuleConfig(
            industry="BANKING",
            risk_type="FRAUD",
            model_version="credit-risk-v1",
            features=["transaction_amount", "anomaly_score", "velocity", "account_age"],
            policy_bands={"LOW": (0.0, 0.15), "MEDIUM": (0.15, 0.45), "HIGH": (0.45, 1.0)},
        ),
    },
}


def get_module_config(industry: str, risk_type: str) -> RiskModuleConfig | None:
    return DEFAULT_MODULE_CONFIGS.get(industry, {}).get(risk_type)
