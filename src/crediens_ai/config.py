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
    "equipment-risk-v1": {
        "model_family": "Gradient-boosted risk model",
        "description": "Reference implementation for manufacturing equipment risk",
    },
    "credit-risk-v1": {
        "model_family": "Late-fusion ensemble",
        "description": "Reference implementation for credit risk",
    },
}
