import json
from typing import Any, Dict

from src.crediens_ai.risk.engine import RiskAssessmentEngine


def main() -> None:
    engine = RiskAssessmentEngine()
    example_input = {
        "industry": "IT",
        "risk_type": "PROJECT",
        "data": {
            "team_size": 12,
            "bug_count": 45,
            "critical_bugs": 8,
            "pending_tasks": 28,
            "test_coverage": 61,
            "technical_debt": "High",
        },
        "model_version": "project-risk-v1",
    }
    result = engine.evaluate(example_input)
    summary = {
        "industry": result["industry"],
        "risk_type": result["risk_type"],
        "risk_probability": result["risk_probability"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "compliance_status": result["compliance_status"],
        "top_factors": result["top_factors"],
    }
    print("CrediLens AI Dashboard")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
