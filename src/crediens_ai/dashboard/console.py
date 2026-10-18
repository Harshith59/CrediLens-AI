from __future__ import annotations

from typing import Any, Dict


class ConsoleDashboard:
    """Simple console dashboard summary for operational review."""

    @staticmethod
    def render(assessment: Dict[str, Any]) -> str:
        lines = [
            "CrediLens AI Dashboard",
            f"Industry: {assessment.get('industry')}",
            f"Risk Type: {assessment.get('risk_type')}",
            f"Risk Probability: {assessment.get('risk_probability')}",
            f"Risk Level: {assessment.get('risk_level')}",
            f"Decision: {assessment.get('decision')}",
            f"Compliance Status: {assessment.get('compliance_status')}",
            "Top Factors: " + ", ".join(assessment.get("top_factors", [])),
        ]
        return "\n".join(lines)
