from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class DashboardApp:
    """Minimal dashboard summary object for the blueprint."""

    @staticmethod
    def build_summary(assessment: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "CrediLens AI Dashboard",
            "industry": assessment.get("industry"),
            "risk_type": assessment.get("risk_type"),
            "risk_probability": assessment.get("risk_probability"),
            "risk_level": assessment.get("risk_level"),
            "decision": assessment.get("decision"),
            "compliance_status": assessment.get("compliance_status"),
            "top_factors": assessment.get("top_factors", []),
        }

    @staticmethod
    def page() -> str:
        return Path(__file__).with_name("page.html").read_text(encoding="utf-8")
