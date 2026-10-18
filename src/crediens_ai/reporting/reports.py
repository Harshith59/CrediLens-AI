from __future__ import annotations

from io import BytesIO
from html import escape
from typing import Any, Dict

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class ReportBuilder:
    """Build JSON and HTML report payloads for assessment outputs."""

    @staticmethod
    def build_demo_json_report(
        *,
        industry: str,
        risk_type: str,
        inputs: Dict[str, Any],
        risk_probability: float,
        risk_level: str,
        decision: str,
        scoring_method: str,
        scoring_method_description: str,
        timestamp: str,
        recommendation: str,
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "report_type": "PUBLIC_DEMO",
            "demo": True,
            "disclaimer": (
                "Synthetic demonstration only. Do not use this result for real-world "
                "credit, fraud, safety, or other consequential decisions."
            ),
            "industry": industry,
            "risk_type": risk_type,
            "risk_probability": risk_probability,
            "risk_level": risk_level,
            "decision": decision,
            "scoring_method": scoring_method,
            "scoring_method_description": scoring_method_description,
            "timestamp": timestamp,
            "inputs": dict(inputs),
            "explanation": dict(explanation),
            "recommendations": [recommendation],
        }

    @staticmethod
    def build_json_report(assessment: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "assessment_id": assessment.get("audit_id"),
            "application_id": assessment.get("application_id"),
            "industry": assessment.get("industry"),
            "risk_type": assessment.get("risk_type"),
            "risk_probability": assessment.get("risk_probability"),
            "risk_level": assessment.get("risk_level"),
            "decision": assessment.get("decision"),
            "compliance_status": assessment.get("compliance_status"),
            "top_factors": assessment.get("top_factors", []),
            "model_version": assessment.get("model_version"),
            "scoring_method": assessment.get("scoring_method"),
            "timestamp": assessment.get("timestamp"),
            "governance": assessment.get("governance", {}),
            "fairness": assessment.get("fairness", {}),
        }

    @staticmethod
    def build_html_report(assessment: Dict[str, Any]) -> str:
        report = assessment
        return (
            "<html><body><h1>CrediLens AI Risk Report</h1>"
            f"<p><strong>Industry:</strong> {escape(str(report.get('industry', '')))}</p>"
            f"<p><strong>Risk Type:</strong> {escape(str(report.get('risk_type', '')))}</p>"
            f"<p><strong>Probability:</strong> {escape(str(report.get('risk_probability', '')))}</p>"
            f"<p><strong>Risk Level:</strong> {escape(str(report.get('risk_level', '')))}</p>"
            f"<p><strong>Decision:</strong> {escape(str(report.get('decision', '')))}</p>"
            f"<p><strong>Compliance:</strong> {escape(str(report.get('compliance_status', '')))}</p>"
            "</body></html>"
        )

    @staticmethod
    def build_png_report(assessment: Dict[str, Any]) -> bytes:
        figure = Figure(figsize=(10, 4.5), constrained_layout=True)
        canvas = FigureCanvasAgg(figure)
        score_axis, factors_axis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})

        score = float(assessment.get("risk_probability", 0.0))
        risk_level = str(assessment.get("risk_level", "UNKNOWN"))
        score_axis.barh([risk_level], [score], color={"LOW": "#14804a", "MEDIUM": "#bd7500", "HIGH": "#c43737"}.get(risk_level, "#536579"))
        score_axis.set_xlim(0, 1)
        score_axis.set_xlabel("Risk score")
        score_axis.set_title("Assessment risk score")
        score_axis.text(min(score + 0.03, 0.78), 0, f"{score:.1%}", va="center", fontweight="bold")

        input_data = assessment.get("decision_trace", {}).get("input_snapshot", {}).get("data", {})
        factors = [
            (name, float(value))
            for name, value in input_data.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ][:8]
        if factors:
            names = [name for name, _ in reversed(factors)]
            values = [value for _, value in reversed(factors)]
            maximum = max(abs(value) for value in values) or 1.0
            normalized = [value / maximum for value in values]
            factors_axis.barh(names, normalized, color="#3674a5")
            factors_axis.set_xlim(min(-1.05, min(normalized) - 0.05), 1.05)
            factors_axis.set_xlabel("Relative input value (scaled)")
        else:
            factors_axis.text(0.5, 0.5, "No numeric input factors available", ha="center", va="center")
            factors_axis.set_axis_off()
        factors_axis.set_title("Numeric input indicators")
        figure.suptitle(
            f"{assessment.get('industry', '')} / {assessment.get('risk_type', '')} — "
            f"{assessment.get('decision', '')}"
        )

        output = BytesIO()
        canvas.print_png(output)
        return output.getvalue()
