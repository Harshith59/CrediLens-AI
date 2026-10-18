from __future__ import annotations

from typing import Any, Dict


class ComplianceEngine:
    """Evaluate governance checks and retain evidence for each rule."""

    @staticmethod
    def evaluate(assessment: Dict[str, Any]) -> Dict[str, Any]:
        explanation = assessment.get("explainability", {})
        fairness = assessment.get("fairness", {})
        evidence = assessment.get("evidence", {})
        decision_trace = assessment.get("decision_trace", {})
        risk_probability = assessment.get("risk_probability")
        risk_level = assessment.get("risk_level")
        risk_policy = assessment.get("risk_policy", {})
        matching_bands = [
            band
            for band, (lower, upper) in risk_policy.items()
            if isinstance(risk_probability, (int, float)) and lower <= risk_probability <= upper
        ]
        expected_policy = (
            "HUMAN_REVIEW"
            if risk_probability is not None and risk_probability >= assessment.get("human_review_threshold", 0.55)
            else "MONITOR"
            if risk_probability is not None and risk_probability >= assessment.get("monitor_threshold", 0.4)
            else "APPROVE"
        )
        review_complete = assessment.get("decision") != "HUMAN_REVIEW" or bool(
            decision_trace.get("human_decision") and decision_trace.get("reviewer")
        )
        checks: Dict[str, bool] = {
            "explanation_available": bool(explanation.get("global") and explanation.get("local")),
            "required_input_fields_present": (
                assessment.get("audit", {}).get("input_validation_result") == "VALID"
                and not evidence.get("missing_fields")
            ),
            "model_version_recorded": bool(assessment.get("model_version")),
            "prediction_timestamp_recorded": bool(assessment.get("timestamp")),
            "audit_metadata_present": bool(assessment.get("audit")),
            "fairness_status_available": bool(
                fairness.get("status") or assessment.get("fairness_status") == "NOT_APPLICABLE"
            ),
            "human_review_requirement_satisfied": review_complete,
            "decision_policy_applied_consistently": (
                bool(matching_bands)
                and risk_level in matching_bands
                and assessment.get("decision") == expected_policy
            ),
        }

        failed = [
            name for name, passed in checks.items()
            if not passed and name != "human_review_requirement_satisfied"
        ]
        status = "NON_COMPLIANT" if failed else "COMPLIANT"
        if status == "COMPLIANT" and (
            not checks["human_review_requirement_satisfied"]
            or (risk_level == "HIGH" and not decision_trace.get("human_decision"))
            or fairness.get("status") in {"ALERT", "REVIEW_REQUIRED"}
        ):
            status = "REVIEW_REQUIRED"

        results = {
            name: {
                "status": "PASS" if passed else "FAIL",
                "evidence": {
                    "risk_probability": risk_probability,
                    "risk_level": risk_level,
                    "decision": assessment.get("decision"),
                },
            }
            for name, passed in checks.items()
        }
        return {
            "status": status,
            "checks": checks,
            "results": results,
            "notes": "Automated governance checks produced a compliance decision.",
        }
