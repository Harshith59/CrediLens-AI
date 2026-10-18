import unittest

from src.crediens_ai.reporting.reports import ReportBuilder
from src.crediens_ai.risk.engine import RiskAssessmentEngine


class GovernanceAndReportingTests(unittest.TestCase):
    def test_assessment_compliance_includes_rule_evidence_and_review_status(self):
        result = RiskAssessmentEngine().evaluate({
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
        })

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["compliance_status"], "REVIEW_REQUIRED")
        self.assertTrue(result["governance"]["checks"]["required_input_fields_present"])
        self.assertTrue(result["governance"]["checks"]["decision_policy_applied_consistently"])
        self.assertIn("evidence", result["governance"]["results"]["explanation_available"])

    def test_html_report_escapes_untrusted_text(self):
        report = ReportBuilder.build_html_report({
            "industry": "<script>alert(1)</script>",
            "risk_type": "PROJECT",
            "risk_probability": 0.1,
            "risk_level": "LOW",
            "decision": "APPROVE",
            "compliance_status": "COMPLIANT",
        })

        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)


if __name__ == "__main__":
    unittest.main()
