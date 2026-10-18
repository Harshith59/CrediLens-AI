import unittest

from src.crediens_ai.config.platform import get_module_config
from src.crediens_ai.risk.engine import RiskAssessmentEngine


class ConfigHooksTests(unittest.TestCase):
    def test_module_config_returns_expected_values(self):
        config = get_module_config("IT", "PROJECT")
        self.assertIsNotNone(config)
        self.assertEqual(config.model_version, "project-risk-v1")
        self.assertIn("bug_count", config.features)

    def test_unknown_module_is_none(self):
        self.assertIsNone(get_module_config("HEALTHCARE", "RISK"))

    def test_all_supported_modules_have_config_entries(self):
        expected = {
            ("IT", "PROJECT"): "project-risk-v1",
            ("IT", "CYBER"): "project-risk-v1",
            ("IT", "OPERATIONAL"): "operational-risk-v1",
            ("MANUFACTURING", "EQUIPMENT"): "equipment-risk-v1",
            ("MANUFACTURING", "SUPPLY_CHAIN"): "equipment-risk-v1",
            ("BANKING", "CREDIT"): "credit-risk-v1",
            ("BANKING", "FRAUD"): "credit-risk-v1",
        }
        for (industry, risk_type), model_version in expected.items():
            config = get_module_config(industry, risk_type)
            self.assertIsNotNone(config)
            self.assertEqual(config.model_version, model_version)

    def test_engine_exposes_configured_module_defaults(self):
        engine = RiskAssessmentEngine()
        self.addCleanup(engine.close)
        module = engine.resolve_module("IT", "PROJECT")
        self.assertEqual(module["model_version"], "project-risk-v1")
        self.assertIn("bug_count", module["features"])

    def test_engine_returns_evidence_and_trace_record(self):
        engine = RiskAssessmentEngine()
        self.addCleanup(engine.close)
        result = engine.evaluate({
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
        self.assertIn("evidence", result)
        self.assertIn("confidence", result["evidence"])
        self.assertIn("missing_fields", result["evidence"])
        self.assertIn("decision_trace", result)
        self.assertIn("input_snapshot", result["decision_trace"])
        self.assertEqual(result["fairness"]["status"], "NOT_ASSESSED")
        self.assertIsNone(result["fairness"]["gap"])

    def test_engine_records_human_review_decision(self):
        engine = RiskAssessmentEngine()
        self.addCleanup(engine.close)
        result = engine.evaluate({
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
            "application_id": "review-456",
        })
        updated = engine.record_human_decision(
            result["audit_id"],
            reviewer="analyst-01",
            reviewer_comment="Escalated due to current backlog risk.",
            final_decision="APPROVED_WITH_CONDITIONS",
        )
        self.assertEqual(updated["decision_trace"]["human_decision"], "APPROVED_WITH_CONDITIONS")
        self.assertEqual(updated["decision_trace"]["reviewer_comment"], "Escalated due to current backlog risk.")
        self.assertEqual(updated["governance"]["checks"]["human_review_requirement_satisfied"], True)


if __name__ == "__main__":
    unittest.main()
