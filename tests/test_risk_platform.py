import unittest

import pandas as pd

from src.crediens_ai.data.pipeline import SyntheticDataBuilder
from src.crediens_ai.models.trainer import RiskModelTrainer
from src.crediens_ai.risk.engine import RiskAssessmentEngine


class RiskPlatformTests(unittest.TestCase):
    def setUp(self):
        self.engine = RiskAssessmentEngine()

    def test_it_project_high_risk(self):
        payload = {
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
        }
        result = self.engine.evaluate(payload)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertGreaterEqual(result["risk_probability"], 0.5)
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["decision"], "HUMAN_REVIEW")

    def test_banking_low_risk(self):
        payload = {
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {
                "loan_amount": 50000,
                "existing_debt": 20000,
                "credit_history": "Good",
                "income": 90000,
                "employment_years": 10,
            },
        }
        result = self.engine.evaluate(payload)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertLess(result["risk_probability"], 0.5)
        self.assertIn(result["risk_level"], ["LOW", "MEDIUM"])

    def test_module_policy_bands_control_risk_classification(self):
        credit_result = self.engine.evaluate({
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {
                "existing_debt": 30000,
                "loan_amount": 0,
                "credit_history": "Good",
                "income": 50000,
                "employment_years": 10,
            },
        })
        fraud_result = self.engine.evaluate({
            "industry": "BANKING",
            "risk_type": "FRAUD",
            "data": {
                "transaction_amount": 0,
                "anomaly_score": 0.34444444444444444,
                "velocity": 0,
                "account_age": 60,
            },
        })

        self.assertEqual(credit_result["risk_probability"], 0.155)
        self.assertEqual(credit_result["risk_level"], "LOW")
        self.assertEqual(fraud_result["risk_probability"], 0.155)
        self.assertEqual(fraud_result["risk_level"], "MEDIUM")

    def test_each_supported_module_uses_its_own_risk_features(self):
        cybersecurity = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "CYBER",
            "data": {
                "failed_logins": 100,
                "vulnerabilities": 10,
                "critical_vulnerabilities": 5,
                "patch_age": 90,
                "mfa_coverage": 20,
            },
        })
        supply_chain = self.engine.evaluate({
            "industry": "MANUFACTURING",
            "risk_type": "SUPPLY_CHAIN",
            "data": {
                "supplier_reliability": 30,
                "delivery_delay": 20,
                "inventory_level": 10,
                "supplier_concentration": 60,
            },
        })
        fraud = self.engine.evaluate({
            "industry": "BANKING",
            "risk_type": "FRAUD",
            "data": {
                "transaction_amount": 50000,
                "anomaly_score": 0.9,
                "velocity": 8,
                "account_age": 2,
            },
        })

        for result in (cybersecurity, supply_chain, fraud):
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["scoring_method"], "HEURISTIC")
            self.assertGreater(result["risk_probability"], 0)

    def test_input_schema_rejects_missing_unknown_and_invalid_values(self):
        base_data = {
            "team_size": 12,
            "bug_count": 45,
            "critical_bugs": 8,
            "pending_tasks": 28,
            "test_coverage": 61,
            "technical_debt": "High",
        }
        missing = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {key: value for key, value in base_data.items() if key != "bug_count"},
        })
        extra = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {**base_data, "unexpected": 1},
        })
        invalid_category = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {**base_data, "technical_debt": "Extreme"},
        })
        invalid_number = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {**base_data, "bug_count": float("inf")},
        })

        self.assertEqual(missing["status"], "INVALID_INPUT")
        self.assertIn("Missing required fields", missing["errors"][0])
        self.assertIn("Unexpected fields", extra["errors"][0])
        self.assertIn("technical_debt must be one of", invalid_category["errors"][0])
        self.assertIn("bug_count must be a finite number", invalid_number["errors"][0])

    def test_policy_band_boundaries_have_no_gaps(self):
        bands = self.engine.resolve_module("IT", "PROJECT")["policy"]

        self.assertEqual(self.engine._assign_risk_band(0.2, bands), "LOW")
        self.assertEqual(self.engine._assign_risk_band(0.2001, bands), "MEDIUM")
        self.assertEqual(self.engine._assign_risk_band(0.5, bands), "MEDIUM")
        self.assertEqual(self.engine._assign_risk_band(0.5001, bands), "HIGH")

    def test_invalid_input_rejected(self):
        result = self.engine.evaluate({"industry": "IT"})
        self.assertEqual(result["status"], "INVALID_INPUT")
        self.assertIn("Risk type must be a non-empty string", result["errors"])

    def test_registered_model_drives_matching_assessment(self):
        dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=300)
        training_result = RiskModelTrainer().train(dataset)
        self.engine.register_model(
            industry="IT",
            risk_type="PROJECT",
            model_version="project-risk-v1",
            model=training_result["model"],
            feature_columns=training_result["feature_columns"],
        )
        payload = {
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
        }

        result = self.engine.evaluate(payload)
        expected_probability = training_result["model"].predict_proba(
            pd.DataFrame([payload["data"]])
        )[0, 1]

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["scoring_method"], "MODEL")
        self.assertAlmostEqual(result["risk_probability"], expected_probability)
        self.assertEqual(result["audit"]["scoring_method"], "MODEL")

    def test_unregistered_module_uses_explicit_heuristic_fallback(self):
        result = self.engine.evaluate({
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {
                "income": 90000,
                "loan_amount": 50000,
                "credit_history": "Good",
                "existing_debt": 20000,
                "employment_years": 10,
            },
        })

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["scoring_method"], "HEURISTIC")

    def test_registered_model_rejects_missing_features(self):
        dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=300)
        training_result = RiskModelTrainer().train(dataset)
        self.engine.register_model(
            industry="IT",
            risk_type="PROJECT",
            model_version="project-risk-v1",
            model=training_result["model"],
            feature_columns=training_result["feature_columns"],
        )

        result = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {"team_size": 12},
        })

        self.assertEqual(result["status"], "INVALID_INPUT")
        self.assertIn("Missing required fields", result["errors"][0])

    def test_unregistered_requested_model_version_is_rejected(self):
        result = self.engine.evaluate({
            "industry": "IT",
            "risk_type": "PROJECT",
            "model_version": "unknown-model",
            "data": {
                "team_size": 12,
                "bug_count": 45,
                "critical_bugs": 8,
                "pending_tasks": 28,
                "test_coverage": 61,
                "technical_debt": "High",
            },
        })

        self.assertEqual(result["status"], "INVALID_INPUT")
        self.assertIn("Model version is not registered", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
