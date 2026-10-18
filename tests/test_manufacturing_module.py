import unittest

from src.crediens_ai.modules.manufacturing import ManufacturingRiskModule


class ManufacturingModuleTests(unittest.TestCase):
    def test_equipment_scoring_and_band(self):
        payload = {
            "machine_age": 8,
            "vibration": 7,
            "temperature": 6,
            "maintenance_history": "Overdue",
            "operating_hours": 9000,
        }
        score = ManufacturingRiskModule.score_equipment(payload)
        self.assertGreater(score, 0.5)
        self.assertEqual(ManufacturingRiskModule.risk_band(score), "HIGH")

    def test_required_features_are_present(self):
        self.assertIn("machine_age", ManufacturingRiskModule.required_features())
        self.assertIn("maintenance_history", ManufacturingRiskModule.required_features())


if __name__ == "__main__":
    unittest.main()
