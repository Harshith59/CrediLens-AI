import unittest

from src.crediens_ai.data.pipeline import SyntheticDataBuilder
from src.crediens_ai.models.trainer import RiskModelTrainer


class TrainingPipelineTests(unittest.TestCase):
    def test_synthetic_dataset_has_expected_columns(self):
        dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=250)
        self.assertIn("team_size", dataset.columns)
        self.assertIn("technical_debt", dataset.columns)
        self.assertIn("target", dataset.columns)
        self.assertEqual(len(dataset), 250)

    def test_trainer_builds_model_and_reports_metrics(self):
        dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=300)
        trainer = RiskModelTrainer()
        result = trainer.train(dataset)
        self.assertIn("accuracy", result["metrics"])
        self.assertGreater(result["metrics"]["accuracy"], 0.7)
        self.assertIn("model", result)
        self.assertEqual(
            set(result["model_metrics"]),
            {"lightgbm", "xgboost", "random_forest"},
        )
        self.assertAlmostEqual(sum(result["fusion_weights"].values()), 1.0)
        self.assertTrue(all(weight >= 0 for weight in result["fusion_weights"].values()))
        self.assertEqual(result["cross_validation"]["folds"], 5)
        self.assertIn("pr_auc", result["metrics"])
        self.assertIn("brier_score", result["metrics"])
        self.assertEqual(len(result["metrics"]["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
