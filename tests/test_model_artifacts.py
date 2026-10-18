import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient

from src.crediens_ai.api.app import app, engine
from src.crediens_ai.data.pipeline import SyntheticDataBuilder
from src.crediens_ai.models.artifacts import load_model_artifact, save_model_artifact
from src.crediens_ai.models.trainer import RiskModelTrainer


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=300)
        self.training_result = RiskModelTrainer().train(self.dataset)

    def test_model_artifact_round_trips_metadata_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "it" / "project-risk-v1.joblib"
            save_model_artifact(
                path,
                industry="IT",
                risk_type="PROJECT",
                model_version="project-risk-v1",
                model=self.training_result["model"],
                feature_columns=self.training_result["feature_columns"],
                metrics=self.training_result["metrics"],
                model_metrics=self.training_result["model_metrics"],
            )

            artifact = load_model_artifact(path)

        self.assertEqual(artifact["industry"], "IT")
        self.assertEqual(artifact["risk_type"], "PROJECT")
        self.assertEqual(artifact["feature_columns"], self.training_result["feature_columns"])
        self.assertEqual(artifact["metrics"], self.training_result["metrics"])
        self.assertEqual(artifact["model_metrics"], self.training_result["model_metrics"])
        self.assertEqual(artifact["fusion_weights"], self.training_result["fusion_weights"])
        self.assertEqual(
            artifact["model"].predict_proba(
                self.dataset[self.training_result["feature_columns"]].iloc[:1]
            ).shape,
            (1, 2),
        )

    def test_api_loads_configured_artifact_and_uses_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project-risk-v1.joblib"
            save_model_artifact(
                path,
                industry="IT",
                risk_type="PROJECT",
                model_version="project-risk-v1",
                model=self.training_result["model"],
                feature_columns=self.training_result["feature_columns"],
                metrics=self.training_result["metrics"],
                model_metrics=self.training_result["model_metrics"],
            )
            original_models = engine._registered_models.copy()
            original_metadata = engine.model_metadata.copy()
            try:
                with patch.dict("os.environ", {
                    "CREDILENS_MODEL_ARTIFACT": str(path),
                    "CREDILENS_API_KEYS": (
                        '{"local-analyst-key-123456789012345678901234":'
                        '{"role":"analyst","subject":"local-analyst"}}'
                    ),
                }):
                    with TestClient(app) as client:
                        client.headers["X-API-Key"] = "local-analyst-key-123456789012345678901234"
                        model_info = client.get("/api/model-info").json()
                        result = client.post("/api/risk/predict", json={
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
                        }).json()
            finally:
                engine._registered_models = original_models
                engine.model_metadata.clear()
                engine.model_metadata.update(original_metadata)

        self.assertIn(
            {
                "industry": "IT",
                "risk_type": "PROJECT",
                "model_version": "project-risk-v1",
                "feature_columns": self.training_result["feature_columns"],
            },
            model_info["registered_models"],
        )
        self.assertEqual(result["scoring_method"], "MODEL")

    def test_loader_rejects_unknown_artifact_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.joblib"
            joblib.dump({"artifact_format": "unknown"}, path)

            with self.assertRaisesRegex(ValueError, "Unsupported or invalid"):
                load_model_artifact(path)

    def test_save_requires_a_probability_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "predict_proba"):
                save_model_artifact(
                    Path(directory) / "invalid.joblib",
                    industry="IT",
                    risk_type="PROJECT",
                    model_version="project-risk-v1",
                    model=object(),
                    feature_columns=["team_size"],
                    metrics={},
                )


if __name__ == "__main__":
    unittest.main()
