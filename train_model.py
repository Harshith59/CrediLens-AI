from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.crediens_ai.data.pipeline import SyntheticDataBuilder
from src.crediens_ai.models.artifacts import save_model_artifact
from src.crediens_ai.models.trainer import RiskModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the synthetic IT project-risk reference model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "it" / "project-risk-v1.joblib",
        help="Path for the trained model artifact",
    )
    args = parser.parse_args()

    dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=400)
    training_result = RiskModelTrainer().train(dataset)
    artifact_path = save_model_artifact(
        args.output,
        industry="IT",
        risk_type="PROJECT",
        model_version="project-risk-v1",
        model=training_result["model"],
        feature_columns=training_result["feature_columns"],
        metrics=training_result["metrics"],
        model_metrics=training_result["model_metrics"],
    )
    print(json.dumps({
        "artifact": str(artifact_path),
        "metrics": training_result["metrics"],
        "model_metrics": training_result["model_metrics"],
        "fusion_weights": training_result["fusion_weights"],
        "cross_validation": training_result["cross_validation"],
        "training_rows": training_result["n_train"],
        "test_rows": training_result["n_test"],
    }, indent=2))


if __name__ == "__main__":
    main()
