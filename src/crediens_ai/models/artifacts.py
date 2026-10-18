from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import joblib


ARTIFACT_FORMAT = "credilens-model-v1"


def save_model_artifact(
    path: str | Path,
    *,
    industry: str,
    risk_type: str,
    model_version: str,
    model: Any,
    feature_columns: List[str],
    metrics: Dict[str, Any],
    model_metrics: Dict[str, Dict[str, Any]] | None = None,
) -> Path:
    destination = Path(path)
    if not industry or not risk_type or not model_version:
        raise ValueError("Industry, risk type, and model version are required")
    if not feature_columns or any(not isinstance(column, str) or not column for column in feature_columns):
        raise ValueError("At least one valid feature column is required")
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Model artifacts require a model that implements predict_proba")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=destination.suffix or ".joblib",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump({
            "artifact_format": ARTIFACT_FORMAT,
            "industry": industry,
            "risk_type": risk_type,
            "model_version": model_version,
            "model": model,
            "feature_columns": list(feature_columns),
            "metrics": dict(metrics),
            "model_metrics": dict(model_metrics or {}),
            "fusion_weights": dict(getattr(model, "weights", {})),
        }, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_model_artifact(path: str | Path) -> Dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported or invalid CrediLens model artifact")
    for key in ("industry", "risk_type", "model_version"):
        if not isinstance(artifact.get(key), str) or not artifact[key]:
            raise ValueError(f"Model artifact is missing a valid {key}")
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model artifact is missing feature columns")
    if any(not isinstance(column, str) or not column for column in feature_columns):
        raise ValueError("Model artifact contains invalid feature columns")
    if not callable(getattr(artifact.get("model"), "predict_proba", None)):
        raise ValueError("Model artifact does not contain a predict_proba model")
    if not isinstance(artifact.get("metrics"), dict):
        raise ValueError("Model artifact is missing training metrics")
    if not isinstance(artifact.get("model_metrics", {}), dict):
        raise ValueError("Model artifact contains invalid per-model metrics")
    if not isinstance(artifact.get("fusion_weights", {}), dict):
        raise ValueError("Model artifact contains invalid fusion weights")
    return artifact
