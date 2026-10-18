from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    from lime import lime_tabular
except Exception:  # pragma: no cover
    lime_tabular = None


class ExplanationEngine:
    """Generate global and local feature explanations using SHAP/LIME when available."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def _feature_names(self, X: Sequence[Sequence[float]]) -> List[str]:
        if hasattr(X, "columns"):
            return list(X.columns)
        if len(X) == 0:
            return []
        row = X[0]
        if isinstance(row, dict):
            return list(row.keys())
        return [f"feature_{index}" for index in range(len(row))]

    def _prepare_numeric_matrix(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if len(X) == 0:
            return np.asarray([], dtype=float).reshape(0, 0)

        if isinstance(X, np.ndarray):
            rows = X.tolist()
        else:
            rows = list(X)

        if len(rows) == 0:
            return np.asarray([], dtype=float).reshape(0, 0)

        feature_count = len(rows[0])
        category_maps: List[Dict[Any, float]] = []
        for column_index in range(feature_count):
            unique_values = []
            for row in rows:
                value = row[column_index]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    continue
                if value is None:
                    continue
                unique_values.append(value)
            if unique_values:
                category_maps.append({val: idx + 1 for idx, val in enumerate(sorted(set(unique_values), key=lambda v: str(v)))})
            else:
                category_maps.append({})

        matrix = []
        for row in rows:
            converted = []
            for column_index, value in enumerate(row):
                if value is None:
                    converted.append(0.0)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    converted.append(float(value))
                else:
                    converted.append(float(category_maps[column_index].get(value, 0.0)))
            matrix.append(converted)
        return np.asarray(matrix, dtype=float)

    def shap_global(self, X: Sequence[Sequence[float]]) -> Dict[str, Any]:
        feature_names = self._feature_names(X)
        array = self._prepare_numeric_matrix(X)
        if array.size == 0:
            feature_importance = {name: 0.0 for name in feature_names}
        elif self.model is None:
            feature_importance = {
                feature_names[idx]: float(np.abs(array[:, idx]).mean())
                for idx in range(array.shape[1])
            }
        elif shap is not None:
            explainer = shap.Explainer(self.model, array)
            values = explainer(array)
            if hasattr(values, "values"):
                importances = np.abs(values.values)
                if importances.ndim > 1:
                    importances = importances.mean(axis=tuple(range(1, importances.ndim)))
            else:
                importances = np.abs(np.asarray(values))
                if importances.ndim > 1:
                    importances = importances.mean(axis=tuple(range(1, importances.ndim)))
            feature_importance = {name: float(score) for name, score in zip(feature_names, importances)}
        else:
            if hasattr(self.model, "feature_importances_"):
                feature_importance = {name: float(score) for name, score in zip(feature_names, self.model.feature_importances_)}
            else:
                feature_importance = {name: 0.0 for name in feature_names}
        sorted_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        return {
            "feature_importance": dict(sorted_importance),
            "top_features": [name for name, _ in sorted_importance[: min(5, len(sorted_importance))]],
        }

    def lime_local(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> Dict[str, Any]:
        feature_names = self._feature_names(X)
        array = self._prepare_numeric_matrix(X)
        if self.model is None:
            row = array[0] if len(array) > 0 else np.zeros(len(feature_names))
            feature_importance = {name: float(abs(value)) for name, value in zip(feature_names, row)}
            sorted_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
            return {
                "feature_importance": dict(sorted_importance),
                "top_features": [name for name, _ in sorted_importance[: min(5, len(sorted_importance))]],
            }

        if lime_tabular is not None and len(array) > 0:
            explainer = lime_tabular.LimeTabularExplainer(
                array,
                feature_names=feature_names,
                class_names=["low_risk", "high_risk"],
                mode="classification",
                random_state=42,
            )
            row = array[0]
            explanation = explainer.explain_instance(row, lambda values: self.model.predict_proba(values), num_features=len(feature_names), num_samples=200)
            explanation_dict = dict(explanation.as_list())
            return {"feature_importance": explanation_dict, "top_features": list(explanation_dict.keys())[:5]}

        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(feature_names, self.model.feature_importances_))
        else:
            importances = {name: 1.0 / max(1, len(feature_names)) for name in feature_names}
        sorted_importance = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        return {
            "feature_importance": dict(sorted_importance),
            "top_features": [name for name, _ in sorted_importance[: min(5, len(sorted_importance))]],
        }
