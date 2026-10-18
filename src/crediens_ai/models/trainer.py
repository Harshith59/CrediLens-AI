from __future__ import annotations

from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class WeightedLateFusionModel:
    """Average fitted binary classifiers with validation-selected weights."""

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]) -> None:
        self.models = models
        self.weights = weights
        self.classes_ = np.asarray([0, 1])
        self.model_family = "Weighted late-fusion ensemble"

    @staticmethod
    def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("A fitted risk classifier must include the positive class label 1")
        class_index = classes.index(1)
        return np.asarray(model.predict_proba(X)[:, class_index], dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = sum(
            self.weights[name] * self._positive_probability(model, X)
            for name, model in self.models.items()
        )
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RiskModelTrainer:
    """Compare tree-based tabular models and fuse them without test-set leakage."""

    def __init__(self, target_column: str = "target") -> None:
        self.target_column = target_column

    def _make_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = [column for column in X.columns if column not in numeric_features]
        transformers = []
        if numeric_features:
            transformers.append(
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features)
            )
        if categorical_features:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical_features,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _base_estimators() -> Dict[str, Any]:
        return {
            "lightgbm": LGBMClassifier(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                max_depth=8,
                random_state=42,
                n_jobs=1,
            ),
        }

    @staticmethod
    def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
        return WeightedLateFusionModel._positive_probability(model, X)

    @staticmethod
    def _metrics(y_true: pd.Series, probabilities: np.ndarray) -> Dict[str, Any]:
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
            "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        }

    @staticmethod
    def _select_fusion_weights(
        y_validation: pd.Series,
        out_of_fold_probabilities: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        model_names = list(out_of_fold_probabilities)
        best_score = -1.0
        best_weights: tuple[float, ...] | None = None
        grid = [step / 10 for step in range(11)]
        for first, second in product(grid, repeat=2):
            third = round(1.0 - first - second, 10)
            if third < 0:
                continue
            weights = (first, second, third)
            combined = sum(
                weight * out_of_fold_probabilities[name]
                for name, weight in zip(model_names, weights)
            )
            score = roc_auc_score(y_validation, combined)
            if score > best_score:
                best_score = float(score)
                best_weights = weights
        if best_weights is None:
            raise RuntimeError("Unable to select non-negative late-fusion weights")
        return {name: weight for name, weight in zip(model_names, best_weights)}

    def train(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        if self.target_column not in dataset.columns:
            raise ValueError(f"Dataset is missing target column '{self.target_column}'")
        feature_columns = [column for column in dataset.columns if column != self.target_column]
        if not feature_columns:
            raise ValueError("Dataset must contain at least one feature column")
        if dataset.empty or dataset[self.target_column].isna().any():
            raise ValueError("Dataset must contain rows with non-missing target labels")
        labels = set(dataset[self.target_column].unique())
        if labels != {0, 1}:
            raise ValueError("Binary risk targets must use both class labels 0 and 1")

        X = dataset[feature_columns]
        y = dataset[self.target_column].astype(int)
        class_counts = y.value_counts()
        if int(class_counts.min()) < 3:
            raise ValueError("Each target class must contain at least three rows for stratified training and validation")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        train_class_count = int(y_train.value_counts().min())
        fold_count = min(5, train_class_count)
        if fold_count < 2:
            raise ValueError("Training split does not contain enough examples per class for cross-validation")
        cross_validation = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)

        preprocessor = self._make_preprocessor(X_train)
        base_estimators = self._base_estimators()
        fitted_models: Dict[str, Any] = {}
        out_of_fold_probabilities: Dict[str, np.ndarray] = {}
        for name, estimator in base_estimators.items():
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("classifier", estimator),
            ])
            out_of_fold_probabilities[name] = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cross_validation,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            fitted_model = clone(pipeline)
            fitted_model.fit(X_train, y_train)
            fitted_models[name] = fitted_model

        fusion_weights = self._select_fusion_weights(y_train, out_of_fold_probabilities)
        model = WeightedLateFusionModel(fitted_models, fusion_weights)
        model_probabilities = {
            name: self._positive_probability(fitted, X_test)
            for name, fitted in fitted_models.items()
        }
        model_metrics = {
            name: self._metrics(y_test, probabilities)
            for name, probabilities in model_probabilities.items()
        }
        fused_probabilities = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics(y_test, fused_probabilities)
        return {
            "model": model,
            "metrics": metrics,
            "model_metrics": model_metrics,
            "fusion_weights": fusion_weights,
            "cross_validation": {
                "folds": fold_count,
                "fusion_roc_auc": round(float(roc_auc_score(
                    y_train,
                    sum(
                        fusion_weights[name] * out_of_fold_probabilities[name]
                        for name in fitted_models
                    ),
                )), 4),
            },
            "feature_columns": feature_columns,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, model: Any, row: Dict[str, Any]) -> float:
        sample = pd.DataFrame([row])
        return float(self._positive_probability(model, sample)[0])
