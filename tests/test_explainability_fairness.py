import unittest

from sklearn.ensemble import RandomForestClassifier

from src.crediens_ai.explainability.explain import ExplanationEngine
from src.crediens_ai.fairness.metrics import FairnessEvaluator


class ExplainabilityAndFairnessTests(unittest.TestCase):
    def test_explanations_return_feature_contributions(self):
        X = [[1, 0], [0, 1], [1, 1], [0, 0]]
        y = [0, 1, 1, 0]
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        explanation = ExplanationEngine(model)
        shap_result = explanation.shap_global(X)
        lime_result = explanation.lime_local(X, y)

        self.assertIn("feature_importance", shap_result)
        self.assertIn("top_features", shap_result)
        self.assertIn("feature_importance", lime_result)
        self.assertGreater(len(lime_result["feature_importance"]), 0)

    def test_fairness_metrics_detect_group_gap(self):
        protected = ["A", "A", "B", "B", "A", "B"]
        scores = [0.9, 0.8, 0.2, 0.1, 0.95, 0.15]
        evaluator = FairnessEvaluator()
        result = evaluator.demographic_parity_gap(protected, scores, threshold=0.5)

        self.assertGreater(result["gap"], 0)
        self.assertIn("group_rates", result)
        self.assertIn("status", result)


if __name__ == "__main__":
    unittest.main()
