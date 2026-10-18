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
        self.assertIn("method", shap_result)
        self.assertTrue(shap_result["attribution_available"])
        self.assertIn("feature_importance", lime_result)
        self.assertTrue(lime_result["attribution_available"])
        self.assertGreater(len(lime_result["feature_importance"]), 0)

    def test_unavailable_explanations_are_labelled_as_input_magnitude_proxy(self):
        explanation = ExplanationEngine(None)

        global_result = explanation.shap_global([[5, 1]])
        local_result = explanation.lime_local([[5, 1]], [1])

        self.assertEqual(global_result["method"], "input_magnitude_proxy")
        self.assertFalse(global_result["attribution_available"])
        self.assertEqual(local_result["method"], "input_magnitude_proxy")
        self.assertFalse(local_result["attribution_available"])

    def test_fairness_metrics_detect_group_gap(self):
        protected = ["A", "A", "B", "B", "A", "B"]
        scores = [0.9, 0.8, 0.2, 0.1, 0.95, 0.15]
        evaluator = FairnessEvaluator()
        result = evaluator.demographic_parity_gap(protected, scores, threshold=0.5)

        self.assertGreater(result["gap"], 0)
        self.assertIn("group_rates", result)
        self.assertIn("status", result)

    def test_fairness_requires_sufficient_samples_from_multiple_groups(self):
        evaluator = FairnessEvaluator()

        missing_group = evaluator.demographic_parity_gap(["A"], [0.9])
        insufficient_group = evaluator.demographic_parity_gap(["A", "A", "B"], [0.9, 0.8, 0.1])

        self.assertEqual(missing_group["status"], "NOT_ASSESSED")
        self.assertIsNone(missing_group["gap"])
        self.assertEqual(insufficient_group["status"], "NOT_ASSESSED")
        self.assertIn("not evidence", insufficient_group["notes"])

    def test_fairness_rejects_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            FairnessEvaluator.demographic_parity_gap(["A"], [float("nan")])


if __name__ == "__main__":
    unittest.main()
