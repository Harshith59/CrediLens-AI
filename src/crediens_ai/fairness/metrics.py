from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List


class FairnessEvaluator:
    """Compute simple group fairness metrics used in governance reviews."""

    @staticmethod
    def demographic_parity_gap(
        protected: Iterable[str],
        scores: Iterable[float],
        threshold: float = 0.5,
        min_group_size: int = 2,
    ) -> Dict[str, Any]:
        protected_list = list(protected)
        scores_list = list(scores)
        if len(protected_list) != len(scores_list):
            raise ValueError("protected and scores must have equal length")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a finite value between 0 and 1")
        if min_group_size < 1:
            raise ValueError("min_group_size must be positive")

        by_group: Dict[str, List[float]] = defaultdict(list)
        for group, score in zip(protected_list, scores_list):
            if not isinstance(group, str) or not group.strip():
                raise ValueError("protected group labels must be non-empty strings")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("scores must be finite probabilities between 0 and 1")
            try:
                valid_score = math.isfinite(score) and 0 <= score <= 1
            except OverflowError:
                valid_score = False
            if not valid_score:
                raise ValueError("scores must be finite probabilities between 0 and 1")
            by_group[group].append(score)

        sample_counts = {group: len(values) for group, values in by_group.items()}
        if len(by_group) < 2 or any(count < min_group_size for count in sample_counts.values()):
            return {
                "gap": None,
                "group_rates": {},
                "sample_counts": sample_counts,
                "status": "NOT_ASSESSED",
                "threshold": threshold,
                "notes": (
                    "At least two groups with the minimum sample size are required. "
                    "This metric is a screening measure, not evidence that a system is unbiased."
                ),
            }

        group_rates = {
            group: sum(1 for score in values if score >= threshold) / len(values)
            for group, values in by_group.items()
        }
        gap = max(group_rates.values()) - min(group_rates.values()) if group_rates else 0.0
        if gap > 0.35:
            status = "ALERT"
        elif gap > 0.2:
            status = "REVIEW_REQUIRED"
        else:
            status = "PASS"

        return {
            "gap": round(float(gap), 4),
            "group_rates": {group: round(float(rate), 4) for group, rate in group_rates.items()},
            "sample_counts": sample_counts,
            "status": status,
            "threshold": threshold,
            "notes": "This demographic-parity gap is a screening measure, not evidence that a system is unbiased.",
        }
