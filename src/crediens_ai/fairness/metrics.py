from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


class FairnessEvaluator:
    """Compute simple group fairness metrics used in governance reviews."""

    @staticmethod
    def demographic_parity_gap(protected: Iterable[str], scores: Iterable[float], threshold: float = 0.5) -> Dict[str, Any]:
        protected_list = list(protected)
        scores_list = list(scores)
        if len(protected_list) != len(scores_list):
            raise ValueError("protected and scores must have equal length")

        by_group: Dict[str, List[float]] = defaultdict(list)
        for group, score in zip(protected_list, scores_list):
            by_group[group].append(score)

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
            "status": status,
            "threshold": threshold,
        }
