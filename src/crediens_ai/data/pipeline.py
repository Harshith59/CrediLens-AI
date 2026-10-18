from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticDataBuilder:
    """Create synthetic tabular data for the IT project risk module."""

    @staticmethod
    def generate_it_project_dataset(num_rows: int = 500, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        team_size = rng.integers(4, 25, size=num_rows)
        bug_count = rng.integers(5, 90, size=num_rows)
        critical_bugs = rng.integers(0, 15, size=num_rows)
        pending_tasks = rng.integers(5, 70, size=num_rows)
        test_coverage = rng.integers(20, 95, size=num_rows)
        technical_debt = rng.choice(["Low", "Medium", "High", "Critical"], size=num_rows, p=[0.45, 0.25, 0.2, 0.1])

        debt_score = np.select(
            [technical_debt == "Low", technical_debt == "Medium", technical_debt == "High", technical_debt == "Critical"],
            [0.0, 0.3, 0.7, 1.1],
            default=0.0,
        )

        signal = (
            0.024 * bug_count
            + 0.12 * critical_bugs
            + 0.03 * pending_tasks
            + 0.025 * (25 - team_size)
            - 0.022 * test_coverage
            + 0.8 * debt_score
        )
        signal += rng.normal(0.0, 0.35, size=num_rows)
        probability = 1 / (1 + np.exp(-signal))
        target = (probability >= 0.58).astype(int)

        dataset = pd.DataFrame(
            {
                "team_size": team_size,
                "bug_count": bug_count,
                "critical_bugs": critical_bugs,
                "pending_tasks": pending_tasks,
                "test_coverage": test_coverage,
                "technical_debt": technical_debt,
                "target": target,
            }
        )
        return dataset
