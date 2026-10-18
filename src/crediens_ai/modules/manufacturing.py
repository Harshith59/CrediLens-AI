from __future__ import annotations

from typing import Any, Dict, List


class ManufacturingRiskModule:
    """Manufacturing equipment risk scoring module aligned to the blueprint."""

    @staticmethod
    def score_equipment(data: Dict[str, Any]) -> float:
        base = 0.0
        base += data.get("machine_age", 0) * 0.01
        base += data.get("vibration", 0) * 0.08
        base += data.get("temperature", 0) * 0.02
        base += data.get("operating_hours", 0) / 1000.0 * 0.12
        if data.get("maintenance_history") in {"Overdue", "Late"}:
            base += 0.2
        return max(0.0, min(1.0, round(base, 4)))

    @staticmethod
    def risk_band(probability: float) -> str:
        if probability >= 0.51:
            return "HIGH"
        if probability >= 0.21:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def required_features() -> List[str]:
        return [
            "machine_age",
            "vibration",
            "temperature",
            "maintenance_history",
            "operating_hours",
        ]
