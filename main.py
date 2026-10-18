import json

from src.crediens_ai.data.pipeline import SyntheticDataBuilder
from src.crediens_ai.models.trainer import RiskModelTrainer
from src.crediens_ai.risk.engine import RiskAssessmentEngine


def main() -> None:
    dataset = SyntheticDataBuilder.generate_it_project_dataset(num_rows=400)
    trainer = RiskModelTrainer()
    training_result = trainer.train(dataset)
    print("MODEL TRAINING SUMMARY")
    print(json.dumps(training_result["metrics"], indent=2))

    engine = RiskAssessmentEngine()
    engine.register_model(
        industry="IT",
        risk_type="PROJECT",
        model_version="project-risk-v1",
        model=training_result["model"],
        feature_columns=training_result["feature_columns"],
    )
    scenarios = [
        {
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {
                "team_size": 12,
                "bug_count": 45,
                "critical_bugs": 8,
                "pending_tasks": 28,
                "test_coverage": 61,
                "technical_debt": "High",
            },
        },
        {
            "industry": "MANUFACTURING",
            "risk_type": "EQUIPMENT",
            "data": {
                "machine_age": 14,
                "vibration": 8.4,
                "temperature": 92,
                "maintenance_history": "Overdue",
                "operating_hours": 4800,
            },
        },
        {
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {
                "income": 90000,
                "loan_amount": 50000,
                "credit_history": "Good",
                "existing_debt": 20000,
            },
        },
    ]

    print("\nCONFIG-DRIVEN MODULE SUMMARY")
    for industry in engine.registry.get_supported_industries():
        for risk_type in engine.registry.get_risk_types(industry):
            module = engine.resolve_module(industry, risk_type)
            if module:
                print(f"- {industry}/{risk_type}: model={module.get('default_model_version') or module.get('model_version')} features={len(module.get('features', []))}")

    print("\nASSESSMENT OUTPUT")
    for scenario in scenarios:
        result = engine.evaluate(scenario)
        print(json.dumps({
            "industry": result["industry"],
            "risk_type": result["risk_type"],
            "risk_probability": result["risk_probability"],
            "risk_level": result["risk_level"],
            "decision": result["decision"],
            "model_version": result["model_version"],
            "scoring_method": result["scoring_method"],
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
