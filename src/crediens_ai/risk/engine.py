from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd

from src.crediens_ai.audit.store import AuditStore
from src.crediens_ai.compliance.rules import ComplianceEngine
from src.crediens_ai.config import DEFAULT_MODEL_METADATA, DEFAULT_POLICY_BANDS
from src.crediens_ai.config.platform import get_module_config
from src.crediens_ai.dashboard.app import DashboardApp
from src.crediens_ai.explainability.explain import ExplanationEngine
from src.crediens_ai.fairness.metrics import FairnessEvaluator
from src.crediens_ai.industry.registry import IndustryRegistry
from src.crediens_ai.reporting.reports import ReportBuilder


class RiskAssessmentEngine:
    """Core risk-evaluation engine for the CrediLens AI blueprint."""

    def __init__(self, audit_store: AuditStore | None = None) -> None:
        self.model_metadata = {version: metadata.copy() for version, metadata in DEFAULT_MODEL_METADATA.items()}
        self.policy_bands = DEFAULT_POLICY_BANDS
        self.registry = IndustryRegistry()
        self.compliance_engine = ComplianceEngine()
        self.audit_store = audit_store or AuditStore()
        self.explanation_engine = ExplanationEngine(model=None)
        self.fairness_engine = FairnessEvaluator()
        self._registered_models: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    def register_model(
        self,
        industry: str,
        risk_type: str,
        model_version: str,
        model: Any,
        feature_columns: List[str],
    ) -> None:
        if not industry or not risk_type or not model_version:
            raise ValueError("Industry, risk type, and model version are required")
        if not feature_columns:
            raise ValueError("At least one model feature is required")
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("Registered models must implement predict_proba")
        self._registered_models[(industry, risk_type, model_version)] = {
            "model": model,
            "feature_columns": list(feature_columns),
        }

    def get_registered_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "industry": industry,
                "risk_type": risk_type,
                "model_version": model_version,
                "feature_columns": list(registration["feature_columns"]),
            }
            for (industry, risk_type, model_version), registration in sorted(self._registered_models.items())
        ]

    def resolve_module(self, industry: str, risk_type: str) -> Dict[str, Any]:
        configured = get_module_config(industry, risk_type)
        module = self.registry.get_module(industry, risk_type)
        if configured is not None:
            merged = {
                "industry": industry,
                "risk_type": risk_type,
                "features": list(configured.features),
                "default_model": configured.model_version,
                "default_model_version": configured.model_version,
                "model_version": configured.model_version,
                "policy": self.policy_bands,
                "categorical_features": configured.categorical_features,
                "numeric_bounds": configured.numeric_bounds,
                "monitor_threshold": configured.monitor_threshold,
                "human_review_threshold": configured.human_review_threshold,
            }
            if module:
                for key, value in module.items():
                    if key == "features":
                        continue
                    if key not in merged or key in {"default_model", "default_model_version", "model_version", "policy"}:
                        merged[key] = value
            return merged
        if module:
            return module
        return {}

    def _validate_inputs(self, payload: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not isinstance(payload.get("industry"), str) or not payload["industry"].strip():
            errors.append("Industry must be a non-empty string")
        if not isinstance(payload.get("risk_type"), str) or not payload["risk_type"].strip():
            errors.append("Risk type must be a non-empty string")
        if "data" not in payload or not isinstance(payload["data"], dict):
            errors.append("Data payload must be a dictionary")
        return errors

    def _risk_score(self, data: Dict[str, Any], industry: str, risk_type: str) -> float:
        base = 0.0
        if industry == "IT" and risk_type == "PROJECT":
            base += data.get("bug_count", 0) * 0.006
            base += data.get("critical_bugs", 0) * 0.02
            base += data.get("pending_tasks", 0) * 0.004
            base += data.get("team_size", 0) * 0.008
            base += data.get("test_coverage", 0) * -0.002
            debt = data.get("technical_debt", "Low")
            if isinstance(debt, str):
                debt_map = {"Low": 0.05, "Medium": 0.12, "High": 0.18, "Critical": 0.28}
                base += debt_map.get(debt, 0.0)
        elif industry == "IT" and risk_type == "CYBER":
            base += data.get("failed_logins", 0) * 0.004
            base += data.get("vulnerabilities", 0) * 0.008
            base += data.get("critical_vulnerabilities", 0) * 0.035
            base += data.get("patch_age", 0) * 0.003
            base += max(0.0, 100.0 - data.get("mfa_coverage", 100)) * 0.003
        elif industry == "IT" and risk_type == "OPERATIONAL":
            base += data.get("code_complexity", 0) * 0.004
            base += data.get("bug_density", 0) * 0.025
            base += data.get("open_issues", 0) * 0.004
            base += max(0.0, 100.0 - data.get("sprint_velocity", 100)) * 0.002
            base += data.get("failed_deployments", 0) * 0.03
            base += max(0.0, 100.0 - data.get("test_coverage", 100)) * 0.002
            debt_map = {"Low": 0.02, "Medium": 0.06, "High": 0.12, "Critical": 0.2}
            base += debt_map.get(data.get("technical_debt"), 0.0)
        elif industry == "BANKING" and risk_type == "CREDIT":
            base += data.get("existing_debt", 0) / 100000.0 * 0.35
            base += data.get("loan_amount", 0) / 100000.0 * 0.2
            credit_history = data.get("credit_history", "Fair")
            quality_map = {"Poor": 0.3, "Fair": 0.15, "Good": 0.05, "Excellent": 0.02}
            base += quality_map.get(credit_history, 0.1)
            base += max(0.0, 10.0 - data.get("employment_years", 10)) * 0.005
        elif industry == "BANKING" and risk_type == "FRAUD":
            base += data.get("transaction_amount", 0) / 100000.0 * 0.08
            base += data.get("anomaly_score", 0) * 0.45
            base += data.get("velocity", 0) * 0.025
            base += max(0.0, 60.0 - data.get("account_age", 60)) * 0.003
        elif industry == "MANUFACTURING" and risk_type == "EQUIPMENT":
            base += data.get("machine_age", 0) * 0.01
            base += data.get("vibration", 0) * 0.08
            base += data.get("temperature", 0) * 0.02
            base += data.get("operating_hours", 0) / 1000.0 * 0.12
            if data.get("maintenance_history") in {"Overdue", "Late"}:
                base += 0.2
        elif industry == "MANUFACTURING" and risk_type == "SUPPLY_CHAIN":
            base += max(0.0, 100.0 - data.get("supplier_reliability", 100)) * 0.004
            base += data.get("delivery_delay", 0) * 0.012
            base += max(0.0, 100.0 - data.get("inventory_level", 100)) * 0.003
            base += data.get("supplier_concentration", 0) * 0.003
        else:
            raise ValueError(f"No heuristic scorer is configured for {industry}/{risk_type}")

        capped = max(0.0, min(1.0, base))
        return round(capped, 4)

    def _assign_risk_band(
        self,
        probability: float,
        policy_bands: Dict[str, tuple[float, float]],
    ) -> str:
        for band, (lower, upper) in policy_bands.items():
            if lower <= probability <= upper:
                return band
        raise ValueError(f"Risk probability {probability} is not covered by the configured policy bands")

    def _determine_policy(self, probability: float, module: Dict[str, Any]) -> str:
        if probability >= module.get("human_review_threshold", 0.55):
            return "HUMAN_REVIEW"
        if probability >= module.get("monitor_threshold", 0.4):
            return "MONITOR"
        return "APPROVE"

    def _build_compliance_result(self, risk_band: str, policy: str) -> Dict[str, Any]:
        status = "COMPLIANT"
        if risk_band == "HIGH" or policy == "HUMAN_REVIEW":
            status = "REVIEW_REQUIRED"
        return {
            "status": status,
            "checks": {
                "explanation_available": True,
                "required_input_fields_present": True,
                "model_version_recorded": True,
                "prediction_timestamp_recorded": True,
                "audit_metadata_present": True,
                "fairness_status_available": True,
                "human_review_requirement_satisfied": policy == "HUMAN_REVIEW",
                "decision_policy_applied_consistently": True,
            },
            "notes": "Automated governance checks passed or flagged for human review.",
        }

    def _build_audit_record(self, payload: Dict[str, Any], probability: float, risk_band: str, policy: str) -> Dict[str, Any]:
        return {
            "application_id": payload.get("application_id", "demo-app-001"),
            "organization": payload.get("organization", "demo-org"),
            "industry": payload["industry"],
            "risk_type": payload["risk_type"],
            "model_version": payload.get("model_version", "project-risk-v1"),
            "feature_version": "feature-v1",
            "ruleset_version": "ruleset-v1",
            "input_validation_result": "VALID",
            "final_risk_probability": probability,
            "risk_band": risk_band,
            "policy_outcome": policy,
            "fairness_status": "NOT_ASSESSED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewer": payload.get("reviewer"),
        }

    def _build_evidence_record(
        self,
        data: Dict[str, Any],
        module: Dict[str, Any],
        risk_probability: float,
        risk_level: str,
        top_factors: List[str],
        compliance: Dict[str, Any],
    ) -> Dict[str, Any]:
        required_fields = list(module.get("features", []))
        missing_fields = [field for field in required_fields if field not in data]
        confidence = 0.9
        if missing_fields:
            confidence -= min(0.5, 0.08 * len(missing_fields))
        if risk_level == "HIGH":
            confidence -= 0.05
        confidence = max(0.1, min(0.99, confidence))
        factors = []
        for factor in top_factors[:5]:
            value = data.get(factor)
            factors.append({
                "feature": factor,
                "value": value,
                "evidence_strength": "strong" if factor in top_factors[:3] else "supporting",
                "direction": "increases_risk" if value is not None else "unknown",
            })
        return {
            "confidence": round(confidence, 2),
            "missing_fields": missing_fields,
            "summary": (
                f"{risk_level} risk assessment based on {', '.join(top_factors[:3]) or 'available indicators'} "
                f"with {len(missing_fields)} missing required field(s)."
            ),
            "factors": factors,
            "policy_checks": compliance.get("checks", {}),
            "evidence_status": "partial" if missing_fields else "complete",
        }

    def _build_decision_trace(
        self,
        payload: Dict[str, Any],
        module: Dict[str, Any],
        probability: float,
        risk_band: str,
        policy: str,
        evidence: Dict[str, Any],
        compliance: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "input_snapshot": payload,
            "module_definition": {
                "industry": module.get("industry"),
                "risk_type": module.get("risk_type"),
                "features": module.get("features", []),
                "model_version": module.get("model_version") or module.get("default_model_version") or module.get("default_model"),
                "policy": module.get("policy", {}),
                "monitor_threshold": module.get("monitor_threshold", 0.4),
                "human_review_threshold": module.get("human_review_threshold", 0.55),
            },
            "prediction": {
                "risk_probability": probability,
                "risk_band": risk_band,
                "decision": policy,
            },
            "explanation_summary": evidence.get("summary"),
            "policy_checks": compliance.get("checks", {}),
            "reviewer_comment": payload.get("reviewer_comment"),
            "human_decision": payload.get("final_decision"),
            "audit_record_id": payload.get("application_id", "demo-app-001"),
        }

    def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self.evaluate(payload)
        if result.get("status") != "SUCCESS":
            return result
        return {
            "assessment_id": result.get("audit_id"),
            "industry": result["industry"],
            "risk_type": result["risk_type"],
            "risk_probability": result["risk_probability"],
            "risk_level": result["risk_level"],
            "explainability": result.get("explainability", {}),
            "fairness": result.get("fairness", {}),
        }

    def get_audit_record(self, audit_id: str) -> Dict[str, Any]:
        try:
            return self.audit_store.get(audit_id)
        except KeyError:
            return {"status": "NOT_FOUND", "audit_id": audit_id}

    def get_compliance_record(self, audit_id: str) -> Dict[str, Any]:
        record = self.get_audit_record(audit_id)
        if record.get("status") == "NOT_FOUND":
            return record
        governance = record.get("governance")
        if governance is not None:
            return governance
        return {"status": "COMPLIANT", "checks": {}, "notes": "No governance data found for this audit record."}

    def record_human_decision(
        self,
        audit_id: str,
        reviewer: str | None = None,
        reviewer_comment: str | None = None,
        final_decision: str | None = None,
    ) -> Dict[str, Any]:
        record = self.get_audit_record(audit_id)
        if record.get("status") == "NOT_FOUND":
            return record

        if reviewer is not None:
            record["reviewer"] = reviewer
        if reviewer_comment is not None:
            record["reviewer_comment"] = reviewer_comment
        if final_decision is not None:
            record["final_decision"] = final_decision
            record["decision_trace"] = record.get("decision_trace", {})
            record["decision_trace"]["human_decision"] = final_decision
            record["decision_trace"]["reviewer_comment"] = reviewer_comment
            record["decision_trace"]["reviewer"] = reviewer
        compliance_context = {
            **record,
            "decision_trace": record.get("decision_trace", {}),
            "risk_policy": record.get("decision_trace", {}).get("module_definition", {}).get("policy", {}),
            "human_review_threshold": record.get("decision_trace", {}).get("module_definition", {}).get("human_review_threshold", 0.55),
            "monitor_threshold": record.get("decision_trace", {}).get("module_definition", {}).get("monitor_threshold", 0.4),
            "explainability": record.get("explainability", {}),
            "fairness": record.get("fairness", {}),
            "evidence": record.get("evidence", {}),
        }
        record["governance"] = self.compliance_engine.evaluate(compliance_context)
        record["compliance_status"] = record["governance"]["status"]
        self.audit_store.update(audit_id, record)
        return record

    def evaluate(self, payload: Dict[str, Any], *, persist_audit: bool = True) -> Dict[str, Any]:
        errors = self._validate_inputs(payload)
        if errors:
            return {"status": "INVALID_INPUT", "errors": errors}

        industry = payload["industry"]
        industry = industry.strip().upper()
        risk_type = payload["risk_type"].strip().upper()
        module = self.resolve_module(industry, risk_type)
        if not module:
            return {
                "status": "INVALID_INPUT",
                "errors": [f"Unsupported industry/risk combination: {industry}/{risk_type}"],
            }

        data = payload["data"]
        expected_features = list(module.get("features", []))
        missing_fields = [feature for feature in expected_features if feature not in data]
        if missing_fields:
            return {
                "status": "INVALID_INPUT",
                "errors": [f"Missing required fields: {', '.join(missing_fields)}"],
            }
        unexpected_fields = [feature for feature in data if feature not in expected_features]
        if unexpected_fields:
            return {
                "status": "INVALID_INPUT",
                "errors": [f"Unexpected fields: {', '.join(unexpected_fields)}"],
            }
        categorical_features = module.get("categorical_features", {})
        for feature in expected_features:
            value = data[feature]
            if feature in categorical_features:
                if not isinstance(value, str) or value not in categorical_features[feature]:
                    return {
                        "status": "INVALID_INPUT",
                        "errors": [f"{feature} must be one of: {', '.join(categorical_features[feature])}"],
                    }
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                return {
                    "status": "INVALID_INPUT",
                    "errors": [f"{feature} must be a finite number"],
                }
            else:
                try:
                    finite_value = math.isfinite(value)
                except OverflowError:
                    finite_value = False
                if not finite_value:
                    return {
                        "status": "INVALID_INPUT",
                        "errors": [f"{feature} must be a finite number"],
                    }
                lower, upper = module.get("numeric_bounds", {}).get(feature, (0, None))
                if value < lower:
                    return {
                        "status": "INVALID_INPUT",
                        "errors": [f"{feature} must be at least {lower}"],
                    }
                if upper is not None and value > upper:
                    return {
                        "status": "INVALID_INPUT",
                        "errors": [f"{feature} must be at most {upper}"],
                    }
        payload = {**payload, "industry": industry, "risk_type": risk_type}
        model_version = payload.get("model_version") or module.get(
            "default_model",
            module.get("default_model_version", "project-risk-v1"),
        )
        registered_model = self._registered_models.get((industry, risk_type, model_version))
        scoring_method = "HEURISTIC"
        if payload.get("model_version") and registered_model is None:
            return {
                "status": "INVALID_INPUT",
                "errors": [f"Model version is not registered for {industry}/{risk_type}: {model_version}"],
            }
        if registered_model:
            feature_columns = registered_model["feature_columns"]
            missing_features = [feature for feature in feature_columns if feature not in data]
            if missing_features:
                return {
                    "status": "INVALID_INPUT",
                    "errors": [f"Missing required model features: {', '.join(missing_features)}"],
                }
            sample = pd.DataFrame([{feature: data[feature] for feature in feature_columns}])
            try:
                probabilities = registered_model["model"].predict_proba(sample)
                probability = float(probabilities[0][1])
            except (ValueError, IndexError, TypeError, OverflowError) as error:
                return {"status": "INVALID_INPUT", "errors": [f"Model input is invalid: {error}"]}
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                return {
                    "status": "INVALID_INPUT",
                    "errors": ["Model returned a risk probability outside the valid range [0, 1]"],
                }
            scoring_method = "MODEL"
        else:
            probability = self._risk_score(data, industry, risk_type)
        risk_band = self._assign_risk_band(probability, module.get("policy") or self.policy_bands)
        policy = self._determine_policy(probability, module)
        audit_payload = {**payload, "model_version": model_version}
        audit = self._build_audit_record(audit_payload, probability, risk_band, policy)
        audit["scoring_method"] = scoring_method
        audit_id = self.audit_store.add(audit) if persist_audit else f"demo-{uuid4()}"
        top_factors = self._top_factors(data, industry, risk_type)
        explanation = self.explanation_engine.shap_global([[data.get(factor, 0) for factor in top_factors]])
        local_explanation = self.explanation_engine.lime_local([[data.get(factor, 0) for factor in top_factors]], [1])
        protected_group = payload.get("protected_group")
        fairness = {
            "status": "NOT_ASSESSED",
            "gap": None,
            "group_rates": {},
            "sample_counts": {},
            "notes": "Fairness requires sufficient observations across multiple groups; no group data was provided.",
        }
        if protected_group is not None:
            fairness = self.fairness_engine.demographic_parity_gap([protected_group], [probability], threshold=0.5)
        audit["fairness_status"] = fairness["status"]
        assessment = {
            "status": "SUCCESS",
            "industry": industry,
            "risk_type": risk_type,
            "model_version": model_version,
            "scoring_method": scoring_method,
            "risk_probability": probability,
            "risk_level": risk_band,
            "decision": policy,
            "fairness_status": fairness["status"],
            "timestamp": audit["timestamp"],
            "audit": audit,
            "top_factors": top_factors,
            "explainability": {
                "global": explanation,
                "local": local_explanation,
            },
            "fairness": fairness,
            "risk_policy": module.get("policy", self.policy_bands),
            "monitor_threshold": module.get("monitor_threshold", 0.4),
            "human_review_threshold": module.get("human_review_threshold", 0.55),
        }
        compliance = self.compliance_engine.evaluate(assessment)
        assessment["compliance_status"] = compliance["status"]
        assessment["governance"] = compliance
        assessment["model_metadata"] = self.model_metadata.get(assessment["model_version"], {})
        assessment["evidence"] = self._build_evidence_record(data, module, probability, risk_band, top_factors, compliance)
        assessment["decision_trace"] = self._build_decision_trace(audit_payload, module, probability, risk_band, policy, assessment["evidence"], compliance)
        assessment["audit_id"] = audit_id
        assessment["report"] = ReportBuilder.build_json_report(assessment)
        assessment["html_report"] = ReportBuilder.build_html_report(assessment)
        assessment["dashboard"] = DashboardApp.build_summary(assessment)
        if persist_audit:
            self.audit_store.update(audit_id, {**audit, **assessment})
        return assessment

    def _top_factors(self, data: Dict[str, Any], industry: str, risk_type: str) -> List[str]:
        module = self.resolve_module(industry, risk_type)
        return [feature for feature in module.get("features", []) if feature in data]
