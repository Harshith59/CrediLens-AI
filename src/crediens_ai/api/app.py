import atexit
import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.crediens_ai.audit.store import AuditStore
from src.crediens_ai.dashboard.app import DashboardApp
from src.crediens_ai.industry.registry import IndustryRegistry
from src.crediens_ai.models.artifacts import load_model_artifact
from src.crediens_ai.reporting.reports import ReportBuilder
from src.crediens_ai.risk.engine import RiskAssessmentEngine


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry: str = Field(..., description="Industry name such as IT or BANKING")
    risk_type: str = Field(..., description="Risk type, such as PROJECT or CREDIT")
    data: Dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None
    organization: str | None = None
    application_id: str | None = None


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry: str = Field(..., description="Industry name")
    risk_type: str = Field(..., description="Risk type")
    data: Dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None


class DemoAssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry: str = Field(..., min_length=1, max_length=32)
    risk_type: str = Field(..., min_length=1, max_length=32)
    data: Dict[str, Any] = Field(..., max_length=20)

    @field_validator("data")
    @classmethod
    def validate_demo_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        for name, value in data.items():
            if not isinstance(name, str) or len(name) > 80:
                raise ValueError("Input field names must be strings of at most 80 characters")
            if isinstance(value, str) and len(value) > 100:
                raise ValueError(f"{name} must be at most 100 characters")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"{name} must be a simple string or number")
        return data


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_comment: str | None = Field(default=None, max_length=2000)
    final_decision: Literal["APPROVED", "REJECTED", "APPROVED_WITH_CONDITIONS", "ESCALATED"]


def require_role(required_role: Literal["analyst", "reviewer"]):
    def dependency(api_key: str | None = Header(default=None, alias="X-API-Key")) -> Dict[str, str]:
        configured_keys = os.environ.get("CREDILENS_API_KEYS")
        if not configured_keys:
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        try:
            key_records = json.loads(configured_keys)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=503, detail="API authentication configuration is invalid") from error
        if not isinstance(key_records, dict):
            raise HTTPException(status_code=503, detail="API authentication configuration is invalid")

        matched_principal: Dict[str, str] | None = None
        for configured_key, principal in key_records.items():
            if (
                not isinstance(configured_key, str)
                or len(configured_key) < 32
                or not isinstance(principal, dict)
                or principal.get("role") not in {"analyst", "reviewer", "admin"}
                or not isinstance(principal.get("subject"), str)
                or not principal["subject"].strip()
            ):
                raise HTTPException(status_code=503, detail="API authentication configuration is invalid")
            if api_key is not None and secrets.compare_digest(api_key, configured_key):
                matched_principal = {
                    "role": principal["role"],
                    "subject": principal["subject"],
                }

        if matched_principal is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        permitted_roles = {
            "analyst": {"analyst", "reviewer", "admin"},
            "reviewer": {"reviewer", "admin"},
        }
        if matched_principal["role"] not in permitted_roles[required_role]:
            raise HTTPException(status_code=403, detail="Insufficient role for this operation")
        return matched_principal

    return dependency


audit_database_path = os.environ.get("CREDILENS_AUDIT_DB", ".credilens/audit.sqlite3")
engine = RiskAssessmentEngine(audit_store=AuditStore(audit_database_path))
atexit.register(engine.close)

DEMO_SAMPLE_CASES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "BANKING": {
        "CREDIT": {
            "income": 90000,
            "loan_amount": 50000,
            "credit_history": "Good",
            "existing_debt": 20000,
            "employment_years": 10,
        },
        "FRAUD": {
            "transaction_amount": 2400,
            "anomaly_score": 0.72,
            "velocity": 4,
            "account_age": 18,
        },
    },
    "IT": {
        "CYBER": {
            "failed_logins": 24,
            "vulnerabilities": 7,
            "critical_vulnerabilities": 1,
            "patch_age": 30,
            "mfa_coverage": 82,
        },
        "OPERATIONAL": {
            "code_complexity": 55,
            "bug_density": 2.4,
            "open_issues": 18,
            "sprint_velocity": 75,
            "technical_debt": "Medium",
            "test_coverage": 68,
            "failed_deployments": 2,
        },
        "PROJECT": {
            "team_size": 12,
            "bug_count": 45,
            "critical_bugs": 8,
            "pending_tasks": 28,
            "test_coverage": 61,
            "technical_debt": "High",
        },
    },
    "MANUFACTURING": {
        "EQUIPMENT": {
            "machine_age": 14,
            "vibration": 8.4,
            "temperature": 92,
            "maintenance_history": "Overdue",
            "operating_hours": 4800,
        },
        "SUPPLY_CHAIN": {
            "supplier_reliability": 85,
            "delivery_delay": 5,
            "inventory_level": 45,
            "supplier_concentration": 60,
        },
    },
}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    artifact_path = os.environ.get("CREDILENS_MODEL_ARTIFACT")
    if artifact_path:
        artifact = load_model_artifact(artifact_path)
        engine.register_model(
            industry=artifact["industry"],
            risk_type=artifact["risk_type"],
            model_version=artifact["model_version"],
            model=artifact["model"],
            feature_columns=artifact["feature_columns"],
        )
        engine.model_metadata[artifact["model_version"]] = {
            "model_family": getattr(artifact["model"], "model_family", type(artifact["model"]).__name__),
            "training_metrics": artifact["metrics"],
            "model_metrics": artifact.get("model_metrics", {}),
            "fusion_weights": artifact.get("fusion_weights", {}),
        }
    yield


app = FastAPI(title="CrediLens AI", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    if request.url.path != "/api/demo/assessments":
        return await request_validation_exception_handler(request, error)
    safe_errors = [
        {key: value for key, value in item.items() if key != "input"}
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(safe_errors)},
    )


@app.middleware("http")
async def limit_demo_request_size(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.url.path == "/api/demo/assessments":
        content_length = request.headers.get("content-length")
        if content_length is None:
            return JSONResponse(status_code=411, content={"detail": "Content-Length is required"})
        try:
            payload_size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
        if payload_size > 16384:
            return JSONResponse(status_code=413, content={"detail": "Demo assessment payload exceeds 16 KB"})
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(
        DashboardApp.page(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/catalog")
def public_catalog() -> Dict[str, Any]:
    industries = engine.registry.get_supported_industries()
    modules = {
        industry: engine.registry.get_risk_types(industry)
        for industry in industries
    }
    return {
        "industries": industries,
        "modules": modules,
        "module_count": sum(len(risk_types) for risk_types in modules.values()),
    }


@app.get("/api/demo/modules")
def demo_modules() -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    for industry in engine.registry.get_supported_industries():
        modules[industry] = {}
        for risk_type in engine.registry.get_risk_types(industry):
            module = engine.resolve_module(industry, risk_type)
            modules[industry][risk_type] = {
                "features": module.get("features", []),
                "categorical_features": module.get("categorical_features", {}),
                "numeric_bounds": module.get("numeric_bounds", {}),
                "sample": DEMO_SAMPLE_CASES[industry][risk_type],
            }
    return {"modules": modules, "sample_data_label": "Synthetic demonstration inputs"}


@app.post("/api/demo/assessments")
def demo_assessment(payload: DemoAssessmentRequest) -> Dict[str, Any]:
    result = engine.evaluate(
        {
            "industry": payload.industry,
            "risk_type": payload.risk_type,
            "data": payload.data,
        },
        persist_audit=False,
    )
    if result.get("status") == "INVALID_INPUT":
        raise HTTPException(status_code=422, detail=result["errors"])

    scoring_method = result["scoring_method"]
    if scoring_method == "MODEL":
        method_description = (
            "A registered trained model produced this prediction score. Feature-level causal attribution "
            "is not configured for this public demonstration."
        )
    else:
        method_description = (
            "A rule-based heuristic produced this score; it is not a trained-model prediction. "
            "Inputs are listed for transparency, not as causal feature attributions."
        )

    recommendations = {
        "HUMAN_REVIEW": "Policy requires qualified human review before a real-world decision.",
        "MONITOR": "Policy suggests monitoring and reassessing as conditions change.",
        "APPROVE": (
            "Configured thresholds did not trigger monitoring or human review. "
            "Independently review the evidence before any real-world decision."
        ),
    }
    explanation = {
        "status": "LIMITED",
        "attribution_available": False,
        "summary": method_description,
        "limitation": (
            "No feature-level SHAP or LIME attribution is configured for this public report."
        ),
    }
    recommendation = recommendations[result["decision"]]
    report = ReportBuilder.build_demo_json_report(
        industry=result["industry"],
        risk_type=result["risk_type"],
        inputs=payload.data,
        risk_probability=result["risk_probability"],
        risk_level=result["risk_level"],
        decision=result["decision"],
        scoring_method=scoring_method,
        scoring_method_description=method_description,
        timestamp=result["timestamp"],
        recommendation=recommendation,
        explanation=explanation,
    )
    return {
        "status": "SUCCESS",
        "industry": result["industry"],
        "risk_type": result["risk_type"],
        "risk_probability": result["risk_probability"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "scoring_method": scoring_method,
        "method_description": method_description,
        "input_factors": [
            {"name": name, "value": value}
            for name, value in payload.data.items()
        ],
        "recommendation": recommendation,
        "explanation": explanation,
        "disclaimer": report["disclaimer"],
        "automated_check_status": result["compliance_status"],
        "timestamp": result["timestamp"],
        "report": report,
        "demo": True,
        "persisted": False,
    }


@app.get("/api/model-info", dependencies=[Depends(require_role("analyst"))])
def model_info() -> Dict[str, Any]:
    return {
        "model_versions": engine.model_metadata,
        "supported_industries": engine.registry.get_supported_industries(),
        "registered_models": engine.get_registered_models(),
    }


@app.post("/api/risk/predict", dependencies=[Depends(require_role("analyst"))])
def predict(payload: RiskRequest) -> Dict[str, Any]:
    result = engine.evaluate(payload.model_dump())
    if result.get("status") == "INVALID_INPUT":
        raise HTTPException(status_code=422, detail=result["errors"])
    return result


@app.post("/api/risk/explain", dependencies=[Depends(require_role("analyst"))])
def explain(payload: ExplainRequest) -> Dict[str, Any]:
    result = engine.explain(payload.model_dump())
    if result.get("status") == "INVALID_INPUT":
        raise HTTPException(status_code=422, detail=result["errors"])
    return result


def _get_record_or_404(audit_id: str) -> Dict[str, Any]:
    record = engine.get_audit_record(audit_id)
    if record.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record


@app.get("/api/risk/{risk_id}", dependencies=[Depends(require_role("analyst"))])
def get_risk(risk_id: str) -> Dict[str, Any]:
    return _get_record_or_404(risk_id)


@app.get("/api/audit", dependencies=[Depends(require_role("analyst"))])
def list_audits(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    industry: str | None = None,
    risk_type: str | None = None,
    decision: str | None = None,
) -> Dict[str, Any]:
    records = engine.audit_store.list(
        limit=limit,
        offset=offset,
        industry=industry.upper() if industry else None,
        risk_type=risk_type.upper() if risk_type else None,
        decision=decision.upper() if decision else None,
    )
    return {
        "items": [
            {
                key: record.get(key)
                for key in (
                    "audit_id",
                    "application_id",
                    "organization",
                    "industry",
                    "risk_type",
                    "timestamp",
                    "risk_probability",
                    "risk_level",
                    "decision",
                    "compliance_status",
                    "scoring_method",
                )
            }
            for record in records
        ],
        "total": engine.audit_store.count(
            industry=industry.upper() if industry else None,
            risk_type=risk_type.upper() if risk_type else None,
            decision=decision.upper() if decision else None,
        ),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/audit/{audit_id}", dependencies=[Depends(require_role("analyst"))])
def get_audit(audit_id: str) -> Dict[str, Any]:
    return _get_record_or_404(audit_id)


@app.get("/api/audit/{audit_id}/events", dependencies=[Depends(require_role("analyst"))])
def get_audit_events(audit_id: str) -> Dict[str, Any]:
    _get_record_or_404(audit_id)
    return {"audit_id": audit_id, "events": engine.audit_store.events(audit_id)}


@app.get("/api/compliance/{audit_id}", dependencies=[Depends(require_role("analyst"))])
def get_compliance(audit_id: str) -> Dict[str, Any]:
    _get_record_or_404(audit_id)
    return engine.get_compliance_record(audit_id)


@app.get("/api/reports/{audit_id}/json", dependencies=[Depends(require_role("analyst"))])
def get_json_report(audit_id: str) -> Dict[str, Any]:
    record = _get_record_or_404(audit_id)
    return ReportBuilder.build_json_report(record)


@app.get("/api/reports/{audit_id}/html", dependencies=[Depends(require_role("analyst"))])
def get_html_report(audit_id: str) -> HTMLResponse:
    record = _get_record_or_404(audit_id)
    return HTMLResponse(
        ReportBuilder.build_html_report(record),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/reports/{audit_id}/png", dependencies=[Depends(require_role("analyst"))])
def get_png_report(audit_id: str) -> Response:
    record = _get_record_or_404(audit_id)
    return Response(
        content=ReportBuilder.build_png_report(record),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/audit/{audit_id}/decision", dependencies=[Depends(require_role("analyst"))])
def get_decision(audit_id: str) -> Dict[str, Any]:
    record = _get_record_or_404(audit_id)
    return {
        "audit_id": audit_id,
        "status": record.get("status", "SUCCESS"),
        "decision": record.get("decision"),
        "decision_trace": record.get("decision_trace", {}),
        "evidence": record.get("evidence", {}),
    }


@app.post("/api/audit/{audit_id}/review")
def review_decision(
    audit_id: str,
    payload: ReviewDecisionRequest,
    principal: Dict[str, str] = Depends(require_role("reviewer")),
) -> Dict[str, Any]:
    record = engine.record_human_decision(
        audit_id,
        reviewer=principal["subject"],
        reviewer_comment=payload.reviewer_comment,
        final_decision=payload.final_decision,
    )
    if record.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Audit record not found")
    return record


@app.get("/api/summary", dependencies=[Depends(require_role("analyst"))])
def project_summary() -> Dict[str, Any]:
    records = engine.audit_store.list(limit=1)
    return {
        "total_audits": engine.audit_store.count(),
        "pending_human_review": engine.audit_store.pending_review_count(),
        "total_industries": len(engine.registry.get_supported_industries()),
        "supported_industries": engine.registry.get_supported_industries(),
        "module_count": sum(
            len(IndustryRegistry.get_risk_types(industry))
            for industry in IndustryRegistry.get_supported_industries()
        ),
        "latest_audit_id": records[0].get("audit_id") if records else None,
        "latest_decision": records[0].get("decision") if records else None,
    }


@app.get("/api/system/status", dependencies=[Depends(require_role("analyst"))])
def system_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "application": "CrediLens AI",
        "version": "0.1.0",
        "total_audits": engine.audit_store.count(),
        "pending_human_review": engine.audit_store.pending_review_count(),
        "supported_industries": engine.registry.get_supported_industries(),
        "module_count": sum(len(IndustryRegistry.get_risk_types(industry)) for industry in IndustryRegistry.get_supported_industries()),
    }


@app.get("/api/config/industries", dependencies=[Depends(require_role("analyst"))])
def industries() -> Dict[str, Any]:
    return {
        "industries": {
            industry: {"risk_types": IndustryRegistry.get_risk_types(industry)}
            for industry in IndustryRegistry.get_supported_industries()
        }
    }


@app.get("/api/config/risk-types", dependencies=[Depends(require_role("analyst"))])
def risk_types() -> Dict[str, Any]:
    return {"risk_types": {industry: IndustryRegistry.get_risk_types(industry) for industry in IndustryRegistry.get_supported_industries()}}


@app.get("/api/config/modules", dependencies=[Depends(require_role("analyst"))])
def modules() -> Dict[str, Any]:
    response: Dict[str, Any] = {}
    for industry in IndustryRegistry.get_supported_industries():
        response[industry] = {}
        for risk_type in IndustryRegistry.get_risk_types(industry):
            module = engine.resolve_module(industry, risk_type)
            if module:
                response[industry][risk_type] = {
                    "model_version": module.get("default_model_version") or module.get("model_version") or module.get("default_model"),
                    "features": module.get("features", []),
                    "categorical_features": module.get("categorical_features", {}),
                    "policy": module.get("policy", {}),
                    "monitor_threshold": module.get("monitor_threshold", 0.4),
                    "human_review_threshold": module.get("human_review_threshold", 0.55),
                }
    return {"modules": response}
