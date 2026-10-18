import os
import json
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field

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
