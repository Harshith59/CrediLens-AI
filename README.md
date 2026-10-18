# CrediLens AI

CrediLens AI is a multi-industry risk intelligence platform blueprint implemented as a runnable starter project. The design follows the blueprint for explainable AI risk assessment, governance, compliance, and audit workflows across credit, IT, manufacturing, and other configurable domains.

## What is included

- Configurable industry and risk-type metadata
- Module-aware required features, value validation, risk scoring, risk bands, and policy decisions
- Optional fitted-model inference registered per industry and risk type, with an explicit heuristic fallback for modules without a trained model
- Governance and compliance checks with per-rule evidence
- SQLite-persisted audit records and review-event history
- Minimal FastAPI-based REST API
- Example CLI runner
- Regression tests for the core engine

## Project layout

```text
CrediLens-AI/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   └── crediens_ai/
│       ├── __init__.py
│       ├── config.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── app.py
│       ├── risk/
│       │   ├── __init__.py
│       │   └── engine.py
│       └── tests/
│           └── __init__.py
└── tests/
    └── test_risk_platform.py
```

## Quick start

```bash
cd CrediLens-AI
C:/ProgramData/anaconda3/python.exe -m pip install -r requirements.txt

# Run the demo assessment flow
C:/ProgramData/anaconda3/python.exe main.py

# Train and save the synthetic IT project-risk reference model
C:/ProgramData/anaconda3/python.exe train_model.py --output models/it/project-risk-v1.joblib

# Run the regression suite
C:/ProgramData/anaconda3/python.exe -m unittest discover -s tests -v
```

## API access and browser dashboard

The API requires an analyst, reviewer, or admin API key for private data and configuration endpoints. `/health`, the static `/dashboard` shell, and the metadata-only `/api/catalog` are public; the catalog lists supported industries and risk modules but does not expose audit or model data. Generate high-entropy keys locally and configure the key-to-role map in the environment before starting the service. Do not commit keys to source control.

```powershell
$analystKey = "$([guid]::NewGuid().ToString('N'))$([guid]::NewGuid().ToString('N'))"
$reviewerKey = "$([guid]::NewGuid().ToString('N'))$([guid]::NewGuid().ToString('N'))"
$apiKeys = @{}
$apiKeys[$analystKey] = @{ role = "analyst"; subject = "local-analyst" }
$apiKeys[$reviewerKey] = @{ role = "reviewer"; subject = "local-reviewer" }
$env:CREDILENS_API_KEYS = $apiKeys | ConvertTo-Json -Compress

# Optional: set this after training if the API should use the fitted model.
$env:CREDILENS_MODEL_ARTIFACT = "models\it\project-risk-v1.joblib"

# Optional: use a different SQLite audit database path.
# $env:CREDILENS_AUDIT_DB = "C:\data\credilens-audit.sqlite3"

& "C:\ProgramData\anaconda3\python.exe" start_server.py
```

Open `http://127.0.0.1:8000/dashboard` to browse available risk modules without connecting. Paste an analyst/reviewer key to unlock assessments and private audit/model data. Enter the key itself, not the full JSON map configured in the environment. Analyst keys can submit assessments and read reports/audit history. Reviewer and admin keys can also record a human outcome; the reviewer identity comes from the configured key subject, not from the request body.

## Deploy a free prototype on Render

The repository includes a Render Blueprint (`render.yaml`) for a free Python web service. To deploy it:

1. Sign in to Render and create a new Blueprint instance.
2. Connect the private `Harshith59/CrediLens-AI` GitHub repository and select its `main` branch.
3. Confirm the service plan is **Free**. Do not add a persistent disk or upgrade the service if you need to avoid paid resources.
4. When prompted, set `CREDILENS_API_KEYS` in Render's secret environment-variable form. Do not put API keys in this repository or share them in chat. It must be a JSON map from keys of at least 32 characters to role/subject objects, for example:

   ```json
   {
     "<analyst-key>": {"role": "analyst", "subject": "render-analyst"},
     "<reviewer-key>": {"role": "reviewer", "subject": "render-reviewer"}
   }
   ```

5. After deployment, open the generated `https://<service-name>.onrender.com/dashboard` URL and use the analyst or reviewer key.

The Free plan is intended for prototypes, not production: Render spins down inactive free web services, and their filesystems are ephemeral. This app's SQLite audit history will be lost on restarts, spin-downs, or redeployments. No persistent disk or paid database is configured. Free usage quotas apply; check [Render's current free plan limitations](https://render.com/docs/free) and billing controls before deploying. If the API-key secret is missing or invalid, protected API endpoints return HTTP 503; check that `CREDILENS_API_KEYS` is a JSON object mapping each key (at least 32 characters) to a valid role and non-empty subject.

## API overview

The service provides:

- `GET /health`
- `GET /dashboard`
- `GET /api/catalog` (public supported-industry and risk-module names)
- `GET /api/model-info`
- `GET /api/config/modules`
- `POST /api/risk/predict`
- `POST /api/risk/explain`
- `GET /api/audit?limit=20&offset=0`
- `GET /api/audit/{audit_id}`
- `GET /api/audit/{audit_id}/events`
- `GET /api/audit/{audit_id}/decision`
- `POST /api/audit/{audit_id}/review`
- `GET /api/compliance/{audit_id}`
- `GET /api/reports/{audit_id}/json`
- `GET /api/reports/{audit_id}/html`
- `GET /api/reports/{audit_id}/png`
- `GET /api/summary`
- `GET /api/system/status`

Send the configured key using the `X-API-Key` header. Prediction input must contain every feature declared by the selected module and no undeclared fields. Invalid schemas return HTTP 422; missing audit IDs return HTTP 404.

Example payload:

```json
{
  "industry": "IT",
  "risk_type": "PROJECT",
  "data": {
    "team_size": 12,
    "bug_count": 45,
    "critical_bugs": 8,
    "pending_tasks": 28,
    "test_coverage": 61,
    "technical_debt": "High"
  },
  "application_id": "demo-001"
}
```

## Notes

This project is intentionally a clean, modular foundation for the blueprint. It is not production-grade for regulated decision-making without legal, domain, and security review.

## Registering a trained model

`RiskAssessmentEngine` can use a fitted classifier when the model is registered for the exact industry, risk type, and version. The demo in `main.py` trains the IT project model, registers it, and then uses it for IT project assessments. Other modules continue to use heuristic scoring until a model is trained and registered for that specific module.

```python
engine.register_model(
    industry="IT",
    risk_type="PROJECT",
    model_version="project-risk-v1",
    model=training_result["model"],
    feature_columns=training_result["feature_columns"],
)
```

`train_model.py` writes a versioned joblib artifact containing the fitted pipeline, feature schema, module identity, model version, and evaluation metrics. Set `CREDILENS_MODEL_ARTIFACT` to that trusted file before API startup to register it; `/api/model-info` then reports the loaded model and metrics. If the variable is unset, the API does not train a model and continues using the explicitly reported heuristic fallback. Only load artifacts from trusted sources: joblib files can execute code during loading.

## Audit storage

The API stores assessment records and review events in SQLite at `.credilens/audit.sqlite3` by default. Set `CREDILENS_AUDIT_DB` to use a different SQLite file. Audit IDs are generated independently of application IDs; the audit events endpoint exposes the original assessment and subsequent review snapshots. The database contains submitted assessment data and should be protected accordingly. PostgreSQL and production access controls remain deployment work.

## Training and model evaluation

The reference trainer compares LightGBM, XGBoost, and Random Forest models. It generates out-of-fold probabilities from stratified folds of the training split, selects non-negative late-fusion weights that sum to one using those training-only validation probabilities, then evaluates the selected ensemble and each base model once on the held-out test split. Metrics include ROC-AUC, PR-AUC, precision, recall, F1, accuracy, Brier score, and confusion matrix. The default training command uses synthetic IT project data for demonstration; these metrics are not evidence of real-world or regulated-use performance.

The PNG report visualizes the score and scaled numeric input indicators; it is not a SHAP explanation or a causal attribution. SHAP/LIME integration, real-data pipeline validation, drift monitoring, and jurisdiction-specific rules still need domain validation before production use.
