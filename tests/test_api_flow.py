import os
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from src.crediens_ai.api.app import app


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
        self.auth_environment = patch.dict(os.environ, {
            "CREDILENS_API_KEYS": (
                '{"local-analyst-key-123456789012345678901234":'
                '{"role":"analyst","subject":"local-analyst"},'
                '"local-reviewer-key-123456789012345678901234":'
                '{"role":"reviewer","subject":"local-reviewer"}}'
            ),
        })
        self.auth_environment.start()
        self.client = TestClient(app)
        self.client.headers["X-API-Key"] = "local-analyst-key-123456789012345678901234"

    def tearDown(self):
        self.client.close()
        self.auth_environment.stop()

    def test_predict_and_explain_endpoints(self):
        payload = {
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
        }
        predict_response = self.client.post("/api/risk/predict", json=payload)
        self.assertEqual(predict_response.status_code, 200)
        body = predict_response.json()
        self.assertIn("risk_probability", body)
        self.assertIn("decision", body)

        explain_response = self.client.post("/api/risk/explain", json={"industry": "IT", "risk_type": "PROJECT", "data": payload["data"]})
        self.assertEqual(explain_response.status_code, 200)
        explain_body = explain_response.json()
        self.assertIn("explainability", explain_body)

    def test_home_redirects_to_dashboard(self):
        response = self.client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/dashboard")

    def test_config_and_audit_endpoints(self):
        config_response = self.client.get("/api/config/industries")
        self.assertEqual(config_response.status_code, 200)
        self.assertIn("industries", config_response.json())

        payload = {
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {
                "loan_amount": 50000,
                "existing_debt": 20000,
                "credit_history": "Good",
                "income": 90000,
                "employment_years": 10,
            },
            "application_id": "audit-123",
        }
        predict_response = self.client.post("/api/risk/predict", json=payload)
        self.assertEqual(predict_response.status_code, 200)
        audit_id = predict_response.json()["audit_id"]

        audit_response = self.client.get(f"/api/audit/{audit_id}")
        self.assertEqual(audit_response.status_code, 200)
        self.assertEqual(audit_response.json()["application_id"], "audit-123")

        summary_response = self.client.get("/api/summary")
        self.assertEqual(summary_response.status_code, 200)
        self.assertIn("total_audits", summary_response.json())

        status_response = self.client.get("/api/system/status")
        self.assertEqual(status_response.status_code, 200)
        self.assertIn("status", status_response.json())

        decision_response = self.client.get(f"/api/audit/{audit_id}/decision")
        self.assertEqual(decision_response.status_code, 200)
        self.assertIn("decision_trace", decision_response.json())

        events_response = self.client.get(f"/api/audit/{audit_id}/events")
        self.assertEqual(events_response.status_code, 200)
        self.assertEqual(events_response.json()["events"][0]["event_type"], "ASSESSMENT_CREATED")

        report_response = self.client.get(f"/api/reports/{audit_id}/json")
        html_report_response = self.client.get(f"/api/reports/{audit_id}/html")
        png_report_response = self.client.get(f"/api/reports/{audit_id}/png")
        audit_list_response = self.client.get("/api/audit?industry=BANKING&limit=5")
        dashboard_response = self.client.get("/dashboard")
        self.assertEqual(report_response.status_code, 200)
        self.assertEqual(report_response.json()["assessment_id"], audit_id)
        self.assertEqual(html_report_response.status_code, 200)
        self.assertTrue(html_report_response.text.startswith("<html>"))
        self.assertEqual(png_report_response.status_code, 200)
        self.assertEqual(png_report_response.content[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(audit_list_response.status_code, 200)
        self.assertTrue(all(item["industry"] == "BANKING" for item in audit_list_response.json()["items"]))
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn("CrediLens AI", dashboard_response.text)

    def test_invalid_assessments_and_missing_audits_use_http_errors(self):
        invalid_response = self.client.post("/api/risk/predict", json={
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {"bug_count": 12},
        })
        missing_audit_response = self.client.get("/api/audit/not-a-real-audit")

        self.assertEqual(invalid_response.status_code, 422)
        self.assertIn("Missing required fields", str(invalid_response.json()["detail"]))
        self.assertEqual(missing_audit_response.status_code, 404)

    def test_review_requires_reviewer_role_and_uses_authenticated_identity(self):
        assessment = self.client.post("/api/risk/predict", json={
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
        }).json()
        audit_id = assessment["audit_id"]
        review_payload = {
            "final_decision": "APPROVED_WITH_CONDITIONS",
            "reviewer_comment": "Approved with additional monitoring.",
        }

        analyst_response = self.client.post(f"/api/audit/{audit_id}/review", json=review_payload)
        self.client.headers["X-API-Key"] = "local-reviewer-key-123456789012345678901234"
        reviewer_response = self.client.post(f"/api/audit/{audit_id}/review", json=review_payload)

        self.assertEqual(analyst_response.status_code, 403)
        self.assertEqual(reviewer_response.status_code, 200)
        self.assertEqual(reviewer_response.json()["reviewer"], "local-reviewer")

    def test_api_requires_a_configured_key(self):
        self.client.headers.pop("X-API-Key")
        response = self.client.get("/api/model-info")

        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
