import os
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from src.crediens_ai.api.app import app, engine


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

    def test_public_catalog_is_available_without_authentication(self):
        self.client.headers.pop("X-API-Key")

        catalog_response = self.client.get("/api/catalog")
        demo_modules_response = self.client.get("/api/demo/modules")
        protected_response = self.client.get("/api/config/modules")
        private_audit_response = self.client.get("/api/audit")
        dashboard_response = self.client.get("/dashboard")

        self.assertEqual(catalog_response.status_code, 200)
        catalog = catalog_response.json()
        self.assertEqual(catalog["industries"], ["BANKING", "IT", "MANUFACTURING"])
        self.assertEqual(catalog["module_count"], 7)
        self.assertEqual(set(catalog["modules"]), set(catalog["industries"]))
        self.assertEqual(catalog["modules"]["IT"], ["CYBER", "OPERATIONAL", "PROJECT"])
        self.assertEqual(demo_modules_response.status_code, 200)
        demo_modules = demo_modules_response.json()["modules"]
        for industry, risk_types in catalog["modules"].items():
            self.assertEqual(set(demo_modules[industry]), set(risk_types))
            for risk_type, module in demo_modules[industry].items():
                self.assertEqual(set(module["sample"]), set(module["features"]))
        self.assertEqual(protected_response.status_code, 401)
        self.assertEqual(private_audit_response.status_code, 401)
        self.assertEqual(dashboard_response.status_code, 200)

    def test_public_demo_assessment_uses_real_engine_without_private_audit_writes(self):
        self.client.headers.pop("X-API-Key")
        sample = self.client.get("/api/demo/modules").json()["modules"]["IT"]["PROJECT"]["sample"]
        audit_count_before = engine.audit_store.count()

        response = self.client.post("/api/demo/assessments", json={
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": sample,
        })

        self.assertEqual(response.status_code, 200)
        assessment = response.json()
        self.assertEqual(assessment["status"], "SUCCESS")
        self.assertEqual(assessment["scoring_method"], "HEURISTIC")
        self.assertGreaterEqual(assessment["risk_probability"], 0.0)
        self.assertLessEqual(assessment["risk_probability"], 1.0)
        self.assertEqual(len(assessment["input_factors"]), len(sample))
        self.assertIn("not a trained-model prediction", assessment["method_description"])
        self.assertTrue(assessment["demo"])
        self.assertFalse(assessment["persisted"])
        self.assertNotIn("audit_id", assessment)
        self.assertEqual(engine.audit_store.count(), audit_count_before)

    def test_every_public_demo_module_accepts_its_synthetic_sample(self):
        self.client.headers.pop("X-API-Key")
        modules = self.client.get("/api/demo/modules").json()["modules"]
        audit_count_before = engine.audit_store.count()

        for industry, risk_types in modules.items():
            for risk_type, module in risk_types.items():
                with self.subTest(industry=industry, risk_type=risk_type):
                    response = self.client.post("/api/demo/assessments", json={
                        "industry": industry,
                        "risk_type": risk_type,
                        "data": module["sample"],
                    })

                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["status"], "SUCCESS")
                    self.assertEqual(response.json()["industry"], industry)
                    self.assertEqual(response.json()["risk_type"], risk_type)
                    self.assertEqual(response.json()["persisted"], False)

        self.assertEqual(engine.audit_store.count(), audit_count_before)

    def test_public_demo_validation_and_protected_data_boundary(self):
        self.client.headers.pop("X-API-Key")

        unsupported = self.client.post("/api/demo/assessments", json={
            "industry": "HEALTHCARE",
            "risk_type": "RISK",
            "data": {},
        })
        incomplete = self.client.post("/api/demo/assessments", json={
            "industry": "BANKING",
            "risk_type": "CREDIT",
            "data": {"loan_amount": 1000},
        })
        out_of_range = self.client.post("/api/demo/assessments", json={
            "industry": "IT",
            "risk_type": "PROJECT",
            "data": {
                "team_size": 1,
                "bug_count": 1,
                "critical_bugs": 0,
                "pending_tasks": 0,
                "test_coverage": 101,
                "technical_debt": "Low",
            },
        })
        invalid_number = self.client.post(
            "/api/demo/assessments",
            content=(
                '{"industry":"IT","risk_type":"PROJECT","data":{'
                '"team_size":1,"bug_count":NaN,"critical_bugs":0,'
                '"pending_tasks":0,"test_coverage":100,"technical_debt":"Low"}}'
            ),
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(unsupported.status_code, 422)
        self.assertEqual(incomplete.status_code, 422)
        self.assertEqual(out_of_range.status_code, 422)
        self.assertEqual(invalid_number.status_code, 422)
        self.assertEqual(self.client.get("/api/summary").status_code, 401)
        self.assertEqual(self.client.get("/api/system/status").status_code, 401)

    def test_public_demo_rejects_oversized_request_bodies(self):
        self.client.headers.pop("X-API-Key")

        response = self.client.post(
            "/api/demo/assessments",
            content=" " * 16385,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 413)
        self.assertNotIn("Traceback", response.text)

    def test_public_demo_works_when_protected_workspace_configuration_is_invalid(self):
        self.client.headers.pop("X-API-Key")
        sample = self.client.get("/api/demo/modules").json()["modules"]["BANKING"]["CREDIT"]["sample"]
        with patch.dict(os.environ, {"CREDILENS_API_KEYS": "invalid-json"}):
            public_response = self.client.post("/api/demo/assessments", json={
                "industry": "BANKING",
                "risk_type": "CREDIT",
                "data": sample,
            })
            protected_response = self.client.get("/api/model-info")

        self.assertEqual(public_response.status_code, 200)
        self.assertEqual(protected_response.status_code, 503)
        self.assertNotIn("invalid-json", protected_response.text)

    def test_protected_endpoint_rejects_invalid_api_key(self):
        self.client.headers["X-API-Key"] = "not-a-valid-key"

        response = self.client.get("/api/model-info")

        self.assertEqual(response.status_code, 401)
        self.assertNotIn("not-a-valid-key", response.text)

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
