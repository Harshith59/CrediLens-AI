import tempfile
import unittest
from pathlib import Path

from src.crediens_ai.audit.store import AuditStore


class AuditStoreTests(unittest.TestCase):
    def test_records_persist_and_review_updates_append_events(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "audit.sqlite3"
            store = AuditStore(database_path)
            record_id = store.add({
                "application_id": "application-123",
                "decision": "HUMAN_REVIEW",
            })
            store.update(record_id, {
                "application_id": "application-123",
                "decision": "HUMAN_REVIEW",
                "final_decision": "APPROVED_WITH_CONDITIONS",
            })
            store.close()

            reopened = AuditStore(database_path)
            try:
                record = reopened.get(record_id)
                events = reopened.events(record_id)
            finally:
                reopened.close()

        self.assertEqual(record["audit_id"], record_id)
        self.assertEqual(record["application_id"], "application-123")
        self.assertEqual(record["final_decision"], "APPROVED_WITH_CONDITIONS")
        self.assertEqual([event["event_type"] for event in events], [
            "ASSESSMENT_CREATED",
            "HUMAN_REVIEW_RECORDED",
        ])

    def test_audit_ids_are_unique_when_application_ids_repeat(self):
        store = AuditStore()
        try:
            first = store.add({"application_id": "same-application"})
            second = store.add({"application_id": "same-application"})
            records = store.list()
        finally:
            store.close()

        self.assertNotEqual(first, second)
        self.assertEqual(len(records), 2)

    def test_update_of_unknown_record_fails_explicitly(self):
        store = AuditStore()
        try:
            with self.assertRaises(KeyError):
                store.update("missing-audit", {})
        finally:
            store.close()

    def test_list_supports_filters_and_pagination(self):
        store = AuditStore()
        try:
            store.add({"industry": "IT", "risk_type": "PROJECT", "decision": "APPROVE"})
            store.add({"industry": "BANKING", "risk_type": "CREDIT", "decision": "HUMAN_REVIEW"})
            store.add({"industry": "BANKING", "risk_type": "FRAUD", "decision": "APPROVE"})

            page = store.list(limit=1, offset=0, industry="BANKING")
            total = store.count(industry="BANKING")
        finally:
            store.close()

        self.assertEqual(len(page), 1)
        self.assertEqual(page[0]["industry"], "BANKING")
        self.assertEqual(total, 2)


if __name__ == "__main__":
    unittest.main()
