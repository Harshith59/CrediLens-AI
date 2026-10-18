from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


class AuditStore:
    """SQLite-backed audit records with an append-only event history."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            timeout=30,
        )
        self._connection.row_factory = sqlite3.Row
        if self.database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_records (
                audit_id TEXT PRIMARY KEY,
                application_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY (audit_id) REFERENCES audit_records(audit_id)
            );
            CREATE INDEX IF NOT EXISTS idx_audit_events_audit_id
                ON audit_events (audit_id, event_id);
            """
        )
        self._connection.commit()

    @staticmethod
    def _serialize(record: Dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _deserialize(payload: str) -> Dict[str, Any]:
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("Stored audit record must be a JSON object")
        return record

    def add(self, record: Dict[str, Any]) -> str:
        record_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        stored_record = {**record, "audit_id": record_id}
        serialized = self._serialize(stored_record)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO audit_records (audit_id, application_id, created_at, updated_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, record.get("application_id"), created_at, created_at, serialized),
            )
            self._connection.execute(
                """
                INSERT INTO audit_events (audit_id, event_type, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (record_id, "ASSESSMENT_CREATED", created_at, serialized),
            )
        return record_id

    def update(self, record_id: str, record: Dict[str, Any]) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        stored_record = {**record, "audit_id": record_id}
        serialized = self._serialize(stored_record)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE audit_records
                SET application_id = ?, updated_at = ?, payload = ?
                WHERE audit_id = ?
                """,
                (record.get("application_id"), updated_at, serialized, record_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(record_id)
            event_type = "HUMAN_REVIEW_RECORDED" if record.get("final_decision") else "ASSESSMENT_UPDATED"
            self._connection.execute(
                """
                INSERT INTO audit_events (audit_id, event_type, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (record_id, event_type, updated_at, serialized),
            )

    def get(self, record_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM audit_records WHERE audit_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return self._deserialize(row["payload"])

    def list(
        self,
        limit: int | None = 100,
        offset: int = 0,
        industry: str | None = None,
        risk_type: str | None = None,
        decision: str | None = None,
    ) -> List[Dict[str, Any]]:
        filters = []
        parameters: List[Any] = []
        for key, value in (("industry", industry), ("risk_type", risk_type), ("decision", decision)):
            if value is not None:
                filters.append(f"json_extract(payload, '$.{key}') = ?")
                parameters.append(value)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        pagination = ""
        if limit is not None:
            pagination = "LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT payload FROM audit_records
                {where_clause}
                ORDER BY created_at DESC, audit_id DESC
                {pagination}
                """,
                parameters,
            ).fetchall()
        return [self._deserialize(row["payload"]) for row in rows]

    def count(
        self,
        industry: str | None = None,
        risk_type: str | None = None,
        decision: str | None = None,
    ) -> int:
        filters = []
        parameters: List[Any] = []
        for key, value in (("industry", industry), ("risk_type", risk_type), ("decision", decision)):
            if value is not None:
                filters.append(f"json_extract(payload, '$.{key}') = ?")
                parameters.append(value)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) AS count FROM audit_records {where_clause}",
                parameters,
            ).fetchone()
        return int(row["count"])

    def pending_review_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM audit_records
                WHERE json_extract(payload, '$.decision') = 'HUMAN_REVIEW'
                  AND COALESCE(json_extract(payload, '$.decision_trace.human_decision'), '') = ''
                """
            ).fetchone()
        return int(row["count"])

    def events(self, record_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_type, created_at, payload
                FROM audit_events
                WHERE audit_id = ?
                ORDER BY event_id
                """,
                (record_id,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "record": self._deserialize(row["payload"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
