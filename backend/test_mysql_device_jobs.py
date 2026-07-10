import unittest
from unittest.mock import patch

import mysql_db


class _FakeCursor:
    def __init__(self, row):
        self.row = row
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        params = tuple(params or ())
        self.queries.append((query, params))
        self.assert_placeholder_count(query, params)

    def fetchone(self):
        return dict(self.row)

    @staticmethod
    def assert_placeholder_count(query, params):
        expected = query.count("%s")
        if expected != len(params):
            raise AssertionError(f"SQL expects {expected} params, received {len(params)}")


class _FakeConnection:
    def __init__(self, row):
        self.cursor_obj = _FakeCursor(row)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


class DeviceJobSqlTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "job_id": "devjob_test",
            "session_id": "session_test",
            "status": "queued",
            "phase": "waiting",
            "progress_percent": 0,
            "attempt_count": 0,
            "patient_json": '{"patient_id":"P001"}',
            "parse_warnings": "[]",
            "error_retryable": 0,
        }
        self.conn = _FakeConnection(self.row)

    def test_create_device_job_sql_parameters_match(self):
        with patch.object(mysql_db, "get_conn", return_value=self.conn):
            job = mysql_db.create_device_job(
                job_id="devjob_test",
                device_id="device_001",
                session_id="session_test",
                assessment_id="A001",
                patient_id="P001",
                package_name="input.zip",
                package_hash="abc",
                institution="device",
                input_path="/tmp/device_jobs/devjob_test/bundle.zip",
                patient_json={"patient_id": "P001"},
                parse_warnings=[],
                n_trials=3,
                idempotency_key="device_001:A001",
            )

        self.assertEqual(job["patient_json"]["patient_id"], "P001")
        self.assertIn("INSERT INTO device_jobs", self.conn.cursor_obj.queries[0][0])

    def test_update_device_job_supports_retry_reset(self):
        with patch.object(mysql_db, "get_conn", return_value=self.conn):
            mysql_db.update_device_job(
                "devjob_test",
                status="queued",
                phase="waiting",
                progress_percent=0,
                clear_error=True,
                increment_attempt=True,
                reset_timestamps=True,
            )

        update_sql = self.conn.cursor_obj.queries[0][0]
        self.assertIn("attempt_count=attempt_count+1", update_sql)
        self.assertIn("started_at=NULL", update_sql)


if __name__ == "__main__":
    unittest.main()
