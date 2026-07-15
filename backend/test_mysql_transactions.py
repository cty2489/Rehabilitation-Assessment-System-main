import unittest
from unittest.mock import patch

import mysql_db


class _TransactionCursor:
    def __init__(self, fail_children=False):
        self.fail_children = fail_children
        self.lastrowid = 0
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.queries.append((query, tuple(params or ())))
        if "INSERT INTO assessments" in query:
            self.lastrowid = 42

    def executemany(self, query, rows):
        self.queries.append((query, tuple(rows)))
        if self.fail_children:
            raise RuntimeError("child insert failed")

    def fetchone(self):
        return {"id": 7}


class _TransactionConnection:
    def __init__(self, fail_children=False):
        self.cursor_obj = _TransactionCursor(fail_children)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _save(conn):
    with patch.object(mysql_db, "get_conn", return_value=conn):
        return mysql_db.save_assessment_bundle(
            {
                "patient_id": "P001",
                "name": "测试患者",
                "sex": "男",
                "age": 60,
                "diagnosis": "脑梗死",
                "paralysis_side": "左",
                "disease_days": 30,
            },
            "session-1",
            {"FMA_UE": 16, "BI": 0, "hand_tone": "0", "hand_function": 6},
            "report",
            "generated",
            source="device",
            trials=[{"trial_index": 1, "action_name": "伸腕"}],
        )


def _enroll(conn):
    with (
        patch.object(mysql_db, "get_conn", return_value=conn),
        patch.object(mysql_db, "get_patient", return_value={"id": 7}) as get_patient,
    ):
        result = mysql_db.enroll_patient(
            {
                "patient_id": "P001",
                "name": "测试患者",
                "sex": "男",
                "age": 60,
                "diagnosis": "脑梗死",
                "paralysis_side": "左",
                "disease_days": 30,
            },
            {
                "FMA_UE": 16,
                "BI": 0,
                "hand_tone": "0",
                "hand_function": 6,
                "trials": [{"trial_index": 1, "action_name": "伸腕"}],
            },
        )
        get_patient.assert_called_once_with(7)
        return result


class MysqlTransactionTests(unittest.TestCase):
    def test_complete_bundle_commits_once(self):
        conn = _TransactionConnection()
        self.assertEqual(_save(conn), 42)
        self.assertTrue(conn.committed)
        self.assertFalse(conn.rolled_back)
        self.assertTrue(conn.closed)

    def test_child_failure_rolls_back_parent_rows(self):
        conn = _TransactionConnection(fail_children=True)
        with self.assertRaisesRegex(RuntimeError, "child insert failed"):
            _save(conn)
        self.assertFalse(conn.committed)
        self.assertTrue(conn.rolled_back)
        self.assertTrue(conn.closed)

    def test_timezone_aware_manifest_time_is_stored_as_shanghai_time(self):
        self.assertEqual(
            mysql_db._to_dt("2026-07-14T20:29:00Z"),
            "2026-07-15 04:29:00",
        )

    def test_enrollment_and_first_assessment_commit_together(self):
        conn = _TransactionConnection()
        self.assertEqual(_enroll(conn), {"id": 7})
        self.assertTrue(conn.committed)
        self.assertFalse(conn.rolled_back)

    def test_enrollment_rolls_back_when_trial_insert_fails(self):
        conn = _TransactionConnection(fail_children=True)
        with self.assertRaisesRegex(RuntimeError, "child insert failed"):
            _enroll(conn)
        self.assertFalse(conn.committed)
        self.assertTrue(conn.rolled_back)


if __name__ == "__main__":
    unittest.main()
