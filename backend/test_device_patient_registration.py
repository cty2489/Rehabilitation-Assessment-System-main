import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

import mysql_db
from device_patient_policy import DevicePatientPolicyError, resolve_device_patient
from schemas import DevicePatientRegistrationRequest


def _payload(**overrides):
    payload = {
        "patient_id": "DEV001_0001",
        "name": "测试患者",
        "sex": "男",
        "age": 62,
        "diagnosis": "脑梗死",
        "paralysis_side": "左",
        "disease_days": 120,
        "hand_brunnstrom_stage": "III",
    }
    payload.update(overrides)
    return payload


def _row(**overrides):
    row = {
        "id": 17,
        "patient_id": "DEV001_0001",
        "name": "测试患者",
        "sex": "男",
        "age": 62,
        "diagnosis": "脑梗死",
        "paralysis_side": "左",
        "disease_days": 120,
        "hand_function": 3,
        "source": "device-enroll",
        "created_at": "2026-07-19 10:00:00",
        "updated_at": "2026-07-19 10:00:00",
    }
    row.update(overrides)
    return row


class _RegistrationCursor:
    def __init__(self, row, created):
        self.row = row
        self.created = created
        self.rowcount = 0
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        params = tuple(params or ())
        self.queries.append((query, params))
        if "INSERT INTO patients" in query:
            self.rowcount = 1 if self.created else 0

    def fetchone(self):
        return dict(self.row)


class _RegistrationConnection:
    def __init__(self, row, created):
        self.cursor_obj = _RegistrationCursor(row, created)
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


class DevicePatientSchemaTests(unittest.TestCase):
    def test_patient_id_is_trimmed_and_uppercased(self):
        parsed = DevicePatientRegistrationRequest(**_payload(patient_id=" dev001_0001 "))
        self.assertEqual(parsed.patient_id, "DEV001_0001")

    def test_patient_id_requires_device_prefix_and_sequence(self):
        with self.assertRaises(ValidationError):
            DevicePatientRegistrationRequest(**_payload(patient_id="P001"))

    def test_hand_brunnstrom_stage_accepts_only_roman_one_to_six(self):
        parsed = DevicePatientRegistrationRequest(**_payload())
        self.assertEqual(parsed.hand_brunnstrom_stage, "III")
        for invalid in ("3", "VII", "iii"):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                DevicePatientRegistrationRequest(
                    **_payload(hand_brunnstrom_stage=invalid)
                )

    def test_hand_brunnstrom_stage_remains_optional_for_old_devices(self):
        parsed = DevicePatientRegistrationRequest(
            **_payload(hand_brunnstrom_stage=None)
        )
        self.assertIsNone(parsed.hand_brunnstrom_stage)


class DevicePatientRegistrationSqlTests(unittest.TestCase):
    def test_existing_patient_table_gets_hand_function_column(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {"Field": name}
            for name in ("id", "birth_date", "id_number", "phone", "onset_date")
        ]

        mysql_db._ensure_patient_schema(cursor)

        cursor.execute.assert_any_call(
            "ALTER TABLE patients ADD COLUMN hand_function INT"
        )

    def test_first_registration_commits_one_patient(self):
        conn = _RegistrationConnection(_row(), created=True)
        with patch.object(mysql_db, "get_conn", return_value=conn) as get_conn:
            patient, created = mysql_db.register_device_patient(_payload())

        get_conn.assert_called_once_with(autocommit=False)
        self.assertTrue(created)
        self.assertEqual(patient["patient_id"], "DEV001_0001")
        self.assertEqual(patient["hand_function"], 3)
        self.assertEqual(patient["hand_brunnstrom_stage"], "III")
        self.assertTrue(conn.committed)
        self.assertFalse(conn.rolled_back)
        self.assertTrue(conn.closed)
        insert_sql = conn.cursor_obj.queries[0][0]
        self.assertIn("'device-enroll'", insert_sql)
        self.assertIn("ON DUPLICATE KEY UPDATE", insert_sql)
        insert_params = conn.cursor_obj.queries[0][1]
        self.assertEqual(insert_params[7], 3)

    def test_identical_retry_returns_existing_patient(self):
        conn = _RegistrationConnection(_row(), created=False)
        with patch.object(mysql_db, "get_conn", return_value=conn):
            patient, created = mysql_db.register_device_patient(_payload())

        self.assertFalse(created)
        self.assertEqual(patient["id"], 17)
        self.assertTrue(conn.committed)
        self.assertFalse(conn.rolled_back)

    def test_identity_conflict_rolls_back(self):
        conn = _RegistrationConnection(_row(name="另一位患者"), created=False)
        with (
            patch.object(mysql_db, "get_conn", return_value=conn),
            self.assertRaises(mysql_db.PatientRegistrationConflict) as raised,
        ):
            mysql_db.register_device_patient(_payload())

        self.assertEqual(raised.exception.fields, ("name",))
        self.assertFalse(conn.committed)
        self.assertTrue(conn.rolled_back)
        self.assertTrue(conn.closed)

    def test_registration_retry_does_not_overwrite_clinical_profile(self):
        existing = _row(
            diagnosis="脑出血",
            paralysis_side="右",
            disease_days=30,
            hand_function=4,
        )
        conn = _RegistrationConnection(existing, created=False)
        with patch.object(mysql_db, "get_conn", return_value=conn):
            patient, created = mysql_db.register_device_patient(_payload())

        self.assertFalse(created)
        self.assertEqual(patient["diagnosis"], "脑出血")
        self.assertEqual(patient["paralysis_side"], "右")
        self.assertEqual(patient["hand_function"], 4)
        self.assertEqual(patient["hand_brunnstrom_stage"], "IV")
        self.assertTrue(conn.committed)


class DevicePatientAssessmentPolicyTests(unittest.TestCase):
    def test_request_and_manifest_patient_ids_must_match(self):
        with self.assertRaises(DevicePatientPolicyError) as raised:
            resolve_device_patient(
                requested_patient_id="DEV001_0001",
                manifest_patient_id="DEV001_0002",
                enrolled=None,
                require_registered=False,
                request_profile=_payload(),
                manifest_profile={},
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.code, "PATIENT_ID_MISMATCH")

    def test_strict_mode_rejects_unknown_patient(self):
        with self.assertRaises(DevicePatientPolicyError) as raised:
            resolve_device_patient(
                requested_patient_id="DEV001_0001",
                manifest_patient_id="DEV001_0001",
                enrolled=None,
                require_registered=True,
                request_profile={},
                manifest_profile={},
            )
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.code, "PATIENT_NOT_FOUND")

    def test_strict_mode_uses_cloud_profile_only(self):
        patient = resolve_device_patient(
            requested_patient_id=None,
            manifest_patient_id="DEV001_0001",
            enrolled=_row(),
            require_registered=True,
            request_profile=_payload(name="请求中的错误姓名"),
            manifest_profile=_payload(name="manifest中的错误姓名"),
        )
        self.assertEqual(patient.name, "测试患者")
        self.assertEqual(patient.diagnosis, "脑梗死")

    def test_compatibility_mode_can_use_legacy_request_profile(self):
        patient = resolve_device_patient(
            requested_patient_id="DEV001_0001",
            manifest_patient_id=None,
            enrolled=None,
            require_registered=False,
            request_profile=_payload(),
            manifest_profile={},
        )
        self.assertEqual(patient.patient_id, "DEV001_0001")
        self.assertEqual(patient.name, "测试患者")


if __name__ == "__main__":
    unittest.main()
