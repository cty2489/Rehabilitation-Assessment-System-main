"""MySQL persistence for rehabilitation assessment workflows.

This module is the single business store for the browser assessment workflow,
the task-interface workflow, and the device API:

* ``patients``    — one row per business ``patient_id`` (minimal basic info,
                    enrolled by the hospital or auto-created from a device manifest).
* ``assessments`` — many rows per patient: the hospital's first record at
                    enrollment + every later device assessment.

Stdlib-style raw SQL via PyMySQL (no ORM).
Connection config comes from ``.env`` (MYSQL_HOST / MYSQL_PORT / MYSQL_USER /
MYSQL_PASSWORD / MYSQL_DB). PyMySQL is imported defensively so callers can
surface a clear 503 when MySQL is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime
from numbers import Real
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

try:  # defensive: callers surface a clear MySQLUnavailable if missing
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # pragma: no cover - exercised only when dep missing
    pymysql = None  # type: ignore[assignment]
    DictCursor = None  # type: ignore[assignment]

_init_lock = threading.Lock()
_initialized = False

_PATIENT_EXTRA_COLUMNS: Dict[str, str] = {
    "hand_function": "INT",
    "birth_date": "VARCHAR(32)",
    "id_number": "VARCHAR(32)",
    "phone": "VARCHAR(32)",
    "onset_date": "VARCHAR(32)",
}

_PATIENT_CORE = (
    "name",
    "sex",
    "age",
    "diagnosis",
    "disease_days",
    "paralysis_side",
    "hand_function",
)
_PATIENT_EXTENDED = ("birth_date", "id_number", "phone", "onset_date")
_PATIENT_EDITABLE = _PATIENT_CORE + _PATIENT_EXTENDED

_ASSESSMENT_EXTRA_COLUMNS: Dict[str, str] = {
    "institution": "VARCHAR(16)",
    "n_trials": "INT",
    "package_hash": "VARCHAR(64)",
    "parse_warnings": "LONGTEXT",
    "prediction_json": "LONGTEXT",
    "model_version": "VARCHAR(255)",
    "llm_provider": "VARCHAR(32)",
    "llm_model": "VARCHAR(128)",
    "patient_snapshot": "LONGTEXT",
    "quality_json": "LONGTEXT",
    "validation_status": "VARCHAR(64)",
    "report_generation": "VARCHAR(32)",
}

_ASSESSMENT_INDEXES: Dict[str, str] = {
    "idx_assess_external": "CREATE INDEX idx_assess_external ON assessments(assessment_id)",
    "idx_assess_session": "CREATE INDEX idx_assess_session ON assessments(session_id)",
}

_DEVICE_JOB_EXTRA_COLUMNS: Dict[str, str] = {
    "institution": "VARCHAR(16)",
    "input_path": "VARCHAR(1024)",
    "patient_json": "LONGTEXT",
    "parse_warnings": "LONGTEXT",
    "n_trials": "INT",
    "idempotency_key": "VARCHAR(255)",
    "phase": "VARCHAR(32) NOT NULL DEFAULT 'waiting'",
    "progress_percent": "INT NOT NULL DEFAULT 0",
    "status_message": "VARCHAR(255)",
    "error_code": "VARCHAR(64)",
    "error_retryable": "TINYINT(1) NOT NULL DEFAULT 0",
    "attempt_count": "INT NOT NULL DEFAULT 0",
}

_DEVICE_JOB_INDEXES: Dict[str, str] = {
    "uniq_device_job_idempotency": (
        "CREATE UNIQUE INDEX uniq_device_job_idempotency "
        "ON device_jobs(device_id, idempotency_key)"
    ),
    "idx_device_job_created": "CREATE INDEX idx_device_job_created ON device_jobs(created_at)",
}


class MySQLUnavailable(RuntimeError):
    """Raised when MySQL / pymysql is not usable; callers surface a clear error."""


class PatientRegistrationConflict(ValueError):
    """Raised when one patient_id is reused for different identity data."""

    def __init__(self, fields: List[str]):
        self.fields = tuple(fields)
        super().__init__(f"患者编号对应的身份字段冲突：{', '.join(fields)}")


_BRUNNSTROM_STAGES = ("I", "II", "III", "IV", "V", "VI")


def _hand_function_from_stage(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return _BRUNNSTROM_STAGES.index(str(value)) + 1
    except ValueError:
        return None


def _stage_from_hand_function(value: Any) -> Optional[str]:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= numeric <= len(_BRUNNSTROM_STAGES):
        return _BRUNNSTROM_STAGES[numeric - 1]
    return None


def ping() -> bool:
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
                return bool(row and int(row.get("ok", 0)) == 1)
        finally:
            conn.close()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Config / connection                                                          #
# --------------------------------------------------------------------------- #
def _config() -> Dict[str, Any]:
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "db": os.environ.get("MYSQL_DB", "rehab_mysql"),
    }


def get_conn(with_db: bool = True, autocommit: bool = True):
    if pymysql is None:
        raise MySQLUnavailable("未安装 pymysql，无法连接 MySQL（pip install pymysql）")
    cfg = _config()
    kwargs: Dict[str, Any] = {
        "host": cfg["host"],
        "port": cfg["port"],
        "user": cfg["user"],
        "password": cfg["password"],
        "charset": "utf8mb4",
        "autocommit": autocommit,
        "cursorclass": DictCursor,
    }
    if with_db:
        kwargs["database"] = cfg["db"]
    try:
        return pymysql.connect(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise MySQLUnavailable(f"连接 MySQL 失败：{exc}") from exc


# --------------------------------------------------------------------------- #
# Time helpers                                                                 #
# --------------------------------------------------------------------------- #
def now_dt() -> str:
    """Beijing-time 'YYYY-MM-DD HH:MM:SS' for MySQL DATETIME columns.

    MySQL DATETIME is timezone-naive; storing UTC here makes the UI look eight
    hours behind in this China-facing deployment.
    """
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _to_dt(value: Any) -> Optional[str]:
    """Best-effort parse of a manifest timestamp (ISO-8601, possibly with tz)
    into a MySQL DATETIME string. Returns None if it can't be parsed."""
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return None


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict or an object (PatientInfo / PredictionResult)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _norm(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Stringify DATETIME values so rows are JSON/Pydantic friendly."""
    if row is None:
        return None
    for k, v in list(row.items()):
        if isinstance(v, datetime):
            row[k] = v.strftime("%Y-%m-%d %H:%M:%S")
    return row


def _norm_patient(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    row = _norm(row)
    if row is None:
        return None
    for key in ("name", "sex", "diagnosis", "paralysis_side"):
        if row.get(key) is None:
            row[key] = ""
    row["hand_brunnstrom_stage"] = _stage_from_hand_function(
        row.get("hand_function")
    )
    return row


def _json_value(value: Optional[str]) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except Exception:  # noqa: BLE001
        return value


def _norm_assessment_detail(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    row = _norm(row)
    if row is None:
        return None
    for key in ("biomarkers", "parse_warnings", "prediction_json", "patient_snapshot", "quality_json"):
        row[key] = _json_value(row.get(key))
    return row


def _norm_device_job(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    row = _norm(row)
    if row is None:
        return None
    row["patient_json"] = _json_value(row.get("patient_json"))
    row["parse_warnings"] = _json_value(row.get("parse_warnings")) or []
    row["error_retryable"] = bool(row.get("error_retryable"))
    return row


def _number_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _fetch_trials(cur, assessment_id: int) -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT id, trial_index, assessment_type, action_name, eeg_file, emg_file,
               eeg_name, emg_name, status, note, created_at
        FROM assessment_trials
        WHERE assessment_db_id=%s
        ORDER BY COALESCE(trial_index, id), id
        """,
        (assessment_id,),
    )
    return [_norm(r) for r in cur.fetchall()]


def _fetch_biomarker_items(cur, assessment_id: int) -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT id, group_key, group_label, marker_key, marker_name, value_text,
               value_num, unit, ref_range, n_valid, available, note, created_at
        FROM assessment_biomarkers
        WHERE assessment_db_id=%s
        ORDER BY FIELD(group_key, 'emg', 'eeg', 'imu'), id
        """,
        (assessment_id,),
    )
    rows = []
    for row in cur.fetchall():
        normed = _norm(row)
        normed["available"] = bool(normed.get("available"))
        rows.append(normed)
    return rows


def _attach_assessment_children(cur, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    row = _norm_assessment_detail(row)
    if row is None:
        return None
    assessment_id = int(row["id"])
    snapshot = row.get("patient_snapshot")
    if isinstance(snapshot, dict):
        for key in ("patient_id", "name", "sex", "age", "diagnosis", "paralysis_side", "disease_days"):
            if snapshot.get(key) is not None:
                row[key] = snapshot[key]
    row["trials"] = _fetch_trials(cur, assessment_id)
    row["biomarker_items"] = _fetch_biomarker_items(cur, assessment_id)
    return row


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
_CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
  id              BIGINT PRIMARY KEY AUTO_INCREMENT,
  patient_id      VARCHAR(64)  NOT NULL UNIQUE,
  name            VARCHAR(128),
  sex             VARCHAR(8),
  age             INT,
  diagnosis       VARCHAR(255),
  paralysis_side  VARCHAR(8),
  disease_days    INT,
  hand_function   INT,
  birth_date      VARCHAR(32),
  id_number       VARCHAR(32),
  phone           VARCHAR(32),
  onset_date      VARCHAR(32),
  source          VARCHAR(16),
  created_at      DATETIME NOT NULL,
  updated_at      DATETIME NOT NULL
) CHARACTER SET utf8mb4
"""

_CREATE_ASSESSMENTS = """
CREATE TABLE IF NOT EXISTS assessments (
  id              BIGINT PRIMARY KEY AUTO_INCREMENT,
  patient_db_id   BIGINT NOT NULL,
  source          VARCHAR(16) NOT NULL,
  assessment_id   VARCHAR(64),
  session_id      VARCHAR(32),
  package_name    VARCHAR(255),
  institution     VARCHAR(16),
  n_trials        INT,
  package_hash    VARCHAR(64),
  created_at      DATETIME NOT NULL,
  assessment_time DATETIME,
  fma_ue          FLOAT NOT NULL,
  bi              FLOAT NOT NULL,
  hand_tone       VARCHAR(8) NOT NULL,
  hand_function   INT NOT NULL,
  report          MEDIUMTEXT,
  report_status   VARCHAR(16) NOT NULL,
  biomarkers      LONGTEXT,
  parse_warnings  LONGTEXT,
  prediction_json LONGTEXT,
  model_version   VARCHAR(255),
  llm_provider    VARCHAR(32),
  llm_model       VARCHAR(128),
  patient_snapshot LONGTEXT,
  quality_json    LONGTEXT,
  validation_status VARCHAR(64),
  report_generation VARCHAR(32),
  CONSTRAINT fk_assess_patient FOREIGN KEY (patient_db_id)
    REFERENCES patients(id) ON DELETE CASCADE,
  INDEX idx_assess_patient (patient_db_id),
  INDEX idx_assess_created (created_at),
  INDEX idx_assess_external (assessment_id),
  INDEX idx_assess_session (session_id)
) CHARACTER SET utf8mb4
"""

_CREATE_ASSESSMENT_TRIALS = """
CREATE TABLE IF NOT EXISTS assessment_trials (
  id                BIGINT PRIMARY KEY AUTO_INCREMENT,
  assessment_db_id  BIGINT NOT NULL,
  trial_index       INT,
  assessment_type   VARCHAR(32),
  action_name       VARCHAR(128),
  eeg_file          VARCHAR(512),
  emg_file          VARCHAR(512),
  eeg_name          VARCHAR(255),
  emg_name          VARCHAR(255),
  status            VARCHAR(32),
  note              VARCHAR(255),
  created_at        DATETIME NOT NULL,
  CONSTRAINT fk_trial_assessment FOREIGN KEY (assessment_db_id)
    REFERENCES assessments(id) ON DELETE CASCADE,
  INDEX idx_trial_assessment (assessment_db_id)
) CHARACTER SET utf8mb4
"""

_CREATE_ASSESSMENT_BIOMARKERS = """
CREATE TABLE IF NOT EXISTS assessment_biomarkers (
  id                BIGINT PRIMARY KEY AUTO_INCREMENT,
  assessment_db_id  BIGINT NOT NULL,
  group_key         VARCHAR(32),
  group_label       VARCHAR(128),
  marker_key        VARCHAR(128) NOT NULL,
  marker_name       VARCHAR(255),
  value_text        VARCHAR(64),
  value_num         DOUBLE,
  unit              VARCHAR(64),
  ref_range         VARCHAR(255),
  n_valid           INT,
  available         TINYINT(1),
  note              TEXT,
  created_at        DATETIME NOT NULL,
  CONSTRAINT fk_biomarker_assessment FOREIGN KEY (assessment_db_id)
    REFERENCES assessments(id) ON DELETE CASCADE,
  UNIQUE KEY uniq_biomarker_marker (assessment_db_id, marker_key),
  INDEX idx_biomarker_assessment (assessment_db_id),
  INDEX idx_biomarker_key (marker_key)
) CHARACTER SET utf8mb4
"""

_CREATE_DEVICE_JOBS = """
CREATE TABLE IF NOT EXISTS device_jobs (
  job_id            VARCHAR(64) PRIMARY KEY,
  device_id         VARCHAR(128),
  session_id        VARCHAR(32),
  assessment_db_id  BIGINT,
  assessment_id     VARCHAR(64),
  patient_id        VARCHAR(64),
  package_name      VARCHAR(255),
  package_hash      VARCHAR(64),
  institution       VARCHAR(16),
  input_path        VARCHAR(1024),
  patient_json      LONGTEXT,
  parse_warnings    LONGTEXT,
  n_trials          INT,
  idempotency_key   VARCHAR(255),
  status            VARCHAR(32) NOT NULL,
  phase             VARCHAR(32) NOT NULL DEFAULT 'waiting',
  progress_percent  INT NOT NULL DEFAULT 0,
  status_message    VARCHAR(255),
  error_message     TEXT,
  error_code        VARCHAR(64),
  error_retryable   TINYINT(1) NOT NULL DEFAULT 0,
  attempt_count     INT NOT NULL DEFAULT 0,
  created_at        DATETIME NOT NULL,
  started_at        DATETIME,
  completed_at      DATETIME,
  delivered_at      DATETIME,
  updated_at        DATETIME NOT NULL,
  CONSTRAINT fk_device_job_assessment FOREIGN KEY (assessment_db_id)
    REFERENCES assessments(id) ON DELETE SET NULL,
  INDEX idx_device_job_session (session_id),
  INDEX idx_device_job_assessment (assessment_db_id),
  INDEX idx_device_job_patient (patient_id),
  INDEX idx_device_job_status (status),
  INDEX idx_device_job_created (created_at),
  UNIQUE KEY uniq_device_job_idempotency (device_id, idempotency_key)
) CHARACTER SET utf8mb4
"""

_CREATE_DEVICE_CREDENTIALS = """
CREATE TABLE IF NOT EXISTS device_credentials (
  id                BIGINT PRIMARY KEY AUTO_INCREMENT,
  device_id         VARCHAR(128) NOT NULL UNIQUE,
  label             VARCHAR(128),
  access_scope      VARCHAR(16) NOT NULL DEFAULT 'device',
  token_hash        CHAR(64) NOT NULL UNIQUE,
  token_hint        VARCHAR(32) NOT NULL,
  status            VARCHAR(16) NOT NULL DEFAULT 'active',
  source            VARCHAR(32) NOT NULL DEFAULT 'admin',
  created_by        VARCHAR(128),
  created_at        DATETIME NOT NULL,
  updated_at        DATETIME NOT NULL,
  last_used_at      DATETIME,
  rotated_at        DATETIME,
  revoked_at        DATETIME,
  INDEX idx_device_credential_status (status)
) CHARACTER SET utf8mb4
"""


def _ensure_patient_schema(cur) -> None:
    cur.execute("SHOW COLUMNS FROM patients")
    existing = {row["Field"] for row in cur.fetchall()}
    for name, ddl in _PATIENT_EXTRA_COLUMNS.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE patients ADD COLUMN {name} {ddl}")


def _ensure_assessment_schema(cur) -> None:
    cur.execute("SHOW COLUMNS FROM assessments")
    existing = {row["Field"] for row in cur.fetchall()}
    for name, ddl in _ASSESSMENT_EXTRA_COLUMNS.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE assessments ADD COLUMN {name} {ddl}")

    cur.execute("SHOW INDEX FROM assessments")
    indexes = {row["Key_name"] for row in cur.fetchall()}
    for name, ddl in _ASSESSMENT_INDEXES.items():
        if name not in indexes:
            cur.execute(ddl)


def _ensure_device_job_schema(cur) -> None:
    cur.execute("SHOW COLUMNS FROM device_jobs")
    existing = {row["Field"] for row in cur.fetchall()}
    for name, ddl in _DEVICE_JOB_EXTRA_COLUMNS.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE device_jobs ADD COLUMN {name} {ddl}")

    cur.execute("SHOW INDEX FROM device_jobs")
    index_rows = cur.fetchall()
    indexes = {row["Key_name"] for row in index_rows}
    idempotency_columns = [
        row["Column_name"]
        for row in sorted(index_rows, key=lambda item: int(item.get("Seq_in_index") or 0))
        if row["Key_name"] == "uniq_device_job_idempotency"
    ]
    if idempotency_columns and idempotency_columns != ["device_id", "idempotency_key"]:
        cur.execute("DROP INDEX uniq_device_job_idempotency ON device_jobs")
        indexes.discard("uniq_device_job_idempotency")
    for name, ddl in _DEVICE_JOB_INDEXES.items():
        if name not in indexes:
            cur.execute(ddl)


def _biomarker_insert_rows(
    assessment_id: int,
    biomarkers: Optional[Dict[str, Any]],
    ts: str,
) -> List[tuple]:
    if not biomarkers:
        return []
    rows = []
    for group in biomarkers.get("groups", []) or []:
        group_key = group.get("key")
        group_label = group.get("label")
        for marker in group.get("markers", []) or []:
            raw_value = marker.get("value")
            value_num = _number_value(raw_value)
            value_text = None if raw_value is None else str(raw_value)
            rows.append(
                (
                    assessment_id,
                    group_key,
                    group_label,
                    marker.get("key"),
                    marker.get("name"),
                    value_text,
                    value_num,
                    marker.get("unit"),
                    marker.get("ref_range"),
                    marker.get("n_valid"),
                    1 if marker.get("available") else 0,
                    marker.get("note"),
                    ts,
                )
            )
    return rows


def _backfill_assessment_children(cur) -> None:
    """Populate new normalized child tables for assessments saved before them."""
    cur.execute("SELECT id, n_trials, package_name, biomarkers FROM assessments")
    rows = cur.fetchall()
    ts = now_dt()
    for row in rows:
        assessment_id = int(row["id"])

        cur.execute(
            "SELECT COUNT(*) AS c FROM assessment_trials WHERE assessment_db_id=%s",
            (assessment_id,),
        )
        if int(cur.fetchone()["c"]) == 0 and row.get("n_trials"):
            n_trials = int(row["n_trials"])
            cur.executemany(
                """
                INSERT INTO assessment_trials
                  (assessment_db_id, trial_index, assessment_type, action_name,
                   eeg_file, emg_file, eeg_name, emg_name, status, note, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        assessment_id,
                        idx,
                        "active",
                        f"trial_{idx}",
                        row.get("package_name"),
                        row.get("package_name"),
                        None,
                        None,
                        "backfilled",
                        "历史记录回填：原始 trial 文件名未单独保存",
                        ts,
                    )
                    for idx in range(1, n_trials + 1)
                ],
            )

        cur.execute(
            "SELECT COUNT(*) AS c FROM assessment_biomarkers WHERE assessment_db_id=%s",
            (assessment_id,),
        )
        if int(cur.fetchone()["c"]) == 0 and row.get("biomarkers"):
            biomarker_rows = _biomarker_insert_rows(
                assessment_id,
                _json_value(row.get("biomarkers")),
                ts,
            )
            if biomarker_rows:
                cur.executemany(
                    """
                    INSERT INTO assessment_biomarkers
                      (assessment_db_id, group_key, group_label, marker_key, marker_name,
                       value_text, value_num, unit, ref_range, n_valid, available, note,
                       created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    biomarker_rows,
                )


def init_db() -> None:
    """Create the database (if missing) and both tables. Idempotent.

    Raises ``MySQLUnavailable`` if MySQL can't be reached; startup logs a warning
    and business APIs surface a clear 503.
    """
    global _initialized
    with _init_lock:
        if _initialized:
            return
        cfg = _config()
        conn = get_conn(with_db=False)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{cfg['db']}` CHARACTER SET utf8mb4"
                )
        finally:
            conn.close()
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_PATIENTS)
                cur.execute(_CREATE_ASSESSMENTS)
                cur.execute(_CREATE_ASSESSMENT_TRIALS)
                cur.execute(_CREATE_ASSESSMENT_BIOMARKERS)
                cur.execute(_CREATE_DEVICE_JOBS)
                cur.execute(_CREATE_DEVICE_CREDENTIALS)
                _ensure_patient_schema(cur)
                _ensure_assessment_schema(cur)
                _ensure_device_job_schema(cur)
                _backfill_assessment_children(cur)
        finally:
            conn.close()
        _initialized = True


# --------------------------------------------------------------------------- #
# Patients                                                                     #
# --------------------------------------------------------------------------- #
def upsert_patient(patient: Any, source: str = "device-auto") -> int:
    """Insert or update a patient by business key ``patient_id``. ``source`` is
    only applied on first insert (an existing enrollment source is preserved).
    ``patient`` is a PatientInfo or a dict. Returns the patient row id."""
    ts = now_dt()
    pid = _field(patient, "patient_id")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients
                  (patient_id, name, sex, age, diagnosis, paralysis_side,
                   disease_days, source, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  name=VALUES(name), sex=VALUES(sex),
                  age=COALESCE(VALUES(age), age),
                  diagnosis=COALESCE(NULLIF(VALUES(diagnosis), ''), diagnosis),
                  paralysis_side=COALESCE(NULLIF(VALUES(paralysis_side), ''), paralysis_side),
                  disease_days=COALESCE(VALUES(disease_days), disease_days),
                  updated_at=VALUES(updated_at)
                """,
                (
                    pid,
                    _field(patient, "name"),
                    _field(patient, "sex"),
                    _field(patient, "age"),
                    _field(patient, "diagnosis"),
                    _field(patient, "paralysis_side"),
                    _field(patient, "disease_days"),
                    source,
                    ts,
                    ts,
                ),
            )
            cur.execute("SELECT id FROM patients WHERE patient_id=%s", (pid,))
            row = cur.fetchone()
    finally:
        conn.close()
    return int(row["id"])


def get_patient_by_business_id(patient_id: str) -> Optional[Dict[str, Any]]:
    """Return the patient basic-info row for a business ``patient_id`` (or None).
    Used by the task-interface parse step to refill the form from the档案."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE patient_id=%s", (patient_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm_patient(row)


def register_device_patient(patient: Any) -> tuple[Dict[str, Any], bool]:
    """Create a device patient exactly once and reject identity collisions.

    ``patient_id`` is the idempotency key. Repeating the same registration
    returns the existing row without mutating it. Name, sex and age define the
    identity fields for collision detection; clinical profile changes use the
    explicit patient-update workflow instead of registration retries.
    """
    ts = now_dt()
    values = {
        "patient_id": _field(patient, "patient_id"),
        "name": _field(patient, "name"),
        "sex": _field(patient, "sex"),
        "age": _field(patient, "age"),
        "diagnosis": _field(patient, "diagnosis"),
        "paralysis_side": _field(patient, "paralysis_side"),
        "disease_days": _field(patient, "disease_days"),
        "hand_function": _hand_function_from_stage(
            _field(patient, "hand_brunnstrom_stage")
        ),
    }
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients
                  (patient_id, name, sex, age, diagnosis, paralysis_side,
                   disease_days, hand_function, source, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'device-enroll', %s, %s)
                ON DUPLICATE KEY UPDATE patient_id=VALUES(patient_id)
                """,
                (
                    values["patient_id"],
                    values["name"],
                    values["sex"],
                    values["age"],
                    values["diagnosis"],
                    values["paralysis_side"],
                    values["disease_days"],
                    values["hand_function"],
                    ts,
                    ts,
                ),
            )
            created = cur.rowcount == 1
            cur.execute(
                "SELECT * FROM patients WHERE patient_id=%s FOR UPDATE",
                (values["patient_id"],),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("patient row was not available after device registration")
            if not created:
                conflicts = [
                    key
                    for key in ("name", "sex", "age")
                    if row.get(key) != values[key]
                ]
                if conflicts:
                    raise PatientRegistrationConflict(conflicts)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    normalized = _norm_patient(row)
    if normalized is None:  # pragma: no cover - guarded above
        raise RuntimeError("patient row normalization failed after device registration")
    return normalized, created


def get_patient(patient_db_id: int) -> Optional[Dict[str, Any]]:
    """Patient row + assessment count + full assessment list."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE id=%s", (patient_db_id,))
            row = cur.fetchone()
            if row is None:
                return None
            patient = _norm_patient(row)
            cur.execute(
                """
                SELECT COUNT(*) AS c, MAX(created_at) AS last_assessed_at
                FROM assessments WHERE patient_db_id=%s
                """,
                (patient_db_id,),
            )
            summary = cur.fetchone()
            patient["assessment_count"] = int(summary["c"])
            patient["last_assessed_at"] = _norm({"v": summary["last_assessed_at"]})["v"]
            cur.execute(
                """
                SELECT id, source, assessment_id, session_id, package_name,
                       institution, n_trials, package_hash, created_at,
                       assessment_time, fma_ue, hand_tone, hand_function,
                       report, report_status, biomarkers, parse_warnings,
                       prediction_json, model_version, llm_provider, llm_model,
                       patient_snapshot, quality_json, validation_status,
                       report_generation
                FROM assessments WHERE patient_db_id=%s
                ORDER BY created_at DESC, id DESC
                """,
                (patient_db_id,),
            )
            patient["assessments"] = [
                _attach_assessment_children(cur, r) for r in cur.fetchall()
            ]
    finally:
        conn.close()
    return patient


def update_patient(patient_db_id: int, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update editable patient profile fields and return the full patient row."""
    updates = {k: v for k, v in fields.items() if k in _PATIENT_EDITABLE}
    if not updates:
        return get_patient(patient_db_id)

    updates["updated_at"] = now_dt()
    cols = ", ".join(f"{k}=%s" for k in updates)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE patients SET {cols} WHERE id=%s",
                (*updates.values(), patient_db_id),
            )
            if cur.rowcount == 0:
                return None
    finally:
        conn.close()
    return get_patient(patient_db_id)


# --------------------------------------------------------------------------- #
# Device credentials                                                           #
# --------------------------------------------------------------------------- #
_DEVICE_CREDENTIAL_PUBLIC_COLUMNS = """
    id, device_id, label, access_scope, token_hint, status, source,
    created_by, created_at, updated_at, last_used_at, rotated_at, revoked_at
"""


def _device_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def ensure_device_credential(
    *,
    device_id: str,
    label: str,
    access_scope: str,
    token_hash: str,
    token_hint: str,
    source: str = "env-migrated",
    created_by: str = "system",
) -> Dict[str, Any]:
    """Insert an environment credential once without reactivating later revokes."""
    existing = get_device_credential_by_device_id(device_id)
    if existing is not None:
        return existing
    return create_device_credential(
        device_id=device_id,
        label=label,
        access_scope=access_scope,
        token_hash=token_hash,
        token_hint=token_hint,
        source=source,
        created_by=created_by,
    )


def create_device_credential(
    *,
    device_id: str,
    label: str,
    access_scope: str,
    token_hash: str,
    token_hint: str,
    source: str = "admin",
    created_by: str = "admin",
) -> Dict[str, Any]:
    ts = now_dt()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO device_credentials
                  (device_id, label, access_scope, token_hash, token_hint,
                   status, source, created_by, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, 'active', %s, %s, %s, %s)
                """,
                (
                    device_id,
                    label,
                    access_scope,
                    token_hash,
                    token_hint,
                    source,
                    created_by,
                    ts,
                    ts,
                ),
            )
            credential_id = int(cur.lastrowid)
    finally:
        conn.close()
    credential = get_device_credential(credential_id)
    if credential is None:
        raise MySQLUnavailable(f"设备凭证创建后无法读取：{device_id}")
    return credential


def get_device_credential(credential_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_DEVICE_CREDENTIAL_PUBLIC_COLUMNS} FROM device_credentials WHERE id=%s",
                (credential_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm(row)


def get_device_credential_by_device_id(device_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_DEVICE_CREDENTIAL_PUBLIC_COLUMNS} FROM device_credentials WHERE device_id=%s",
                (device_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm(row)


def list_device_credentials() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_DEVICE_CREDENTIAL_PUBLIC_COLUMNS},
                       CASE WHEN access_scope='shared'
                         THEN (SELECT COUNT(*) FROM device_jobs)
                         ELSE (SELECT COUNT(*) FROM device_jobs j WHERE j.device_id=device_credentials.device_id)
                       END AS job_count,
                       CASE WHEN access_scope='shared'
                         THEN (SELECT MAX(created_at) FROM device_jobs)
                         ELSE (SELECT MAX(j.created_at) FROM device_jobs j WHERE j.device_id=device_credentials.device_id)
                       END AS last_job_at
                FROM device_credentials
                ORDER BY access_scope='shared' DESC, created_at, id
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_norm(row) for row in rows]


def authenticate_device_credential(token: str) -> tuple[Optional[Dict[str, Any]], int]:
    """Return an active credential and total configured rows.

    Once at least one row exists, the database is authoritative and callers
    must not fall back to environment plaintext tokens.
    """
    digest = _device_token_hash(token)
    now = now_dt()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM device_credentials")
            configured = int(cur.fetchone()["c"])
            cur.execute(
                """
                SELECT id, device_id, access_scope, status
                FROM device_credentials
                WHERE token_hash=%s
                LIMIT 1
                """,
                (digest,),
            )
            row = cur.fetchone()
            if row and row.get("status") == "active":
                cur.execute(
                    "UPDATE device_credentials SET last_used_at=%s WHERE id=%s",
                    (now, row["id"]),
                )
            else:
                row = None
    finally:
        conn.close()
    return _norm(row), configured


def update_device_credential(
    credential_id: int,
    *,
    label: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    fields: Dict[str, Any] = {"updated_at": now_dt()}
    if label is not None:
        fields["label"] = label
    if status is not None:
        fields["status"] = status
    cols = ", ".join(f"{key}=%s" for key in fields)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE device_credentials SET {cols} WHERE id=%s AND status<>'revoked'",
                (*fields.values(), credential_id),
            )
    finally:
        conn.close()
    return get_device_credential(credential_id)


def rotate_device_credential(
    credential_id: int,
    *,
    token_hash: str,
    token_hint: str,
) -> Optional[Dict[str, Any]]:
    ts = now_dt()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_credentials
                SET token_hash=%s, token_hint=%s, status='active',
                    rotated_at=%s, revoked_at=NULL, updated_at=%s
                WHERE id=%s
                """,
                (token_hash, token_hint, ts, ts, credential_id),
            )
    finally:
        conn.close()
    return get_device_credential(credential_id)


def revoke_device_credential(credential_id: int) -> Optional[Dict[str, Any]]:
    ts = now_dt()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_credentials
                SET status='revoked', revoked_at=COALESCE(revoked_at, %s), updated_at=%s
                WHERE id=%s
                """,
                (ts, ts, credential_id),
            )
    finally:
        conn.close()
    return get_device_credential(credential_id)


# --------------------------------------------------------------------------- #
# Device API jobs                                                              #
# --------------------------------------------------------------------------- #
def create_device_job(
    *,
    job_id: str,
    device_id: Optional[str],
    session_id: str,
    assessment_id: Optional[str],
    patient_id: str,
    package_name: Optional[str],
    package_hash: Optional[str],
    institution: str,
    input_path: str,
    patient_json: Dict[str, Any],
    parse_warnings: Optional[List[str]],
    n_trials: int,
    idempotency_key: Optional[str] = None,
    status: str = "queued",
) -> Dict[str, Any]:
    """Create one device upload/analysis job."""
    ts = now_dt()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO device_jobs
                  (job_id, device_id, session_id, assessment_id, patient_id,
                   package_name, package_hash, institution, input_path,
                   patient_json, parse_warnings, n_trials, idempotency_key,
                   status, phase, progress_percent, status_message,
                   created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, 'waiting', 0, %s, %s, %s)
                """,
                (
                    job_id,
                    device_id,
                    session_id,
                    assessment_id,
                    patient_id,
                    package_name,
                    package_hash,
                    institution,
                    input_path,
                    json.dumps(patient_json, ensure_ascii=False),
                    json.dumps(parse_warnings or [], ensure_ascii=False),
                    n_trials,
                    idempotency_key,
                    status,
                    "已接收评估数据，等待处理",
                    ts,
                    ts,
                ),
            )
    finally:
        conn.close()
    job = get_device_job(job_id)
    if job is None:
        raise MySQLUnavailable(f"设备任务创建后无法读取：{job_id}")
    return job


def update_device_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    progress_percent: Optional[int] = None,
    status_message: Optional[str] = None,
    assessment_db_id: Optional[int] = None,
    error_message: Optional[str] = None,
    error_code: Optional[str] = None,
    error_retryable: Optional[bool] = None,
    clear_error: bool = False,
    increment_attempt: bool = False,
    reset_timestamps: bool = False,
    mark_started: bool = False,
    mark_completed: bool = False,
    mark_delivered: bool = False,
) -> Optional[Dict[str, Any]]:
    """Update one device job and return the normalized row."""
    now = now_dt()
    fields: Dict[str, Any] = {"updated_at": now}
    if status is not None:
        fields["status"] = status
    if phase is not None:
        fields["phase"] = phase
    if progress_percent is not None:
        fields["progress_percent"] = max(0, min(100, int(progress_percent)))
    if status_message is not None:
        fields["status_message"] = status_message
    if assessment_db_id is not None:
        fields["assessment_db_id"] = assessment_db_id
    if error_message is not None:
        fields["error_message"] = error_message
    if error_code is not None:
        fields["error_code"] = error_code
    if error_retryable is not None:
        fields["error_retryable"] = 1 if error_retryable else 0
    if clear_error:
        fields["error_message"] = None
        fields["error_code"] = None
        fields["error_retryable"] = 0
    if mark_started:
        fields["started_at"] = now
    if mark_completed:
        fields["completed_at"] = now
    if mark_delivered:
        fields["delivered_at"] = now

    assignments = [f"{k}=%s" for k in fields]
    if increment_attempt:
        assignments.append("attempt_count=attempt_count+1")
    if reset_timestamps:
        assignments.extend(("started_at=NULL", "completed_at=NULL", "delivered_at=NULL"))
    cols = ", ".join(assignments)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE device_jobs SET {cols} WHERE job_id=%s", (*fields.values(), job_id))
    finally:
        conn.close()
    return get_device_job(job_id)


def get_device_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM device_jobs WHERE job_id=%s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm_device_job(row)


def find_device_job_by_idempotency_key(
    idempotency_key: str,
    device_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM device_jobs
                WHERE idempotency_key=%s AND (device_id <=> %s)
                LIMIT 1
                """,
                (idempotency_key, device_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm_device_job(row)


def find_reusable_device_job(
    *,
    package_hash: str,
    patient_id: str,
    device_id: Optional[str],
    assessment_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Find the newest non-failed submission for the same logical package."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM device_jobs
                WHERE package_hash=%s AND patient_id=%s
                  AND (device_id <=> %s) AND (assessment_id <=> %s)
                  AND status IN ('queued', 'running', 'completed', 'delivered')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (package_hash, patient_id, device_id, assessment_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm_device_job(row)


def list_recoverable_device_jobs() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM device_jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at, job_id
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_norm_device_job(row) for row in rows]


def list_device_job_retention_records() -> List[Dict[str, Any]]:
    """Return the small job subset needed by filesystem retention cleanup."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT job_id, status FROM device_jobs")
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def assessment_id_for_session(session_id: str) -> Optional[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM assessments WHERE session_id=%s ORDER BY id DESC LIMIT 1",
                (session_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return int(row["id"]) if row else None


def list_patients() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.*,
                       COUNT(a.id)       AS assessment_count,
                       MAX(a.created_at) AS last_assessed_at
                FROM patients p
                LEFT JOIN assessments a ON a.patient_db_id = p.id
                GROUP BY p.id
                ORDER BY MAX(a.created_at) IS NULL, MAX(a.created_at) DESC, p.updated_at DESC
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_norm_patient(r) for r in rows]


def delete_patient(patient_db_id: int) -> int:
    """Delete a patient, its device jobs, and cascading assessments atomically."""
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT patient_id FROM patients WHERE id=%s", (patient_db_id,))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return 0
            cur.execute("DELETE FROM device_jobs WHERE patient_id=%s", (row["patient_id"],))
            cur.execute("DELETE FROM patients WHERE id=%s", (patient_db_id,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def assessment_ids_for_patient(patient_db_id: int) -> List[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM assessments WHERE patient_db_id=%s", (patient_db_id,))
            return [int(row["id"]) for row in cur.fetchall()]
    finally:
        conn.close()


def device_jobs_for_patient(patient_db_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.* FROM device_jobs j
                JOIN patients p ON p.patient_id=j.patient_id
                WHERE p.id=%s
                """,
                (patient_db_id,),
            )
            return [_norm_device_job(row) for row in cur.fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Assessments                                                                  #
# --------------------------------------------------------------------------- #
def insert_assessment(
    patient_db_id: int,
    session_id: Optional[str],
    predictions: Any,
    report: Optional[str],
    report_status: str,
    *,
    source: str,
    package_name: Optional[str] = None,
    assessment_id: Optional[str] = None,
    assessment_time: Optional[str] = None,
    created_at: Optional[str] = None,
    biomarkers: Optional[str] = None,
    institution: Optional[str] = None,
    n_trials: Optional[int] = None,
    package_hash: Optional[str] = None,
    parse_warnings: Optional[str] = None,
    prediction_json: Optional[str] = None,
    model_version: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> int:
    """Insert one assessment row.

    The database still has a NOT NULL ``bi`` column for legacy records. Current
    online inference no longer serves BI, so callers may pass a compatibility
    placeholder while user-facing output remains focused on FMA-UE, hand tone
    and Brunnstrom hand function.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO assessments
                  (patient_db_id, source, assessment_id, session_id, package_name,
                   institution, n_trials, package_hash, created_at, assessment_time,
                   fma_ue, bi, hand_tone, hand_function, report, report_status,
                   biomarkers, parse_warnings, prediction_json, model_version,
                   llm_provider, llm_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    patient_db_id,
                    source,
                    assessment_id,
                    session_id,
                    package_name,
                    institution,
                    n_trials,
                    package_hash,
                    created_at or now_dt(),
                    _to_dt(assessment_time),
                    float(_field(predictions, "FMA_UE")),
                    float(_field(predictions, "BI")),
                    str(_field(predictions, "hand_tone")),
                    int(_field(predictions, "hand_function")),
                    report,
                    report_status,
                    biomarkers,
                    parse_warnings,
                    prediction_json,
                    model_version,
                    llm_provider,
                    llm_model,
                ),
            )
            new_id = cur.lastrowid
    finally:
        conn.close()
    return int(new_id)


def replace_assessment_trials(assessment_id: int, trials: Optional[List[Dict[str, Any]]]) -> int:
    """Replace normalized movement/trial rows for one assessment."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM assessment_trials WHERE assessment_db_id=%s", (assessment_id,))
            if not trials:
                return 0
            ts = now_dt()
            rows = []
            for idx, trial in enumerate(trials, start=1):
                rows.append(
                    (
                        assessment_id,
                        trial.get("trial_index") or idx,
                        trial.get("assessment_type"),
                        trial.get("action_name") or trial.get("action"),
                        trial.get("eeg_file") or trial.get("eeg_path"),
                        trial.get("emg_imu_file") or trial.get("emg_file") or trial.get("emg_path"),
                        trial.get("eeg_name"),
                        trial.get("emg_name"),
                        trial.get("status") or "used",
                        trial.get("note"),
                        ts,
                    )
                )
            cur.executemany(
                """
                INSERT INTO assessment_trials
                  (assessment_db_id, trial_index, assessment_type, action_name,
                   eeg_file, emg_file, eeg_name, emg_name, status, note, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            return len(rows)
    finally:
        conn.close()


def replace_assessment_biomarkers(assessment_id: int, biomarkers: Optional[Dict[str, Any]]) -> int:
    """Replace normalized 26-biomarker rows for one assessment."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM assessment_biomarkers WHERE assessment_db_id=%s",
                (assessment_id,),
            )
            if not biomarkers:
                return 0

            rows = _biomarker_insert_rows(assessment_id, biomarkers, now_dt())
            if not rows:
                return 0
            cur.executemany(
                """
                INSERT INTO assessment_biomarkers
                  (assessment_db_id, group_key, group_label, marker_key, marker_name,
                   value_text, value_num, unit, ref_range, n_valid, available, note,
                   created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            return len(rows)
    finally:
        conn.close()


def save_assessment_bundle(
    patient: Any,
    session_id: Optional[str],
    predictions: Any,
    report: Optional[str],
    report_status: str,
    *,
    source: str,
    package_name: Optional[str] = None,
    assessment_id: Optional[str] = None,
    assessment_time: Optional[str] = None,
    institution: Optional[str] = None,
    n_trials: Optional[int] = None,
    package_hash: Optional[str] = None,
    parse_warnings: Optional[List[str]] = None,
    prediction_payload: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    trials: Optional[List[Dict[str, Any]]] = None,
    biomarkers: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    validation_status: Optional[str] = None,
    report_generation: Optional[str] = None,
) -> int:
    """Persist one complete assessment atomically and return its row id.

    Existing patient master data is never changed by an assessment upload. The
    submitted patient information is retained as an immutable assessment-time
    snapshot for historical reports and exports.
    """
    ts = now_dt()
    patient_snapshot = (
        patient.model_dump() if hasattr(patient, "model_dump") else dict(patient)
    )
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients
                  (patient_id, name, sex, age, diagnosis, paralysis_side,
                   disease_days, source, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE patient_id=VALUES(patient_id)
                """,
                (
                    _field(patient, "patient_id"),
                    _field(patient, "name"),
                    _field(patient, "sex"),
                    _field(patient, "age"),
                    _field(patient, "diagnosis"),
                    _field(patient, "paralysis_side"),
                    _field(patient, "disease_days"),
                    f"{source}-auto",
                    ts,
                    ts,
                ),
            )
            cur.execute(
                "SELECT id FROM patients WHERE patient_id=%s",
                (_field(patient, "patient_id"),),
            )
            patient_row = cur.fetchone()
            if not patient_row:
                raise RuntimeError("patient row was not available after insert")
            patient_db_id = int(patient_row["id"])

            cur.execute(
                """
                INSERT INTO assessments
                  (patient_db_id, source, assessment_id, session_id, package_name,
                   institution, n_trials, package_hash, created_at, assessment_time,
                   fma_ue, bi, hand_tone, hand_function, report, report_status,
                   biomarkers, parse_warnings, prediction_json, model_version,
                   llm_provider, llm_model, patient_snapshot, quality_json,
                   validation_status, report_generation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    patient_db_id,
                    source,
                    assessment_id,
                    session_id,
                    package_name,
                    institution,
                    n_trials,
                    package_hash,
                    ts,
                    _to_dt(assessment_time),
                    float(_field(predictions, "FMA_UE")),
                    float(_field(predictions, "BI") or 0.0),
                    str(_field(predictions, "hand_tone")),
                    int(_field(predictions, "hand_function")),
                    report,
                    report_status,
                    json.dumps(biomarkers, ensure_ascii=False) if biomarkers else None,
                    json.dumps(parse_warnings or [], ensure_ascii=False),
                    json.dumps(prediction_payload or {}, ensure_ascii=False),
                    model_version,
                    llm_provider,
                    llm_model,
                    json.dumps(patient_snapshot, ensure_ascii=False),
                    json.dumps(quality or {}, ensure_ascii=False),
                    validation_status,
                    report_generation,
                ),
            )
            new_assessment_id = int(cur.lastrowid)

            trial_rows = []
            for index, trial in enumerate(trials or [], start=1):
                trial_rows.append(
                    (
                        new_assessment_id,
                        trial.get("trial_index") or index,
                        trial.get("assessment_type"),
                        trial.get("action_name") or trial.get("action"),
                        trial.get("eeg_file") or trial.get("eeg_path"),
                        trial.get("emg_imu_file") or trial.get("emg_file") or trial.get("emg_path"),
                        trial.get("eeg_name"),
                        trial.get("emg_name"),
                        trial.get("status") or "used",
                        trial.get("note"),
                        ts,
                    )
                )
            if trial_rows:
                cur.executemany(
                    """
                    INSERT INTO assessment_trials
                      (assessment_db_id, trial_index, assessment_type, action_name,
                       eeg_file, emg_file, eeg_name, emg_name, status, note, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    trial_rows,
                )

            biomarker_rows = _biomarker_insert_rows(new_assessment_id, biomarkers, ts)
            if biomarker_rows:
                cur.executemany(
                    """
                    INSERT INTO assessment_biomarkers
                      (assessment_db_id, group_key, group_label, marker_key, marker_name,
                       value_text, value_num, unit, ref_range, n_valid, available, note,
                       created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    biomarker_rows,
                )
        conn.commit()
        return new_assessment_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
def latest_assessment_for_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Most recent assessment (three served indicators + biomarkers) for a business
    ``patient_id`` — the report's 变化趋势 column reads this before inserting the
    current visit. Returns None if the patient has no prior assessment."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.fma_ue, a.hand_tone, a.hand_function,
                       a.biomarkers, a.created_at, a.assessment_time
                FROM assessments a
                JOIN patients p ON p.id = a.patient_db_id
                WHERE p.patient_id = %s
                ORDER BY COALESCE(a.assessment_time, a.created_at) DESC, a.id DESC
                LIMIT 1
                """,
                (patient_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _norm(row)


def list_assessments(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM assessments")
            total = int(cur.fetchone()["c"])
            cur.execute(
                """
                SELECT a.id, a.created_at, a.patient_db_id, p.patient_id,
                       COALESCE(p.name, '') AS name,
                       a.source, a.assessment_id, a.session_id, a.package_name,
                       a.institution, a.n_trials, a.package_hash, a.assessment_time,
                       a.fma_ue, a.hand_tone, a.hand_function, a.report_status,
                       a.model_version, a.llm_provider, a.llm_model,
                       a.validation_status, a.report_generation
                FROM assessments a
                JOIN patients p ON p.id = a.patient_db_id
                ORDER BY a.created_at DESC, a.id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            items = [_norm(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return {"total": total, "items": items}


def get_assessment(assessment_id: int) -> Optional[Dict[str, Any]]:
    """Return one structured assessment row, including report and JSON payloads."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.*, p.patient_id, p.name, p.sex, p.age, p.diagnosis,
                       p.paralysis_side, p.disease_days
                FROM assessments a
                JOIN patients p ON p.id = a.patient_db_id
                WHERE a.id=%s
                """,
                (assessment_id,),
            )
            row = cur.fetchone()
            detail = _attach_assessment_children(cur, row)
    finally:
        conn.close()
    return detail


def stats_summary() -> Dict[str, Any]:
    """Aggregate dashboard/statistics metrics from the MySQL business store."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM patients")
            patient_count = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) AS c FROM assessments")
            assessment_count = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) AS c FROM assessments WHERE report_status='failed'")
            report_failed = int(cur.fetchone()["c"])

            cur.execute(
                """
                SELECT COALESCE(NULLIF(diagnosis, ''), '未填写') AS diagnosis,
                       COUNT(*) AS c
                FROM patients
                GROUP BY COALESCE(NULLIF(diagnosis, ''), '未填写')
                """
            )
            diag_rows = cur.fetchall()

            cur.execute(
                """
                SELECT hand_function, COUNT(*) AS c
                FROM assessments
                GROUP BY hand_function
                """
            )
            hand_rows = cur.fetchall()

            cur.execute("SELECT AVG(fma_ue) AS fma FROM assessments")
            avg_row = cur.fetchone()

            cur.execute(
                """
                SELECT DATE(created_at) AS date, COUNT(*) AS count
                FROM assessments
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 30
                """
            )
            day_rows = cur.fetchall()
    finally:
        conn.close()

    fma = avg_row["fma"] if avg_row else None
    return {
        "patient_count": patient_count,
        "assessment_count": assessment_count,
        "report_failed_count": report_failed,
        "diagnosis_distribution": {
            str(r["diagnosis"]): int(r["c"]) for r in diag_rows
        },
        "hand_function_distribution": {
            str(r["hand_function"]): int(r["c"]) for r in hand_rows
        },
        "avg_fma_ue": round(float(fma), 1) if fma is not None else None,
        "assessments_by_day": [
            {"date": str(r["date"]), "count": int(r["count"])}
            for r in reversed(day_rows)
        ],
    }


def delete_assessment(assessment_id: int) -> int:
    """Delete one assessment and its associated device jobs atomically."""
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM device_jobs WHERE assessment_db_id=%s", (assessment_id,))
            cur.execute("DELETE FROM assessments WHERE id=%s", (assessment_id,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def device_jobs_for_assessment(assessment_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM device_jobs WHERE assessment_db_id=%s", (assessment_id,))
            return [_norm_device_job(row) for row in cur.fetchall()]
    finally:
        conn.close()


def all_assessment_ids() -> List[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM assessments")
            return [int(row["id"]) for row in cur.fetchall()]
    finally:
        conn.close()


def completed_device_jobs() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM device_jobs WHERE assessment_db_id IS NOT NULL")
            return [_norm_device_job(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_all_assessments() -> int:
    """Clear assessments and their completed device jobs atomically."""
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM device_jobs WHERE assessment_db_id IS NOT NULL")
            cur.execute("DELETE FROM assessments")
            deleted = cur.rowcount
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Enrollment (hospital → MySQL): basic info + first assessment record          #
# --------------------------------------------------------------------------- #
def enroll_patient(basic: Dict[str, Any], first: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enroll a patient: upsert basic info (source='enrollment') and, if a first
    assessment is supplied (the hospital's first record, manually entered),
    insert the patient, assessment, trials, and biomarkers in one transaction.
    Returns the full patient record.
    """
    ts = now_dt()
    conn = get_conn(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients
                  (patient_id, name, sex, age, diagnosis, paralysis_side,
                   disease_days, source, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'enrollment', %s, %s)
                ON DUPLICATE KEY UPDATE
                  name=VALUES(name), sex=VALUES(sex),
                  age=COALESCE(VALUES(age), age),
                  diagnosis=COALESCE(NULLIF(VALUES(diagnosis), ''), diagnosis),
                  paralysis_side=COALESCE(NULLIF(VALUES(paralysis_side), ''), paralysis_side),
                  disease_days=COALESCE(VALUES(disease_days), disease_days),
                  updated_at=VALUES(updated_at)
                """,
                (
                    _field(basic, "patient_id"),
                    _field(basic, "name"),
                    _field(basic, "sex"),
                    _field(basic, "age"),
                    _field(basic, "diagnosis"),
                    _field(basic, "paralysis_side"),
                    _field(basic, "disease_days"),
                    ts,
                    ts,
                ),
            )
            cur.execute(
                "SELECT id FROM patients WHERE patient_id=%s",
                (_field(basic, "patient_id"),),
            )
            patient_row = cur.fetchone()
            if not patient_row:
                raise RuntimeError("patient row was not available after enrollment")
            patient_db_id = int(patient_row["id"])

            if first:
                biomarkers = _json_value(first.get("biomarkers"))
                parse_warnings = _json_value(first.get("parse_warnings")) or []
                prediction_payload = _json_value(first.get("prediction_json")) or {}
                report = first.get("report")
                cur.execute(
                    """
                    INSERT INTO assessments
                      (patient_db_id, source, assessment_id, session_id, package_name,
                       institution, n_trials, package_hash, created_at, assessment_time,
                       fma_ue, bi, hand_tone, hand_function, report, report_status,
                       biomarkers, parse_warnings, prediction_json, model_version,
                       llm_provider, llm_model, patient_snapshot, quality_json,
                       validation_status, report_generation)
                    VALUES (%s, 'hospital', %s, NULL, NULL, 'hospital', %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, NULL, 'manual_clinical_input', %s)
                    """,
                    (
                        patient_db_id,
                        first.get("assessment_id"),
                        first.get("n_trials"),
                        first.get("package_hash"),
                        ts,
                        _to_dt(first.get("assessment_time")),
                        float(_field(first, "FMA_UE")),
                        float(_field(first, "BI") or 0.0),
                        str(_field(first, "hand_tone")),
                        int(_field(first, "hand_function")),
                        report,
                        "generated" if report else "manual",
                        json.dumps(biomarkers, ensure_ascii=False) if biomarkers else None,
                        json.dumps(parse_warnings, ensure_ascii=False),
                        json.dumps(prediction_payload, ensure_ascii=False),
                        first.get("model_version"),
                        first.get("llm_provider"),
                        first.get("llm_model"),
                        json.dumps(basic, ensure_ascii=False),
                        "manual" if report else None,
                    ),
                )
                assessment_db_id = int(cur.lastrowid)

                trial_rows = []
                for index, trial in enumerate(first.get("trials") or [], start=1):
                    trial_rows.append(
                        (
                            assessment_db_id,
                            trial.get("trial_index") or index,
                            trial.get("assessment_type"),
                            trial.get("action_name") or trial.get("action"),
                            trial.get("eeg_file") or trial.get("eeg_path"),
                            trial.get("emg_imu_file") or trial.get("emg_file") or trial.get("emg_path"),
                            trial.get("eeg_name"),
                            trial.get("emg_name"),
                            trial.get("status") or "used",
                            trial.get("note"),
                            ts,
                        )
                    )
                if trial_rows:
                    cur.executemany(
                        """
                        INSERT INTO assessment_trials
                          (assessment_db_id, trial_index, assessment_type, action_name,
                           eeg_file, emg_file, eeg_name, emg_name, status, note, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        trial_rows,
                    )

                biomarker_rows = _biomarker_insert_rows(assessment_db_id, biomarkers, ts)
                if biomarker_rows:
                    cur.executemany(
                        """
                        INSERT INTO assessment_biomarkers
                          (assessment_db_id, group_key, group_label, marker_key, marker_name,
                           value_text, value_num, unit, ref_range, n_valid, available, note,
                           created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        biomarker_rows,
                    )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_patient(patient_db_id)  # type: ignore[return-value]


__all__ = [
    "MySQLUnavailable",
    "PatientRegistrationConflict",
    "init_db",
    "get_conn",
    "now_dt",
    "upsert_patient",
    "get_patient_by_business_id",
    "register_device_patient",
    "get_patient",
    "update_patient",
    "list_patients",
    "delete_patient",
    "insert_assessment",
    "replace_assessment_trials",
    "replace_assessment_biomarkers",
    "latest_assessment_for_patient",
    "ensure_device_credential",
    "create_device_credential",
    "get_device_credential",
    "get_device_credential_by_device_id",
    "list_device_credentials",
    "authenticate_device_credential",
    "update_device_credential",
    "rotate_device_credential",
    "revoke_device_credential",
    "create_device_job",
    "update_device_job",
    "get_device_job",
    "find_device_job_by_idempotency_key",
    "find_reusable_device_job",
    "list_recoverable_device_jobs",
    "list_device_job_retention_records",
    "assessment_id_for_session",
    "list_assessments",
    "get_assessment",
    "stats_summary",
    "delete_assessment",
    "delete_all_assessments",
    "enroll_patient",
]
