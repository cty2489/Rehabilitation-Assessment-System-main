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

import os
import json
import math
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
    "birth_date": "VARCHAR(32)",
    "id_number": "VARCHAR(32)",
    "phone": "VARCHAR(32)",
    "onset_date": "VARCHAR(32)",
}

_PATIENT_CORE = ("name", "sex", "age", "diagnosis", "disease_days", "paralysis_side")
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
}

_ASSESSMENT_INDEXES: Dict[str, str] = {
    "idx_assess_external": "CREATE INDEX idx_assess_external ON assessments(assessment_id)",
    "idx_assess_session": "CREATE INDEX idx_assess_session ON assessments(session_id)",
}


class MySQLUnavailable(RuntimeError):
    """Raised when MySQL / pymysql is not usable; callers surface a clear error."""


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


def get_conn(with_db: bool = True):
    if pymysql is None:
        raise MySQLUnavailable("未安装 pymysql，无法连接 MySQL（pip install pymysql）")
    cfg = _config()
    kwargs: Dict[str, Any] = {
        "host": cfg["host"],
        "port": cfg["port"],
        "user": cfg["user"],
        "password": cfg["password"],
        "charset": "utf8mb4",
        "autocommit": True,
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
        return datetime.fromisoformat(str(value).strip()).strftime("%Y-%m-%d %H:%M:%S")
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
    for key in ("biomarkers", "parse_warnings", "prediction_json"):
        row[key] = _json_value(row.get(key))
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
  status            VARCHAR(32) NOT NULL,
  error_message     TEXT,
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
  INDEX idx_device_job_status (status)
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
                _ensure_patient_schema(cur)
                _ensure_assessment_schema(cur)
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
                  name=VALUES(name), sex=VALUES(sex), age=VALUES(age),
                  diagnosis=VALUES(diagnosis), paralysis_side=VALUES(paralysis_side),
                  disease_days=VALUES(disease_days), updated_at=VALUES(updated_at)
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
                       assessment_time, fma_ue, bi, hand_tone, hand_function,
                       report, report_status, biomarkers, parse_warnings,
                       prediction_json, model_version, llm_provider, llm_model
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
                   package_name, package_hash, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id,
                    device_id,
                    session_id,
                    assessment_id,
                    patient_id,
                    package_name,
                    package_hash,
                    status,
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
    assessment_db_id: Optional[int] = None,
    error_message: Optional[str] = None,
    mark_started: bool = False,
    mark_completed: bool = False,
    mark_delivered: bool = False,
) -> Optional[Dict[str, Any]]:
    """Update one device job and return the normalized row."""
    now = now_dt()
    fields: Dict[str, Any] = {"updated_at": now}
    if status is not None:
        fields["status"] = status
    if assessment_db_id is not None:
        fields["assessment_db_id"] = assessment_db_id
    if error_message is not None:
        fields["error_message"] = error_message
    if mark_started:
        fields["started_at"] = now
    if mark_completed:
        fields["completed_at"] = now
    if mark_delivered:
        fields["delivered_at"] = now

    cols = ", ".join(f"{k}=%s" for k in fields)
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
    return _norm(row)


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
    """Delete a patient (assessments cascade). Returns rows affected."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM patients WHERE id=%s", (patient_db_id,))
            return cur.rowcount
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


def latest_assessment_for_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Most recent assessment (4 indicators + biomarkers) for a business
    ``patient_id`` — the report's 变化趋势 column reads this before inserting the
    current visit. Returns None if the patient has no prior assessment."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.fma_ue, a.bi, a.hand_tone, a.hand_function,
                       a.biomarkers, a.created_at
                FROM assessments a
                JOIN patients p ON p.id = a.patient_db_id
                WHERE p.patient_id = %s
                ORDER BY a.created_at DESC, a.id DESC
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
                       a.fma_ue, a.bi, a.hand_tone, a.hand_function, a.report_status,
                       a.model_version, a.llm_provider, a.llm_model
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

            cur.execute("SELECT AVG(fma_ue) AS fma, AVG(bi) AS bi FROM assessments")
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
    bi = avg_row["bi"] if avg_row else None
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
        "avg_bi": round(float(bi), 1) if bi is not None else None,
        "assessments_by_day": [
            {"date": str(r["date"]), "count": int(r["count"])}
            for r in reversed(day_rows)
        ],
    }


def delete_assessment(assessment_id: int) -> int:
    """Delete one assessment row by primary key. Returns rows affected."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM assessments WHERE id=%s", (assessment_id,))
            return cur.rowcount
    finally:
        conn.close()


def delete_all_assessments() -> int:
    """Clear every assessment row (test cleanup). Returns rows affected."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM assessments")
            return cur.rowcount
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Enrollment (hospital → MySQL): basic info + first assessment record          #
# --------------------------------------------------------------------------- #
def enroll_patient(basic: Dict[str, Any], first: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enroll a patient: upsert basic info (source='enrollment') and, if a first
    assessment is supplied (the hospital's first record, manually entered),
    insert it (source='hospital'). Returns the full patient record."""
    pid = upsert_patient(basic, source="enrollment")
    if first:
        assessment_id = insert_assessment(
            pid,
            None,
            first,
            first.get("report"),
            "generated" if first.get("report") else "manual",
            source="hospital",
            assessment_id=first.get("assessment_id"),
            assessment_time=first.get("assessment_time"),
            biomarkers=first.get("biomarkers"),
            institution="hospital",
            n_trials=first.get("n_trials"),
            package_hash=first.get("package_hash"),
            parse_warnings=first.get("parse_warnings"),
            prediction_json=first.get("prediction_json"),
            model_version=first.get("model_version"),
            llm_provider=first.get("llm_provider"),
            llm_model=first.get("llm_model"),
        )
        replace_assessment_trials(assessment_id, first.get("trials"))
        replace_assessment_biomarkers(assessment_id, _json_value(first.get("biomarkers")))
    return get_patient(pid)  # type: ignore[return-value]


__all__ = [
    "MySQLUnavailable",
    "init_db",
    "get_conn",
    "now_dt",
    "upsert_patient",
    "get_patient_by_business_id",
    "get_patient",
    "update_patient",
    "list_patients",
    "delete_patient",
    "insert_assessment",
    "replace_assessment_trials",
    "replace_assessment_biomarkers",
    "latest_assessment_for_patient",
    "list_assessments",
    "get_assessment",
    "stats_summary",
    "delete_assessment",
    "delete_all_assessments",
    "enroll_patient",
]
