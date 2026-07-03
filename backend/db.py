"""SQLite persistence for patients and assessment records.

Stdlib-only (no ORM). Two tables — ``patients`` (one row per business
``patient_id``) and ``assessments`` (many rows per patient). Connections are
opened per-operation with ``check_same_thread=False`` so the inference worker
thread and the request handlers can both write/read; WAL mode keeps the single
writer + readers happy on a single host.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

DB_PATH = Path(__file__).resolve().parent / "rehab.db"

_init_lock = threading.Lock()
_initialized = False

# Columns the upsert may overwrite on a repeat assessment. Extended fields
# (birth_date/id_number/phone/onset_date) are intentionally excluded so manually
# supplemented info survives re-assessment.
_PATIENT_CORE = ("name", "sex", "age", "diagnosis", "disease_days", "paralysis_side")
_PATIENT_EXTENDED = ("birth_date", "id_number", "phone", "onset_date")
_PATIENT_EDITABLE = _PATIENT_CORE + _PATIENT_EXTENDED


def now_iso() -> str:
    """Beijing-time ISO-8601 timestamp, second precision, no microseconds."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        with get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                  id              INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id      TEXT NOT NULL UNIQUE,
                  name            TEXT NOT NULL,
                  sex             TEXT NOT NULL,
                  age             INTEGER,
                  diagnosis       TEXT NOT NULL,
                  disease_days    INTEGER,
                  paralysis_side  TEXT NOT NULL,
                  birth_date      TEXT,
                  id_number       TEXT,
                  phone           TEXT,
                  onset_date      TEXT,
                  created_at      TEXT NOT NULL,
                  updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assessments (
                  id              INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_db_id   INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                  session_id      TEXT,
                  created_at      TEXT NOT NULL,
                  fma_ue          REAL NOT NULL,
                  bi              REAL NOT NULL,
                  hand_tone       TEXT NOT NULL,
                  hand_function   INTEGER NOT NULL,
                  report          TEXT,
                  report_status   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assessments_patient
                  ON assessments(patient_db_id);
                CREATE INDEX IF NOT EXISTS idx_assessments_created
                  ON assessments(created_at);
                """
            )
            # Idempotent migration: add the biomarkers JSON column if missing
            # (older DBs created before the multi-section report). SQLite has no
            # "ADD COLUMN IF NOT EXISTS", so probe the schema first.
            cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(assessments)").fetchall()
            }
            if "biomarkers" not in cols:
                conn.execute("ALTER TABLE assessments ADD COLUMN biomarkers TEXT")
        _initialized = True


# --------------------------------------------------------------------------- #
# Patients                                                                     #
# --------------------------------------------------------------------------- #
def upsert_patient(patient: Any) -> int:
    """Insert or update a patient by business key ``patient_id``.

    ``patient`` is a PatientInfo (has patient_id/name/sex/age/diagnosis/
    disease_days/paralysis_side). Extended fields are never touched here.
    Returns the patient row id.
    """
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO patients
              (patient_id, name, sex, age, diagnosis, disease_days,
               paralysis_side, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
              name=excluded.name,
              sex=excluded.sex,
              age=excluded.age,
              diagnosis=excluded.diagnosis,
              disease_days=excluded.disease_days,
              paralysis_side=excluded.paralysis_side,
              updated_at=excluded.updated_at
            """,
            (
                patient.patient_id,
                patient.name,
                patient.sex,
                patient.age,
                patient.diagnosis,
                patient.disease_days,
                patient.paralysis_side,
                ts,
                ts,
            ),
        )
        row = conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?", (patient.patient_id,)
        ).fetchone()
    return int(row["id"])


def list_patients() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.*,
                   COUNT(a.id)        AS assessment_count,
                   MAX(a.created_at)  AS last_assessed_at
            FROM patients p
            LEFT JOIN assessments a ON a.patient_db_id = p.id
            GROUP BY p.id
            ORDER BY (last_assessed_at IS NULL), last_assessed_at DESC, p.updated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_patient(patient_db_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_db_id,)
        ).fetchone()
        if row is None:
            return None
        patient = dict(row)
        patient["assessment_count"] = conn.execute(
            "SELECT COUNT(*) AS c FROM assessments WHERE patient_db_id = ?",
            (patient_db_id,),
        ).fetchone()["c"]
        patient["last_assessed_at"] = conn.execute(
            "SELECT MAX(created_at) AS m FROM assessments WHERE patient_db_id = ?",
            (patient_db_id,),
        ).fetchone()["m"]
        patient["assessments"] = list_assessments_for_patient(patient_db_id, conn)
    return patient


def get_patient_by_business_id(patient_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    return get_patient(int(row["id"])) if row else None


def update_patient(patient_db_id: int, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    updates = {k: v for k, v in fields.items() if k in _PATIENT_EDITABLE}
    if updates:
        updates["updated_at"] = now_iso()
        cols = ", ".join(f"{k} = ?" for k in updates)
        with get_conn() as conn:
            cur = conn.execute(
                f"UPDATE patients SET {cols} WHERE id = ?",
                (*updates.values(), patient_db_id),
            )
            if cur.rowcount == 0:
                return None
    return get_patient(patient_db_id)


# --------------------------------------------------------------------------- #
# Assessments                                                                  #
# --------------------------------------------------------------------------- #
def insert_assessment(
    patient_db_id: int,
    session_id: Optional[str],
    predictions: Any,
    report: Optional[str],
    report_status: str,
    created_at: Optional[str] = None,
    biomarkers: Optional[str] = None,
) -> int:
    """Insert one assessment row. ``predictions`` is a PredictionResult.

    ``biomarkers`` is the JSON-serialised digital-biomarker dict (or None) so a
    later visit can compute biomarker change-trends.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO assessments
              (patient_db_id, session_id, created_at, fma_ue, bi,
               hand_tone, hand_function, report, report_status, biomarkers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_db_id,
                session_id,
                created_at or now_iso(),
                float(predictions.FMA_UE),
                float(predictions.BI),
                str(predictions.hand_tone),
                int(predictions.hand_function),
                report,
                report_status,
                biomarkers,
            ),
        )
    return int(cur.lastrowid)


def latest_assessment_for_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent assessment (4 indicators + biomarkers) for a
    business ``patient_id``, or None if the patient has no prior assessment.

    Used for the report's 变化趋势 column. Returns a plain dict with keys
    ``fma_ue/bi/hand_tone/hand_function/biomarkers/created_at``.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT a.fma_ue, a.bi, a.hand_tone, a.hand_function,
                   a.biomarkers, a.created_at
            FROM assessments a
            JOIN patients p ON p.id = a.patient_db_id
            WHERE p.patient_id = ?
            ORDER BY a.created_at DESC, a.id DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
    return dict(row) if row else None


def list_assessments_for_patient(
    patient_db_id: int, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    own = conn is None
    conn = conn or get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, created_at, fma_ue, bi, hand_tone,
                   hand_function, report, report_status
            FROM assessments
            WHERE patient_db_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (patient_db_id,),
        ).fetchall()
    finally:
        if own:
            conn.close()
    return [dict(r) for r in rows]


def list_all_assessments(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM assessments").fetchone()["c"]
        rows = conn.execute(
            """
            SELECT a.id, a.created_at, a.patient_db_id,
                   p.patient_id, p.name,
                   a.fma_ue, a.bi, a.hand_tone, a.hand_function, a.report_status
            FROM assessments a
            JOIN patients p ON p.id = a.patient_db_id
            ORDER BY a.created_at DESC, a.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return {"total": int(total), "items": [dict(r) for r in rows]}


# --------------------------------------------------------------------------- #
# Stats                                                                        #
# --------------------------------------------------------------------------- #
def stats_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        patient_count = conn.execute("SELECT COUNT(*) AS c FROM patients").fetchone()["c"]
        assessment_count = conn.execute(
            "SELECT COUNT(*) AS c FROM assessments"
        ).fetchone()["c"]
        report_failed = conn.execute(
            "SELECT COUNT(*) AS c FROM assessments WHERE report_status = 'failed'"
        ).fetchone()["c"]

        diag_rows = conn.execute(
            "SELECT diagnosis, COUNT(*) AS c FROM patients GROUP BY diagnosis"
        ).fetchall()
        hand_rows = conn.execute(
            "SELECT hand_function, COUNT(*) AS c FROM assessments GROUP BY hand_function"
        ).fetchall()
        avg_row = conn.execute(
            "SELECT AVG(fma_ue) AS fma, AVG(bi) AS bi FROM assessments"
        ).fetchone()
        day_rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count
            FROM assessments
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
            """
        ).fetchall()

    return {
        "patient_count": int(patient_count),
        "assessment_count": int(assessment_count),
        "report_failed_count": int(report_failed),
        "diagnosis_distribution": {r["diagnosis"]: int(r["c"]) for r in diag_rows},
        "hand_function_distribution": {
            str(r["hand_function"]): int(r["c"]) for r in hand_rows
        },
        "avg_fma_ue": round(avg_row["fma"], 1) if avg_row["fma"] is not None else None,
        "avg_bi": round(avg_row["bi"], 1) if avg_row["bi"] is not None else None,
        "assessments_by_day": [
            {"date": r["date"], "count": int(r["count"])} for r in reversed(day_rows)
        ],
    }
