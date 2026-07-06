"""FastAPI entrypoint for the rehabilitation assessment platform.

Three endpoints:
  POST /api/assess                       — accept multipart files + patient info, return session_id
  GET  /api/assess/{session_id}/stream   — SSE stream of progress events
  GET  /api/assess/{session_id}/result   — cached final result (reconnect fallback)

The full inference pipeline runs in a worker thread; events are pushed onto a
per-session queue.Queue that the SSE coroutine drains asynchronously.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import secrets
import shutil
import time
import tempfile
import threading
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import db
import mysql_db
from assessment_export import ensure_assessment_export, export_filename, file_info
from eval_package import INSTITUTIONS, read_eval_package, safe_extract_zip
from inference import CHECKPOINTS, SENTINEL, ModelRegistry, error_event, run_pipeline
from report import REPORT_MODEL, llm_provider, remote_url, stream_report
from schemas import (
    AssessmentOverview,
    AssessmentResult,
    AssessSessionResponse,
    AuthLoginRequest,
    AuthLoginResponse,
    EnrollmentRequest,
    MysqlAssessmentDetail,
    MysqlAssessmentList,
    PatientDetail,
    PatientInfo,
    PatientSummary,
    PatientUpdate,
    PredictionResult,
    StatsSummary,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

SESSION_ROOT = Path(tempfile.gettempdir()) / "rehab_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

UPLOAD_CHUNK_BYTES = 1024 * 1024


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        print(f"[startup][warn] invalid {name}={raw!r}; using {default}")
        return default


MAX_UPLOAD_FILE_BYTES = _env_int("MAX_UPLOAD_FILE_BYTES", 2 * 1024 * 1024 * 1024)
MAX_TRIALS = _env_int("MAX_TRIALS", 30)
MAX_ZIP_BYTES = _env_int("MAX_ZIP_BYTES", 4 * 1024 * 1024 * 1024)
MAX_ZIP_EXTRACTED_BYTES = _env_int("MAX_ZIP_EXTRACTED_BYTES", 10 * 1024 * 1024 * 1024)
MAX_ZIP_MEMBERS = _env_int("MAX_ZIP_MEMBERS", 2000)
SESSION_TTL_HOURS = _env_int("SESSION_TTL_HOURS", 168)
_LAST_SESSION_CLEANUP = 0.0


# --------------------------------------------------------------------------- #
# In-process session registry. Keys: session_id → SessionState.               #
# --------------------------------------------------------------------------- #
class SessionState:
    def __init__(self, session_id: str, patient: PatientInfo, eeg_paths: List[Path],
                 emg_paths: List[Path], institution: str = "hospital",
                 persist_target: str = "sqlite", package_name: Optional[str] = None,
                 assessment_id: Optional[str] = None, assessment_time: Optional[str] = None,
                 n_trials: Optional[int] = None, package_hash: Optional[str] = None,
                 parse_warnings: Optional[List[str]] = None,
                 trial_details: Optional[List[Dict[str, Any]]] = None,
                 device_job_id: Optional[str] = None):
        self.session_id = session_id
        self.patient = patient
        self.eeg_paths = eeg_paths
        self.emg_paths = emg_paths
        self.institution = institution
        # Where this session's result is persisted. MySQL is the formal business
        # store; SQLite is kept only as a legacy/local fallback target.
        self.persist_target = persist_target
        self.package_name = package_name
        self.assessment_id = assessment_id
        self.assessment_time = assessment_time
        self.n_trials = n_trials if n_trials is not None else len(eeg_paths)
        self.package_hash = package_hash
        self.parse_warnings = parse_warnings or []
        self.trial_details = trial_details or _trial_details_from_paths(eeg_paths, emg_paths)
        self.device_job_id = device_job_id
        self.assessment_db_id: Optional[int] = None
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.result: Optional[AssessmentResult] = None
        self.started: bool = False
        self.lock = threading.Lock()


def _dl_model_version() -> str:
    return ";".join(f"{task}:{path.name}" for task, path in CHECKPOINTS.items())


def _llm_model_name() -> str:
    provider = llm_provider()
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if provider == "remote":
        return remote_url()
    return os.environ.get("LLM_MODEL_ID", "")


def _trial_details_from_paths(eeg_paths: List[Path], emg_paths: List[Path]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for idx, (eeg_path, emg_path) in enumerate(zip(eeg_paths, emg_paths), start=1):
        details.append(
            {
                "assessment_type": "active",
                "action_name": f"trial_{idx}",
                "trial_index": idx,
                "eeg_file": eeg_path.name,
                "emg_imu_file": emg_path.name,
                "eeg_name": eeg_path.name,
                "emg_name": emg_path.name,
                "status": "used",
            }
        )
    return details


SESSIONS: Dict[str, SessionState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_old_sessions(force=True)

    db.init_db()
    print(f"[startup] SQLite ready at {db.DB_PATH}")

    # MySQL store for the device-end (task-interface) workflow. Best-effort: a
    # missing MySQL/pymysql only warns so the SQLite「康复评估」flow still serves.
    try:
        mysql_db.init_db()
        print("[startup] MySQL ready (device-end task-interface store)")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup][warn] MySQL not ready: {exc}")

    registry = ModelRegistry()
    print(f"[startup] loading CMK-AGN models onto {registry.device}...")
    registry.load_all()
    print(f"[startup] loaded {len(registry.models)} models: {list(registry.models.keys())}")
    app.state.registry = registry

    # Report generation: either call a remote cloud-GPU LLM service
    # (LLM_REMOTE_URL set — nothing to load locally) or load the QLoRA model
    # in-process. In-process load failures (e.g. no CUDA / missing deps) don't
    # crash startup — DL predictions still serve, and report generation
    # surfaces a clear per-session error.
    app.state.report_model = REPORT_MODEL
    _provider = llm_provider()
    _remote = remote_url()
    if _provider == "deepseek":
        print("[startup] report: DeepSeek API mode (no local LLM load)")
    elif _remote:
        print(f"[startup] report: remote mode → {_remote} (no local LLM load)")
    else:
        try:
            print(f"[startup] loading report LLM from {REPORT_MODEL.adapter_dir}...")
            REPORT_MODEL.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[startup][warn] report LLM not loaded: {exc}")

    yield
    # No teardown needed; torch frees memory on process exit.


app = FastAPI(title="Rehabilitation Assessment Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _admin_settings() -> tuple[str, str, str]:
    return (
        os.environ.get("APP_ADMIN_USER", "rehabdemo").strip(),
        os.environ.get("APP_ADMIN_PASSWORD", ""),
        os.environ.get("APP_AUTH_TOKEN", ""),
    )


def _require_admin(authorization: Optional[str] = Header(None)) -> None:
    _, _, expected_token = _admin_settings()
    if not expected_token:
        raise HTTPException(status_code=503, detail="后端鉴权未配置：缺少 APP_AUTH_TOKEN")

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="请先登录后再执行该操作",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=403,
            detail="登录凭证无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _require_device(
    authorization: Optional[str] = Header(None),
    x_device_token: Optional[str] = Header(None, alias="X-Device-Token"),
) -> None:
    expected_token = os.environ.get("DEVICE_API_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(status_code=503, detail="设备端鉴权未配置：缺少 DEVICE_API_TOKEN")

    token = (x_device_token or "").strip()
    if not token:
        scheme, _, bearer = (authorization or "").partition(" ")
        if scheme.lower() == "bearer":
            token = bearer.strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="缺少设备端 Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=403,
            detail="设备端凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{num}B"


def _cleanup_old_sessions(force: bool = False) -> None:
    global _LAST_SESSION_CLEANUP
    now = time.time()
    if not force and now - _LAST_SESSION_CLEANUP < 3600:
        return
    _LAST_SESSION_CLEANUP = now
    ttl_seconds = SESSION_TTL_HOURS * 3600
    for child in SESSION_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
            if age > ttl_seconds:
                shutil.rmtree(child, ignore_errors=True)
        except OSError as exc:
            print(f"[cleanup][warn] failed to inspect {child}: {exc}")


def _save_uploads(files: List[UploadFile], destdir: Path, prefix: str) -> List[Path]:
    if len(files) > MAX_TRIALS:
        raise HTTPException(status_code=413, detail=f"单次最多上传 {MAX_TRIALS} 组 trial")
    destdir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i, uf in enumerate(files):
        suffix = Path(uf.filename or f"{prefix}_{i}.csv").suffix or ".csv"
        target = destdir / f"{prefix}_{i:02d}{suffix}"
        written = 0
        with target.open("wb") as fh:
            while True:
                chunk = uf.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_FILE_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{uf.filename or target.name} 超过单文件限制 {_human_bytes(MAX_UPLOAD_FILE_BYTES)}",
                    )
                fh.write(chunk)
        out.append(target)
    return out


def _worker(state: SessionState, registry: ModelRegistry, report_model) -> None:
    """Run the full pipeline + report generation on a worker thread."""
    try:
        if state.device_job_id:
            try:
                mysql_db.update_device_job(
                    state.device_job_id,
                    status="running",
                    mark_started=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark running: {exc}")

        predictions_raw = run_pipeline(
            state.eeg_paths, state.emg_paths, registry, state.queue,
            affected_side=state.patient.paralysis_side,
            institution=state.institution,
        )

        predictions = PredictionResult(
            FMA_UE=float(predictions_raw["FMA_UE"]),
            BI=float(predictions_raw["BI"]),
            hand_tone=str(predictions_raw["hand_tone"]),
            hand_function=int(predictions_raw["hand_function"]),
        )
        biomarkers = predictions_raw.get("_biomarkers")

        # Persist to the store this session targets: MySQL for the device-end
        # task-interface workflow, SQLite for the康复评估 page.
        store = mysql_db if state.persist_target == "mysql" else db

        # Previous assessment (for the report's 变化趋势 column) — read BEFORE we
        # insert this one so it reflects the prior visit, not the current.
        try:
            history = store.latest_assessment_for_patient(state.patient.patient_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[history][warn] {exc}")
            history = None

        try:
            report_text = stream_report(
                state.patient,
                predictions,
                state.queue,
                biomarkers=biomarkers,
                history=history,
                report_model=report_model,
            )
        except Exception:
            # Report generation failed — predictions are still kept and the SSE
            # error event was already emitted by stream_report.
            report_text = None

        state.result = AssessmentResult(
            session_id=state.session_id,
            patient_info=state.patient,
            predictions=predictions,
            report=report_text,
        )

        # Persist. A failed/empty report still saves the record with
        # report_status='failed' so it shows up in the records list. DB errors
        # are isolated so they never break the SSE `done` event.
        try:
            report_status = "generated" if report_text else "failed"
            bio_json = json.dumps(biomarkers, ensure_ascii=False) if biomarkers else None
            prediction_json = json.dumps(
                {
                    "FMA_UE": predictions.FMA_UE,
                    "BI": predictions.BI,
                    "hand_tone": predictions.hand_tone,
                    "hand_function": predictions.hand_function,
                },
                ensure_ascii=False,
            )
            if store is mysql_db:
                # Device assessment → MySQL. upsert auto-creates the patient
                # (source='device-auto') when not yet enrolled by the hospital.
                pid = mysql_db.upsert_patient(state.patient, source=f"{state.institution}-auto")
                assessment_db_id = mysql_db.insert_assessment(
                    pid,
                    state.session_id,
                    predictions,
                    report_text,
                    report_status,
                    source=state.institution,
                    package_name=state.package_name,
                    assessment_id=state.assessment_id,
                    assessment_time=state.assessment_time,
                    biomarkers=bio_json,
                    institution=state.institution,
                    n_trials=state.n_trials,
                    package_hash=state.package_hash,
                    parse_warnings=json.dumps(state.parse_warnings, ensure_ascii=False),
                    prediction_json=prediction_json,
                    model_version=_dl_model_version(),
                    llm_provider=llm_provider(),
                    llm_model=_llm_model_name(),
                )
                state.assessment_db_id = int(assessment_db_id)
                mysql_db.replace_assessment_trials(assessment_db_id, state.trial_details)
                mysql_db.replace_assessment_biomarkers(assessment_db_id, biomarkers)
            else:
                pid = db.upsert_patient(state.patient)
                db.insert_assessment(
                    pid,
                    state.session_id,
                    predictions,
                    report_text,
                    report_status,
                    biomarkers=bio_json,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[persist][warn] failed to save assessment {state.session_id}: {exc}")

        if state.device_job_id:
            try:
                if state.assessment_db_id is not None:
                    mysql_db.update_device_job(
                        state.device_job_id,
                        status="completed",
                        assessment_db_id=state.assessment_db_id,
                        mark_completed=True,
                    )
                else:
                    mysql_db.update_device_job(
                        state.device_job_id,
                        status="failed",
                        error_message="评估已结束，但未写入 MySQL 评估记录，无法生成设备端导出文件。",
                        mark_completed=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark completed: {exc}")

        state.queue.put({"type": "done"})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if state.device_job_id:
            try:
                mysql_db.update_device_job(
                    state.device_job_id,
                    status="failed",
                    error_message=str(exc),
                    mark_completed=True,
                )
            except Exception as job_exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark failed: {job_exc}")
        state.queue.put(error_event(f"会话 {state.session_id} 失败：{exc}"))
    finally:
        state.queue.put(SENTINEL)


def _start_session_worker(state: SessionState) -> None:
    registry: ModelRegistry = app.state.registry
    report_model = app.state.report_model
    with state.lock:
        if not state.started:
            state.started = True
            threading.Thread(
                target=_worker, args=(state, registry, report_model), daemon=True
            ).start()


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.post("/api/assess", response_model=AssessSessionResponse)
async def create_assessment(
    patient_id: str = Form(...),
    name: str = Form(...),
    sex: str = Form(...),
    age: Optional[int] = Form(None),
    diagnosis: str = Form(...),
    disease_days: Optional[int] = Form(None),
    paralysis_side: str = Form(...),
    eeg_files: List[UploadFile] = File(...),
    emg_files: List[UploadFile] = File(...),
    _admin: None = Depends(_require_admin),
):
    _cleanup_old_sessions()
    if len(eeg_files) == 0 or len(emg_files) == 0:
        raise HTTPException(status_code=422, detail="必须至少上传一对 EEG / EMG 文件")
    if len(eeg_files) != len(emg_files):
        raise HTTPException(
            status_code=422,
            detail=f"EEG 与 EMG 文件数量不匹配：{len(eeg_files)} vs {len(emg_files)}",
        )

    try:
        patient = PatientInfo(
            patient_id=patient_id,
            name=name,
            sex=sex,  # type: ignore[arg-type]
            age=age,
            diagnosis=diagnosis,
            disease_days=disease_days,
            paralysis_side=paralysis_side,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex[:12]
    destdir = SESSION_ROOT / session_id
    eeg_paths = _save_uploads(eeg_files, destdir / "eeg", "eeg")
    emg_paths = _save_uploads(emg_files, destdir / "emg", "emg")

    SESSIONS[session_id] = SessionState(
        session_id,
        patient,
        eeg_paths,
        emg_paths,
        persist_target="mysql",
        institution="hospital",
    )
    return AssessSessionResponse(session_id=session_id, n_trials=len(eeg_paths))


# --------------------------------------------------------------------------- #
# 任务一与任务三对接接口：离线 zip 数据包导入 + 在线设备端占位                  #
# --------------------------------------------------------------------------- #
def _save_and_extract_zip(upload: UploadFile, institution: str) -> "tuple[Path, Any, str]":
    """Persist the uploaded zip, extract it (zip-slip safe), and read the bundle.

    Returns ``(bundle_root, EvalPackage, package_sha256)``. Raises HTTPException(422) on bad input.
    """
    if institution not in INSTITUTIONS:
        raise HTTPException(status_code=422, detail=f"未知机构类型：{institution}")
    if not (upload.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="请上传 .zip 压缩包")

    _cleanup_old_sessions()
    work = SESSION_ROOT / f"pkg_{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / "bundle.zip"
    digest = hashlib.sha256()
    written = 0
    with zip_path.open("wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_ZIP_BYTES:
                fh.close()
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"压缩包超过限制 {_human_bytes(MAX_ZIP_BYTES)}",
                )
            digest.update(chunk)
            fh.write(chunk)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="压缩包不是有效的 zip 文件") from exc

    total_uncompressed = sum(info.file_size for info in members)
    if len(members) > MAX_ZIP_MEMBERS:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"压缩包内文件数超过限制 {MAX_ZIP_MEMBERS}")
    if total_uncompressed > MAX_ZIP_EXTRACTED_BYTES:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(
            status_code=413,
            detail=f"压缩包解压后超过限制 {_human_bytes(MAX_ZIP_EXTRACTED_BYTES)}",
        )
    try:
        root = safe_extract_zip(zip_path, work / "extracted")
        pkg = read_eval_package(root, institution=institution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return root, pkg, digest.hexdigest()


@app.post("/api/task-interface/parse")
async def task_interface_parse(
    institution: str = Form(...),
    package: UploadFile = File(...),
    _admin: None = Depends(_require_admin),
):
    """Parse-only preview: unzip + read manifest, return trial count + patient
    prefill + warnings so the UI can pre-fill the form before running.

    If the manifest's patient_id is already enrolled in MySQL, the stored basic
    info overrides the manifest prefill and ``enrolled`` is set so the UI shows
    "该患者已入组，已按档案回填"."""
    _root, pkg, package_hash = _save_and_extract_zip(package, institution)
    prefill = dict(pkg.patient_prefill)
    enrolled = False
    pid = (prefill.get("patient_id") or "").strip()
    if pid:
        try:
            record = mysql_db.get_patient_by_business_id(pid)
        except Exception as exc:  # noqa: BLE001
            print(f"[mysql][warn] enrollment lookup failed: {exc}")
            record = None
        if record:
            enrolled = True
            for key in ("name", "sex", "age", "diagnosis", "paralysis_side", "disease_days"):
                if record.get(key) is not None:
                    prefill[key] = record[key]
    return {
        "institution": pkg.institution,
        "n_trials": pkg.n_trials,
        "patient_prefill": prefill,
        "manifest_summary": pkg.manifest_summary,
        "warnings": pkg.warnings,
        "package_hash": package_hash,
        "enrolled": enrolled,
    }


@app.post("/api/task-interface/offline", response_model=AssessSessionResponse)
async def task_interface_offline(
    institution: str = Form(...),
    package: UploadFile = File(...),
    patient_id: str = Form(...),
    name: str = Form(...),
    sex: str = Form(...),
    age: Optional[int] = Form(None),
    diagnosis: str = Form(...),
    disease_days: Optional[int] = Form(None),
    paralysis_side: str = Form(...),
    _admin: None = Depends(_require_admin),
):
    """Offline mode: ingest a zip bundle and start a session reusing the existing
    inference/report/SSE machinery. The client then streams progress from the
    shared ``/api/assess/{session_id}/stream`` endpoint.

    Results from this device-end workflow persist to **MySQL** (not the SQLite
    used by the 康复评估 page). The manifest's assessment_id / time and the zip
    filename are carried into the record for traceability."""
    _root, pkg, package_hash = _save_and_extract_zip(package, institution)
    if pkg.n_trials == 0:
        detail = "数据包中没有可用的 trial。" + ("；".join(pkg.warnings) if pkg.warnings else "")
        raise HTTPException(status_code=422, detail=detail)

    try:
        patient = PatientInfo(
            patient_id=patient_id,
            name=name,
            sex=sex,  # type: ignore[arg-type]
            age=age,
            diagnosis=diagnosis,
            disease_days=disease_days,
            paralysis_side=paralysis_side,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = SessionState(
        session_id, patient, pkg.eeg_paths, pkg.emg_paths,
        institution=pkg.institution,
        persist_target="mysql",
        package_name=package.filename,
        assessment_id=pkg.manifest_summary.get("assessment_id"),
        assessment_time=pkg.manifest_summary.get("assessment_time"),
        n_trials=pkg.n_trials,
        package_hash=package_hash,
        parse_warnings=pkg.warnings,
        trial_details=pkg.trial_details,
    )
    return AssessSessionResponse(session_id=session_id, n_trials=pkg.n_trials)


@app.get("/api/task-interface/online/status")
async def task_interface_online_status(_admin: None = Depends(_require_admin)):
    """Online mode placeholder: the wearable device real-time acquisition
    interface is not wired yet. Exposes a configurable device URL (env
    DEVICE_STREAM_URL) and a 'pending integration' status for the UI."""
    device_url = os.environ.get("DEVICE_STREAM_URL", "").strip()
    return {
        "status": "pending",
        "device_url": device_url,
        "message": "设备端实时采集接口待对接，当前仅支持离线数据包导入。",
    }


# --------------------------------------------------------------------------- #
# Device-to-cloud HTTPS API                                                   #
# --------------------------------------------------------------------------- #
def _nonblank(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _valid_sex(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in {"男", "女"} else "男"


def _valid_side(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in {"左", "右"} else "左"


def _device_job_files(job_id: str) -> Dict[str, str]:
    base = f"/api/device/v1/jobs/{job_id}"
    return {
        "json": f"{base}/result.json",
        "pdf": f"{base}/report.pdf",
        "zip": f"{base}/export.zip",
    }


def _format_device_job(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "job_id": job.get("job_id"),
        "device_id": job.get("device_id"),
        "session_id": job.get("session_id"),
        "assessment_db_id": job.get("assessment_db_id"),
        "assessment_id": job.get("assessment_id"),
        "patient_id": job.get("patient_id"),
        "package_name": job.get("package_name"),
        "package_hash": job.get("package_hash"),
        "status": job.get("status"),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "delivered_at": job.get("delivered_at"),
        "updated_at": job.get("updated_at"),
    }
    if job.get("status") in {"completed", "delivered"} and job.get("assessment_db_id"):
        payload["files"] = _device_job_files(str(job["job_id"]))
    return payload


def _device_job_or_404(job_id: str) -> Dict[str, Any]:
    try:
        job = mysql_db.get_device_job(job_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="设备任务不存在")
    return job


def _completed_device_assessment_id(job: Dict[str, Any]) -> int:
    if job.get("status") not in {"completed", "delivered"}:
        raise HTTPException(status_code=409, detail=f"任务尚未完成，当前状态：{job.get('status')}")
    assessment_db_id = job.get("assessment_db_id")
    if not assessment_db_id:
        raise HTTPException(status_code=409, detail="任务已结束但未生成评估记录")
    return int(assessment_db_id)


@app.post("/api/device/v1/assessments")
async def device_create_assessment(
    package: UploadFile = File(...),
    institution: str = Form("device"),
    device_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    diagnosis: Optional[str] = Form(None),
    disease_days: Optional[int] = Form(None),
    paralysis_side: Optional[str] = Form(None),
    _device: None = Depends(_require_device),
):
    """Device-side machine API: upload one active-assessment zip and start a
    background analysis job. The device polls ``/jobs/{job_id}`` and downloads
    the generated export files when the job reaches ``completed``."""
    institution = (institution or "device").strip().lower()
    _root, pkg, package_hash = _save_and_extract_zip(package, institution)
    if pkg.n_trials == 0:
        detail = "数据包中没有可用的 active trial。" + ("；".join(pkg.warnings) if pkg.warnings else "")
        raise HTTPException(status_code=422, detail=detail)

    prefill = dict(pkg.patient_prefill)
    business_pid = _nonblank(patient_id, prefill.get("patient_id"))
    if not business_pid:
        raise HTTPException(status_code=422, detail="缺少 patient_id，表单或 manifest.json 至少提供一个")

    enrolled: Dict[str, Any] = {}
    try:
        enrolled = mysql_db.get_patient_by_business_id(business_pid) or {}
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc

    try:
        patient = PatientInfo(
            patient_id=business_pid,
            name=_nonblank(name, enrolled.get("name"), prefill.get("name"), business_pid),
            sex=_valid_sex(_nonblank(sex, enrolled.get("sex"), prefill.get("sex"))),
            age=age if age is not None else enrolled.get("age") or prefill.get("age"),
            diagnosis=_nonblank(diagnosis, enrolled.get("diagnosis"), prefill.get("diagnosis"), "未填写"),
            disease_days=(
                disease_days
                if disease_days is not None
                else enrolled.get("disease_days") or prefill.get("disease_days")
            ),
            paralysis_side=_valid_side(_nonblank(paralysis_side, enrolled.get("paralysis_side"), prefill.get("paralysis_side"))),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex[:12]
    job_id = f"devjob_{uuid.uuid4().hex[:16]}"
    state = SessionState(
        session_id, patient, pkg.eeg_paths, pkg.emg_paths,
        institution=pkg.institution,
        persist_target="mysql",
        package_name=package.filename,
        assessment_id=pkg.manifest_summary.get("assessment_id"),
        assessment_time=pkg.manifest_summary.get("assessment_time"),
        n_trials=pkg.n_trials,
        package_hash=package_hash,
        parse_warnings=pkg.warnings,
        trial_details=pkg.trial_details,
        device_job_id=job_id,
    )
    SESSIONS[session_id] = state

    try:
        job = mysql_db.create_device_job(
            job_id=job_id,
            device_id=(device_id or pkg.manifest_summary.get("device_id") or "").strip() or None,
            session_id=session_id,
            assessment_id=pkg.manifest_summary.get("assessment_id"),
            patient_id=patient.patient_id,
            package_name=package.filename,
            package_hash=package_hash,
        )
    except mysql_db.MySQLUnavailable as exc:
        SESSIONS.pop(session_id, None)
        raise _mysql_guard(exc) from exc

    _start_session_worker(state)
    response = _format_device_job(job)
    response["status_url"] = f"/api/device/v1/jobs/{job_id}"
    response["n_trials"] = pkg.n_trials
    response["parse_warnings"] = pkg.warnings
    return response


@app.get("/api/device/v1/jobs/{job_id}")
async def device_get_job(job_id: str, _device: None = Depends(_require_device)):
    return _format_device_job(_device_job_or_404(job_id))


@app.get("/api/device/v1/jobs/{job_id}/result.json")
async def device_download_result_json(job_id: str, _device: None = Depends(_require_device)):
    job = _device_job_or_404(job_id)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.result_json,
        media_type="application/json",
        filename=export_filename(assessment, "json"),
    )


@app.get("/api/device/v1/jobs/{job_id}/report.pdf")
async def device_download_report_pdf(job_id: str, _device: None = Depends(_require_device)):
    job = _device_job_or_404(job_id)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.report_pdf,
        media_type="application/pdf",
        filename=export_filename(assessment, "pdf"),
    )


@app.get("/api/device/v1/jobs/{job_id}/export.zip")
async def device_download_export_zip(job_id: str, _device: None = Depends(_require_device)):
    job = _device_job_or_404(job_id)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.export_zip,
        media_type="application/zip",
        filename=export_filename(assessment, "zip"),
    )


@app.post("/api/device/v1/jobs/{job_id}/ack")
async def device_ack_job(job_id: str, _device: None = Depends(_require_device)):
    _completed_device_assessment_id(_device_job_or_404(job_id))
    try:
        job = mysql_db.update_device_job(job_id, status="delivered", mark_delivered=True)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="设备任务不存在")
    return _format_device_job(job)


@app.get("/api/assess/{session_id}/stream")
async def stream_assessment(session_id: str):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在或已过期")

    # Kick off worker only once per session.
    _start_session_worker(state)

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, state.queue.get)
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                break
            if item is SENTINEL or item.get("__sentinel__"):
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/assess/{session_id}/result", response_model=AssessmentResult)
async def get_result(session_id: str):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在")
    if state.result is None:
        raise HTTPException(status_code=425, detail="评估尚未完成")
    return state.result


@app.get("/api/assess/{session_id}/report.docx")
async def get_report_docx(session_id: str):
    """Render the session's Markdown report into a downloadable .docx."""
    state = SESSIONS.get(session_id)
    if state is None or state.result is None:
        raise HTTPException(status_code=404, detail="报告不存在或评估尚未完成")
    md = state.result.report
    if not md:
        raise HTTPException(status_code=404, detail="该会话没有可导出的报告")
    from report_docx import markdown_to_docx_bytes

    data = markdown_to_docx_bytes(md)
    pid = state.patient.patient_id or "patient"
    filename = f"rehab_report_{pid}_{session_id[:6]}.docx"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": list(app.state.registry.models.keys())}


@app.post("/api/auth/login", response_model=AuthLoginResponse)
async def auth_login(payload: AuthLoginRequest):
    expected_user, expected_password, token = _admin_settings()
    if not expected_password or not token:
        raise HTTPException(status_code=503, detail="后端鉴权未配置")
    if not (
        secrets.compare_digest(payload.username, expected_user)
        and secrets.compare_digest(payload.password, expected_password)
    ):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return AuthLoginResponse(access_token=token, user=expected_user)


# --------------------------------------------------------------------------- #
# Patient management / records / stats (MySQL-backed business store)           #
# --------------------------------------------------------------------------- #
@app.get("/api/patients", response_model=List[PatientSummary])
async def list_patients(_admin: None = Depends(_require_admin)):
    try:
        return mysql_db.list_patients()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.get("/api/patients/{patient_db_id}", response_model=PatientDetail)
async def get_patient(patient_db_id: int, _admin: None = Depends(_require_admin)):
    try:
        patient = mysql_db.get_patient(patient_db_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if patient is None:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@app.patch("/api/patients/{patient_db_id}", response_model=PatientDetail)
async def update_patient(
    patient_db_id: int,
    payload: PatientUpdate,
    _admin: None = Depends(_require_admin),
):
    fields = payload.model_dump(exclude_unset=True)
    try:
        patient = mysql_db.update_patient(patient_db_id, fields)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if patient is None:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@app.get("/api/assessments", response_model=AssessmentOverview)
async def list_assessments(
    limit: int = 50,
    offset: int = 0,
    _admin: None = Depends(_require_admin),
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    try:
        return mysql_db.list_assessments(limit=limit, offset=offset)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.get("/api/stats/summary", response_model=StatsSummary)
async def stats_summary(_admin: None = Depends(_require_admin)):
    try:
        return mysql_db.stats_summary()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


# --------------------------------------------------------------------------- #
# Device-end (task-interface) MySQL store: enrollment + records + cleanup      #
# --------------------------------------------------------------------------- #
def _mysql_guard(exc: Exception) -> HTTPException:
    """Map a MySQL outage to a clear 503 for the UI."""
    return HTTPException(status_code=503, detail=f"MySQL 不可用：{exc}")


def _mysql_assessment_or_404(assessment_id: int) -> Dict[str, Any]:
    try:
        assessment = mysql_db.get_assessment(assessment_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


def _assessment_export_bundle(assessment_id: int, force: bool = False):
    assessment = _mysql_assessment_or_404(assessment_id)
    try:
        return assessment, ensure_assessment_export(assessment, force=force)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成导出文件失败：{exc}") from exc


@app.post("/api/mysql/enroll")
async def mysql_enroll(payload: EnrollmentRequest, _admin: None = Depends(_require_admin)):
    """医院入组：写入患者基本信息 + 可选的第一次评估记录（手工分数）。"""
    basic = {
        "patient_id": payload.patient_id,
        "name": payload.name,
        "sex": payload.sex,
        "age": payload.age,
        "diagnosis": payload.diagnosis,
        "paralysis_side": payload.paralysis_side,
        "disease_days": payload.disease_days,
    }
    first = None
    if None not in (payload.fma_ue, payload.bi, payload.hand_tone, payload.hand_function):
        first = {
            "FMA_UE": payload.fma_ue,
            "BI": payload.bi,
            "hand_tone": payload.hand_tone,
            "hand_function": payload.hand_function,
            "assessment_time": payload.assessment_time,
            "report": payload.report,
        }
    try:
        return mysql_db.enroll_patient(basic, first)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.get("/api/mysql/patients")
async def mysql_list_patients(_admin: None = Depends(_require_admin)):
    try:
        return mysql_db.list_patients()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.get("/api/mysql/patients/{patient_db_id}")
async def mysql_get_patient(patient_db_id: int, _admin: None = Depends(_require_admin)):
    try:
        patient = mysql_db.get_patient(patient_db_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/api/mysql/assessments", response_model=MysqlAssessmentList)
async def mysql_list_assessments(
    limit: int = 50,
    offset: int = 0,
    _admin: None = Depends(_require_admin),
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    try:
        return mysql_db.list_assessments(limit=limit, offset=offset)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.get("/api/mysql/assessments/{assessment_id}", response_model=MysqlAssessmentDetail)
async def mysql_get_assessment(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    return _mysql_assessment_or_404(assessment_id)


@app.post("/api/mysql/assessments/{assessment_id}/exports/regenerate")
async def mysql_regenerate_assessment_export(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    _assessment, bundle = _assessment_export_bundle(assessment_id, force=True)
    return {
        "manifest": bundle.manifest,
        "zip": file_info("export.zip", bundle.export_zip),
    }


@app.get("/api/mysql/assessments/{assessment_id}/export.json")
async def mysql_download_assessment_json(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.result_json,
        media_type="application/json",
        filename=export_filename(assessment, "json"),
    )


@app.get("/api/mysql/assessments/{assessment_id}/report.pdf")
async def mysql_download_assessment_pdf(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.report_pdf,
        media_type="application/pdf",
        filename=export_filename(assessment, "pdf"),
    )


@app.get("/api/mysql/assessments/{assessment_id}/export.zip")
async def mysql_download_assessment_zip(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.export_zip,
        media_type="application/zip",
        filename=export_filename(assessment, "zip"),
    )


@app.delete("/api/mysql/assessments/{assessment_id}")
async def mysql_delete_assessment(
    assessment_id: int,
    _admin: None = Depends(_require_admin),
):
    try:
        deleted = mysql_db.delete_assessment(assessment_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if deleted == 0:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"deleted": deleted}


@app.delete("/api/mysql/assessments")
async def mysql_clear_assessments(_admin: None = Depends(_require_admin)):
    """清空全部设备评估记录（测试期清理）。"""
    try:
        deleted = mysql_db.delete_all_assessments()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    return {"deleted": deleted}


@app.delete("/api/mysql/patients/{patient_db_id}")
async def mysql_delete_patient(
    patient_db_id: int,
    _admin: None = Depends(_require_admin),
):
    """删除患者及其全部评估记录（级联，测试期清理）。"""
    try:
        deleted = mysql_db.delete_patient(patient_db_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if deleted == 0:
        raise HTTPException(status_code=404, detail="患者不存在")
    return {"deleted": deleted}


# --------------------------------------------------------------------------- #
# CLI entry: `python -m backend.main` or `uvicorn backend.main:app --reload`.  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
