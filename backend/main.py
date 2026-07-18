"""FastAPI entrypoint for the rehabilitation assessment platform.

Three endpoints:
  POST /api/assess                       — accept multipart files + patient info, return session_id
  GET  /api/assess/{session_id}/stream   — SSE stream of progress events
  GET  /api/assess/{session_id}/result   — cached final result (reconnect fallback)

The full inference pipeline runs in a worker thread; events are appended to a
per-session replayable stream consumed by the SSE endpoint.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import knowledge_admin
import llm_settings
import mysql_db
from admin_auth import browser_origin_allowed, issue_session_token, verify_session_token
from assessment_queue import AssessmentQueue
from assessment_export import (
    delete_assessment_export,
    ensure_assessment_export,
    export_filename,
    file_info,
)
from device_auth import (
    DeviceCredential,
    DeviceTokenConfigError,
    authenticate_device_token,
    credential_count,
    generate_device_token,
    parse_named_tokens,
    token_digest,
    token_hint,
)
from device_patient_policy import DevicePatientPolicyError, resolve_device_patient
from eval_package import INSTITUTIONS, locate_manifest_root, read_eval_package, safe_extract_zip
from inference import (
    CHECKPOINTS,
    SENTINEL,
    AssessmentCancelled,
    ModelRegistry,
    error_event,
    run_pipeline,
)
from report import REPORT_MODEL, llm_model_name, llm_provider, remote_url, stream_report
from schemas import (
    AssessmentOverview,
    AssessmentResult,
    AssessSessionResponse,
    AuthLoginRequest,
    AuthLoginResponse,
    DeviceCredentialCreate,
    DeviceCredentialUpdate,
    DevicePatientRegistrationRequest,
    DevicePatientRegistrationResponse,
    EnrollmentRequest,
    LlmModelSettingsUpdate,
    LlmSettingsUpdate,
    MysqlAssessmentDetail,
    MysqlAssessmentList,
    PatientDetail,
    PatientInfo,
    PatientSummary,
    PatientUpdate,
    PredictionResult,
    StatsSummary,
)
from session_events import SessionEventStream

load_dotenv(Path(__file__).resolve().parent / ".env")

APP_VERSION = os.environ.get("APP_VERSION", "unreleased").strip() or "unreleased"
APP_BUILD_COMMIT = os.environ.get("APP_BUILD_COMMIT", "unknown").strip() or "unknown"

SESSION_ROOT = Path(tempfile.gettempdir()) / "rehab_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE_JOB_ROOT = Path(
    os.environ.get(
        "DEVICE_JOB_ROOT",
        str(Path(__file__).resolve().parents[2] / "device_jobs"),
    )
)
DEVICE_JOB_ROOT.mkdir(parents=True, exist_ok=True)

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ADMIN_SESSION_COOKIE = os.environ.get("APP_SESSION_COOKIE_NAME", "rehab_admin_session").strip()
ADMIN_SESSION_TTL_SECONDS = _env_int("APP_SESSION_TTL_SECONDS", 8 * 60 * 60)
ALLOW_LEGACY_ADMIN_BEARER = _env_flag("ALLOW_LEGACY_ADMIN_BEARER")
ALLOW_LEGACY_DEVICE_TOKEN = _env_flag("ALLOW_LEGACY_DEVICE_TOKEN")
DEVICE_REQUIRE_REGISTERED_PATIENT = _env_flag("DEVICE_REQUIRE_REGISTERED_PATIENT")
LOGIN_RATE_LIMIT = _env_int("APP_LOGIN_RATE_LIMIT", 5)
LOGIN_RATE_WINDOW_SECONDS = _env_int("APP_LOGIN_RATE_WINDOW_SECONDS", 5 * 60)
_LOGIN_FAILURES: Dict[str, List[float]] = {}
_LOGIN_FAILURES_LOCK = threading.Lock()


MAX_UPLOAD_FILE_BYTES = _env_int("MAX_UPLOAD_FILE_BYTES", 512 * 1024 * 1024)
MAX_SESSION_UPLOAD_BYTES = _env_int("MAX_SESSION_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)
MAX_TRIALS = _env_int("MAX_TRIALS", 30)
MAX_ZIP_BYTES = _env_int("MAX_ZIP_BYTES", 1024 * 1024 * 1024)
MAX_ZIP_EXTRACTED_BYTES = _env_int("MAX_ZIP_EXTRACTED_BYTES", 4 * 1024 * 1024 * 1024)
MAX_ZIP_MEMBERS = _env_int("MAX_ZIP_MEMBERS", 500)
MAX_ZIP_COMPRESSION_RATIO = _env_int("MAX_ZIP_COMPRESSION_RATIO", 200)
MIN_FREE_DISK_BYTES = _env_int("MIN_FREE_DISK_BYTES", 2 * 1024 * 1024 * 1024)
SESSION_TTL_HOURS = _env_int("SESSION_TTL_HOURS", 168)
DEVICE_INPUT_TTL_HOURS = _env_int("DEVICE_INPUT_TTL_HOURS", 168)
_LAST_SESSION_CLEANUP = 0.0
_LAST_DEVICE_INPUT_CLEANUP = 0.0


# --------------------------------------------------------------------------- #
# In-process session registry. Keys: session_id → SessionState.               #
# --------------------------------------------------------------------------- #
class SessionState:
    def __init__(self, session_id: str, patient: PatientInfo, eeg_paths: List[Path],
                 emg_paths: List[Path], institution: str = "hospital",
                 persist_target: str = "mysql", package_name: Optional[str] = None,
                 assessment_id: Optional[str] = None, assessment_time: Optional[str] = None,
                 n_trials: Optional[int] = None, package_hash: Optional[str] = None,
                 parse_warnings: Optional[List[str]] = None,
                 trial_details: Optional[List[Dict[str, Any]]] = None,
                 device_job_id: Optional[str] = None,
                 temporary_work_dir: Optional[Path] = None):
        self.session_id = session_id
        self.patient = patient
        self.eeg_paths = eeg_paths
        self.emg_paths = emg_paths
        self.institution = institution
        # MySQL is the single business store for all assessment flows.
        self.persist_target = persist_target
        self.package_name = package_name
        self.assessment_id = assessment_id
        self.assessment_time = assessment_time
        self.n_trials = n_trials if n_trials is not None else len(eeg_paths)
        self.package_hash = package_hash
        self.parse_warnings = parse_warnings or []
        self.trial_details = trial_details or _trial_details_from_paths(eeg_paths, emg_paths)
        self.device_job_id = device_job_id
        self.temporary_work_dir = temporary_work_dir
        self.assessment_db_id: Optional[int] = None
        self.report_provider: Optional[str] = None
        self.report_model_id: Optional[str] = None
        self.queue = SessionEventStream()
        self.result: Optional[AssessmentResult] = None
        self.started: bool = False
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.created_monotonic = time.monotonic()
        self.finished_monotonic: Optional[float] = None


def _dl_model_version() -> str:
    versions: List[str] = [f"app:{APP_VERSION}@{APP_BUILD_COMMIT}"]
    for task, path in CHECKPOINTS.items():
        if not path.is_file():
            versions.append(f"{task}:{path.name}@missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        versions.append(f"{task}:{path.name}@{digest.hexdigest()[:12]}")
    return ";".join(versions)


def _llm_model_name() -> str:
    return llm_model_name()


def _trial_details_from_paths(eeg_paths: List[Path], emg_paths: List[Path]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for idx, (eeg_path, emg_path) in enumerate(zip(eeg_paths, emg_paths), start=1):
        details.append(
            {
                "assessment_type": "active",
                "action_name": f"trial_{idx}",
                "trial_index": idx,
                "model_task_index": idx - 1,
                "model_trial_index": 0,
                "eeg_file": eeg_path.name,
                "emg_imu_file": emg_path.name,
                "eeg_name": eeg_path.name,
                "emg_name": emg_path.name,
                "status": "used",
            }
        )
    return details


SESSIONS: Dict[str, SessionState] = {}
SESSION_SCHEDULER = AssessmentQueue()


def _migrate_env_device_credentials() -> None:
    """Seed plaintext environment credentials into the hashed MySQL store once."""
    legacy_token = os.environ.get("DEVICE_API_TOKEN", "").strip()
    named_raw = os.environ.get("DEVICE_API_TOKENS_JSON", "")
    named = parse_named_tokens(named_raw)
    if legacy_token and ALLOW_LEGACY_DEVICE_TOKEN:
        mysql_db.ensure_device_credential(
            device_id="legacy_shared",
            label="旧共享设备码",
            access_scope="shared",
            token_hash=token_digest(legacy_token),
            token_hint=token_hint(legacy_token),
        )
    for device_id, token in named.items():
        mysql_db.ensure_device_credential(
            device_id=device_id,
            label=device_id,
            access_scope="device",
            token_hash=token_digest(token),
            token_hint=token_hint(token),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_old_sessions(force=True)
    app.state.mysql_ready = False
    app.state.dl_ready = False
    app.state.report_ready = False

    # MySQL is required for patient, assessment, export, and device-job records.
    # Startup warns instead of crashing so /api/health can still explain service
    # state, but business APIs will return a clear 503 if MySQL is unavailable.
    try:
        mysql_db.init_db()
        app.state.mysql_ready = True
        print("[startup] MySQL ready (business store)")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup][warn] MySQL not ready: {exc}")
    else:
        try:
            _migrate_env_device_credentials()
        except Exception as exc:  # noqa: BLE001
            print(f"[startup][warn] device credential migration failed: {exc}")
        _cleanup_old_device_inputs(force=True)

    registry = ModelRegistry()
    print(f"[startup] loading CMK-AGN models onto {registry.device}...")
    registry.load_all()
    app.state.dl_ready = len(registry.models) == len(CHECKPOINTS)
    app.state.dl_model_version = _dl_model_version()
    print(f"[startup] loaded {len(registry.models)} models: {list(registry.models.keys())}")
    app.state.registry = registry

    # Report generation follows the saved System Management selection when it
    # exists; .env remains a fallback for older deployments. In-process load
    # failures (e.g. no CUDA / missing deps) don't crash startup — DL predictions
    # still serve, and report generation surfaces a clear per-session error.
    app.state.report_model = REPORT_MODEL
    _provider = llm_provider()
    if _provider == "deepseek":
        app.state.report_ready = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
        print("[startup] report: DeepSeek API mode (no local LLM load)")
    elif _provider == "remote":
        _remote = remote_url()
        print(f"[startup] report: remote mode → {_remote} (no local LLM load)")
        app.state.report_ready = bool(_remote)
    else:
        try:
            print(f"[startup] report: local mode selected, loading active report LLM...")
            REPORT_MODEL.load()
            app.state.report_ready = REPORT_MODEL.loaded
        except Exception as exc:  # noqa: BLE001
            print(f"[startup][warn] report LLM not loaded: {exc}")

    SESSION_SCHEDULER.start(_run_scheduled_state)
    _restore_device_jobs()

    yield
    SESSION_SCHEDULER.stop()
    # Torch frees model memory when the process exits.


app = FastAPI(title="Rehabilitation Assessment Platform", lifespan=lifespan)

_cors_origins = [
    value.strip()
    for value in os.environ.get("APP_CORS_ORIGINS", "").split(",")
    if value.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Device-Token"],
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


def _authoritative_patient(candidate: PatientInfo) -> PatientInfo:
    """Use an enrolled hospital profile as the patient master for assessments."""
    try:
        enrolled = mysql_db.get_patient_by_business_id(candidate.patient_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if not enrolled:
        return candidate
    values = candidate.model_dump()
    for key in ("name", "sex", "age", "diagnosis", "disease_days", "paralysis_side"):
        value = enrolled.get(key)
        if value not in (None, ""):
            values[key] = value
    return PatientInfo(**values)


def _browser_write_origin_allowed(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    forwarded_host = request.headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.headers.get("host", "")
    return browser_origin_allowed(
        request.headers.get("origin", ""),
        request.headers.get("referer", ""),
        f"{scheme}://{host}",
        _cors_origins,
    )


def _require_admin(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> None:
    expected_user, _, expected_token = _admin_settings()
    if not expected_token:
        raise HTTPException(status_code=503, detail="后端鉴权未配置：缺少 APP_AUTH_TOKEN")

    scheme, _, bearer = (authorization or "").partition(" ")
    bearer = bearer.strip() if scheme.lower() == "bearer" else ""
    cookie = request.cookies.get(ADMIN_SESSION_COOKIE, "").strip()
    if not bearer and not cookie:
        raise HTTPException(
            status_code=401,
            detail="请先登录后再执行该操作",
            headers={"WWW-Authenticate": "Bearer"},
        )
    valid_bearer = bool(
        bearer
        and (
            (ALLOW_LEGACY_ADMIN_BEARER and secrets.compare_digest(bearer, expected_token))
            or verify_session_token(bearer, expected_user, expected_token)
        )
    )
    valid_cookie = bool(
        cookie and verify_session_token(cookie, expected_user, expected_token)
    )
    if not valid_bearer and not valid_cookie:
        raise HTTPException(
            status_code=403,
            detail="登录凭证无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if (
        valid_cookie
        and not valid_bearer
        and request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
    ):
        if not _browser_write_origin_allowed(request):
            raise HTTPException(status_code=403, detail="请求来源校验失败，请刷新页面后重试")


def _require_device(
    authorization: Optional[str] = Header(None),
    x_device_token: Optional[str] = Header(None, alias="X-Device-Token"),
) -> DeviceCredential:
    legacy_token = os.environ.get("DEVICE_API_TOKEN", "")
    named_tokens_json = os.environ.get("DEVICE_API_TOKENS_JSON", "")
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

    try:
        stored, stored_count = mysql_db.authenticate_device_credential(token)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if stored_count > 0:
        if stored is None:
            raise HTTPException(
                status_code=403,
                detail="设备端凭证无效或已停用",
                headers={"WWW-Authenticate": "Bearer"},
            )
        shared = stored.get("access_scope") == "shared"
        if shared and not ALLOW_LEGACY_DEVICE_TOKEN:
            raise HTTPException(
                status_code=403,
                detail="旧共享设备码已停用，请在系统管理中为设备生成独立设备码",
            )
        return DeviceCredential(
            device_id=None if shared else str(stored["device_id"]),
            legacy=shared,
        )

    accepted_legacy_token = legacy_token if ALLOW_LEGACY_DEVICE_TOKEN else ""
    try:
        configured = credential_count(accepted_legacy_token, named_tokens_json)
    except DeviceTokenConfigError as exc:
        raise HTTPException(status_code=503, detail=f"设备端鉴权配置错误：{exc}") from exc
    if configured == 0:
        raise HTTPException(status_code=503, detail="设备端鉴权未配置")
    try:
        credential = authenticate_device_token(token, accepted_legacy_token, named_tokens_json)
    except DeviceTokenConfigError as exc:
        raise HTTPException(status_code=503, detail=f"设备端鉴权配置错误：{exc}") from exc
    if credential is None:
        raise HTTPException(
            status_code=403,
            detail="设备端凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credential


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
    monotonic_now = time.monotonic()
    for session_id, state in list(SESSIONS.items()):
        finished = state.finished_monotonic
        if finished is not None and monotonic_now - finished > ttl_seconds:
            SESSIONS.pop(session_id, None)


def _cleanup_old_device_inputs(force: bool = False) -> None:
    """Remove terminal/orphaned device uploads after the recovery window."""
    global _LAST_DEVICE_INPUT_CLEANUP
    now = time.time()
    if not force and now - _LAST_DEVICE_INPUT_CLEANUP < 3600:
        return
    try:
        records = {
            str(row.get("job_id") or ""): row
            for row in mysql_db.list_device_job_retention_records()
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup][warn] device input retention scan skipped: {exc}")
        return
    _LAST_DEVICE_INPUT_CLEANUP = now
    ttl_seconds = DEVICE_INPUT_TTL_HOURS * 3600
    terminal = {"completed", "delivered", "failed"}
    for child in DEVICE_JOB_ROOT.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        try:
            if now - child.stat().st_mtime <= ttl_seconds:
                continue
            job = records.get(child.name)
            if job is not None and str(job.get("status") or "") not in terminal:
                continue
            shutil.rmtree(child, ignore_errors=True)
        except OSError as exc:
            print(f"[cleanup][warn] failed to remove stale device input {child}: {exc}")


def _save_uploads(
    files: List[UploadFile],
    destdir: Path,
    prefix: str,
    *,
    byte_budget: Optional[int] = None,
) -> List[Path]:
    if len(files) > MAX_TRIALS:
        raise HTTPException(status_code=413, detail=f"单次最多上传 {MAX_TRIALS} 组 trial")
    destdir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destdir).free < MIN_FREE_DISK_BYTES:
        raise HTTPException(status_code=507, detail="服务器可用磁盘空间不足，请稍后重试")
    out: List[Path] = []
    total_written = 0
    allowed_suffixes = {".csv", ".bdf"} if prefix == "eeg" else {".csv"}
    for i, uf in enumerate(files):
        suffix = (Path(uf.filename or f"{prefix}_{i}.csv").suffix or ".csv").lower()
        if suffix not in allowed_suffixes:
            allowed = " / ".join(sorted(allowed_suffixes))
            raise HTTPException(
                status_code=422,
                detail=f"{uf.filename or '上传文件'} 格式不支持，{prefix.upper()} 仅接受 {allowed}",
            )
        target = destdir / f"{prefix}_{i:02d}{suffix}"
        written = 0
        with target.open("wb") as fh:
            while True:
                chunk = uf.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                total_written += len(chunk)
                if written > MAX_UPLOAD_FILE_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{uf.filename or target.name} 超过单文件限制 {_human_bytes(MAX_UPLOAD_FILE_BYTES)}",
                    )
                if byte_budget is not None and total_written > byte_budget:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"本次评估上传总量超过限制 {_human_bytes(MAX_SESSION_UPLOAD_BYTES)}",
                    )
                if shutil.disk_usage(destdir).free - len(chunk) < MIN_FREE_DISK_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=507,
                        detail="服务器磁盘空间不足，已停止上传并清理临时文件",
                    )
                fh.write(chunk)
        out.append(target)
    return out


def _read_saved_zip(zip_path: Path, work: Path, institution: str):
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="压缩包不是有效的 zip 文件") from exc

    total_uncompressed = sum(info.file_size for info in members)
    suspicious = [
        info.filename
        for info in members
        if info.file_size > 0 and info.file_size / max(info.compress_size, 1) > MAX_ZIP_COMPRESSION_RATIO
    ]
    if suspicious:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=413, detail="压缩包包含异常压缩比文件，已拒绝解压")
    if len(members) > MAX_ZIP_MEMBERS:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"压缩包内文件数超过限制 {MAX_ZIP_MEMBERS}")
    if total_uncompressed > MAX_ZIP_EXTRACTED_BYTES:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(
            status_code=413,
            detail=f"压缩包解压后超过限制 {_human_bytes(MAX_ZIP_EXTRACTED_BYTES)}",
        )
    if shutil.disk_usage(work).free - total_uncompressed < MIN_FREE_DISK_BYTES:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=507, detail="磁盘空间不足以安全解压该数据包")
    try:
        root = safe_extract_zip(zip_path, work / "extracted")
        pkg = read_eval_package(root, institution=institution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return root, pkg


def _cached_upload(upload_id: str, institution: str) -> tuple[Path, Any, str, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", str(upload_id or "")):
        raise HTTPException(status_code=422, detail="upload_id 格式无效")
    work = (SESSION_ROOT / f"upload_{upload_id}").resolve()
    if SESSION_ROOT.resolve() not in work.parents:
        raise HTTPException(status_code=422, detail="upload_id 路径无效")
    metadata_path = work / "upload.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("institution") != institution:
            raise HTTPException(status_code=409, detail="upload_id 与当前机构类型不一致")
        root = locate_manifest_root(work / "extracted")
        pkg = read_eval_package(root, institution=institution)
        return work, pkg, str(metadata["package_hash"]), str(metadata.get("filename") or "bundle.zip")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="解析缓存已过期，请重新上传数据包") from exc
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"解析缓存损坏：{exc}") from exc


class AssessmentPersistenceError(RuntimeError):
    """The assessment finished computation but could not be committed."""


def _device_failure_details(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, FileNotFoundError):
        return "INPUT_FILE_MISSING", False
    if isinstance(exc, (ValueError, zipfile.BadZipFile)):
        return "INVALID_SIGNAL_DATA", False
    if isinstance(exc, AssessmentPersistenceError):
        return "PERSISTENCE_FAILED", True
    return "ANALYSIS_FAILED", True


def _worker(state: SessionState, registry: ModelRegistry, report_model) -> None:
    """Run the full pipeline + report generation on a worker thread."""
    try:
        if state.device_job_id:
            try:
                mysql_db.update_device_job(
                    state.device_job_id,
                    status="running",
                    phase="dl_inference",
                    progress_percent=5,
                    status_message="正在解析信号并运行深度学习评估",
                    clear_error=True,
                    increment_attempt=True,
                    mark_started=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark running: {exc}")

        predictions_raw = run_pipeline(
            state.eeg_paths, state.emg_paths, registry, state.queue,
            affected_side=state.patient.paralysis_side,
            institution=state.institution,
            trial_details=state.trial_details,
            cancel_event=state.cancel_event,
        )

        predictions = PredictionResult(
            FMA_UE=float(predictions_raw["FMA_UE"]),
            # BI is no longer served as a user-facing online model. Keep a
            # compatibility value for legacy DB columns/schemas.
            BI=float(predictions_raw.get("BI", 0.0)),
            hand_tone=str(predictions_raw["hand_tone"]),
            hand_function=int(predictions_raw["hand_function"]),
        )
        biomarkers = predictions_raw.get("_biomarkers")
        quality = predictions_raw.get("_quality") or {}
        validation_status = str(
            predictions_raw.get("_validation_status") or "research_assessment"
        )

        if state.device_job_id:
            try:
                mysql_db.update_device_job(
                    state.device_job_id,
                    phase="llm_reporting",
                    progress_percent=65,
                    status_message="深度学习评分已完成，正在生成康复报告",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to update report phase: {exc}")

        # Every flow (康复评估 page / device / task-interface) persists to MySQL;
        # the SQLite legacy store has been retired.
        store = mysql_db

        # Previous assessment (for the report's 变化趋势 column) — read BEFORE we
        # insert this one so it reflects the prior visit, not the current.
        try:
            history = store.latest_assessment_for_patient(state.patient.patient_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[history][warn] {exc}")
            history = None

        if state.cancel_event.is_set():
            raise AssessmentCancelled("评估任务已取消")
        report_generation = "failed"
        try:
            report_text, report_generation = stream_report(
                state.patient,
                predictions,
                state.queue,
                biomarkers=biomarkers,
                history=history,
                report_model=report_model,
                assessment_context={
                    "rag_correlation_id": state.session_id,
                    "institution": state.institution,
                    "assessment_type": "active",
                    "quality": quality,
                    "validation_status": validation_status,
                },
            )
            if state.cancel_event.is_set():
                raise AssessmentCancelled("评估任务已取消")
            app.state.report_ready = report_generation == "llm"
        except AssessmentCancelled:
            raise
        except Exception:
            # Report generation failed — predictions are still kept and the SSE
            # error event was already emitted by stream_report.
            report_text = None
            app.state.report_ready = False

        result = AssessmentResult(
            session_id=state.session_id,
            patient_info=state.patient,
            predictions=predictions,
            report=report_text,
            quality=quality,
            validation_status=validation_status,
        )

        # Persist before exposing a completed result. A failed/empty report is
        # still a valid assessment row with report_status='failed', while a DB
        # transaction failure must fail the session instead of emitting `done`.
        try:
            if state.device_job_id:
                mysql_db.update_device_job(
                    state.device_job_id,
                    phase="exporting",
                    progress_percent=90,
                    status_message="正在保存评估结果并准备导出文件",
                )
            report_status = "generated" if report_text else "failed"
            prediction_payload = {
                "FMA_UE": predictions.FMA_UE,
                "hand_tone": predictions.hand_tone,
                "hand_function": predictions.hand_function,
            }
            if store is mysql_db:
                assessment_db_id = mysql_db.save_assessment_bundle(
                    state.patient,
                    state.session_id,
                    predictions,
                    report_text,
                    report_status,
                    source=state.institution,
                    package_name=state.package_name,
                    assessment_id=state.assessment_id,
                    assessment_time=state.assessment_time,
                    institution=state.institution,
                    n_trials=state.n_trials,
                    package_hash=state.package_hash,
                    parse_warnings=state.parse_warnings,
                    prediction_payload=prediction_payload,
                    model_version=app.state.dl_model_version,
                    llm_provider=state.report_provider or llm_provider(),
                    llm_model=state.report_model_id or _llm_model_name(),
                    report_generation=report_generation,
                    trials=state.trial_details,
                    biomarkers=biomarkers,
                    quality=quality,
                    validation_status=validation_status,
                )
                state.assessment_db_id = int(assessment_db_id)
        except Exception as exc:  # noqa: BLE001
            raise AssessmentPersistenceError(
                f"评估结果写入 MySQL 失败：{exc}"
            ) from exc

        state.result = result

        if state.device_job_id:
            try:
                if state.assessment_db_id is not None:
                    mysql_db.update_device_job(
                        state.device_job_id,
                        status="completed",
                        phase="finished",
                        progress_percent=100,
                        status_message=(
                            "评估完成，可以下载结果"
                            if report_text
                            else "评分完成，但大模型报告生成失败；请审阅导出内容"
                        ),
                        assessment_db_id=state.assessment_db_id,
                        clear_error=True,
                        mark_completed=True,
                    )
                else:
                    mysql_db.update_device_job(
                        state.device_job_id,
                        status="failed",
                        phase="failed",
                        progress_percent=100,
                        status_message="评估结果保存失败",
                        error_message="评估已结束，但未写入 MySQL 评估记录，无法生成设备端导出文件。",
                        error_code="PERSISTENCE_FAILED",
                        error_retryable=True,
                        mark_completed=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark completed: {exc}")

        state.queue.put({"type": "done"})
    except AssessmentCancelled as exc:
        if state.device_job_id:
            try:
                mysql_db.update_device_job(
                    state.device_job_id,
                    status="failed",
                    phase="cancelled",
                    progress_percent=100,
                    status_message="评估任务已取消",
                    error_message=str(exc),
                    error_code="CANCELLED",
                    error_retryable=False,
                    mark_completed=True,
                )
            except Exception as job_exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark cancellation: {job_exc}")
        state.queue.put({"type": "cancelled", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if state.device_job_id:
            try:
                error_code, retryable = _device_failure_details(exc)
                mysql_db.update_device_job(
                    state.device_job_id,
                    status="failed",
                    phase="failed",
                    progress_percent=100,
                    status_message="云端评估失败",
                    error_message=str(exc),
                    error_code=error_code,
                    error_retryable=retryable,
                    mark_completed=True,
                )
            except Exception as job_exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark failed: {job_exc}")
        state.queue.put(error_event(f"会话 {state.session_id} 失败：{exc}"))
    finally:
        state.finished_monotonic = time.monotonic()
        state.queue.put(SENTINEL)
        if state.temporary_work_dir is not None:
            try:
                work = state.temporary_work_dir.resolve()
                if SESSION_ROOT.resolve() in work.parents:
                    shutil.rmtree(work, ignore_errors=True)
            except OSError as exc:
                print(f"[cleanup][warn] failed to remove session upload: {exc}")


def _run_scheduled_state(state: SessionState) -> None:
    registry: ModelRegistry = app.state.registry
    report_model = app.state.report_model
    _worker(state, registry, report_model)


def _start_session_worker(state: SessionState) -> None:
    with state.lock:
        if not state.started:
            state.started = True
            state.report_provider = llm_provider()
            state.report_model_id = _llm_model_name()
            snapshot = SESSION_SCHEDULER.enqueue(state.session_id, state)
            if snapshot.queue_ahead > 0:
                state.queue.put({
                    "type": "assessment_queued",
                    "ahead": snapshot.queue_ahead,
                })


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
        patient = _authoritative_patient(patient)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex
    destdir = SESSION_ROOT / session_id
    try:
        eeg_paths = _save_uploads(
            eeg_files,
            destdir / "eeg",
            "eeg",
            byte_budget=MAX_SESSION_UPLOAD_BYTES,
        )
        eeg_bytes = sum(path.stat().st_size for path in eeg_paths)
        emg_paths = _save_uploads(
            emg_files,
            destdir / "emg",
            "emg",
            byte_budget=max(0, MAX_SESSION_UPLOAD_BYTES - eeg_bytes),
        )
    except Exception:
        shutil.rmtree(destdir, ignore_errors=True)
        raise

    SESSIONS[session_id] = SessionState(
        session_id,
        patient,
        eeg_paths,
        emg_paths,
        persist_target="mysql",
        institution="hospital",
        temporary_work_dir=destdir,
    )
    return AssessSessionResponse(session_id=session_id, n_trials=len(eeg_paths))


# --------------------------------------------------------------------------- #
# 任务一与任务三对接接口：离线 zip 数据包导入 + 在线设备端占位                  #
# --------------------------------------------------------------------------- #
def _save_and_extract_zip(
    upload: UploadFile,
    institution: str,
    work_dir: Optional[Path] = None,
) -> "tuple[Path, Any, str]":
    """Persist the uploaded zip, extract it (zip-slip safe), and read the bundle.

    Returns ``(bundle_root, EvalPackage, package_sha256)``. Raises HTTPException(422) on bad input.
    """
    if institution not in INSTITUTIONS:
        raise HTTPException(status_code=422, detail=f"未知机构类型：{institution}")
    if not (upload.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="请上传 .zip 压缩包")

    _cleanup_old_sessions()
    work = work_dir or SESSION_ROOT / f"pkg_{uuid.uuid4().hex[:12]}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < MIN_FREE_DISK_BYTES:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=507, detail="服务器可用磁盘空间不足，请稍后重试")
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
        root, pkg = _read_saved_zip(zip_path, work, institution)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root, pkg, digest.hexdigest()


async def _save_and_extract_raw_zip(
    request: Request,
    institution: str,
    filename: Optional[str] = None,
    work_dir: Optional[Path] = None,
) -> "tuple[Path, Any, str, str]":
    """Persist a raw ``application/zip`` request body and parse it.

    This is a compatibility path for embedded clients that cannot easily send a
    multipart form. Patient metadata can be supplied through query parameters or
    headers; the zip itself is still validated exactly like multipart uploads.
    """
    if institution not in INSTITUTIONS:
        raise HTTPException(status_code=422, detail=f"未知机构类型：{institution}")

    clean_name = Path(filename or "device_upload.zip").name
    if not clean_name.lower().endswith(".zip"):
        clean_name = f"{clean_name}.zip"

    _cleanup_old_sessions()
    work = work_dir or SESSION_ROOT / f"pkg_{uuid.uuid4().hex[:12]}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < MIN_FREE_DISK_BYTES:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=507, detail="服务器可用磁盘空间不足，请稍后重试")
    zip_path = work / "bundle.zip"
    digest = hashlib.sha256()
    written = 0
    with zip_path.open("wb") as fh:
        async for chunk in request.stream():
            if not chunk:
                continue
            written += len(chunk)
            if written > MAX_ZIP_BYTES:
                fh.close()
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{clean_name} 超过压缩包限制 {_human_bytes(MAX_ZIP_BYTES)}",
                )
            digest.update(chunk)
            fh.write(chunk)
    if written == 0:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=422, detail="请求体为空，请上传 zip 二进制内容")

    try:
        root, pkg = _read_saved_zip(zip_path, work, institution)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root, pkg, digest.hexdigest(), clean_name


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
    upload_id = uuid.uuid4().hex
    work = SESSION_ROOT / f"upload_{upload_id}"
    _root, pkg, package_hash = _save_and_extract_zip(package, institution, work_dir=work)
    (work / "upload.json").write_text(
        json.dumps(
            {
                "upload_id": upload_id,
                "institution": institution,
                "filename": package.filename or "bundle.zip",
                "package_hash": package_hash,
                "created_at": int(time.time()),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
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
        "upload_id": upload_id,
        "enrolled": enrolled,
    }


@app.post("/api/task-interface/offline", response_model=AssessSessionResponse)
async def task_interface_offline(
    institution: str = Form(...),
    package: Optional[UploadFile] = File(None),
    upload_id: Optional[str] = Form(None),
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
    if upload_id and package is not None:
        raise HTTPException(status_code=422, detail="upload_id 与 package 只能提供一个")
    if upload_id:
        work, pkg, package_hash, package_name = _cached_upload(upload_id, institution)
    elif package is not None:
        work = SESSION_ROOT / f"upload_{uuid.uuid4().hex}"
        _root, pkg, package_hash = _save_and_extract_zip(package, institution, work_dir=work)
        package_name = package.filename or "bundle.zip"
    else:
        raise HTTPException(status_code=422, detail="请先解析数据包并提供 upload_id")
    if pkg.n_trials == 0:
        detail = "数据包中没有可用的 trial。" + ("；".join(pkg.warnings) if pkg.warnings else "")
        shutil.rmtree(work, ignore_errors=True)
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
        patient = _authoritative_patient(patient)
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = SessionState(
        session_id, patient, pkg.eeg_paths, pkg.emg_paths,
        institution=pkg.institution,
        persist_target="mysql",
        package_name=package_name,
        assessment_id=pkg.manifest_summary.get("assessment_id"),
        assessment_time=pkg.manifest_summary.get("assessment_time"),
        n_trials=pkg.n_trials,
        package_hash=package_hash,
        parse_warnings=pkg.warnings,
        trial_details=pkg.trial_details,
        temporary_work_dir=work,
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
def _device_job_files(job_id: str) -> Dict[str, str]:
    base = f"/api/device/v1/jobs/{job_id}"
    return {
        "json": f"{base}/result.json",
        "pdf": f"{base}/report.pdf",
        "zip": f"{base}/export.zip",
    }


def _format_device_job(job: Dict[str, Any]) -> Dict[str, Any]:
    status = str(job.get("status") or "queued")
    snapshot = SESSION_SCHEDULER.snapshot(str(job.get("session_id") or ""))
    phase = job.get("phase") or ("waiting" if status == "queued" else status)
    progress_percent = int(job.get("progress_percent") or 0)
    status_message = job.get("status_message")
    if status == "queued" and snapshot and snapshot.state == "running":
        # The scheduler can claim a just-enqueued task a few milliseconds before
        # the worker persists its running state. Keep the external state machine
        # coherent during that transition.
        status = "running"
        phase = "dl_inference"
        progress_percent = max(1, progress_percent)
        status_message = "任务已取得处理资源，正在启动评估"
    queue_position = snapshot.queue_position if snapshot and status == "queued" else 0
    queue_ahead = snapshot.queue_ahead if snapshot and status == "queued" else 0
    payload = {
        "schema_version": "rehab.device_job.v1",
        "job_id": job.get("job_id"),
        "device_id": job.get("device_id"),
        "session_id": job.get("session_id"),
        "assessment_db_id": job.get("assessment_db_id"),
        "assessment_id": job.get("assessment_id"),
        "patient_id": job.get("patient_id"),
        "package_name": job.get("package_name"),
        "package_hash": job.get("package_hash"),
        "status": status,
        "phase": phase,
        "queue_position": queue_position,
        "queue_ahead": queue_ahead,
        "progress_percent": progress_percent,
        "poll_after_seconds": 5 if status in {"queued", "running"} else None,
        "message": status_message,
        "attempt_count": int(job.get("attempt_count") or 0),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "delivered_at": job.get("delivered_at"),
        "updated_at": job.get("updated_at"),
    }
    if status == "failed":
        payload["error"] = {
            "code": job.get("error_code") or "ANALYSIS_FAILED",
            "message": job.get("error_message") or "云端评估失败",
            "retryable": bool(job.get("error_retryable")),
        }
    if job.get("status") in {"completed", "delivered"} and job.get("assessment_db_id"):
        payload["files"] = _device_job_files(str(job["job_id"]))
    return payload


def _bound_device_id(credential: DeviceCredential, requested_device_id: Optional[str]) -> Optional[str]:
    requested = (requested_device_id or "").strip() or None
    if credential.device_id and requested and requested != credential.device_id:
        raise HTTPException(status_code=403, detail="设备凭证与 device_id 不匹配")
    return credential.device_id or requested


def _device_job_or_404(job_id: str, credential: DeviceCredential) -> Dict[str, Any]:
    try:
        job = mysql_db.get_device_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="设备任务不存在")
        if credential.device_id and job.get("device_id") != credential.device_id:
            raise HTTPException(status_code=403, detail="该设备凭证无权访问此任务")
        if job and job.get("status") in {"queued", "running"} and job.get("session_id"):
            saved_assessment_id = mysql_db.assessment_id_for_session(str(job["session_id"]))
            if saved_assessment_id is not None:
                job = mysql_db.update_device_job(
                    job_id,
                    status="completed",
                    phase="finished",
                    progress_percent=100,
                    status_message="评估已完成，可以下载结果",
                    assessment_db_id=saved_assessment_id,
                    clear_error=True,
                    mark_completed=True,
                ) or job
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    return job


def _completed_device_assessment_id(job: Dict[str, Any]) -> int:
    if job.get("status") not in {"completed", "delivered"}:
        raise HTTPException(status_code=409, detail=f"任务尚未完成，当前状态：{job.get('status')}")
    assessment_db_id = job.get("assessment_db_id")
    if not assessment_db_id:
        raise HTTPException(status_code=409, detail="任务已结束但未生成评估记录")
    return int(assessment_db_id)


def _request_text(request: Request, key: str, default: Optional[str] = None) -> Optional[str]:
    """Read raw-upload metadata from query params first, then X-* headers."""
    value = request.query_params.get(key)
    if value not in (None, ""):
        return str(value)
    header = request.headers.get(f"X-{key.replace('_', '-')}")
    if header not in (None, ""):
        return str(header)
    return default


def _device_submission_response(job: Dict[str, Any], *, deduplicated: bool) -> Dict[str, Any]:
    response = _format_device_job(job)
    response["status_url"] = f"/api/device/v1/jobs/{job['job_id']}"
    response["n_trials"] = job.get("n_trials")
    response["parse_warnings"] = job.get("parse_warnings") or []
    response["deduplicated"] = deduplicated
    return response


def _cleanup_delivered_device_input(job: Dict[str, Any]) -> None:
    raw_path = str(job.get("input_path") or "").strip()
    if not raw_path:
        return
    try:
        input_path = Path(raw_path).resolve()
        root = DEVICE_JOB_ROOT.resolve()
        if root not in input_path.parents:
            print(f"[device_job][warn] refusing to clean path outside DEVICE_JOB_ROOT: {input_path}")
            return
        shutil.rmtree(input_path.parent, ignore_errors=True)
    except OSError as exc:
        print(f"[device_job][warn] failed to clean delivered input: {exc}")


def _create_device_assessment_job(
    *,
    job_id: str,
    input_path: Path,
    package_name: Optional[str],
    pkg: Any,
    package_hash: str,
    idempotency_key: Optional[str] = None,
    device_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    name: Optional[str] = None,
    sex: Optional[str] = None,
    age: Optional[int] = None,
    diagnosis: Optional[str] = None,
    disease_days: Optional[int] = None,
    paralysis_side: Optional[str] = None,
) -> Dict[str, Any]:
    if pkg.n_trials == 0:
        detail = "数据包中没有可用的 active trial。" + ("；".join(pkg.warnings) if pkg.warnings else "")
        shutil.rmtree(input_path.parent, ignore_errors=True)
        raise HTTPException(status_code=422, detail=detail)

    prefill = dict(pkg.patient_prefill)
    requested_pid = str(patient_id or "").strip()
    manifest_pid = str(prefill.get("patient_id") or "").strip()
    business_pid = requested_pid or manifest_pid

    try:
        enrolled = mysql_db.get_patient_by_business_id(business_pid) if business_pid else None
    except mysql_db.MySQLUnavailable as exc:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        raise _mysql_guard(exc) from exc

    try:
        patient = resolve_device_patient(
            requested_patient_id=requested_pid,
            manifest_patient_id=manifest_pid,
            enrolled=enrolled,
            require_registered=DEVICE_REQUIRE_REGISTERED_PATIENT,
            request_profile={
                "name": name,
                "sex": sex,
                "age": age,
                "diagnosis": diagnosis,
                "disease_days": disease_days,
                "paralysis_side": paralysis_side,
            },
            manifest_profile=prefill,
        )
    except DevicePatientPolicyError as exc:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    effective_device_id = (
        device_id or pkg.manifest_summary.get("device_id") or ""
    ).strip() or None
    effective_assessment_id = pkg.manifest_summary.get("assessment_id")
    normalized_key = (idempotency_key or "").strip() or None
    if normalized_key and len(normalized_key) > 255:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Idempotency-Key 最长为 255 个字符")

    try:
        existing = (
            mysql_db.find_device_job_by_idempotency_key(normalized_key, effective_device_id)
            if normalized_key
            else mysql_db.find_reusable_device_job(
                package_hash=package_hash,
                patient_id=patient.patient_id,
                device_id=effective_device_id,
                assessment_id=effective_assessment_id,
            )
        )
    except mysql_db.MySQLUnavailable as exc:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        raise _mysql_guard(exc) from exc

    if existing is not None:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        if normalized_key and existing.get("package_hash") != package_hash:
            raise HTTPException(
                status_code=409,
                detail="相同 Idempotency-Key 已用于另一个数据包，请检查 assessment_id",
            )
        return _device_submission_response(existing, deduplicated=True)

    session_id = uuid.uuid4().hex
    state = SessionState(
        session_id, patient, pkg.eeg_paths, pkg.emg_paths,
        institution=pkg.institution,
        persist_target="mysql",
        package_name=package_name,
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
            device_id=effective_device_id,
            session_id=session_id,
            assessment_id=effective_assessment_id,
            patient_id=patient.patient_id,
            package_name=package_name,
            package_hash=package_hash,
            institution=pkg.institution,
            input_path=str(input_path),
            patient_json=(
                patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
            ),
            parse_warnings=pkg.warnings,
            n_trials=pkg.n_trials,
            idempotency_key=normalized_key,
        )
    except Exception as exc:
        SESSIONS.pop(session_id, None)
        shutil.rmtree(input_path.parent, ignore_errors=True)
        if isinstance(exc, mysql_db.MySQLUnavailable):
            raise _mysql_guard(exc) from exc
        if normalized_key:
            raced = mysql_db.find_device_job_by_idempotency_key(normalized_key, effective_device_id)
            if raced is not None:
                if raced.get("package_hash") != package_hash:
                    raise HTTPException(
                        status_code=409,
                        detail="相同 Idempotency-Key 已用于另一个数据包，请检查 assessment_id",
                    ) from exc
                return _device_submission_response(raced, deduplicated=True)
        raise

    _start_session_worker(state)
    return _device_submission_response(
        mysql_db.get_device_job(job_id) or job,
        deduplicated=False,
    )


def _restore_device_jobs() -> None:
    """Requeue durable device jobs after a backend restart."""
    try:
        jobs = mysql_db.list_recoverable_device_jobs()
    except Exception as exc:  # noqa: BLE001
        print(f"[device_job][warn] recovery scan failed: {exc}")
        return

    for job in jobs:
        job_id = str(job.get("job_id") or "")
        session_id = str(job.get("session_id") or "")
        try:
            saved_assessment_id = mysql_db.assessment_id_for_session(session_id)
            if saved_assessment_id is not None:
                mysql_db.update_device_job(
                    job_id,
                    status="completed",
                    phase="finished",
                    progress_percent=100,
                    status_message="评估已完成，可以下载结果",
                    assessment_db_id=saved_assessment_id,
                    clear_error=True,
                    mark_completed=True,
                )
                continue

            input_path = Path(str(job.get("input_path") or ""))
            patient_payload = job.get("patient_json")
            if not input_path.is_file() or not isinstance(patient_payload, dict):
                raise FileNotFoundError("恢复任务所需的数据包或患者快照不存在")

            work = input_path.parent
            shutil.rmtree(work / "extracted", ignore_errors=True)
            _root, pkg = _read_saved_zip(
                input_path,
                work,
                str(job.get("institution") or "device"),
            )
            if pkg.n_trials == 0:
                raise ValueError("恢复的数据包中没有可用的 active trial")

            patient = PatientInfo(**patient_payload)
            state = SessionState(
                session_id,
                patient,
                pkg.eeg_paths,
                pkg.emg_paths,
                institution=pkg.institution,
                persist_target="mysql",
                package_name=job.get("package_name"),
                assessment_id=job.get("assessment_id"),
                assessment_time=pkg.manifest_summary.get("assessment_time"),
                n_trials=pkg.n_trials,
                package_hash=job.get("package_hash"),
                parse_warnings=job.get("parse_warnings") or pkg.warnings,
                trial_details=pkg.trial_details,
                device_job_id=job_id,
            )
            mysql_db.update_device_job(
                job_id,
                status="queued",
                phase="waiting",
                progress_percent=0,
                status_message="服务恢复后已重新进入队列",
                clear_error=True,
                reset_timestamps=True,
            )
            SESSIONS[session_id] = state
            _start_session_worker(state)
            print(f"[device_job] recovered {job_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[device_job][warn] failed to recover {job_id}: {exc}")
            try:
                mysql_db.update_device_job(
                    job_id,
                    status="failed",
                    phase="failed",
                    progress_percent=100,
                    status_message="服务重启后无法恢复任务，请重新上传",
                    error_message=str(exc),
                    error_code="RECOVERY_INPUT_MISSING",
                    error_retryable=True,
                    mark_completed=True,
                )
            except Exception as update_exc:  # noqa: BLE001
                print(f"[device_job][warn] failed to mark recovery error: {update_exc}")


@app.post(
    "/api/device/v1/patients",
    response_model=DevicePatientRegistrationResponse,
    status_code=201,
)
async def device_register_patient(
    payload: DevicePatientRegistrationRequest,
    response: Response,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    _device: DeviceCredential = Depends(_require_device),
):
    """Register a device-issued patient_id before the first assessment."""
    if idempotency_key is not None and len(idempotency_key.strip()) > 255:
        raise HTTPException(status_code=422, detail="Idempotency-Key 最长为 255 个字符")
    try:
        patient, created = mysql_db.register_device_patient(payload)
    except mysql_db.PatientRegistrationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PATIENT_ID_CONFLICT",
                "message": "该患者编号已对应其他身份信息",
                "fields": list(exc.fields),
            },
        ) from exc
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    response.status_code = 201 if created else 200
    return {
        "schema_version": "rehab.patient.v1",
        "patient_id": patient["patient_id"],
        "created": created,
        "message": "患者注册成功" if created else "患者已注册",
        "created_at": patient["created_at"],
        "updated_at": patient["updated_at"],
    }


@app.post("/api/device/v1/assessments", status_code=202)
async def device_create_assessment(
    request: Request,
    package: Optional[UploadFile] = File(None),
    institution: str = Form("device"),
    device_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    diagnosis: Optional[str] = Form(None),
    disease_days: Optional[int] = Form(None),
    paralysis_side: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    _device: DeviceCredential = Depends(_require_device),
):
    """Device-side machine API: upload one active-assessment zip and start a
    background analysis job. The device polls ``/jobs/{job_id}`` and downloads
    the generated export files when the job reaches ``completed``."""
    _cleanup_old_device_inputs()
    institution = (institution or "device").strip().lower()
    if institution != "device":
        raise HTTPException(status_code=422, detail="设备 API 仅接受 institution=device")
    if package is not None:
        job_id = f"devjob_{uuid.uuid4().hex[:16]}"
        work = DEVICE_JOB_ROOT / job_id
        _root, pkg, package_hash = _save_and_extract_zip(package, institution, work)
        return _create_device_assessment_job(
            job_id=job_id,
            input_path=work / "bundle.zip",
            package_name=package.filename,
            pkg=pkg,
            package_hash=package_hash,
            idempotency_key=idempotency_key,
            device_id=_bound_device_id(_device, device_id),
            patient_id=patient_id,
            name=name,
            sex=sex,
            age=age,
            diagnosis=diagnosis,
            disease_days=disease_days,
            paralysis_side=paralysis_side,
        )

    content_type = request.headers.get("content-type", "").lower()
    if "application/zip" in content_type or "application/octet-stream" in content_type:
        job_id = f"devjob_{uuid.uuid4().hex[:16]}"
        work = DEVICE_JOB_ROOT / job_id
        raw_institution = (_request_text(request, "institution", institution) or "device").strip().lower()
        if raw_institution != "device":
            raise HTTPException(status_code=422, detail="设备 API 仅接受 institution=device")
        filename = _request_text(request, "filename") or request.headers.get("X-Filename")
        _root, pkg, package_hash, package_name = await _save_and_extract_raw_zip(
            request, raw_institution, filename=filename, work_dir=work,
        )
        return _create_device_assessment_job(
            job_id=job_id,
            input_path=work / "bundle.zip",
            package_name=package_name,
            pkg=pkg,
            package_hash=package_hash,
            idempotency_key=idempotency_key,
            device_id=_bound_device_id(_device, _request_text(request, "device_id")),
            patient_id=_request_text(request, "patient_id"),
            name=_request_text(request, "name"),
            sex=_request_text(request, "sex"),
            age=int(_request_text(request, "age")) if (_request_text(request, "age") or "").isdigit() else None,
            diagnosis=_request_text(request, "diagnosis"),
            disease_days=(
                int(_request_text(request, "disease_days"))
                if (_request_text(request, "disease_days") or "").isdigit()
                else None
            ),
            paralysis_side=_request_text(request, "paralysis_side"),
        )

    raise HTTPException(
        status_code=422,
        detail=(
            "上传格式不正确：请使用 multipart/form-data，文件字段名为 package；"
            "或使用 Content-Type: application/zip 直接上传 zip 二进制。"
        ),
    )


@app.post("/api/device/v1/assessments/raw", status_code=202)
async def device_create_assessment_raw(
    request: Request,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    _device: DeviceCredential = Depends(_require_device),
):
    """Raw zip upload variant for embedded clients.

    Send ``Content-Type: application/zip`` and put patient/device metadata in
    query parameters or X-* headers, for example ``?patient_id=P001`` or
    ``X-Device-ID: device_001``.
    """
    _cleanup_old_device_inputs()
    job_id = f"devjob_{uuid.uuid4().hex[:16]}"
    work = DEVICE_JOB_ROOT / job_id
    raw_institution = (_request_text(request, "institution", "device") or "device").strip().lower()
    if raw_institution != "device":
        raise HTTPException(status_code=422, detail="设备 API 仅接受 institution=device")
    filename = _request_text(request, "filename") or request.headers.get("X-Filename")
    _root, pkg, package_hash, package_name = await _save_and_extract_raw_zip(
        request, raw_institution, filename=filename, work_dir=work,
    )
    return _create_device_assessment_job(
        job_id=job_id,
        input_path=work / "bundle.zip",
        package_name=package_name,
        pkg=pkg,
        package_hash=package_hash,
        idempotency_key=idempotency_key,
        device_id=_bound_device_id(_device, _request_text(request, "device_id")),
        patient_id=_request_text(request, "patient_id"),
        name=_request_text(request, "name"),
        sex=_request_text(request, "sex"),
        age=int(_request_text(request, "age")) if (_request_text(request, "age") or "").isdigit() else None,
        diagnosis=_request_text(request, "diagnosis"),
        disease_days=(
            int(_request_text(request, "disease_days"))
            if (_request_text(request, "disease_days") or "").isdigit()
            else None
        ),
        paralysis_side=_request_text(request, "paralysis_side"),
    )


@app.get("/api/device/v1/jobs/{job_id}")
async def device_get_job(job_id: str, _device: DeviceCredential = Depends(_require_device)):
    return _format_device_job(_device_job_or_404(job_id, _device))


@app.get("/api/device/v1/jobs/{job_id}/result.json")
async def device_download_result_json(
    job_id: str,
    _device: DeviceCredential = Depends(_require_device),
):
    job = _device_job_or_404(job_id, _device)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.result_json,
        media_type="application/json",
        filename=export_filename(assessment, "json"),
    )


@app.get("/api/device/v1/jobs/{job_id}/report.pdf")
async def device_download_report_pdf(
    job_id: str,
    _device: DeviceCredential = Depends(_require_device),
):
    job = _device_job_or_404(job_id, _device)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.report_pdf,
        media_type="application/pdf",
        filename=export_filename(assessment, "pdf"),
    )


@app.get("/api/device/v1/jobs/{job_id}/export.zip")
async def device_download_export_zip(
    job_id: str,
    _device: DeviceCredential = Depends(_require_device),
):
    job = _device_job_or_404(job_id, _device)
    assessment_id = _completed_device_assessment_id(job)
    assessment, bundle = _assessment_export_bundle(assessment_id)
    return FileResponse(
        bundle.export_zip,
        media_type="application/zip",
        filename=export_filename(assessment, "zip"),
    )


@app.post("/api/device/v1/jobs/{job_id}/ack")
async def device_ack_job(
    job_id: str,
    _device: DeviceCredential = Depends(_require_device),
):
    current = _device_job_or_404(job_id, _device)
    _completed_device_assessment_id(current)
    if current.get("status") == "delivered":
        _cleanup_delivered_device_input(current)
        return _format_device_job(current)
    try:
        job = mysql_db.update_device_job(
            job_id,
            status="delivered",
            phase="finished",
            progress_percent=100,
            status_message="设备端已确认收到结果",
            mark_delivered=True,
        )
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="设备任务不存在")
    _cleanup_delivered_device_input(job)
    return _format_device_job(job)


@app.get("/api/assess/{session_id}/stream")
async def stream_assessment(
    session_id: str,
    request: Request,
    _admin: None = Depends(_require_admin),
):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在或已过期")

    # Kick off worker only once per session.
    _start_session_worker(state)

    try:
        cursor = max(0, int(request.headers.get("last-event-id", "0") or 0))
    except ValueError:
        cursor = 0

    async def event_generator():
        nonlocal cursor
        loop = asyncio.get_event_loop()
        while True:
            try:
                rows, next_cursor, closed = await loop.run_in_executor(
                    None, state.queue.wait_after, cursor, 15.0
                )
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                break
            if not rows and not closed:
                yield ": keepalive\n\n"
            for event_id, item in rows:
                yield (
                    f"id: {event_id}\n"
                    f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                )
            cursor = next_cursor
            if closed and not rows:
                break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/assess/{session_id}/result", response_model=AssessmentResult)
async def get_result(session_id: str, _admin: None = Depends(_require_admin)):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在")
    if state.result is None:
        raise HTTPException(status_code=425, detail="评估尚未完成")
    return state.result


@app.delete("/api/assess/{session_id}")
async def cancel_assessment(session_id: str, _admin: None = Depends(_require_admin)):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在或已过期")
    if state.result is not None:
        raise HTTPException(status_code=409, detail="评估已经完成，不能取消")

    state.cancel_event.set()
    removed = SESSION_SCHEDULER.cancel_pending(session_id)
    if removed or not state.started:
        state.queue.put({"type": "cancelled", "message": "评估任务已取消"})
        state.queue.put(SENTINEL)
        state.finished_monotonic = time.monotonic()
        if state.temporary_work_dir is not None:
            shutil.rmtree(state.temporary_work_dir, ignore_errors=True)
        return {"status": "cancelled"}
    return {"status": "cancellation_requested"}


@app.get("/api/assess/{session_id}/report.docx")
async def get_report_docx(session_id: str, _admin: None = Depends(_require_admin)):
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
    mysql_ready = bool(getattr(app.state, "mysql_ready", False) and mysql_db.ping())
    dl_ready = bool(getattr(app.state, "dl_ready", False))
    report_ready = bool(getattr(app.state, "report_ready", False))
    return {
        "status": "ok" if mysql_ready and dl_ready and report_ready else "degraded",
        "checks": {
            "mysql": mysql_ready,
            "scoring_models": dl_ready,
            "report_model": report_ready,
        },
        "models_loaded": list(app.state.registry.models.keys()),
        "scoring_model_version": getattr(app.state, "dl_model_version", None),
        "report_provider": llm_provider(),
        "report_model": llm_model_name(),
        "app_version": APP_VERSION,
        "build_commit": APP_BUILD_COMMIT,
    }


@app.get("/api/ready")
async def ready():
    payload = await health()
    if payload["status"] != "ok":
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/settings/llm")
async def get_llm_settings(_admin: None = Depends(_require_admin)):
    return llm_settings.settings_payload(probe=True)


@app.patch("/api/settings/llm")
async def update_llm_settings(
    payload: LlmSettingsUpdate,
    _admin: None = Depends(_require_admin),
):
    if SESSION_SCHEDULER.has_work():
        raise HTTPException(
            status_code=409,
            detail="当前仍有评估任务运行或排队，请在队列清空后切换报告模型",
        )
    try:
        llm_settings.update_active_model(payload.active_model_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # The next local report generation should load the newly selected model.
    REPORT_MODEL.reset()
    app.state.report_ready = False
    return llm_settings.settings_payload(probe=True)


@app.patch("/api/settings/llm/models/{model_id}")
async def update_llm_model_settings(
    model_id: str,
    payload: LlmModelSettingsUpdate,
    _admin: None = Depends(_require_admin),
):
    if SESSION_SCHEDULER.has_work():
        raise HTTPException(
            status_code=409,
            detail="当前仍有评估任务运行或排队，请在队列清空后修改模型配置",
        )
    try:
        llm_settings.update_model_settings(
            model_id,
            payload.model_dump(exclude_unset=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    REPORT_MODEL.reset()
    app.state.report_ready = False
    return llm_settings.settings_payload(probe=True)


# --------------------------------------------------------------------------- #
# Read-only knowledge and evidence governance                                 #
# --------------------------------------------------------------------------- #
def _knowledge_guard(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"知识发布包不可用：{exc}")


@app.get("/api/admin/knowledge/status")
def get_knowledge_status(_admin: None = Depends(_require_admin)):
    return knowledge_admin.status_payload(
        app_version=APP_VERSION,
        build_commit=APP_BUILD_COMMIT,
        report_model=llm_model_name(),
    )


@app.get("/api/admin/knowledge/entries")
def get_knowledge_entries(
    category: Optional[str] = Query(None, max_length=64),
    knowledge_status: Optional[str] = Query(None, max_length=64),
    query: Optional[str] = Query(None, alias="q", max_length=200),
    _admin: None = Depends(_require_admin),
):
    try:
        return knowledge_admin.entries_payload(
            category=category,
            knowledge_status=knowledge_status,
            query=query,
        )
    except knowledge_admin.KnowledgeUnavailable as exc:
        raise _knowledge_guard(exc) from exc


@app.get("/api/admin/knowledge/entries/{knowledge_id}")
def get_knowledge_entry(
    knowledge_id: str,
    _admin: None = Depends(_require_admin),
):
    try:
        return knowledge_admin.entry_payload(knowledge_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="知识条目不存在") from exc
    except knowledge_admin.KnowledgeUnavailable as exc:
        raise _knowledge_guard(exc) from exc


@app.get("/api/admin/knowledge/coverage")
def get_knowledge_coverage(_admin: None = Depends(_require_admin)):
    try:
        return knowledge_admin.coverage_payload()
    except knowledge_admin.KnowledgeUnavailable as exc:
        raise _knowledge_guard(exc) from exc


@app.get("/api/admin/knowledge/sources")
def get_knowledge_sources(_admin: None = Depends(_require_admin)):
    try:
        return knowledge_admin.sources_payload()
    except knowledge_admin.KnowledgeUnavailable as exc:
        raise _knowledge_guard(exc) from exc


# --------------------------------------------------------------------------- #
# Admin device credential management                                          #
# --------------------------------------------------------------------------- #
def _device_credentials_payload() -> Dict[str, Any]:
    return {
        "schema_version": "rehab.device_credentials.v1",
        "items": mysql_db.list_device_credentials(),
    }


@app.get("/api/admin/device-credentials")
async def admin_list_device_credentials(_admin: None = Depends(_require_admin)):
    try:
        return _device_credentials_payload()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.post("/api/admin/device-credentials", status_code=201)
async def admin_create_device_credential(
    payload: DeviceCredentialCreate,
    _admin: None = Depends(_require_admin),
):
    try:
        if mysql_db.get_device_credential_by_device_id(payload.device_id) is not None:
            raise HTTPException(status_code=409, detail="设备 ID 已存在，请使用轮换功能生成新码")
        token = generate_device_token()
        credential = mysql_db.create_device_credential(
            device_id=payload.device_id,
            label=payload.label.strip(),
            access_scope="device",
            token_hash=token_digest(token),
            token_hint=token_hint(token),
            created_by=_admin_settings()[0],
        )
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    return {
        "schema_version": "rehab.device_credential_secret.v1",
        "credential": credential,
        "token": token,
    }


@app.patch("/api/admin/device-credentials/{credential_id}")
async def admin_update_device_credential(
    credential_id: int,
    payload: DeviceCredentialUpdate,
    _admin: None = Depends(_require_admin),
):
    try:
        current = mysql_db.get_device_credential(credential_id)
        if current is None:
            raise HTTPException(status_code=404, detail="设备凭证不存在")
        if current.get("status") == "revoked":
            raise HTTPException(status_code=409, detail="已撤销凭证不能直接启用，请重新生成设备码")
        updated = mysql_db.update_device_credential(
            credential_id,
            label=payload.label.strip() if payload.label is not None else None,
            status=payload.status,
        )
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    return updated


@app.post("/api/admin/device-credentials/{credential_id}/rotate")
async def admin_rotate_device_credential(
    credential_id: int,
    _admin: None = Depends(_require_admin),
):
    try:
        if mysql_db.get_device_credential(credential_id) is None:
            raise HTTPException(status_code=404, detail="设备凭证不存在")
        token = generate_device_token()
        credential = mysql_db.rotate_device_credential(
            credential_id,
            token_hash=token_digest(token),
            token_hint=token_hint(token),
        )
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    return {
        "schema_version": "rehab.device_credential_secret.v1",
        "credential": credential,
        "token": token,
    }


@app.delete("/api/admin/device-credentials/{credential_id}")
async def admin_revoke_device_credential(
    credential_id: int,
    _admin: None = Depends(_require_admin),
):
    try:
        if mysql_db.get_device_credential(credential_id) is None:
            raise HTTPException(status_code=404, detail="设备凭证不存在")
        return mysql_db.revoke_device_credential(credential_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc


@app.post("/api/auth/login", response_model=AuthLoginResponse)
async def auth_login(payload: AuthLoginRequest, request: Request, response: Response):
    expected_user, expected_password, token = _admin_settings()
    if not expected_password or not token:
        raise HTTPException(status_code=503, detail="后端鉴权未配置")
    real_ip = request.headers.get("x-real-ip", "").strip()
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    client_key = real_ip or forwarded or (request.client.host if request.client else "unknown")
    now = time.time()
    with _LOGIN_FAILURES_LOCK:
        recent = [
            value for value in _LOGIN_FAILURES.get(client_key, [])
            if now - value < LOGIN_RATE_WINDOW_SECONDS
        ]
        _LOGIN_FAILURES[client_key] = recent
        if len(recent) >= LOGIN_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="登录尝试过于频繁，请稍后再试",
                headers={"Retry-After": str(LOGIN_RATE_WINDOW_SECONDS)},
            )
    if not (
        secrets.compare_digest(payload.username, expected_user)
        and secrets.compare_digest(payload.password, expected_password)
    ):
        with _LOGIN_FAILURES_LOCK:
            _LOGIN_FAILURES.setdefault(client_key, []).append(now)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    with _LOGIN_FAILURES_LOCK:
        _LOGIN_FAILURES.pop(client_key, None)
    session_token = issue_session_token(expected_user, token, ADMIN_SESSION_TTL_SECONDS)
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip().lower()
    origin_scheme = "https" if request.headers.get("origin", "").lower().startswith("https://") else ""
    response.set_cookie(
        ADMIN_SESSION_COOKIE,
        session_token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=proto == "https" or origin_scheme == "https",
        samesite="strict",
        path="/api",
    )
    response.headers["Cache-Control"] = "no-store"
    return AuthLoginResponse(user=expected_user, expires_in=ADMIN_SESSION_TTL_SECONDS)


@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    if not _browser_write_origin_allowed(request):
        raise HTTPException(status_code=403, detail="请求来源校验失败，请刷新页面后重试")
    response.delete_cookie(ADMIN_SESSION_COOKIE, path="/api")
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


@app.get("/api/auth/session")
async def auth_session(_admin: None = Depends(_require_admin)):
    expected_user, _, _ = _admin_settings()
    return {"user": expected_user}


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
    """医院入组：写入患者基本信息 + 可选的第一次上肢/手功能评估记录。"""
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
    if None not in (payload.fma_ue, payload.hand_tone, payload.hand_function):
        first = {
            "FMA_UE": payload.fma_ue,
            # The legacy assessments table keeps a NOT NULL BI column. BI is no
            # longer user-facing, so new manual enrollments store a neutral
            # compatibility value rather than asking users for an unrelated ADL
            # score.
            "BI": 0.0,
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
    if SESSION_SCHEDULER.has_work():
        raise HTTPException(status_code=409, detail="当前仍有评估任务运行或排队，请稍后再删除记录")
    try:
        jobs = mysql_db.device_jobs_for_assessment(assessment_id)
        deleted = mysql_db.delete_assessment(assessment_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if deleted == 0:
        raise HTTPException(status_code=404, detail="记录不存在")
    delete_assessment_export(assessment_id)
    for job in jobs:
        _cleanup_delivered_device_input(job)
    for session_id, state in list(SESSIONS.items()):
        if state.assessment_db_id == assessment_id:
            SESSIONS.pop(session_id, None)
    return {"deleted": deleted}


@app.delete("/api/mysql/assessments")
async def mysql_clear_assessments(_admin: None = Depends(_require_admin)):
    """清空全部设备评估记录（测试期清理）。"""
    if SESSION_SCHEDULER.has_work():
        raise HTTPException(status_code=409, detail="当前仍有评估任务运行或排队，不能清空记录")
    try:
        assessment_ids = mysql_db.all_assessment_ids()
        jobs = mysql_db.completed_device_jobs()
        deleted = mysql_db.delete_all_assessments()
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    for assessment_id in assessment_ids:
        delete_assessment_export(assessment_id)
    for job in jobs:
        _cleanup_delivered_device_input(job)
    SESSIONS.clear()
    return {"deleted": deleted}


@app.delete("/api/mysql/patients/{patient_db_id}")
async def mysql_delete_patient(
    patient_db_id: int,
    _admin: None = Depends(_require_admin),
):
    """删除患者及其全部评估记录（级联，测试期清理）。"""
    if SESSION_SCHEDULER.has_work():
        raise HTTPException(status_code=409, detail="当前仍有评估任务运行或排队，请稍后再删除患者")
    try:
        patient_record = mysql_db.get_patient(patient_db_id)
        assessment_ids = mysql_db.assessment_ids_for_patient(patient_db_id)
        jobs = mysql_db.device_jobs_for_patient(patient_db_id)
        deleted = mysql_db.delete_patient(patient_db_id)
    except mysql_db.MySQLUnavailable as exc:
        raise _mysql_guard(exc) from exc
    if deleted == 0:
        raise HTTPException(status_code=404, detail="患者不存在")
    for assessment_id in assessment_ids:
        delete_assessment_export(assessment_id)
    for job in jobs:
        _cleanup_delivered_device_input(job)
    removed_ids = set(assessment_ids)
    business_patient_id = str((patient_record or {}).get("patient_id") or "")
    for session_id, state in list(SESSIONS.items()):
        if state.assessment_db_id in removed_ids or state.patient.patient_id == business_patient_id:
            SESSIONS.pop(session_id, None)
    return {"deleted": deleted}


# --------------------------------------------------------------------------- #
# CLI entry: `python -m backend.main` or `uvicorn backend.main:app --reload`.  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
