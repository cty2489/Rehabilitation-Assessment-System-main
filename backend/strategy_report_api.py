"""Production upload-to-report API for the training-strategy feature only."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from eval_package import safe_extract_zip
from strategy_report_production import StrategyPipelineError, internal_result, run_report_pipeline
from strategy_report_renderer import build_pdf, markdown_report, validate_profile


router = APIRouter(prefix="/api/strategy-reports", tags=["strategy-reports"])

RESOURCE_DIR = Path(
    os.environ.get("STRATEGY_REPORT_RESOURCE_DIR", Path(__file__).with_name("strategy_report_resources"))
).resolve()
EXPORT_ROOT = Path(
    os.environ.get(
        "STRATEGY_REPORT_EXPORT_ROOT",
        "/root/autodl-tmp/rehab_project/exports/training_strategy_reports",
    )
).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("STRATEGY_REPORT_MAX_UPLOAD_BYTES", str(1024**3)))
MAX_ZIP_ENTRIES = int(os.environ.get("STRATEGY_REPORT_MAX_ZIP_ENTRIES", "500"))
MAX_EXTRACTED_BYTES = int(os.environ.get("STRATEGY_REPORT_MAX_EXTRACTED_BYTES", str(4 * 1024**3)))
MAX_COMPRESSION_RATIO = int(os.environ.get("STRATEGY_REPORT_MAX_COMPRESSION_RATIO", "200"))
MIN_FREE_DISK_BYTES = int(os.environ.get("STRATEGY_REPORT_MIN_FREE_DISK_BYTES", str(2 * 1024**3)))
MAX_MANIFEST_BYTES = int(os.environ.get("STRATEGY_REPORT_MAX_MANIFEST_BYTES", str(4 * 1024**2)))
JOB_ID_PATTERN = re.compile(r"^sr-\d{8}t\d{6}z-[0-9a-f]{12}$")
DOWNLOAD_FILES = {
    "pdf": "report.pdf",
    "json": "report.json",
    "markdown": "report.md",
    "zip": "report_bundle.zip",
}
PUBLISHABLE = {"PASSED", "WARNING"}
PROCESSING_STATES = {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}
_PIPELINE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitise_upload_name(name: str | None) -> str:
    value = Path(name or "patient_data.zip").name
    value = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", value).strip(" .")
    return value[:180] or "patient_data.zip"


def _inspect_zip(path: Path) -> dict[str, Any]:
    """Bound extraction cost and validate the sole manifest before extraction."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("上传文件不是有效的ZIP数据包") from exc
    with archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.is_dir()]
        if not files or len(files) > MAX_ZIP_ENTRIES:
            raise ValueError("ZIP文件条目数量异常")
        manifests: list[zipfile.ZipInfo] = []
        total_uncompressed = 0
        for info in files:
            raw_name = info.filename.replace("\\", "/")
            pure = PurePosixPath(raw_name)
            if pure.is_absolute() or ".." in pure.parts or not raw_name.strip():
                raise ValueError("ZIP内包含不安全路径")
            if info.flag_bits & 0x1:
                raise ValueError("不支持加密ZIP")
            if info.file_size > 0 and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
                raise ValueError("ZIP内包含异常压缩比文件")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_EXTRACTED_BYTES:
                raise ValueError("ZIP解压后体积超过服务器限制")
            if pure.name.lower() == "manifest.json":
                manifests.append(info)
        if len(manifests) != 1:
            raise ValueError("病人数据包必须且只能包含一个manifest.json")
        manifest_info = manifests[0]
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest.json体积异常")
        try:
            manifest = json.loads(archive.read(manifest_info).decode("utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("manifest.json无法解析") from exc
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json结构无效")
        institution = str(manifest.get("institution") or "hospital").strip().lower()
        if institution not in {"hospital", "device"}:
            institution = "hospital"
        return {
            "entry_count": len(files),
            "total_uncompressed_bytes": total_uncompressed,
            "manifest_path": manifest_info.filename,
            "manifest_schema": str(manifest.get("schema_version") or manifest.get("version") or "未标注"),
            "institution": institution,
        }


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _append_audit(event: dict[str, Any]) -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = EXPORT_ROOT / "generation_events.jsonl"
    with audit_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    os.chmod(audit_path, 0o600)


def _job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    return f"sr-{stamp}-{uuid.uuid4().hex[:12]}"


def _write_private(job_dir: Path, name: str, value: Any) -> Path:
    path = job_dir / name
    _atomic_json(path, value)
    os.chmod(path, 0o600)
    return path


def _status_payload(
    *,
    job_id: str,
    source_name: str,
    source_size: int,
    accepted_at: str,
    processing_status: str,
    current_stage: str,
    progress_percent: int,
    **extra: Any,
) -> dict[str, Any]:
    if processing_status not in PROCESSING_STATES:
        raise ValueError(f"invalid processing status: {processing_status}")
    payload: dict[str, Any] = {
        "schema_version": "rehab.strategy_report_job.v3",
        "job_id": job_id,
        "source_upload_name": source_name,
        "source_upload_size": source_size,
        "accepted_at": accepted_at,
        "generated_at": accepted_at,
        "updated_at": _now(),
        "processing_status": processing_status,
        "current_stage": current_stage,
        "progress_percent": max(0, min(100, int(progress_percent))),
        "publication_status": None,
        "publication_warnings": [],
        "formal_report_created": False,
        "files": {},
    }
    payload.update(extra)
    return payload


def _write_job_status(job_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["schema_version"] = "rehab.strategy_report_job.v3"
    payload["updated_at"] = _now()
    _write_private(job_dir, ".job_status.json", payload)
    return payload


def _patch_job_status(job_dir: Path, **patch: Any) -> dict[str, Any]:
    path = job_dir / ".job_status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(patch)
    return _write_job_status(job_dir, current)


def _validated_job_dir(job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="报告任务不存在")
    job_dir = (EXPORT_ROOT / job_id).resolve()
    if job_dir.parent != EXPORT_ROOT or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="报告任务不存在")
    return job_dir


def _build_bundle(job_dir: Path, profile: dict[str, Any], public_base: str) -> None:
    note = (
        "生成方式：安全解析本次上传ZIP，实际运行DL/临床评分映射、26项biomarker、"
        "QualityGate、Planner、知识图谱、RAG、ReportGenerator、Validator和publication gate。\n"
        "本功能不根据ZIP哈希、文件名、患者编号、CASE编号或评分组合查找预置答案。\n"
        "公开报告不包含真实姓名、医院患者ID、上传文件名或原始信号路径。\n"
    )
    bundle = job_dir / DOWNLOAD_FILES["zip"]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(job_dir / DOWNLOAD_FILES["pdf"], f"{public_base}.pdf")
        archive.write(job_dir / DOWNLOAD_FILES["json"], f"{public_base}.json")
        archive.write(job_dir / DOWNLOAD_FILES["markdown"], f"{public_base}.md")
        archive.writestr("生成说明.txt", note)
        used = {
            action["image_filename"]
            for recommendation in profile["recommendations"]
            if (action := recommendation.get("hand_rehabilitation_action"))
        }
        for filename in sorted(used):
            source = RESOURCE_DIR / "gesture_images" / filename
            if source.is_file():
                archive.write(source, f"手势库图片/{filename}")


def _publication_record(
    *, job_id: str, status: str, warnings: list[str], files: dict[str, str], identity: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "schema_version": "rehab.strategy_report_publication.v1",
        "job_id": job_id,
        "decided_at": _now(),
        "status": status,
        "warnings": warnings,
        "identity": identity or {},
        "output_sha256": files,
        "formal_report_created": status in PUBLISHABLE,
    }


def _generate_report_job(
    *,
    job_id: str,
    job_dir: Path,
    source_path: Path,
    source_name: str,
    source_size: int,
    zip_summary: dict[str, Any],
    registry: Any,
    accepted_at: str,
) -> dict[str, Any]:
    generated_at = accepted_at
    try:
        _patch_job_status(
            job_dir,
            processing_status="RUNNING",
            current_stage="正在解析患者数据包",
            progress_percent=10,
        )
        extract_dir = source_path.parent / "extracted"
        package_root = safe_extract_zip(source_path, extract_dir)
        _patch_job_status(
            job_dir,
            processing_status="RUNNING",
            current_stage="正在计算生物标志物并运行评估流水线",
            progress_percent=25,
        )
        with _PIPELINE_LOCK:
            artifacts = run_report_pipeline(
                package_root=package_root,
                institution=zip_summary["institution"],
                registry=registry,
                resource_dir=RESOURCE_DIR,
            )
        private = internal_result(artifacts)
        private.update(
            {
                "job_id": job_id,
                "source_upload_name": source_name,
                "source_upload_size": source_size,
                "zip_validation": zip_summary,
            }
        )
        _write_private(job_dir, ".internal_result.json", private)
        for event in artifacts.stage_events:
            _append_audit({"job_id": job_id, **event})

        _patch_job_status(
            job_dir,
            processing_status="RUNNING",
            current_stage="正在执行报告发布门检查",
            progress_percent=80,
        )

        if artifacts.publication_status == "MANUAL_REVIEW" or artifacts.profile is None:
            publication = _publication_record(
                job_id=job_id,
                status="MANUAL_REVIEW",
                warnings=artifacts.publication_warnings,
                files={},
                identity=None,
            )
            _write_private(job_dir, ".publication.json", publication)
            _append_audit({"job_id": job_id, "at": _now(), "stage": "formal_output", "status": "blocked", "detail": "MANUAL_REVIEW"})
            result = {
                "schema_version": "rehab.strategy_report_job.v2",
                "job_id": job_id,
                "generated_at": generated_at,
                "publication_status": "MANUAL_REVIEW",
                "publication_warnings": artifacts.publication_warnings,
                "formal_report_created": False,
                "files": {},
            }
            return result

        profile = artifacts.profile
        validate_profile(profile, RESOURCE_DIR)
        identity = profile["report_identity"]
        public_base = f"{identity['report_number']}_康复评估与训练策略报告"
        _patch_job_status(
            job_dir,
            processing_status="RUNNING",
            current_stage="正在生成PDF、JSON和Markdown",
            progress_percent=90,
        )
        _atomic_json(job_dir / DOWNLOAD_FILES["json"], profile)
        _atomic_text(job_dir / DOWNLOAD_FILES["markdown"], markdown_report(profile))
        build_pdf(profile, job_dir / DOWNLOAD_FILES["pdf"], RESOURCE_DIR / "gesture_images")
        _build_bundle(job_dir, profile, public_base)
        hashes = {kind: _sha256_file(job_dir / filename) for kind, filename in DOWNLOAD_FILES.items()}
        publication = _publication_record(
            job_id=job_id,
            status=artifacts.publication_status,
            warnings=artifacts.publication_warnings,
            files=hashes,
            identity=identity,
        )
        _write_private(job_dir, ".publication.json", publication)
        _append_audit({"job_id": job_id, "at": _now(), "stage": "formal_output", "status": "completed", "detail": artifacts.publication_status})
        return {
            "schema_version": "rehab.strategy_report_job.v2",
            "job_id": job_id,
            "case_id": identity["case_id"],
            "report_number": identity["report_number"],
            "patient_code": identity["patient_code"],
            "generated_at": generated_at,
            "strategy_count": len(profile["recommendations"]),
            "biomarker_coverage": profile["biomarker_coverage"],
            "generation_mode": "real_upload_pipeline",
            "publication_status": artifacts.publication_status,
            "publication_warnings": artifacts.publication_warnings,
            "formal_report_created": True,
            "clinical_score_mode": artifacts.score_bundle["mode"],
            "report_llm_called_during_request": True,
            "files": {kind: f"/api/strategy-reports/{job_id}/{kind}" for kind in DOWNLOAD_FILES},
        }
    except Exception as exc:  # noqa: BLE001
        failure = {
            "schema_version": "rehab.strategy_report_internal_failure.v1",
            "job_id": job_id,
            "at": _now(),
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
            "source_upload_name": source_name,
            "zip_validation": zip_summary,
        }
        _write_private(job_dir, ".internal_result.json", failure)
        publication = _publication_record(
            job_id=job_id,
            status="MANUAL_REVIEW",
            warnings=[str(exc)[:500]],
            files={},
            identity=None,
        )
        _write_private(job_dir, ".publication.json", publication)
        _append_audit({"job_id": job_id, "at": _now(), "stage": "publication_gate", "status": "completed", "detail": "MANUAL_REVIEW"})
        _append_audit({"job_id": job_id, "at": _now(), "stage": "formal_output", "status": "blocked", "detail": type(exc).__name__})
        if isinstance(exc, StrategyPipelineError):
            return {
                "schema_version": "rehab.strategy_report_job.v2",
                "job_id": job_id,
                "generated_at": generated_at,
                "publication_status": "MANUAL_REVIEW",
                "publication_warnings": [str(exc)],
                "formal_report_created": False,
                "files": {},
            }
        raise


def _run_accepted_job(
    *,
    job_id: str,
    job_dir: Path,
    work_dir: Path,
    source_path: Path,
    source_name: str,
    source_size: int,
    zip_summary: dict[str, Any],
    registry: Any,
    accepted_at: str,
) -> None:
    """Run after the HTTP 202 response; status is persisted for refresh recovery."""
    try:
        result = _generate_report_job(
            job_id=job_id,
            job_dir=job_dir,
            source_path=source_path,
            source_name=source_name,
            source_size=source_size,
            zip_summary=zip_summary,
            registry=registry,
            accepted_at=accepted_at,
        )
        processing_status = "COMPLETED"
        current_stage = (
            "报告生成完成"
            if result.get("formal_report_created")
            else "需要治疗师人工复核"
        )
        _write_job_status(
            job_dir,
            {
                **_status_payload(
                    job_id=job_id,
                    source_name=source_name,
                    source_size=source_size,
                    accepted_at=accepted_at,
                    processing_status=processing_status,
                    current_stage=current_stage,
                    progress_percent=100,
                ),
                **result,
                "schema_version": "rehab.strategy_report_job.v3",
                "processing_status": processing_status,
                "current_stage": current_stage,
                "progress_percent": 100,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _patch_job_status(
            job_dir,
            processing_status="FAILED",
            current_stage="报告生成失败",
            progress_percent=100,
            publication_status="MANUAL_REVIEW",
            publication_warnings=[str(exc)[:500]],
            formal_report_created=False,
            files={},
            error_message=str(exc)[:500],
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _legacy_completed_status(job_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Expose reports created by the previous synchronous endpoint after a refresh."""
    publication_path = job_dir / ".publication.json"
    internal_path = job_dir / ".internal_result.json"
    if not publication_path.is_file():
        return None
    try:
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        private = json.loads(internal_path.read_text(encoding="utf-8")) if internal_path.is_file() else {}
    except Exception:  # noqa: BLE001
        return None
    source_name = str(private.get("source_upload_name") or "历史报告任务")
    source_size = int(private.get("source_upload_size") or 0)
    accepted_at = str(private.get("created_at") or publication.get("decided_at") or _now())
    status = str(publication.get("status") or "MANUAL_REVIEW")
    payload = _status_payload(
        job_id=job_id,
        source_name=source_name,
        source_size=source_size,
        accepted_at=accepted_at,
        processing_status="COMPLETED",
        current_stage="报告生成完成" if status in PUBLISHABLE else "需要治疗师人工复核",
        progress_percent=100,
        publication_status=status,
        publication_warnings=list(publication.get("warnings") or []),
        formal_report_created=bool(publication.get("formal_report_created")),
        files=(
            {kind: f"/api/strategy-reports/{job_id}/{kind}" for kind in DOWNLOAD_FILES}
            if status in PUBLISHABLE and publication.get("formal_report_created")
            else {}
        ),
    )
    report_path = job_dir / DOWNLOAD_FILES["json"]
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            identity = report.get("report_identity") or {}
            payload.update(
                {
                    "case_id": identity.get("case_id"),
                    "report_number": identity.get("report_number"),
                    "patient_code": identity.get("patient_code"),
                    "strategy_count": len(report.get("recommendations") or []),
                    "biomarker_coverage": report.get("biomarker_coverage"),
                    "clinical_score_mode": (report.get("clinical_fact_card") or {}).get("score_mode"),
                    "generation_mode": "real_upload_pipeline",
                }
            )
        except Exception:  # noqa: BLE001
            pass
    return payload


def _read_job_status(job_dir: Path, job_id: str) -> dict[str, Any]:
    status_path = job_dir / ".job_status.json"
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if payload.get("job_id") == job_id:
                return payload
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="报告任务状态无法读取") from exc
    legacy = _legacy_completed_status(job_dir, job_id)
    if legacy is not None:
        return legacy
    raise HTTPException(status_code=404, detail="报告任务状态不存在")


@router.get("/capabilities")
async def strategy_report_capabilities() -> dict[str, Any]:
    return {
        "schema_version": "rehab.strategy_report_capabilities.v3",
        "template_version": "v11-production",
        "pipeline_mode": "real_upload_pipeline",
        "execution_mode": "asynchronous_job",
        "refresh_recovery_supported": True,
        "matching_policy": "none",
        "preset_case_fixture_used": False,
        "accepts": [".zip"],
        "multiple_upload_supported": True,
        "multiple_upload_policy": "网页逐文件上传并显示独立结果",
        "score_policy": "优先采用本次manifest随附的完整医生临床评分，否则使用本次原始信号DL结果",
        "publication_states": ["PASSED", "WARNING", "MANUAL_REVIEW"],
        "outputs": list(DOWNLOAD_FILES),
    }


@router.get("")
async def list_strategy_reports(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (
            path
            for path in EXPORT_ROOT.iterdir()
            if path.is_dir() and JOB_ID_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    jobs: list[dict[str, Any]] = []
    for job_dir in candidates:
        try:
            jobs.append(_read_job_status(job_dir, job_dir.name))
        except HTTPException:
            continue
        if len(jobs) >= limit:
            break
    return {
        "schema_version": "rehab.strategy_report_job_list.v1",
        "jobs": jobs,
    }


@router.get("/{job_id}")
async def get_strategy_report_job(job_id: str) -> dict[str, Any]:
    return _read_job_status(_validated_job_dir(job_id), job_id)


@router.post("", status_code=202)
async def generate_strategy_report(
    request: Request,
    background_tasks: BackgroundTasks,
    package: UploadFile = File(...),
) -> dict[str, Any]:
    source_name = _sanitise_upload_name(package.filename)
    if not source_name.lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="请选择ZIP格式的病人数据包")
    if not getattr(request.app.state, "dl_ready", False):
        raise HTTPException(status_code=503, detail="DL模型尚未就绪，无法运行真实评估流程")
    if not getattr(request.app.state, "report_ready", False):
        raise HTTPException(status_code=503, detail="报告大模型尚未就绪，无法运行planner_rag")
    incoming_root = EXPORT_ROOT / ".incoming"
    work_root = EXPORT_ROOT / ".work"
    incoming_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    os.chmod(incoming_root, 0o700)
    os.chmod(work_root, 0o700)
    temp_dir = Path(tempfile.mkdtemp(prefix="rehab-strategy-report-", dir=incoming_root))
    source_path = temp_dir / "patient_data.zip"
    size = 0
    try:
        with source_path.open("wb") as handle:
            while True:
                block = await package.read(8 * 1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传文件超过服务器允许的大小")
                if shutil.disk_usage(temp_dir).free - len(block) < MIN_FREE_DISK_BYTES:
                    raise HTTPException(status_code=507, detail="服务器空间不足，已停止上传")
                handle.write(block)
        if not size:
            raise HTTPException(status_code=422, detail="上传文件为空")
        try:
            zip_summary = await run_in_threadpool(_inspect_zip, source_path)
            accepted_at = _now()
            job_id = _job_id()
            job_dir = EXPORT_ROOT / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            work_dir = work_root / job_id
            work_dir.mkdir(parents=False, exist_ok=False)
            persistent_source = work_dir / "patient_data.zip"
            source_path.replace(persistent_source)
            accepted = _status_payload(
                job_id=job_id,
                source_name=source_name,
                source_size=size,
                accepted_at=accepted_at,
                processing_status="QUEUED",
                current_stage="上传完成，等待服务器处理",
                progress_percent=5,
            )
            _write_job_status(job_dir, accepted)
            _append_audit(
                {
                    "job_id": job_id,
                    "at": _now(),
                    "stage": "upload_accept",
                    "status": "completed",
                    "detail": f"bytes={size}",
                }
            )
            background_tasks.add_task(
                _run_accepted_job,
                job_id=job_id,
                job_dir=job_dir,
                work_dir=work_dir,
                source_path=persistent_source,
                source_name=source_name,
                source_size=size,
                zip_summary=zip_summary,
                registry=request.app.state.registry,
                accepted_at=accepted_at,
            )
            return accepted
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await package.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/{job_id}/{kind}")
async def download_strategy_report(job_id: str, kind: str) -> FileResponse:
    if not JOB_ID_PATTERN.fullmatch(job_id) or kind not in DOWNLOAD_FILES:
        raise HTTPException(status_code=404, detail="报告文件不存在")
    job_dir = _validated_job_dir(job_id)
    publication_path = job_dir / ".publication.json"
    if not publication_path.is_file():
        raise HTTPException(status_code=423, detail="报告尚未通过publication gate")
    try:
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=423, detail="报告发布状态不可验证") from exc
    status = str(publication.get("status") or "")
    if status not in PUBLISHABLE or not publication.get("formal_report_created"):
        raise HTTPException(status_code=423, detail="该结果为MANUAL_REVIEW，禁止生成或下载患者正式报告")
    path = job_dir / DOWNLOAD_FILES[kind]
    expected_hash = str((publication.get("output_sha256") or {}).get(kind) or "")
    if not path.is_file() or not expected_hash or _sha256_file(path) != expected_hash:
        raise HTTPException(status_code=423, detail="报告文件未通过完整性校验")
    identity = publication.get("identity") or {}
    report_number = re.sub(r"[^A-Za-z0-9._-]", "_", str(identity.get("report_number") or "匿名报告"))
    suffix = {"pdf": "pdf", "json": "json", "markdown": "md", "zip": "zip"}[kind]
    media_type = {
        "pdf": "application/pdf",
        "json": "application/json",
        "markdown": "text/markdown; charset=utf-8",
        "zip": "application/zip",
    }[kind]
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{report_number}_康复评估与训练策略报告.{suffix}",
        headers={"Cache-Control": "no-store, private", "X-Publication-Status": status},
    )
