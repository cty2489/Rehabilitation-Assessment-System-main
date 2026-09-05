from __future__ import annotations

import asyncio
import inspect
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile

# These unit tests exercise report assembly and job routing, not signal inference.
# Keep their imports compatible with the repository's GPU-free CI environment.
with patch.dict(sys.modules, {"inference": SimpleNamespace(
    run_pipeline=Mock(side_effect=AssertionError("Signal inference is outside these unit tests")),
)}):
    import strategy_report_api
    import strategy_report_production


def _biomarkers(value: float) -> dict:
    keys = [f"metric_{index:02d}" for index in range(26)]
    return {
        "groups": [
            {
                "key": "emg",
                "label": "EMG",
                "markers": [
                    {
                        "key": key,
                        "name": key,
                        "value": value if index == 0 else float(index),
                        "unit": "比值",
                        "n_valid": 1,
                        "available": True,
                    }
                    for index, key in enumerate(keys)
                ],
            }
        ],
        "coverage": {"available": 26, "total": 26, "missing_keys": []},
    }


def test_production_source_has_no_preset_matcher() -> None:
    source = inspect.getsource(strategy_report_api) + inspect.getsource(strategy_report_production)
    forbidden = (
        "case_registry.json",
        "approved_cases",
        "exact_source_zip_sha256",
        "source_zip_sha256_verified",
        "registry[\"entries\"]",
    )
    assert all(token not in source for token in forbidden)


def test_clinical_result_does_not_depend_on_identifiers_or_filename() -> None:
    predictions = {"FMA_UE": 7.4, "hand_tone": "1+", "hand_function": 3}
    first = strategy_report_production._score_bundle(
        {"patient_id": "A", "assessment_id": "CASE001", "filename": "one.zip"}, predictions
    )
    second = strategy_report_production._score_bundle(
        {"patient_id": "B", "assessment_id": "OTHER", "filename": "renamed.zip"}, predictions
    )
    assert first == second
    assert first["fma_hand"] == 7


def test_manifest_clinical_scores_override_dl_without_combination_lookup() -> None:
    manifest = {
        "clinical_scores": {
            "fma_wrist": 4,
            "fma_hand": 9,
            "wrist_mas": "1",
            "hand_mas": "1+",
            "brunnstrom_hand": 4,
        }
    }
    result = strategy_report_production._score_bundle(
        manifest, {"FMA_UE": 2, "hand_tone": "3", "hand_function": 2}
    )
    assert result["mode"] == "doctor_clinical_score"
    assert (result["fma_wrist"], result["fma_hand"], result["brunnstrom_hand"]) == (4, 9, 4)


def test_changed_signal_result_changes_biomarker_row() -> None:
    first, _ = strategy_report_production._biomarker_rows(_biomarkers(0.1))
    second, _ = strategy_report_production._biomarker_rows(_biomarkers(0.2))
    assert first[0]["value_text"] != second[0]["value_text"]


def test_rms_converts_to_microvolts_but_iemg_keeps_v_seconds() -> None:
    payload = {
        "groups": [
            {
                "key": "emg",
                "label": "EMG",
                "markers": [
                    {"key": "resting_emg_level", "name": "RMS", "value": 1.219e-5, "unit": "V(RMS)", "n_valid": 6, "available": True},
                    {"key": "fcr_iemg", "name": "IEMG", "value": 1.219e-5, "unit": "V·s", "n_valid": 6, "available": True},
                ],
            }
        ]
    }
    rows, _ = strategy_report_production._biomarker_rows(payload)
    assert rows[0]["unit"] == "μV"
    assert rows[0]["value_text"] == "12.19"
    assert rows[1]["unit"] == "V·s"
    assert rows[1]["value_text"] == "0.00001219"


def test_manual_review_download_is_blocked_even_if_cached_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = "sr-20260904t120000z-123456789abc"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "report.pdf").write_bytes(b"%PDF-cached")
    (job_dir / ".publication.json").write_text(
        json.dumps(
            {
                "status": "MANUAL_REVIEW",
                "formal_report_created": False,
                "output_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_report_api, "EXPORT_ROOT", tmp_path.resolve())
    with pytest.raises(HTTPException) as caught:
        asyncio.run(strategy_report_api.download_strategy_report(job_id, "pdf"))
    assert caught.value.status_code == 423


def test_upload_returns_queued_job_and_schedules_background_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": "test", "institution": "hospital"}))
    archive_bytes.seek(0)
    upload = UploadFile(filename="new_patient.zip", file=archive_bytes)
    background = BackgroundTasks()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(dl_ready=True, report_ready=True, registry=object())
        )
    )
    monkeypatch.setattr(strategy_report_api, "EXPORT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(strategy_report_api, "MIN_FREE_DISK_BYTES", 0)

    result = asyncio.run(strategy_report_api.generate_strategy_report(request, background, upload))

    assert result["processing_status"] == "QUEUED"
    assert result["publication_status"] is None
    assert len(background.tasks) == 1
    status_path = tmp_path / result["job_id"] / ".job_status.json"
    assert json.loads(status_path.read_text(encoding="utf-8"))["processing_status"] == "QUEUED"
    assert (tmp_path / ".work" / result["job_id"] / "patient_data.zip").is_file()


def test_background_completion_is_persisted_and_work_files_are_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = "sr-20260905t000000z-abcdef123456"
    job_dir = tmp_path / job_id
    work_dir = tmp_path / ".work" / job_id
    job_dir.mkdir()
    work_dir.mkdir(parents=True)
    source = work_dir / "patient_data.zip"
    source.write_bytes(b"fixture")
    accepted_at = "2026-09-05T00:00:00+08:00"
    strategy_report_api._write_job_status(
        job_dir,
        strategy_report_api._status_payload(
            job_id=job_id,
            source_name="patient.zip",
            source_size=7,
            accepted_at=accepted_at,
            processing_status="QUEUED",
            current_stage="queued",
            progress_percent=5,
        ),
    )
    monkeypatch.setattr(
        strategy_report_api,
        "_generate_report_job",
        lambda **_: {
            "job_id": job_id,
            "generated_at": accepted_at,
            "publication_status": "WARNING",
            "publication_warnings": ["fixture warning"],
            "formal_report_created": True,
            "files": {"pdf": f"/api/strategy-reports/{job_id}/pdf"},
        },
    )

    strategy_report_api._run_accepted_job(
        job_id=job_id,
        job_dir=job_dir,
        work_dir=work_dir,
        source_path=source,
        source_name="patient.zip",
        source_size=7,
        zip_summary={},
        registry=object(),
        accepted_at=accepted_at,
    )

    status = json.loads((job_dir / ".job_status.json").read_text(encoding="utf-8"))
    assert status["processing_status"] == "COMPLETED"
    assert status["publication_status"] == "WARNING"
    assert status["formal_report_created"] is True
    assert not work_dir.exists()


def test_previous_synchronous_report_is_recovered_in_job_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = "sr-20260905t010000z-123456abcdef"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / ".publication.json").write_text(
        json.dumps(
            {
                "status": "WARNING",
                "warnings": ["retriever unavailable"],
                "formal_report_created": True,
                "decided_at": "2026-09-05T01:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / ".internal_result.json").write_text(
        json.dumps({"source_upload_name": "old.zip", "source_upload_size": 123}),
        encoding="utf-8",
    )
    (job_dir / "report.json").write_text(
        json.dumps(
            {
                "report_identity": {"case_id": "SR-OLD", "report_number": "SR-OLD-R01", "patient_code": "匿名患者"},
                "recommendations": [{"number": "策略一"}],
                "biomarker_coverage": {"available": 26, "total": 26},
                "clinical_fact_card": {"score_mode": "dl_prediction"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_report_api, "EXPORT_ROOT", tmp_path.resolve())

    result = asyncio.run(strategy_report_api.list_strategy_reports())

    assert result["jobs"][0]["job_id"] == job_id
    assert result["jobs"][0]["processing_status"] == "COMPLETED"
    assert result["jobs"][0]["formal_report_created"] is True
