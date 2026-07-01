"""Institution-agnostic reader for a packaged patient evaluation (zip / folder).

Task one (data acquisition) hands task three (this assessment system) a *bundle*
— a zip containing ``manifest.json`` plus per-trial EEG/EMG-IMU files. Two
institutions currently produce these bundles with different sensor formats:

* **hospital** (``patient_hospital_*``) — Delsys 56-column EMG/IMU csv (named
  muscles, per-signal ``X[s]`` time columns), 32-channel 1000 Hz BDF; one trial
  per action; ``active`` assessment only. Read by the existing BJH loaders.
* **device**   (``patient_*`` from the wearable) — 19-column unified-time EMG/IMU
  csv (``通道1-8`` + ``IMU1-9``), 512 Hz BDF; three trials per action; both
  ``active`` and ``passive``. Read by ``Deeplearning/bjh_io/device_loader.py``.

This module only resolves the **manifest → ordered file paths + patient prefill**;
the actual signal decoding is institution-specific and happens downstream
(``inference.run_pipeline`` / ``biomarkers.extract``) keyed on ``institution``.

The manifest layout (``assessments[].trials[].{eeg_file, emg_imu_file}`` with a
per-block ``assessment_type``) is identical across both institutions, so trial
enumeration here is format-agnostic. Per the product decision we only feed the
**active** assessment into the pipeline; ``passive`` blocks are ignored (which
also sidesteps the device manifest's known ``positive_assessment`` path typos
that only occur inside passive blocks).
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

INSTITUTIONS = ("hospital", "device")


@dataclass
class EvalPackage:
    """Resolved evaluation bundle ready to feed the pipeline."""

    institution: str
    root: Path
    eeg_paths: List[Path]
    emg_paths: List[Path]
    patient_prefill: Dict[str, Any]
    manifest_summary: Dict[str, Any]
    trial_details: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.eeg_paths)


# --------------------------------------------------------------------------- #
# Zip extraction (zip-slip safe)                                              #
# --------------------------------------------------------------------------- #
def safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ``zip_path`` into ``dest_dir`` and return the bundle root — the
    directory that directly contains ``manifest.json``.

    Rejects entries with absolute paths or ``..`` traversal (zip-slip). Tolerates
    a wrapping top-level folder (the common ``unzip`` layout) by locating
    ``manifest.json`` recursively after extraction.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = member.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"压缩包包含非法路径，已拒绝解压：{member}")
            target = (dest_root / name).resolve()
            if not str(target).startswith(str(dest_root)):
                raise ValueError(f"压缩包路径越界，已拒绝解压：{member}")
        zf.extractall(dest_root)

    return _locate_manifest_root(dest_root)


def _locate_manifest_root(extracted: Path) -> Path:
    """Find the directory containing manifest.json (shallowest match wins)."""
    if (extracted / "manifest.json").is_file():
        return extracted
    candidates = sorted(extracted.rglob("manifest.json"), key=lambda p: len(p.parts))
    if not candidates:
        raise ValueError("压缩包内未找到 manifest.json，无法识别评估数据包")
    return candidates[0].parent


# --------------------------------------------------------------------------- #
# Manifest parsing                                                            #
# --------------------------------------------------------------------------- #
def _load_manifest(root: Path) -> Dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"未找到 manifest.json：{path}")
    # Device manifests are UTF-8 with a BOM → utf-8-sig handles both.
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json 解析失败：{exc}") from exc


def _patient_prefill(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Pull whatever patient fields the manifest carries (hospital has more).

    Always returns the full PatientInfo-shaped dict; fields the manifest lacks
    (e.g. device has only patient_id) are left as sensible blanks for the user to
    complete in the UI. ``diagnosis`` / ``paralysis_side`` are never in either
    manifest and must be supplied by the clinician.
    """
    gender_raw = str(manifest.get("patient_gender") or "").strip()
    sex = "男" if gender_raw.startswith("男") else "女" if gender_raw.startswith("女") else "男"

    age: Optional[int] = None
    age_match = re.search(r"\d+", str(manifest.get("patient_age") or ""))
    if age_match:
        age = int(age_match.group())

    return {
        "patient_id": str(manifest.get("patient_id") or "").strip(),
        "name": str(manifest.get("patient_name") or "").strip(),
        "sex": sex,
        "age": age,
        "diagnosis": "",
        "disease_days": None,
        "paralysis_side": "左",
    }


def _manifest_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    dd = manifest.get("data_description", {}) or {}
    return {
        "assessment_id": manifest.get("assessment_id"),
        "assessment_time": manifest.get("assessment_time"),
        "assessment_types": dd.get("assessment_types"),
        "actions_per_type": dd.get("actions_per_type"),
        "trials_per_action": dd.get("trials_per_action"),
        "emg_sampling_rate_hz": dd.get("emg_sampling_rate_hz"),
        "eeg_sampling_rate_hz": dd.get("eeg_sampling_rate_hz"),
        "eeg_channel_count": dd.get("eeg_channel_count"),
    }


def _is_usable(path: Path) -> Tuple[bool, str]:
    """A trial file is usable if it exists and isn't an empty placeholder."""
    if not path.is_file():
        return False, "文件缺失"
    if path.stat().st_size == 0:
        return False, "文件为空(0字节)占位"
    return True, ""


def read_eval_package(
    root: Path,
    institution: str,
    assessment_type: str = "active",
) -> EvalPackage:
    """Resolve a bundle directory into ordered (eeg, emg) path pairs + prefill.

    Only ``assessment_type`` blocks are included. Trials whose EEG or EMG file is
    missing/empty are skipped with a recorded warning (so the pipeline still runs
    on whatever real trials exist). Empty-but-present files are skipped here so a
    device placeholder bundle surfaces a clear "no usable trials" message rather
    than a deep decode crash.
    """
    institution = institution.lower().strip()
    if institution not in INSTITUTIONS:
        raise ValueError(f"未知机构类型：{institution}（应为 hospital / device）")

    root = Path(root)
    manifest = _load_manifest(root)
    warnings: List[str] = []

    eeg_paths: List[Path] = []
    emg_paths: List[Path] = []
    trial_details: List[Dict[str, Any]] = []
    n_active_trials = 0
    for block in manifest.get("assessments", []) or []:
        if str(block.get("assessment_type", "")).lower() != assessment_type:
            continue
        action = block.get("action_name") or block.get("action_id") or "?"
        for trial in block.get("trials", []) or []:
            n_active_trials += 1
            eeg_rel = trial.get("eeg_file")
            emg_rel = trial.get("emg_imu_file")
            if not eeg_rel or not emg_rel:
                warnings.append(f"{action} trial{trial.get('trial_index', '?')}: manifest 缺少文件路径，已跳过")
                continue
            eeg_p = (root / eeg_rel).resolve()
            emg_p = (root / emg_rel).resolve()
            eeg_ok, eeg_why = _is_usable(eeg_p)
            emg_ok, emg_why = _is_usable(emg_p)
            if not eeg_ok or not emg_ok:
                why = eeg_why if not eeg_ok else emg_why
                warnings.append(f"{action} trial{trial.get('trial_index', '?')}: {why}，已跳过")
                continue
            eeg_paths.append(eeg_p)
            emg_paths.append(emg_p)
            trial_details.append(
                {
                    "assessment_type": assessment_type,
                    "action_name": str(action),
                    "trial_index": trial.get("trial_index"),
                    "eeg_file": str(eeg_rel),
                    "emg_imu_file": str(emg_rel),
                    "eeg_name": Path(str(eeg_rel)).name,
                    "emg_name": Path(str(emg_rel)).name,
                    "status": "used",
                }
            )

    if n_active_trials == 0:
        warnings.append(f"manifest 中未找到 {assessment_type} 评估数据")
    if not eeg_paths:
        warnings.append("没有可用的 trial（文件缺失或为空占位），无法运行分析流程")

    return EvalPackage(
        institution=institution,
        root=root,
        eeg_paths=eeg_paths,
        emg_paths=emg_paths,
        patient_prefill=_patient_prefill(manifest),
        manifest_summary=_manifest_summary(manifest),
        trial_details=trial_details,
        warnings=warnings,
    )


__all__ = ["EvalPackage", "safe_extract_zip", "read_eval_package", "INSTITUTIONS"]
