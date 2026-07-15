"""Digital biomarker extraction from the uploaded EEG / EMG / IMU trials.

The DL inference pipeline (``backend/inference.py``) feeds the model
*z-score-normalised* signals, which throws away the physical units we need to
report clinical biomarkers. So this module re-loads the **raw** signals from the
same file paths (reusing the BJH loaders/filters in ``bjh_io.bjh_loader``) and
delegates the actual computation to the project's biomarker program in
``biomarkers/biomarkers.py`` — the single source of truth for the 26-biomarker
formulas. Reference ranges come from ``backend/biomarker_refs.py`` (literature
JSON). Interpretation / treatment advice is the LLM's job (see
``backend/report.py``); this module only produces the deterministic measured
values + their references, grouped EMG / EEG / IMU to mirror the report template.

Three groups (mirroring ``大模型评估报告模板示例.docx``):
* **EMG (肌电, 14)** — resting level, wrist/finger co-contraction (CCI), activation
                   RMS, FCR/FDS/ECU/ED IEMG, flexor-extensor IEMG ratio, burst
                   duration, FCR/FDS/ECU/ED median frequency (MDF).
* **EEG (脑电, 6)** — pathological asymmetry, β corticomuscular coherence, prefrontal
                   θ/β, inter-hemispheric coherence, movement μ/β power change.
* **IMU (运动学, 6)** — SPARC smoothness, ROM proxy, tremor index, wrist flexion/
                    extension peak velocity, finger extension peak velocity.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as scisig

# Reuse the BJH raw loaders + filter helpers (no z-score applied here).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DL_DIR = _PROJECT_ROOT / "Deeplearning"
_BJH_BM_DIR = _PROJECT_ROOT / "biomarkers"
for _p in (_DL_DIR, _BJH_BM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bjh_io.bjh_loader import (  # noqa: E402
    EEG_CHANNELS_BDF_30,
    _notch,
    _butter_bandpass,
    load_eeg,
    load_eeg_bdf,
    load_emg_imu,
)
from bjh_io.device_loader import (  # noqa: E402
    load_device_eeg_bdf,
    load_device_emg_imu,
)

# The biomarker program (single source of truth for the 26 formulas). NOTE: this
# backend module is *itself* named ``biomarkers``, so a plain ``import biomarkers``
# would resolve to *this* file (it's already in sys.modules). Load the project's
# biomarker program explicitly from its file path under a distinct module name.
import importlib.util as _ilu  # noqa: E402

_BJH_BM_FILE = _BJH_BM_DIR / "biomarkers.py"
_spec = _ilu.spec_from_file_location("bjh_biomarkers_program", _BJH_BM_FILE)
bjh_bm = _ilu.module_from_spec(_spec)
sys.modules["bjh_biomarkers_program"] = bjh_bm
_spec.loader.exec_module(bjh_bm)

from biomarker_refs import ref_display  # noqa: E402


# --------------------------------------------------------------------------- #
# Raw signal loading (no z-score).                                             #
# --------------------------------------------------------------------------- #
def _bandpass(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass along time axis (axis 0)."""
    nyq = 0.5 * fs
    hi = min(hi, 0.99 * nyq)
    if not (0 < lo < hi):
        return x
    sos = _butter_bandpass(lo, hi, fs)
    return scisig.sosfiltfilt(sos, x, axis=0)


def _load_raw_trial(eeg_path: Path, emg_path: Path, institution: str = "hospital") -> Dict[str, Any]:
    """Load one trial's raw signals into the ``raw`` dict the biomarker program
    expects (see ``biomarkers/biomarkers.py`` module docstring):

        eeg [T, 30] filtered, NOT z-scored;  emg [T, 4] raw amplitude;
        imu [T, 24] raw (4 muscles × ACC/GYRO).

    EEG: filtered (notch+bandpass+avg-ref for BDF) but NOT z-scored — its scale
    is irrelevant since EEG biomarkers are ratios/coherences.
    EMG/IMU: raw amplitudes straight from the loader (absolute values preserved,
    required for tone/tremor). ``institution`` selects the hospital vs device
    raw loader (device markers are best-effort — see device_loader.py).
    """
    if str(institution).lower() == "device":
        emg_raw, imu_raw, _t_emg, _t_imu, meta = load_device_emg_imu(emg_path)
        emg_fs = float(meta["emg_fs"])
        imu_fs = float(meta["imu_fs"])
        eeg, eeg_fs, eeg_channels = load_device_eeg_bdf(eeg_path)
        return {
            "eeg": np.asarray(eeg, dtype=np.float64),
            "eeg_fs": float(eeg_fs),
            "eeg_channels": list(eeg_channels),
            "emg": np.asarray(emg_raw, dtype=np.float64),
            "emg_fs": emg_fs,
            "imu": np.asarray(imu_raw, dtype=np.float64),
            "imu_fs": imu_fs,
        }

    emg_raw, imu_raw, _t_emg, _t_imu, meta = load_emg_imu(emg_path)
    emg_fs = float(meta["emg_fs"])
    imu_fs = float(meta["imu_fs"])

    if Path(eeg_path).suffix.lower() == ".bdf":
        eeg, eeg_fs, eeg_channels = load_eeg_bdf(eeg_path)
        eeg_channels = list(eeg_channels)
    else:
        eeg, eeg_fs = load_eeg(eeg_path)
        eeg_channels = list(EEG_CHANNELS_BDF_30)
        # CSV path is unfiltered raw → apply notch+bandpass so band powers are clean.
        x = eeg.astype(np.float64, copy=True)
        if 0 < 50.0 < 0.5 * eeg_fs:
            b, a = _notch(50.0, eeg_fs)
            x = scisig.filtfilt(b, a, x, axis=0)
        eeg = _bandpass(x, 1.0, 45.0, eeg_fs).astype(np.float32)

    return {
        "eeg": np.asarray(eeg, dtype=np.float64),
        "eeg_fs": float(eeg_fs),
        "eeg_channels": list(eeg_channels),
        "emg": np.asarray(emg_raw, dtype=np.float64),
        "emg_fs": emg_fs,
        "imu": np.asarray(imu_raw, dtype=np.float64),
        "imu_fs": imu_fs,
    }


# --------------------------------------------------------------------------- #
# Biomarker metadata: name / unit / group for each of the 26 markers.          #
# Grouping mirrors the report template (EMG / EEG / IMU). Units come from the   #
# reference JSON's ``units`` field.                                             #
# --------------------------------------------------------------------------- #
_FDS_NOTE = "指浅屈肌（FDS）信号取自掌长肌（Palmaris Longus）电极位"
_BIOMARKER_META: Dict[str, Dict[str, str]] = {
    # --- EMG (肌电) — 14 项 ---
    "resting_emg_level": {"name": "静息肌电水平", "unit": "V(RMS)", "group": "emg"},
    "wrist_co_contraction_index": {"name": "腕屈伸肌共收缩指数（CCI-腕）", "unit": "比值[0,1]", "group": "emg",
                                   "note": "腕屈肌 FCR 与腕伸肌 ECU 的共收缩；设备特异量"},
    "finger_co_contraction_index": {"name": "指屈伸肌共收缩指数（CCI-指）", "unit": "比值[0,1]", "group": "emg",
                                    "note": f"指浅屈肌 FDS 与指伸肌 ED 的共收缩；{_FDS_NOTE}；设备特异量"},
    "emg_activation_rms": {"name": "肌肉激活幅度（RMS）", "unit": "V(RMS)", "group": "emg"},
    "fcr_iemg": {"name": "桡侧腕屈肌（FCR）积分肌电（IEMG）", "unit": "V·s", "group": "emg"},
    "fds_iemg": {"name": "指浅屈肌（FDS）积分肌电（IEMG）", "unit": "V·s", "group": "emg",
                 "note": _FDS_NOTE},
    "ecu_iemg": {"name": "尺侧腕伸肌（ECU）积分肌电（IEMG）", "unit": "V·s", "group": "emg"},
    "extensor_digitorum_iemg": {"name": "指伸肌（ED）积分肌电（IEMG）", "unit": "V·s", "group": "emg"},
    "flexor_extensor_iemg_ratio": {"name": "屈/伸肌 IEMG 比", "unit": "比值", "group": "emg"},
    "emg_burst_duration": {"name": "肌电爆发持续时间", "unit": "s", "group": "emg"},
    "fcr_mdf": {"name": "桡侧腕屈肌（FCR）中位频率（MDF）", "unit": "Hz", "group": "emg",
                "note": "Welch 谱 20–450Hz 限带累计 50% 功率；设备特异量，降低提示疲劳/募集改变"},
    "fds_mdf": {"name": "指浅屈肌（FDS）中位频率（MDF）", "unit": "Hz", "group": "emg",
                "note": f"Welch 谱 20–450Hz 限带累计 50% 功率；{_FDS_NOTE}；设备特异量"},
    "ecu_mdf": {"name": "尺侧腕伸肌（ECU）中位频率（MDF）", "unit": "Hz", "group": "emg",
                "note": "Welch 谱 20–450Hz 限带累计 50% 功率；设备特异量"},
    "extensor_digitorum_mdf": {"name": "指伸肌（ED）中位频率（MDF）", "unit": "Hz", "group": "emg",
                               "note": "Welch 谱 20–450Hz 限带累计 50% 功率；设备特异量"},
    # --- EEG (脑电) — 6 项 ---
    "pathological_asymmetry_index": {"name": "病理性半球不对称指数（PAI）", "unit": "比值[-1,1]", "group": "eeg",
                                     "note": "基于全段 μ/β 谱功率的静息态不对称"},
    "corticomuscular_coherence_beta": {"name": "皮层-肌肉相干（β带 15–30Hz）", "unit": "相干[0,1]", "group": "eeg",
                                       "note": "EEG 与 EMG 跨模态未精同步，绝对值仅供队列内方向参考"},
    "prefrontal_theta_beta_ratio": {"name": "前额叶 θ/β 比值", "unit": "比值", "group": "eeg"},
    "interhemispheric_motor_coherence": {"name": "半球间运动皮层相干", "unit": "相干[0,1]", "group": "eeg"},
    "movement_mu_power_change": {"name": "运动相关 μ 功率变化", "unit": "相对变化", "group": "eeg",
                                 "note": "以 EMG 包络划窗近似，跨模态未精同步，仅供队列内方向参考"},
    "movement_beta_power_change": {"name": "运动相关 β 功率变化", "unit": "相对变化", "group": "eeg",
                                   "note": "以 EMG 包络划窗近似，跨模态未精同步，仅供队列内方向参考"},
    # --- IMU (运动学) — 6 项 ---
    "movement_smoothness_sparc": {"name": "运动平滑度（SPARC）", "unit": "SPARC", "group": "imu",
                                  "note": "计于加速度模值，与文献速度-SPARC 尺度不同，仅供方向参考"},
    "range_of_motion_proxy": {"name": "关节活动度代理（角速度范围）", "unit": "deg/s(p2p)", "group": "imu",
                              "note": "基于 IMU 陀螺角速度的设备特异估计量"},
    "tremor_index_3_6hz": {"name": "震颤指数（3–6Hz 相对功率）", "unit": "相对功率", "group": "imu"},
    "wrist_flexion_peak_velocity": {"name": "腕屈方向峰值角速度", "unit": "deg/s", "group": "imu",
                                    "note": "ECU 处陀螺主轴 0.3Hz 高通去偏后取负向 |p5|；设备特异估计量"},
    "wrist_extension_peak_velocity": {"name": "腕伸方向峰值角速度", "unit": "deg/s", "group": "imu",
                                      "note": "ECU 处陀螺主轴 0.3Hz 高通去偏后取正向 p95；设备特异估计量"},
    "finger_extension_peak_velocity": {"name": "伸指峰值角速度", "unit": "deg/s", "group": "imu",
                                       "note": "基于 IMU 陀螺角速度的设备特异估计量"},
}

_GROUP_ORDER = ("emg", "eeg", "imu")
_GROUP_LABELS = {
    "emg": "肌电标志物（基于本次主动动作评估）",
    "eeg": "脑电标志物（基于本次主动动作评估）",
    "imu": "运动学标志物（IMU）",
}


def _affected_side_code(side: Optional[str]) -> str:
    """Map patient.paralysis_side (左/右/L/R) → 'L'/'R' for the biomarker program."""
    s = str(side or "").strip().upper()
    if s.startswith("左") or s.startswith("L"):
        return "L"
    if s.startswith("右") or s.startswith("R"):
        return "R"
    # Unknown → default to 'R' (most common); biomarker program raises on bad input.
    return "R"


def _round(key: str, value: float) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    # Tiny EMG voltages (V scale) need scientific-friendly precision; ratios 3dp.
    if abs(value) < 1e-3 and value != 0:
        return float(f"{value:.3e}")
    if abs(value) >= 100:
        return round(float(value), 1)
    return round(float(value), 4)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def extract(
    eeg_paths: Sequence[Path],
    emg_paths: Sequence[Path],
    hand_function_stage: int,
    affected_side: Optional[str] = None,
    institution: str = "hospital",
) -> Dict[str, Any]:
    """Extract the 26 digital biomarkers across all trials, aggregated (nanmean).

    Delegates per-trial computation to ``biomarkers/biomarkers.py``
    (``compute_trial_biomarkers`` + ``aggregate_trials``) and decorates the
    result with Chinese names, units, EMG/EEG/IMU grouping and the literature
    reference-range display string.

    Returns::

        {
          "stage": int,                      # Brunnstrom stage (for report header only)
          "affected_side": "L"|"R",
          "groups": [
            {"key": "emg", "label": "...", "markers": [
                {"key","name","value","unit","ref_range","n_valid","note"?}, ...]},
            ...                              # eeg, imu
          ],
          "flat": {biomarker_key: value, ...},
        }

    Robust to per-trial failures: a trial that errors is skipped; if *all*
    trials fail, every value is None with references intact so the report still
    renders (the LLM / rule-engine notes the missing signals).
    """
    side = _affected_side_code(affected_side)

    per_trial: List[Dict[str, float]] = []
    for eeg_p, emg_p in zip(eeg_paths, emg_paths):
        try:
            raw = _load_raw_trial(Path(eeg_p), Path(emg_p), institution=institution)
            per_trial.append(bjh_bm.compute_trial_biomarkers(raw, side))
        except Exception as exc:  # noqa: BLE001 — one bad trial mustn't kill the report
            print(f"[biomarkers][warn] trial skipped ({eeg_p}): {exc}")
            continue

    if per_trial:
        agg = bjh_bm.aggregate_trials(per_trial)  # {name: {value, n_valid}}
    else:
        agg = {name: {"value": float("nan"), "n_valid": 0} for name in bjh_bm.BIOMARKER_NAMES}

    flat: Dict[str, Any] = {}
    groups_map: Dict[str, List[Dict[str, Any]]] = {g: [] for g in _GROUP_ORDER}
    missing_keys: List[str] = []
    n_available = 0
    for key, meta in _BIOMARKER_META.items():
        info = agg.get(key, {"value": float("nan"), "n_valid": 0})
        value = _round(key, info.get("value", float("nan")))
        n_valid = int(info.get("n_valid", 0))
        # A marker is "available" only if it produced a finite aggregate from at
        # least one trial. Device-format trials legitimately can't compute every
        # marker (coarser/unnamed montage) → those come back None/n_valid=0 and
        # are flagged unavailable so the report doesn't force a fabricated reading.
        available = value is not None and n_valid > 0
        flat[key] = value
        if available:
            n_available += 1
        else:
            missing_keys.append(key)
        marker = {
            "key": key,
            "name": meta["name"],
            "value": value if value is not None else "—",
            "unit": meta["unit"],
            "ref_range": ref_display(key),
            "n_valid": n_valid,
            "available": available,
        }
        if "note" in meta:
            marker["note"] = meta["note"]
        if not available:
            _unavail = "本次数据不足/该格式暂不支持，未解读"
            marker["note"] = f"{marker['note']}；{_unavail}" if marker.get("note") else _unavail
        groups_map[meta["group"]].append(marker)

    groups = [
        {"key": g, "label": _GROUP_LABELS[g], "markers": groups_map[g]}
        for g in _GROUP_ORDER
    ]
    return {
        "stage": int(hand_function_stage),
        "affected_side": side,
        "groups": groups,
        "flat": flat,
        "coverage": {
            "available": n_available,
            "total": len(_BIOMARKER_META),
            "missing_keys": missing_keys,
        },
    }


__all__ = ["extract"]
