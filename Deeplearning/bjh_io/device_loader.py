"""Signal loader for the **device** (wearable) evaluation format.

The hospital format is handled by ``bjh_loader.load_bjh_trial`` (Delsys 56-column
EMG/IMU csv, 32-ch BDF). The wearable device emits a *different* bundle and this
module adapts it onto the **same** ``TriModalSignals`` contract the rest of the
pipeline (alignment → model, biomarkers) already expects:

    emg [T, 4]   imu [T, 24]   eeg [T, 30]

Device raw format (verified against patient_P001_eval_20260629_test, 2026-06):
* ``trial_*_emg_imu.csv`` — one header row, columns::
      EMG采样时间点, EMG通道1..EMG通道8,
      IMU采样时间点, IMU加速度计X/Y/Z, IMU陀螺仪X/Y/Z, IMU预留1/2/3
  i.e. 8 EMG channels (200 Hz) + a single 6-axis IMU (50 Hz, named ACC/GYRO axes
  plus 3 empty reserved columns). EMG and IMU share the file but sample at
  DIFFERENT rates, so each modality has its own timestamp column and the slower
  IMU rows are NaN-padded at the tail — they must be stripped per modality.
* ``trial_*_eeg.bdf`` — 8-channel BDF, 512 Hz, named 10-20 electrodes
  (FP1,FP2,F3,F4,FC3,FC4,C3,C4), all of which exist in the hospital 30-channel
  layout, so they are placed at their correct indices (not blindly front-filled).

⚠️ MAPPING IS BEST-EFFORT AND CENTRALISED HERE. The DL models + biomarker
formulas were designed on the hospital montage (4 named forearm muscles, 6-axis
IMU *per muscle*, 30 scalp EEG channels). The device montage is coarser, so two
structural approximations remain (documented at their call sites): the 8 EMG
channels are unnamed → we take the first 4 as the model's muscle slots; the
single 6-axis IMU is replicated across the 4 muscle slots; 22 of 30 EEG
electrodes are absent → zero-filled. Predictions/biomarkers from device data
must be re-validated once the device channel→muscle spec is available.
See memory: inference-pipeline-config-mismatch.

Older column names (肌电采样时间点 / 通道N / IMUN) from the first placeholder
sample are still accepted via candidate-name resolution, so both parse.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from device_timebase import infer_device_timebase

from .bjh_loader import (
    EEG_CHANNELS_BDF_30,
    EEG_FS_BDF_OUT,
    MOTOR_CHANNELS_FOR_SYNC,
    SYNC_ENV_FS,
    SYNC_MAX_LAG_S,
    SYNC_MIN_PEAK_CORR,
    TriModalSignals,
    _emg_sync_envelope,
    _eeg_mu_beta_envelope,
    _strip_padding,
    _zscore_clip,
    crop_to_common_window,
    estimate_lag_xcorr,
    preprocess_emg,
    preprocess_imu,
)

# Device defaults (used only if the timestamp columns can't yield a rate).
DEVICE_EMG_FS_DEFAULT = 200.0
DEVICE_IMU_FS_DEFAULT = 50.0
DEVICE_EEG_FS_IN = 512.0

# Number of model EMG slots / IMU muscle-slots to fill (hospital contract).
_N_EMG_SLOTS = 4
_N_IMU_MUSCLES = 4

# Column-name candidates (new named format first, old placeholder names second).
_EMG_TIME_CANDIDATES = ("EMG采样时间点", "肌电采样时间点")
_IMU_TIME_CANDIDATES = ("IMU采样时间点",)
# EMG channel i: try "EMG通道{i}" then "通道{i}".
_EMG_CH_CANDIDATES = [(f"EMG通道{i}", f"通道{i}") for i in range(1, 9)]
# IMU 6 motion axes (ACC X/Y/Z + GYRO X/Y/Z), named in the new format. The old
# placeholder used IMU1..6 for the same six axes.
_IMU_AXIS_CANDIDATES = [
    ("IMU加速度计X", "IMU1"),
    ("IMU加速度计Y", "IMU2"),
    ("IMU加速度计Z", "IMU3"),
    ("IMU陀螺仪X", "IMU4"),
    ("IMU陀螺仪Y", "IMU5"),
    ("IMU陀螺仪Z", "IMU6"),
]


def _resolve_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column name present in ``df`` (or None)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


_timebase_from_time = infer_device_timebase


def _fs_from_time(t: np.ndarray, default: float) -> float:
    """Backward-compatible sampling-rate helper."""
    return _timebase_from_time(t, default)[0]


def load_device_emg_imu(
    path: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Parse the device EMG/IMU csv → ``(emg[T,4], imu[T,24], t_emg, t_imu, meta)``.

    EMG (200 Hz) and IMU (50 Hz) live in one csv with separate timestamp columns;
    the slower IMU rows are NaN-padded at the tail. We strip padding per modality
    (via ``_strip_padding``) so each comes out at its native length/rate.

    Mappings (documented assumptions, see module docstring):
    * EMG: 8 unnamed channels → take the first 4 as the model muscle slots
      (FCR/FDS/ECU/ED order). Awaiting the device channel→muscle spec.
    * IMU: one 6-axis IMU (ACC X/Y/Z + GYRO X/Y/Z) replicated across the 4 muscle
      slots → [T, 24] to match the hospital per-muscle IMU contract.
    Raises a clear error on empty placeholder files.
    """
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[0] == 0:
        raise ValueError(
            f"设备端 EMG/IMU 文件为空占位（仅表头，0 行）：{path.name}；请提供真实采集数据"
        )

    # ---- resolve columns (new named format, fall back to old placeholder names) ----
    emg_time_col = _resolve_col(df, _EMG_TIME_CANDIDATES)
    imu_time_col = _resolve_col(df, _IMU_TIME_CANDIDATES)
    emg_cols: List[str] = [c for c in (_resolve_col(df, cand) for cand in _EMG_CH_CANDIDATES) if c]
    imu_axis_cols: List[str] = [c for c in (_resolve_col(df, cand) for cand in _IMU_AXIS_CANDIDATES) if c]
    if emg_time_col is None or not emg_cols:
        raise ValueError(f"设备端 EMG/IMU 文件 {path.name} 缺少 EMG 时间或通道列")
    if imu_time_col is None or len(imu_axis_cols) < 6:
        raise ValueError(f"设备端 EMG/IMU 文件 {path.name} 缺少 IMU 时间或 6 轴数据列")

    # ---- EMG (strip its own padding) ----
    t_emg_raw = df[emg_time_col].to_numpy(dtype=np.float64, copy=False)
    emg_all = np.stack([df[c].to_numpy(dtype=np.float32, copy=False) for c in emg_cols], axis=1)
    emg_all, t_emg = _strip_padding(emg_all, t_emg_raw)
    emg = emg_all[:, :_N_EMG_SLOTS]                       # 8→4 (documented assumption)

    # ---- IMU (slower rate → NaN-padded tail; strip its own padding) ----
    t_imu_raw = df[imu_time_col].to_numpy(dtype=np.float64, copy=False)
    imu6_raw = np.stack([df[c].to_numpy(dtype=np.float32, copy=False) for c in imu_axis_cols[:6]], axis=1)
    imu6, t_imu = _strip_padding(imu6_raw, t_imu_raw)
    imu = np.tile(imu6, (1, _N_IMU_MUSCLES))             # replicate 6 axes across 4 slots → 24

    emg_fs, emg_time_scale, emg_time_unit = _timebase_from_time(
        t_emg, DEVICE_EMG_FS_DEFAULT
    )
    imu_fs, imu_time_scale, imu_time_unit = _timebase_from_time(
        t_imu, DEVICE_IMU_FS_DEFAULT
    )
    meta = {
        "emg_fs": emg_fs,
        "imu_fs": imu_fs,
        "emg_time_unit": emg_time_unit,
        "imu_time_unit": imu_time_unit,
        "duration_emg": (
            float(t_emg[-1] - t_emg[0]) * emg_time_scale if t_emg.size else 0.0
        ),
        "duration_imu": (
            float(t_imu[-1] - t_imu[0]) * imu_time_scale if t_imu.size else 0.0
        ),
        "n_emg_samples": int(emg.shape[0]),
        "n_imu_samples": int(imu.shape[0]),
        "n_emg_channels_raw": int(emg_all.shape[1]),
        "emg_columns": emg_cols,
        "imu_columns": imu_axis_cols[:6],
        "format": "device",
    }
    return emg, imu, t_emg, t_imu, meta


def load_device_eeg_bdf(
    path: Path | str,
    target_fs: float = EEG_FS_BDF_OUT,
    bandpass: Tuple[float, float] = (1.5, 50.0),
    notch: Optional[float] = 50.0,
) -> Tuple[np.ndarray, float, Sequence[str]]:
    """Load the device's 8-channel BDF, filter+resample, and place each electrode
    into the 30-wide hospital EEG channel layout *by name* (absent slots zeroed).

    The device electrodes (FP1,FP2,F3,F4,FC3,FC4,C3,C4) all exist in
    ``EEG_CHANNELS_BDF_30``, so they land at their correct indices (incl. motor
    channels C3/C4) and the returned channel-name list is the full 30-channel
    layout — downstream code that indexes channels by name (e.g.
    ``MOTOR_CHANNELS_FOR_SYNC``, EEG biomarkers) therefore finds the real signals.
    The remaining ~22 electrodes are absent and stay zero-filled (approximation).
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise ValueError(f"未安装 mne，无法读取设备端 BDF：{path}") from exc

    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError(f"设备端 EEG 文件为空占位（0 字节）：{path.name}；请提供真实采集数据")

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    if notch is not None and float(notch) > 0.0:
        raw.notch_filter(freqs=[float(notch)], verbose="ERROR")
    raw.filter(l_freq=float(bandpass[0]), h_freq=float(bandpass[1]), verbose="ERROR")
    if float(target_fs) > 0.0 and abs(raw.info["sfreq"] - float(target_fs)) > 1e-6:
        raw.resample(float(target_fs), npad="auto", verbose="ERROR")

    data = raw.get_data().T.astype(np.float32)  # [T, n_dev]
    fs = float(raw.info["sfreq"])

    target = list(EEG_CHANNELS_BDF_30)
    idx_by_upper = {c.upper(): i for i, c in enumerate(target)}
    eeg = np.zeros((data.shape[0], len(target)), dtype=np.float32)
    unmatched: List[str] = []
    for dev_i, name in enumerate(raw.ch_names):
        ti = idx_by_upper.get(str(name).upper())
        if ti is None:
            unmatched.append(name)
            continue
        eeg[:, ti] = data[:, dev_i]
    if unmatched:
        print(f"[device_loader][warn] EEG 电极名未匹配到 30 导联布局，已忽略：{unmatched}")
    return eeg, fs, tuple(target)


def load_device_trial(
    eeg_path: Path | str,
    emg_path: Path | str,
    eeg_fs: float = DEVICE_EEG_FS_IN,
    preprocess: bool = True,
    eeg_kwargs: Optional[dict] = None,
    emg_kwargs: Optional[dict] = None,
    imu_kwargs: Optional[dict] = None,
) -> TriModalSignals:
    """Device-format counterpart of ``bjh_loader.load_bjh_trial`` (BDF branch).

    Same return contract (``TriModalSignals`` with eeg[T,30]/emg[T,4]/imu[T,24]),
    so it drops straight into ``run_pipeline``'s alignment + model stages.
    """
    eeg_kwargs = eeg_kwargs or {}
    emg_kwargs = emg_kwargs or {}
    imu_kwargs = imu_kwargs or {}

    emg_raw, imu_raw, _t_emg, _t_imu, meta = load_device_emg_imu(emg_path)
    emg_fs = float(meta["emg_fs"])
    imu_fs = float(meta["imu_fs"])

    eeg_proc, eeg_fs_out, eeg_channels = load_device_eeg_bdf(eeg_path, **eeg_kwargs)

    if preprocess:
        emg_proc = preprocess_emg(emg_raw, emg_fs, **emg_kwargs)
        imu_proc = preprocess_imu(imu_raw, imu_fs, **imu_kwargs)
    else:
        emg_proc = emg_raw.astype(np.float32, copy=False)
        imu_proc = imu_raw.astype(np.float32, copy=False)

    emg_env = _emg_sync_envelope(emg_proc, emg_fs, SYNC_ENV_FS)
    eeg_env = _eeg_mu_beta_envelope(eeg_proc, eeg_fs_out, eeg_channels,
                                    MOTOR_CHANNELS_FOR_SYNC, SYNC_ENV_FS)
    lag_s, peak_corr = estimate_lag_xcorr(emg_env, eeg_env, SYNC_ENV_FS, SYNC_MAX_LAG_S)
    if not np.isfinite(peak_corr) or peak_corr < SYNC_MIN_PEAK_CORR:
        sync_fallback = True
        lag_for_crop = 0.0
    else:
        sync_fallback = False
        lag_for_crop = lag_s

    eeg_c, emg_c, imu_c, t0_eeg, t0_peripheral, common = crop_to_common_window(
        eeg_proc, eeg_fs_out, emg_proc, emg_fs, imu_proc, imu_fs, lag_for_crop,
    )

    eeg_out = _zscore_clip(eeg_c) if preprocess else eeg_c.astype(np.float32, copy=False)

    return TriModalSignals(
        eeg=eeg_out,
        emg=emg_c.astype(np.float32, copy=False),
        imu=imu_c.astype(np.float32, copy=False),
        eeg_fs=eeg_fs_out,
        emg_fs=emg_fs,
        imu_fs=imu_fs,
        duration=common,
        metadata={
            "eeg_path": str(eeg_path),
            "emg_path": str(emg_path),
            "eeg_format": "bdf",
            "format": "device",
            "eeg_channels": list(eeg_channels),
            "sync_lag_s": float(lag_s),
            "sync_peak_corr": float(peak_corr) if np.isfinite(peak_corr) else float("nan"),
            "sync_fallback": bool(sync_fallback),
            "common_window_s": float(common),
            "preprocessed": bool(preprocess),
            **{k: v for k, v in meta.items()},
        },
    )


__all__ = [
    "load_device_emg_imu",
    "load_device_eeg_bdf",
    "load_device_trial",
]
