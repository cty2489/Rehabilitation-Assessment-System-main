"""临床可解释的三模态生物标志物计算（纯函数，无 I/O / argparse）。

输入信号均来自 :func:`analysis.common.io.load_trial_raw`：
    eeg  [T, 30]  已滤波重采样、未 z-score，fs≈500 Hz
    emg  [T, 4]   原始去 padding，绝对幅值保留（肌张力/震颤需要），fs≈1250 Hz
    imu  [T, 24]  原始去 padding，4 肌 × (ACC.X/Y/Z, GYRO.X/Y/Z)，fs≈148 Hz

轴约定：loader 返回 **时间优先 [T, C]**。本模块所有公开函数都以 [T, C]（或 1-D [T]）
为输入；内部需要逐通道时取 ``axis=0``。这里 *不* 复用 ``features._eeg_bandpower``
（它要求通道优先 [C, T] 且把 5 个频带 log1p 后展平），而是写聚焦的窄带 Welch，
返回线性物理量，便于解释与半球比值。

生理依据见 paper/Method.md 第 III-B 节（皮层 μ/β、半球不对称、皮层-肌肉相干、
前额叶 θ/β、半球间相干、运动相关 μ/β 功率变化、肌张力基线、共收缩、IEMG、
肌电爆发持续、运动平滑度/ROM/腕角速度/伸指速度/震颤）。

重要说明：raw 路径下 EEG 与 EMG 来自独立时钟设备、未做跨模态同步，故皮层-肌肉
相干（CMC）、半球间相干、运动相关 μ/β 功率变化的*绝对值*仅为近似，宜在队列内比较排名。
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import signal as scisig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _trapz(y: np.ndarray, x: Optional[np.ndarray] = None, dx: float = 1.0) -> np.floating:
    """Compatibility wrapper for NumPy 2.x, where np.trapz was removed."""
    if x is None:
        return np.trapezoid(y, dx=dx)
    return np.trapezoid(y, x)

# --- 通道 / 肌肉 / 频带常量 --------------------------------------------------- #
MOTOR_LEFT: Sequence[str] = ("C3", "FC3", "CP3")
MOTOR_RIGHT: Sequence[str] = ("C4", "FC4", "CP4")
MU_BETA_BAND = (8.0, 30.0)     # μ+β 感觉运动节律
MU_BAND = (8.0, 12.0)          # μ 节律
BETA_BAND = (13.0, 30.0)       # β 节律
THETA_BAND = (4.0, 8.0)        # θ 节律
CMC_BAND = (15.0, 30.0)        # β 皮层-肌肉相干
TREMOR_BAND = (3.0, 6.0)       # 病理性震颤
TREMOR_REF_BAND = (0.5, 15.0)  # 震颤相对功率的分母
PREFRONTAL: Sequence[str] = ("Fp1", "Fp2", "Fz")  # 前额叶 θ/β

# EMG 肌肉序（与 CSV 列顺序一致，已核对 BJH/EMG_new/S1_2_1.csv）
# 0 桡侧腕屈肌(FCR), 1 指浅屈肌(FDS, 取自掌长肌电极位), 2 尺侧腕伸肌(ECU), 3 指伸肌(Extensor Digitorum)
# 注：idx 1 通道在原始 CSV 标注为 Palmaris Longus（掌长肌），按临床约定统一命名/使用为指浅屈肌 FDS。
FCR_IDX = 0                 # 桡侧腕屈肌
FDS_IDX = 1                 # 指浅屈肌（掌长肌电极位）
ECU_IDX = 2                 # 尺侧腕伸肌
EXTENSOR_DIGITORUM_IDX = 3  # 指伸肌
FLEXORS = (0, 1)      # 桡侧腕屈肌, 指浅屈肌  -> 屈肌组
EXTENSORS = (2, 3)    # 尺侧腕伸肌, 指伸肌    -> 伸肌组

# 全部生物标志物名称（输出列顺序的单一真相来源）
BIOMARKER_NAMES = (
    "pathological_asymmetry_index",
    "corticomuscular_coherence_beta",
    "resting_emg_level",
    "wrist_co_contraction_index",
    "finger_co_contraction_index",
    "emg_activation_rms",
    "movement_smoothness_sparc",
    "range_of_motion_proxy",
    "tremor_index_3_6hz",
    # --- 新增 EMG ---
    "fcr_iemg",
    "fds_iemg",
    "ecu_iemg",
    "extensor_digitorum_iemg",
    "flexor_extensor_iemg_ratio",
    "emg_burst_duration",
    # --- 新增 EMG 频域（中位频率 MDF）---
    "fcr_mdf",
    "fds_mdf",
    "ecu_mdf",
    "extensor_digitorum_mdf",
    # --- 新增 EEG ---
    "prefrontal_theta_beta_ratio",
    "interhemispheric_motor_coherence",
    "movement_mu_power_change",
    "movement_beta_power_change",
    # --- 新增 IMU ---
    "wrist_flexion_peak_velocity",
    "wrist_extension_peak_velocity",
    "finger_extension_peak_velocity",
)


# --- 小工具 ------------------------------------------------------------------ #
def _channel_indices(eeg_channels: Sequence[str], names: Sequence[str]) -> List[int]:
    return [eeg_channels.index(n) for n in names if n in eeg_channels]


def _affected_hemisphere(affected_side: str) -> str:
    """受损半球 = 患手对侧。患手 'R' -> 'left' 皮层；'L' -> 'right'。"""
    s = str(affected_side).strip().upper()
    if s.startswith("R"):
        return "left"
    if s.startswith("L"):
        return "right"
    raise ValueError(f"未知 affected_side: {affected_side!r}（应为 'L' 或 'R'）")


def _hemisphere_channels(side: str) -> Sequence[str]:
    return MOTOR_LEFT if side == "left" else MOTOR_RIGHT


def _bandpower_1d(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """单通道窄带功率（Welch + 梯形积分）。x: 1-D [T]。"""
    n = x.shape[0]
    nperseg = int(min(n, max(64, 2 * fs)))
    freqs, psd = scisig.welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return float("nan")
    return float(_trapz(psd[mask], freqs[mask]))


def _envelope(x: np.ndarray, fs: float, lowpass: float = 6.0) -> np.ndarray:
    """整流 + 低通线性包络。x: 1-D [T]。"""
    rect = np.abs(np.asarray(x, dtype=np.float64))
    wn = min(0.99, lowpass / (fs / 2.0))
    b, a = scisig.butter(2, wn, btype="low")
    return scisig.filtfilt(b, a, rect)


# --- EEG（皮层）-------------------------------------------------------------- #
def hemisphere_mu_beta_power(
    eeg: np.ndarray, fs: float, eeg_channels: Sequence[str], side: str
) -> float:
    """某半球 3 个运动通道的 8–30 Hz 线性功率均值。eeg: [T, C]。"""
    eeg = np.asarray(eeg, dtype=np.float64)
    assert eeg.ndim == 2, f"eeg 应为 [T,C]，得到 {eeg.shape}"
    idx = _channel_indices(eeg_channels, _hemisphere_channels(side))
    if not idx:
        return float("nan")
    powers = [_bandpower_1d(eeg[:, i], fs, *MU_BETA_BAND) for i in idx]
    return float(np.nanmean(powers))


def pathological_asymmetry_index(
    eeg: np.ndarray, fs: float, eeg_channels: Sequence[str], affected_side: str
) -> float:
    """病理性半球不对称指数 PAI=(p_健-p_患)/(p_健+p_患)，∈[-1,1]，高=损伤重。

    说明：基于全段 μ/β 谱功率的静息态不对称，不是运动相关 ERD。
    """
    aff = _affected_hemisphere(affected_side)        # 受损半球
    unaff = "right" if aff == "left" else "left"
    p_aff = hemisphere_mu_beta_power(eeg, fs, eeg_channels, aff)
    p_unaff = hemisphere_mu_beta_power(eeg, fs, eeg_channels, unaff)
    denom = p_aff + p_unaff
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((p_unaff - p_aff) / denom)


# --- 皮层-肌肉相干（CMC）----------------------------------------------------- #
def corticomuscular_coherence(
    eeg: np.ndarray,
    eeg_fs: float,
    eeg_channels: Sequence[str],
    emg: np.ndarray,
    emg_fs: float,
    affected_side: str,
) -> float:
    """患侧半球运动通道 vs 患手主动肌整流 EMG 的 β 带（15–30 Hz）相干均值，∈[0,1]。

    EMG 整流后重采样到 eeg_fs，与 EEG 截到等长再算 magnitude-squared coherence。
    绝对值近似（跨模态未同步）；评估只用其队列排名。
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    emg = np.asarray(emg, dtype=np.float64)
    aff = _affected_hemisphere(affected_side)
    idx = _channel_indices(eeg_channels, _hemisphere_channels(aff))
    if not idx:
        return float("nan")
    eeg_motor = eeg[:, idx].mean(axis=1)                       # 1-D @ eeg_fs

    emg_rect = np.abs(emg[:, FLEXORS[0]])                      # 患手主动肌（屈肌）
    frac = Fraction(float(eeg_fs) / float(emg_fs)).limit_denominator(1000)
    emg_rs = scisig.resample_poly(emg_rect, frac.numerator, frac.denominator)

    n = int(min(eeg_motor.shape[0], emg_rs.shape[0]))
    if n < int(2 * eeg_fs):
        return float("nan")
    eeg_motor = eeg_motor[:n]
    emg_rs = emg_rs[:n]

    nperseg = int(min(n, eeg_fs))                             # 1 s 段 -> ~1 Hz 分辨率
    f, cxy = scisig.coherence(eeg_motor, emg_rs, fs=eeg_fs, nperseg=nperseg)
    mask = (f >= CMC_BAND[0]) & (f <= CMC_BAND[1])
    if not mask.any():
        return float("nan")
    return float(np.mean(cxy[mask]))


# --- EMG（外周 / 张力）------------------------------------------------------- #
def resting_emg_level(emg: np.ndarray, fs: float, muscle_idx: int) -> float:
    """静息肌电水平：包络最低 20% 样本的 RMS（绝对幅值，V 级）。高=张力高。"""
    env = _envelope(np.asarray(emg, dtype=np.float64)[:, muscle_idx], fs)
    thr = np.quantile(env, 0.20)
    rest = env[env <= thr]
    if rest.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(rest ** 2)))


def co_contraction_index(
    emg: np.ndarray, fs: float, agonists=FLEXORS, antagonists=EXTENSORS
) -> float:
    """共收缩指数 CCI=Σmin(归一主动肌, 归一拮抗肌)/Σmax(...)，∈[0,1]。高=痉挛/拮抗。

    agonists/antagonists 为肌肉列索引元组；可指定单块肌肉构成一对拮抗关系
    （如腕：FCR vs ECU；指：FDS vs 指伸肌）。
    """
    emg = np.asarray(emg, dtype=np.float64)

    def _norm_group(cols) -> np.ndarray:
        env = np.mean([_envelope(emg[:, c], fs) for c in cols], axis=0)
        rng = env.max() - env.min()
        if rng <= 0:
            return np.zeros_like(env)
        return (env - env.min()) / rng

    a = _norm_group(agonists)
    b = _norm_group(antagonists)
    denom = np.sum(np.maximum(a, b))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.minimum(a, b)) / denom)


def emg_activation_amplitude(emg: np.ndarray, fs: float) -> Dict[str, float]:
    """患手肌肉自主激活幅度：全段 RMS（均跨 4 肌）与包络动态范围 p95-p5。高=驱动强。"""
    emg = np.asarray(emg, dtype=np.float64)
    rms = float(np.sqrt(np.mean(emg ** 2)))
    envs = [_envelope(emg[:, c], fs) for c in range(emg.shape[1])]
    dyn = float(np.mean([np.quantile(e, 0.95) - np.quantile(e, 0.05) for e in envs]))
    return {"rms": rms, "dynamic_range": dyn}


def integrated_emg(emg: np.ndarray, fs: float, muscle_idx: int) -> float:
    """积分肌电 IEMG：整流后对时间梯形积分（∫|EMG|dt），单位 V·s。高=肌肉激活强。"""
    x = np.abs(np.asarray(emg, dtype=np.float64)[:, muscle_idx])
    if x.size < 2:
        return float("nan")
    return float(_trapz(x, dx=1.0 / fs))


def flexor_extensor_iemg_ratio(
    emg: np.ndarray, fs: float, flexors=FLEXORS, extensors=EXTENSORS
) -> float:
    """屈伸肌 IEMG 比 = Σ屈肌 IEMG / Σ伸肌 IEMG。>1 屈肌主导，<1 伸肌主导。"""
    flx = sum(integrated_emg(emg, fs, c) for c in flexors)
    ext = sum(integrated_emg(emg, fs, c) for c in extensors)
    if not np.isfinite(ext) or ext <= 0:
        return float("nan")
    return float(flx / ext)


def median_frequency(
    emg: np.ndarray, fs: float, muscle_idx: int, band: Tuple[float, float] = (20.0, 450.0)
) -> float:
    """肌电中位频率 MDF（Hz）：限带功率谱累计达 50% 总功率的频率。低=肌肉疲劳。

    band 上限按 Nyquist 截断；用 Welch 估功率谱后对累积功率线性插值取中位频率。
    """
    x = np.asarray(emg, dtype=np.float64)[:, muscle_idx]
    n = x.shape[0]
    if n < 4:
        return float("nan")
    nperseg = int(min(n, max(64, 2 * fs)))
    freqs, psd = scisig.welch(x, fs=fs, nperseg=nperseg)
    hi = min(band[1], fs / 2.0)
    mask = (freqs >= band[0]) & (freqs <= hi)
    f, p = freqs[mask], psd[mask]
    total = _trapz(p, f)
    if f.size < 2 or not np.isfinite(total) or total <= 0:
        return float("nan")
    cum = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2.0 * np.diff(f))])
    return float(np.interp(total / 2.0, cum, f))


def emg_burst_duration(emg: np.ndarray, fs: float, muscle_idx: int) -> Dict[str, float]:
    """肌电爆发持续时间：包络阈值（静息基线 p20 + 3·MAD）检测 burst。

    返回 {"mean_burst_s": 平均单次爆发时长(秒), "total_active_frac": 活动样本占比}。
    主标量取 mean_burst_s（高=持续性激活/张力倾向）。
    """
    env = _envelope(np.asarray(emg, dtype=np.float64)[:, muscle_idx], fs)
    baseline = np.quantile(env, 0.20)
    mad = np.median(np.abs(env - np.median(env))) + 1e-12
    thr = baseline + 3.0 * mad
    active = env > thr
    total_active_frac = float(np.mean(active))
    # 连续活动段（run-length）
    if not active.any():
        return {"mean_burst_s": 0.0, "total_active_frac": 0.0}
    edges = np.diff(active.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]
    durations = (ends - starts) / fs
    return {
        "mean_burst_s": float(np.mean(durations)) if durations.size else 0.0,
        "total_active_frac": total_active_frac,
    }


def prefrontal_theta_beta_ratio(
    eeg: np.ndarray, fs: float, eeg_channels: Sequence[str]
) -> float:
    """前额叶 θ(4–8)/β(13–30) 功率比。逐通道求比值后取均值。高=低觉醒/执行抑制。"""
    eeg = np.asarray(eeg, dtype=np.float64)
    idx = _channel_indices(eeg_channels, PREFRONTAL)
    if not idx:
        return float("nan")
    ratios = []
    for i in idx:
        theta = _bandpower_1d(eeg[:, i], fs, *THETA_BAND)
        beta = _bandpower_1d(eeg[:, i], fs, *BETA_BAND)
        if np.isfinite(theta) and np.isfinite(beta) and beta > 0:
            ratios.append(theta / beta)
    return float(np.mean(ratios)) if ratios else float("nan")


def interhemispheric_motor_coherence(
    eeg: np.ndarray, fs: float, eeg_channels: Sequence[str], band=BETA_BAND
) -> float:
    """健-患侧运动皮层相干：左右 motor cluster 均值信号的 β 带相干均值，∈[0,1]。

    左 cluster (C3/FC3/CP3) 与右 cluster (C4/FC4/CP4) 各取均值后算 magnitude-squared
    coherence。高=半球间耦合强（康复中常随功能恢复上升）。
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    li = _channel_indices(eeg_channels, MOTOR_LEFT)
    ri = _channel_indices(eeg_channels, MOTOR_RIGHT)
    if not li or not ri:
        return float("nan")
    left = eeg[:, li].mean(axis=1)
    right = eeg[:, ri].mean(axis=1)
    n = left.shape[0]
    if n < int(2 * fs):
        return float("nan")
    nperseg = int(min(n, fs))
    f, cxy = scisig.coherence(left, right, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return float("nan")
    return float(np.mean(cxy[mask]))


def movement_related_power_change(
    eeg: np.ndarray,
    eeg_fs: float,
    eeg_channels: Sequence[str],
    emg: np.ndarray,
    emg_fs: float,
    affected_side: str,
) -> Dict[str, float]:
    """运动相关 μ/β 功率变化：用 EMG 包络在试次内划分高/低活动窗，以低活动为基线。

    流程：① 患手主动肌(FCR)包络 → 阈值分高/低活动样本；② 按 fs 比例把窗映射到 EEG
    时间轴（跨模态未精同步，与 CMC 同一近似前提）；③ 患侧 motor cluster 在 μ(8–12)
    与 β(13–30) 的 Welch 功率：变化% = (高活动 - 低活动)/低活动。
    μ 通常去同步(负)，β 通常反弹(正)。绝对值仅供队列内排名。
    返回 {"movement_mu_power_change", "movement_beta_power_change"}。
    """
    nan = {"movement_mu_power_change": float("nan"),
           "movement_beta_power_change": float("nan")}
    eeg = np.asarray(eeg, dtype=np.float64)
    emg = np.asarray(emg, dtype=np.float64)
    aff = _affected_hemisphere(affected_side)
    idx = _channel_indices(eeg_channels, _hemisphere_channels(aff))
    if not idx:
        return nan
    eeg_motor = eeg[:, idx].mean(axis=1)
    n_eeg = eeg_motor.shape[0]

    # EMG 包络 -> 高/低活动掩码，映射到 EEG 时间索引
    env = _envelope(emg[:, FCR_IDX], emg_fs)
    hi_thr = np.quantile(env, 0.70)
    lo_thr = np.quantile(env, 0.30)
    t_eeg = np.arange(n_eeg) / eeg_fs
    env_at_eeg = np.interp(t_eeg, np.arange(env.size) / emg_fs, env)
    hi = eeg_motor[env_at_eeg >= hi_thr]
    lo = eeg_motor[env_at_eeg <= lo_thr]
    if hi.size < int(eeg_fs) or lo.size < int(eeg_fs):
        return nan

    out = {}
    for key, band in (("movement_mu_power_change", MU_BAND),
                      ("movement_beta_power_change", BETA_BAND)):
        p_hi = _bandpower_1d(hi, eeg_fs, *band)
        p_lo = _bandpower_1d(lo, eeg_fs, *band)
        if np.isfinite(p_hi) and np.isfinite(p_lo) and p_lo > 0:
            out[key] = float((p_hi - p_lo) / p_lo)
        else:
            out[key] = float("nan")
    return out


# --- IMU（运动学）----------------------------------------------------------- #
def _acc_magnitude(imu: np.ndarray, muscle_idx: int, fs: float) -> np.ndarray:
    """去重力后的加速度模值。imu: [T,24]，肌 m 的 acc=列 m*6:m*6+3。"""
    acc = np.asarray(imu, dtype=np.float64)[:, muscle_idx * 6: muscle_idx * 6 + 3]
    wn = min(0.99, 0.3 / (fs / 2.0))                          # ~0.3 Hz 高通去重力
    b, a = scisig.butter(2, wn, btype="high")
    acc = scisig.filtfilt(b, a, acc, axis=0)
    return np.linalg.norm(acc, axis=1)


def _gyro_magnitude(imu: np.ndarray, muscle_idx: int) -> np.ndarray:
    gyro = np.asarray(imu, dtype=np.float64)[:, muscle_idx * 6 + 3: muscle_idx * 6 + 6]
    return np.linalg.norm(gyro, axis=1)


def movement_smoothness_sparc(speed: np.ndarray, fs: float, fc: float = 10.0) -> float:
    """谱弧长 SPARC（Balasubramanian 2012）。高（趋近 0）=更平滑。speed: 1-D。"""
    speed = np.asarray(speed, dtype=np.float64)
    n = speed.shape[0]
    if n < 4:
        return float("nan")
    nfft = int(2 ** np.ceil(np.log2(n * 2)))
    mag = np.abs(np.fft.rfft(speed, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    if mag.max() <= 0:
        return float("nan")
    mag = mag / mag.max()
    # 截到 [0, fc] 且幅值高于阈值的连续频段
    cut = freq <= fc
    f = freq[cut]
    m = mag[cut]
    keep = m >= 0.05
    if keep.sum() < 2:
        return float("nan")
    f = f[keep]
    m = m[keep]
    df = (f[-1] - f[0]) / (len(f) - 1) if len(f) > 1 else 1.0
    arc = -np.sum(np.sqrt((df / (f[-1] - f[0] + 1e-12)) ** 2 + np.diff(m) ** 2))
    return float(arc)


def dimensionless_jerk(speed: np.ndarray, fs: float) -> float:
    """无量纲 jerk（交叉验证用）。高（趋近 0）=更平滑。"""
    speed = np.asarray(speed, dtype=np.float64)
    dt = 1.0 / fs
    jerk = np.gradient(speed, dt)
    peak = np.max(np.abs(speed))
    T = len(speed) * dt
    if peak <= 0 or T <= 0:
        return float("nan")
    integ = _trapz(jerk ** 2, dx=dt)
    return float(-np.log((T ** 3 / peak ** 2) * integ + 1e-12))


def range_of_motion_proxy(imu: np.ndarray, muscle_idx: int) -> float:
    """ROM 代理：陀螺仪幅值的 p98-p2（角速度范围）。"""
    g = _gyro_magnitude(imu, muscle_idx)
    return float(np.quantile(g, 0.98) - np.quantile(g, 0.02))


def finger_extension_peak_velocity(
    imu: np.ndarray, extensor_sensor_idx: int = EXTENSOR_DIGITORUM_IDX
) -> float:
    """伸指峰值角速度：指伸肌处传感器陀螺幅值 p95（deg/s）。高=伸指运动充分。"""
    g = _gyro_magnitude(imu, extensor_sensor_idx)
    return float(np.quantile(g, 0.95))


def _signed_dominant_gyro(imu: np.ndarray, muscle_idx: int, fs: float) -> np.ndarray:
    """取该传感器方差最大的陀螺轴，高通(~0.3Hz)去安装方向偏置，返回带符号角速度 1-D。

    原始陀螺各轴含较大的传感器安装方向 DC 偏置，去偏置后信号双极，正/负向分别对应
    腕的两个旋转方向（伸 / 屈）。
    """
    gyro = np.asarray(imu, dtype=np.float64)[:, muscle_idx * 6 + 3: muscle_idx * 6 + 6]
    dom = int(np.argmax(gyro.std(axis=0)))
    col = gyro[:, dom]
    wn = min(0.99, 0.3 / (fs / 2.0))                          # ~0.3 Hz 高通去偏置
    b, a = scisig.butter(2, wn, btype="high")
    return scisig.filtfilt(b, a, col)


def wrist_directional_peak_velocity(
    imu: np.ndarray, fs: float, muscle_idx: int = ECU_IDX
) -> Dict[str, float]:
    """腕屈/伸方向峰值角速度（deg/s）：主运动轴去偏置后取正向 p95 与负向 |p5|。

    返回 {"extension": 正向峰值(伸), "flexion": 负向峰值绝对值(屈)}。两方向幅值
    刻画屈/伸运动的对称性。
    """
    sig = _signed_dominant_gyro(imu, muscle_idx, fs)
    return {
        "extension": float(np.quantile(sig, 0.95)),
        "flexion": float(abs(np.quantile(sig, 0.05))),
    }


def tremor_amplitude(acc_mag: np.ndarray, fs: float) -> float:
    """3–6 Hz 相对功率（/0.5–15 Hz 总功率），幅值尺度鲁棒。高=震颤重。"""
    acc_mag = np.asarray(acc_mag, dtype=np.float64)
    band = _bandpower_1d(acc_mag, fs, *TREMOR_BAND)
    ref = _bandpower_1d(acc_mag, fs, *TREMOR_REF_BAND)
    if not np.isfinite(ref) or ref <= 0:
        return float("nan")
    return float(band / ref)


# --- trial / subject 聚合 ---------------------------------------------------- #
def compute_trial_biomarkers(raw: dict, affected_side: str) -> Dict[str, float]:
    """对单个 trial 的 raw 字典计算全部生物标志物，返回扁平 {name: value}。

    raw 形如 :func:`analysis.common.io.load_trial_raw` 的返回。
    EEG 相关项若缺失（如无 mne）会是 NaN，由上层标“不可用”。
    """
    eeg, eeg_fs, ch = raw["eeg"], raw["eeg_fs"], raw["eeg_channels"]
    emg, emg_fs = raw["emg"], raw["emg_fs"]
    imu, imu_fs = raw["imu"], raw["imu_fs"]

    # IMU：对患手 4 个传感器取平均（运动学整体）
    acc_mags = [_acc_magnitude(imu, m, imu_fs) for m in range(imu.shape[1] // 6)]
    acc_mag = np.mean(acc_mags, axis=0)
    rom_vals = [range_of_motion_proxy(imu, m) for m in range(imu.shape[1] // 6)]

    act = emg_activation_amplitude(emg, emg_fs)
    burst = emg_burst_duration(emg, emg_fs, FCR_IDX)
    mpc = movement_related_power_change(eeg, eeg_fs, ch, emg, emg_fs, affected_side)
    wrist_vel = wrist_directional_peak_velocity(imu, imu_fs)
    return {
        "pathological_asymmetry_index":
            pathological_asymmetry_index(eeg, eeg_fs, ch, affected_side),
        "corticomuscular_coherence_beta":
            corticomuscular_coherence(eeg, eeg_fs, ch, emg, emg_fs, affected_side),
        "resting_emg_level": resting_emg_level(emg, emg_fs, FLEXORS[0]),
        # 腕屈/伸：FCR vs ECU；指屈/伸：FDS vs 指伸肌
        "wrist_co_contraction_index":
            co_contraction_index(emg, emg_fs, agonists=(FCR_IDX,), antagonists=(ECU_IDX,)),
        "finger_co_contraction_index":
            co_contraction_index(emg, emg_fs, agonists=(FDS_IDX,),
                                 antagonists=(EXTENSOR_DIGITORUM_IDX,)),
        "emg_activation_rms": act["rms"],
        "movement_smoothness_sparc": movement_smoothness_sparc(acc_mag, imu_fs),
        "range_of_motion_proxy": float(np.nanmean(rom_vals)),
        "tremor_index_3_6hz": tremor_amplitude(acc_mag, imu_fs),
        # --- 新增 EMG（4 块肌肉积分肌电）---
        "fcr_iemg": integrated_emg(emg, emg_fs, FCR_IDX),
        "fds_iemg": integrated_emg(emg, emg_fs, FDS_IDX),
        "ecu_iemg": integrated_emg(emg, emg_fs, ECU_IDX),
        "extensor_digitorum_iemg":
            integrated_emg(emg, emg_fs, EXTENSOR_DIGITORUM_IDX),
        "flexor_extensor_iemg_ratio": flexor_extensor_iemg_ratio(emg, emg_fs),
        "emg_burst_duration": burst["mean_burst_s"],
        # --- 新增 EMG 频域（4 块肌肉中位频率 MDF）---
        "fcr_mdf": median_frequency(emg, emg_fs, FCR_IDX),
        "fds_mdf": median_frequency(emg, emg_fs, FDS_IDX),
        "ecu_mdf": median_frequency(emg, emg_fs, ECU_IDX),
        "extensor_digitorum_mdf":
            median_frequency(emg, emg_fs, EXTENSOR_DIGITORUM_IDX),
        # --- 新增 EEG ---
        "prefrontal_theta_beta_ratio":
            prefrontal_theta_beta_ratio(eeg, eeg_fs, ch),
        "interhemispheric_motor_coherence":
            interhemispheric_motor_coherence(eeg, eeg_fs, ch),
        "movement_mu_power_change": mpc["movement_mu_power_change"],
        "movement_beta_power_change": mpc["movement_beta_power_change"],
        # --- 新增 IMU（腕屈/伸方向峰值角速度，尺侧腕伸肌 ECU 处传感器）---
        "wrist_flexion_peak_velocity": wrist_vel["flexion"],
        "wrist_extension_peak_velocity": wrist_vel["extension"],
        "finger_extension_peak_velocity": finger_extension_peak_velocity(imu),
    }


def aggregate_trials(trial_dicts: List[Dict[str, float]]) -> Dict[str, dict]:
    """跨 trial 取 nanmean，返回 {name: {value, n_valid}}。"""
    out: Dict[str, dict] = {}
    for name in BIOMARKER_NAMES:
        vals = np.array([d.get(name, np.nan) for d in trial_dicts], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[name] = {
            "value": float(np.mean(finite)) if finite.size else float("nan"),
            "n_valid": int(finite.size),
        }
    return out


# --- 人口学（患侧 / 年龄 / 病种 / 病程）-------------------------------------- #
_REHAB_JSON = _REPO_ROOT / "patient_rehab_suggestions_15subjects.json"


def load_demographics(subject_id: str) -> dict:
    """读取 patient_rehab_suggestions_15subjects.json 的 demographics。"""
    num = str(subject_id).lstrip("S")
    with _REHAB_JSON.open("r", encoding="utf-8") as fh:
        subs = json.load(fh)["subjects"]
    if num not in subs:
        raise KeyError(f"无 {subject_id} 的人口学信息")
    return subs[num]["demographics"]
