"""Inference pipeline driver — wraps Deeplearning/ into an SSE-friendly flow.

Loads the served pretrained CMK-AGN models once at startup, then for every
session runs the 6 processing steps (parse → preprocess → alignment →
feature_extract → graph_fusion → inference) on a thread pool, pushing
fine-grained progress events onto a `queue.Queue` consumed by the SSE endpoint.

(CMK-AGN is the public-facing name; the internal backbone class is still
``ADKMDFANTriBackbone`` in Deeplearning/, kept as-is to stay bound to the
trained ``.pth`` checkpoints.)
"""
from __future__ import annotations

import importlib.util
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Wire up Deeplearning/ into the import path so we can reuse predict.py utils. #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DL_DIR = PROJECT_ROOT / "Deeplearning"
DL_MODEL_DIR = PROJECT_ROOT / "DL_model"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from alignment.tri_strategies import align_by_strategy_tri  # noqa: E402
from alignment.wby_dtw import WBYDTWConfig  # noqa: E402
from bjh_io.bjh_loader import (  # noqa: E402
    EEG_CHANNELS,
    EEG_CHANNELS_BDF_30,
    EEG_FS_DEFAULT,
    EMG_MUSCLES,
    IMU_AXES_PER_MUSCLE,
    load_bjh_trial,
)
from bjh_io.device_loader import load_device_trial  # noqa: E402
from clinical_model import ClinicalPredictionModel  # noqa: E402
from task_config import (  # noqa: E402
    ALL_TASK_NAMES,
    clip_regression,
    get_encoder,
    get_task,
)


# Served clinical tasks. Barthel Index (BI) is intentionally excluded from the
# online flow: it measures ADL independence, while this system focuses on upper
# limb/hand motor assessment from EEG/EMG/IMU signals.
SERVED_TASKS: Tuple[str, ...] = ("FMA_UE", "hand_tone", "hand_function")

CHECKPOINTS: Dict[str, Path] = {
    "FMA_UE": DL_MODEL_DIR / "FMA_UE_fold1.pth",
    "hand_tone": DL_MODEL_DIR / "hand_tone_fold2.pth",
    "hand_function": DL_MODEL_DIR / "hand_function_fold3.pth",
}

# Front-facing labels used in the SSE `prediction` event (matches design doc).
# Note: task keys hand_tone / hand_function are kept for backwards-compat with
# the trained checkpoints; clinically they are Hand MAS and Brunnstrom (hand).
PREDICTION_LABELS: Dict[str, Dict[str, Any]] = {
    "FMA_UE": {"label": "FMA手部分数", "range": "0–20"},
    "hand_tone": {"label": "手部肌张力 (Hand MAS)"},
    "hand_function": {"label": "Brunnstrom 分期 (手)"},
}

# --------------------------------------------------------------------------- #
# Physician-readable clinical reasoning for each predicted score.             #
# These translate the raw model output into a one-line clinical reading shown #
# to the rehab physician (semantics kept consistent with the frontend's       #
# HAND_TONE_DESC / BRUNNSTROM_DESC so the UI and the reasoning stay aligned).  #
# The reading lookup tables live in the torch-free `inference_readings` module #
# so report_builder.py can reuse them without importing this heavy stack.      #
# --------------------------------------------------------------------------- #
from inference_readings import (  # noqa: E402
    BRUNNSTROM_READING as _BRUNNSTROM_READING,
    HAND_TONE_READING as _HAND_TONE_READING,
)

# The backend biomarker module (``backend/biomarkers.py``) shares its name with
# the project-root ``biomarkers/`` package, so ``import biomarkers`` is ambiguous
# and can resolve to the wrong one. Load it once by explicit file path under a
# distinct module name so the lookup is deterministic regardless of sys.path.
_BACKEND_BIOMARKERS = None


def _load_backend_biomarkers():
    """Return ``backend/biomarkers.py`` loaded by file path (memoised)."""
    global _BACKEND_BIOMARKERS
    if _BACKEND_BIOMARKERS is None:
        bm_file = Path(__file__).resolve().parent / "biomarkers.py"
        spec = importlib.util.spec_from_file_location("backend_biomarkers", bm_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["backend_biomarkers"] = module
        spec.loader.exec_module(module)
        _BACKEND_BIOMARKERS = module
    return _BACKEND_BIOMARKERS


def _fma_reading(value: float) -> str:
    """FMA-UE hand subscore (0–20) → upper-limb motor impairment band."""
    v = float(value)
    if v <= 5:
        band = "重度上肢运动功能障碍，手部几乎无有效随意运动"
    elif v <= 10:
        band = "中重度上肢运动功能障碍，仅能完成少量粗大随意运动"
    elif v <= 15:
        band = "中度上肢运动功能障碍，存在部分分离运动但精细控制不足"
    else:
        band = "轻度上肢运动功能障碍，手部随意与分离运动大部分保留"
    return f"FMA手部评分 {v:.0f}/20 分，提示{band}"


def clinical_reasoning(task: str, value: Any) -> str:
    """Render a one-line physician-readable reading of a predicted score.

    Combines the actual predicted value with its clinical meaning so the
    reasoning shown to the physician is patient-specific rather than generic.
    """
    if task == "FMA_UE":
        return "临床推理 · " + _fma_reading(value)
    if task == "hand_tone":
        reading = _HAND_TONE_READING.get(str(value), "肌张力分级结果")
        return f"临床推理 · 手部肌张力 Hand MAS {value} 级，{reading}"
    if task == "hand_function":
        try:
            stage = int(value)
        except (TypeError, ValueError):
            stage = value  # type: ignore[assignment]
        reading = _BRUNNSTROM_READING.get(stage, "手功能分期结果")  # type: ignore[arg-type]
        return f"临床推理 · 手功能分期 Brunnstrom {value} 期，{reading}"
    return f"临床推理 · {task} = {value}"


# Default inference loader knobs, matching `predict.py` defaults so the saved
# checkpoints behave identically here.
SEQ_LEN = 256
DTW_LENGTH = 32
ALIGNMENT_MODE = "adk"

SENTINEL: Dict[str, Any] = {"__sentinel__": True}


# --------------------------------------------------------------------------- #
# Event helpers                                                               #
# --------------------------------------------------------------------------- #
def step_start(step: str, label: str) -> Dict[str, Any]:
    return {"type": "step_start", "step": step, "label": label}


def step_detail(step: str, detail: str) -> Dict[str, Any]:
    return {"type": "step_detail", "step": step, "detail": detail}


def step_done(step: str) -> Dict[str, Any]:
    return {"type": "step_done", "step": step}


def prediction_event(task: str, value: Any) -> Dict[str, Any]:
    info = PREDICTION_LABELS.get(task, {})
    event: Dict[str, Any] = {"type": "prediction", "task": task, "value": value}
    event.update(info)
    return event


def error_event(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": message}


# --------------------------------------------------------------------------- #
# Model registry — loaded once at app startup.                                #
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    name: str
    model: ClinicalPredictionModel
    task_type: str       # "regression" | "classification"
    encoder: Any = None  # LabelEncoder | None
    head_kind: str = "ce"  # "ce" | "corn" — governs classification decoding


def _load_one(name: str, ckpt: Path, device: torch.device) -> LoadedModel:
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint for task {name} not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("model_config", {})
    score_min = float(cfg.get("score_min", payload.get("score_min", 0.0)))
    score_max = float(cfg.get("score_max", payload.get("score_max", 0.0)))
    bin_step = float(cfg.get("bin_step", 0.0))
    # The classification head variant the checkpoint was trained with. CORN
    # ("corn") uses a multi-layer MLP head emitting K-1 conditional logits; "ce"
    # uses a plain Linear. Must match or load_state_dict fails (and decoding
    # below must match too).
    head_kind = cfg.get("head_kind", "ce")
    model = ClinicalPredictionModel(
        task_type=payload["task_type"],
        num_classes=payload.get("num_classes") or None,
        eeg_channels=cfg.get("eeg_channels", 32),
        emg_channels=cfg.get("emg_channels", 4),
        imu_channels=cfg.get("imu_channels", 24),
        f=cfg.get("feature", 48),
        te=cfg.get("task_emb", 12),
        p=cfg.get("dropout", 0.15),
        score_min=score_min,
        score_max=score_max,
        bin_step=bin_step,
        head_kind=head_kind,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)

    spec = get_task(name)
    encoder = get_encoder(name) if spec.task_type == "classification" else None
    return LoadedModel(
        name=name,
        model=model,
        task_type=spec.task_type,
        encoder=encoder,
        head_kind=head_kind,
    )


class ModelRegistry:
    """Holds every served model + the device they live on."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, LoadedModel] = {}

    def load_all(self) -> None:
        for name in SERVED_TASKS:
            self.models[name] = _load_one(name, CHECKPOINTS[name], self.device)

    def predict(self, name: str, eeg: torch.Tensor, emg: torch.Tensor, imu: torch.Tensor) -> Any:
        """Predict for ONE subject given aligned tri-modal trials (B, S, C, T).

        - regression  → float scalar (already clipped)
        - classification → original-class label (string for hand_tone, int for hand_function)
        """
        bundle = self.models[name]
        model = bundle.model
        device = self.device

        # Trial / task embedding indices: we lack the original manifest at inference,
        # so feed neutral defaults (0) — they live behind nn.Embedding clamping.
        b, s = eeg.shape[:2]
        task_id = torch.zeros((b, s), dtype=torch.long, device=device)
        trial_id = torch.arange(s, dtype=torch.long, device=device).expand(b, s).contiguous()

        with torch.no_grad():
            out = model(eeg.to(device), emg.to(device), imu.to(device), task_id, trial_id)

        spec = get_task(name)
        if spec.task_type == "regression":
            if isinstance(out, dict):
                out = out["pred"]
            raw = float(out.detach().cpu().numpy().mean())
            return clip_regression(name, raw)

        # Classification: decode the single-subject logits to a class index, then
        # map back to the original label. CORN heads emit K-1 *conditional*
        # logits, so argmax is meaningless — decode by the CORN cumprod rule
        # (matches train.py's _corn_decode); CE heads use plain argmax.
        if bundle.head_kind == "corn":
            cond = torch.sigmoid(out)            # P(y>k | y>k-1)
            cum = torch.cumprod(cond, dim=1)     # P(y>k)
            cls_idx = int((cum > 0.5).long().sum(dim=1).detach().cpu().numpy()[0])
        else:
            cls_idx = int(out.argmax(dim=1).detach().cpu().numpy()[0])
        assert bundle.encoder is not None
        return bundle.encoder.decode(cls_idx)


# --------------------------------------------------------------------------- #
# File validation                                                             #
# --------------------------------------------------------------------------- #
def _validate_eeg_bdf(path: Path) -> None:
    """Lightweight check that a real BDF recording is readable and has enough
    EEG channels — the heavy filter+resample read happens later in
    ``load_bjh_trial``/``load_eeg_bdf`` (cached)."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ValueError(
            f"未安装 mne，无法处理 .bdf 脑电文件 {path.name}，请先 pip install mne"
        ) from exc
    try:
        raw = mne.io.read_raw_bdf(str(path), preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"EEG文件 {path.name} 不是有效的 BDF 文件：{exc}") from exc
    # BDF path drops A1/A2 downstream; require the 30 motor/scalp channels.
    present = [c for c in EEG_CHANNELS_BDF_30 if c in raw.ch_names]
    if len(present) < len(EEG_CHANNELS_BDF_30):
        missing = [c for c in EEG_CHANNELS_BDF_30 if c not in raw.ch_names]
        raise ValueError(f"EEG文件 {path.name} 缺少通道 {missing[:3]} 等")


def _validate_eeg_columns(path: Path) -> None:
    if path.suffix.lower() == ".bdf":
        _validate_eeg_bdf(path)
        return
    import pandas as pd
    df = pd.read_csv(path, nrows=2)
    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"EEG文件 {path.name} 缺少通道 {missing[:3]} 等")


def _validate_emg_columns(path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(path, nrows=2)
    missing_emg = [m for m in EMG_MUSCLES if not any(m in c for c in df.columns)]
    if missing_emg:
        raise ValueError(f"EMG/IMU文件 {path.name} 缺少肌肉数据：{missing_emg[:2]}")


# --------------------------------------------------------------------------- #
# Pipeline entry point                                                        #
# --------------------------------------------------------------------------- #
def run_pipeline(
    eeg_paths: Sequence[Path],
    emg_paths: Sequence[Path],
    registry: ModelRegistry,
    q: "queue.Queue[Dict[str, Any]]",
    affected_side: Optional[str] = None,
    institution: str = "hospital",
) -> Dict[str, Any]:
    """Run the full 6-step inference pipeline and emit progress events.

    ``affected_side`` is the patient's paralysed side (左/右/L/R); it selects the
    affected hemisphere for the EEG biomarkers. ``institution`` ("hospital" |
    "device") selects the per-trial signal loader + column validators so the
    same pipeline serves both data formats. Returns a dict {task:
    prediction_value} for the served tasks.
    """
    try:
        return _run_pipeline_inner(eeg_paths, emg_paths, registry, q, affected_side, institution)
    except Exception as exc:  # noqa: BLE001
        q.put(error_event(f"推理失败：{exc}"))
        raise


def _run_pipeline_inner(
    eeg_paths: Sequence[Path],
    emg_paths: Sequence[Path],
    registry: ModelRegistry,
    q: "queue.Queue[Dict[str, Any]]",
    affected_side: Optional[str] = None,
    institution: str = "hospital",
) -> Dict[str, Any]:
    is_device = str(institution).lower() == "device"
    trial_loader = load_device_trial if is_device else load_bjh_trial
    trial_eeg_fs = 512.0 if is_device else EEG_FS_DEFAULT
    if len(eeg_paths) != len(emg_paths):
        raise ValueError(f"EEG 与 EMG 文件数量不匹配：{len(eeg_paths)} vs {len(emg_paths)}")
    if not eeg_paths:
        raise ValueError("未提供任何 trial 文件")

    n_trials = len(eeg_paths)

    # ── Step 1: parse & validate ──────────────────────────────────────────── #
    q.put(step_start("parse", "采集数据核验"))
    if is_device:
        # Device montage is coarser/unnamed — the hospital column validators don't
        # apply. The device loader raises a clear error on empty/placeholder files.
        q.put(step_detail(
            "parse",
            f"本次评估共纳入 {n_trials} 次动作采集（设备端格式）；核验脑电 BDF 与肌电/IMU 文件是否可读...",
        ))
    else:
        q.put(step_detail(
            "parse",
            f"本次评估共纳入 {n_trials} 次动作采集；核验脑电信号 {len(EEG_CHANNELS)} 个导联是否齐全...",
        ))
        for p in eeg_paths:
            _validate_eeg_columns(p)
        q.put(step_detail(
            "parse",
            f"核验上肢 {len(EMG_MUSCLES)} 块目标肌肉的肌电与运动传感数据是否完整...",
        ))
        for p in emg_paths:
            _validate_emg_columns(p)
    q.put(step_done("parse"))

    # ── Step 2: signal preprocessing ──────────────────────────────────────── #
    q.put(step_start("preprocess", "信号质量处理"))
    for line in (
        "脑电 · 去除基线漂移，滤除 50 Hz 工频干扰，保留与运动相关的脑电节律...",
        "脑电 · 对各导联做稳健归一化，抑制个别噪声导联对整体的影响...",
        "肌电 · 滤除工频干扰并提取肌肉激活包络，反映肌肉用力的强弱与时序...",
        "运动传感 · 去漂移并平滑处理，还原肢体实际运动轨迹...",
    ):
        q.put(step_detail("preprocess", line))

    trials: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for eeg_p, emg_p in zip(eeg_paths, emg_paths):
        sig = trial_loader(eeg_p, emg_p, eeg_fs=trial_eeg_fs, preprocess=True)
        trials.append((sig.eeg, sig.emg, sig.imu))
    q.put(step_done("preprocess"))

    # ── Step 3: tri-modal temporal alignment ──────────────────────────────── #
    q.put(step_start("alignment", "脑–肌–肢信号同步"))
    q.put(step_detail("alignment", "以肢体运动时长为基准，将脑电、肌电与运动信号截取到同一动作时间窗..."))
    q.put(step_detail("alignment", "对齐三路信号的时间进程，使「发出运动指令—肌肉发力—肢体运动」在同一时间轴上可比..."))

    cfg = WBYDTWConfig(output_length=SEQ_LEN, dtw_length=DTW_LENGTH, band_radius=0.15, alpha=0.7, beta=0.3)
    aligned_eeg: List[np.ndarray] = []
    aligned_emg: List[np.ndarray] = []
    aligned_imu: List[np.ndarray] = []
    for eeg, emg, imu in trials:
        a = align_by_strategy_tri(eeg, emg, imu, ALIGNMENT_MODE, cfg)
        aligned_eeg.append(a.eeg_aligned)
        aligned_emg.append(a.emg_aligned)
        aligned_imu.append(a.imu_aligned)
    q.put(step_done("alignment"))

    # ── Step 4: feature extraction (descriptive — actual conv happens inside the model) ──
    q.put(step_start("feature_extract", "运动功能特征提取"))
    for line in (
        "脑电 · 提取与运动准备、执行相关的脑电节律特征，反映中枢的运动意图...",
        "肌电 · 提取各目标肌肉的激活强度与发力时序特征，反映外周的肌肉执行能力...",
        "运动传感 · 提取上肢各节段的运动幅度与平稳度特征，反映实际运动表现...",
    ):
        q.put(step_detail("feature_extract", line))
    q.put(step_done("feature_extract"))

    # ── Step 5: cross-modal graph attention fusion ───────────────────────── #
    q.put(step_start("graph_fusion", "脑–肌–肢协同分析"))
    n_eeg = len(EEG_CHANNELS)
    n_emg = len(EMG_MUSCLES)
    n_imu = n_emg  # one IMU node per muscle (ACC + GYRO 六轴)
    for line in (
        f"将中枢与外周整合为一张运动通路网络：{n_eeg} 个脑电导联、{n_emg} 块目标肌肉、{n_imu} 个上肢运动节段...",
        "建立脑–肌对应关系，评估皮层运动指令能否有效下传并募集到相应肌肉...",
        "按解剖关系建立肌–肢对应关系，评估肌肉发力能否转化为有效的肢体运动...",
        "综合分析整条运动通路：判断「想动—肌肉收缩—肢体动起来」各环节是否衔接顺畅、薄弱环节在哪里...",
        "聚焦动作的关键时段（如发力与运动起始时刻），重点解读最能反映运动能力的片段...",
        "汇总脑、肌、肢三方面证据，形成对手部运动功能的整体判断，供下一步评分使用...",
    ):
        q.put(step_detail("graph_fusion", line))
    q.put(step_done("graph_fusion"))

    # Stack to (1, S, C, T) batch — one subject, S trials per bag.
    eeg_bag = torch.from_numpy(np.stack(aligned_eeg, axis=0)).unsqueeze(0).float()
    emg_bag = torch.from_numpy(np.stack(aligned_emg, axis=0)).unsqueeze(0).float()
    imu_bag = torch.from_numpy(np.stack(aligned_imu, axis=0)).unsqueeze(0).float()

    # ── Step 6: per-task inference ───────────────────────────────────────── #
    q.put(step_start("inference", "康复指标评估"))
    results: Dict[str, Any] = {}
    task_detail = {
        "FMA_UE": "正在评估 FMA 手部运动功能评分...",
        "hand_tone": "正在评估手部肌张力 (Hand MAS) 分级...",
        "hand_function": "正在评估手功能 Brunnstrom 分期...",
    }
    for task in SERVED_TASKS:
        q.put(step_detail("inference", task_detail[task]))
        value = registry.predict(task, eeg_bag, emg_bag, imu_bag)
        results[task] = value
        q.put(prediction_event(task, value))
        # Patient-specific clinical reading of the score just produced.
        q.put(step_detail("inference", clinical_reasoning(task, value)))
    q.put(step_done("inference"))

    # ── Digital biomarker extraction ─────────────────────────────────────── #
    # Re-derive clinical biomarkers (IEMG µV·s, ERD%, ROM°, …) from the RAW
    # signals (biomarkers.extract re-loads them pre-normalisation), using the
    # predicted Brunnstrom stage to pick the per-stage reference ranges. Failure
    # here must NOT break the 4-indicator result, so it's isolated.
    try:
        # NOTE: a plain ``import biomarkers`` is ambiguous — the project root's
        # ``biomarkers/`` *package* (the 26-formula program, no ``extract``) can
        # shadow this backend module ``backend/biomarkers.py`` depending on
        # sys.path order (e.g. report.py inserts the project root at sys.path[0]).
        # That collision surfaces as ``module 'biomarkers' has no attribute
        # 'extract'``. Load the backend module by explicit file path so the right
        # one always wins (same trick backend/biomarkers.py uses for the program).
        bm_mod = _load_backend_biomarkers()
        stage = int(results.get("hand_function", 3))
        bm = bm_mod.extract(list(eeg_paths), list(emg_paths), stage, affected_side,
                            institution=institution)
        results["_biomarkers"] = bm
        cov = bm.get("coverage") or {}
        if cov:
            # Surface how many of the 26 markers were actually computed so the UI
            # can show "N/26" — device bundles legitimately cover only a subset.
            q.put({"type": "biomarker_coverage",
                   "available": cov.get("available"),
                   "total": cov.get("total"),
                   "missing_keys": cov.get("missing_keys", [])})
        flat = bm.get("flat", {})
        q.put(step_detail(
            "feature_extract",
            f"数字生物标志物（26项）· 腕共收缩指数 {flat.get('wrist_co_contraction_index', '—')}、"
            f"指共收缩指数 {flat.get('finger_co_contraction_index', '—')}、"
            f"皮层-肌肉相干(β) {flat.get('corticomuscular_coherence_beta', '—')}、"
            f"运动平滑度SPARC {flat.get('movement_smoothness_sparc', '—')}、"
            f"屈/伸肌IEMG比 {flat.get('flexor_extensor_iemg_ratio', '—')}...",
        ))
    except Exception as exc:  # noqa: BLE001
        # Print the FULL traceback (not just the message) and surface a concise
        # reason on the SSE stream so a silent empty biomarker section never
        # happens unnoticed again.
        print(f"[biomarkers][warn] extraction failed: {exc}")
        traceback.print_exc()
        q.put(step_detail(
            "feature_extract",
            f"数字生物标志物提取失败（{exc}），本次报告将缺少该段落，请检查后端日志。",
        ))
        results["_biomarkers"] = None

    return results
