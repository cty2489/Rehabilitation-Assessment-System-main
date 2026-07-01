"""合理性评估：把每个生物标志物与该病例临床标签做方向一致性核对。

核心思想：每个生物标志物对某个临床标签有*预期方向*（如 PAI 高 → 预期 FMA 低）。
我们在 S1–S5 真实队列内，比较该受试者「生物标志物百分位」与「由标签推出的预期
百分位」是否吻合，给出 合理/部分合理/不合理。

免责声明（务必随报告输出）：队列仅 5 名真实受试者，结果为*定性一致性核对*，
不是统计验证、更非临床诊断。判定词指「标签↔信号内部一致性」。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# MAS（hand_tone）序数映射；'1+' 介于 1 与 2 之间
MAS_ORDER = {"0": 0, "1": 1, "1+": 2, "2": 3, "3": 4, "4": 5}

DISCLAIMER = (
    "队列仅 5 名真实受试者（S1–S5），本结果为定性一致性核对，"
    "非统计验证、非临床诊断；判定词指标签与信号的内部一致性。"
)


@dataclass(frozen=True)
class BiomarkerSpec:
    name: str
    units: str
    site: str            # 部位（半球 / 肌肉），中文可读
    target_label: str    # FMA_UE | BI | hand_tone | hand_function
    expected_dir: int    # +1: 标志物高 -> 标签高；-1: 标志物高 -> 标签低
    rationale_zh: str


SPECS: Sequence[BiomarkerSpec] = (
    BiomarkerSpec("pathological_asymmetry_index", "比值[-1,1]", "受损 vs 健侧运动皮层",
                  "FMA_UE", -1, "PAI高=两半球不对称大=皮层损伤重 → 预期 FMA 低"),
    BiomarkerSpec("corticomuscular_coherence_beta", "相干[0,1]", "受损半球–患手主动肌",
                  "FMA_UE", +1, "CMC高=皮层-肌肉驱动强 → 预期 FMA 高"),
    BiomarkerSpec("resting_emg_level", "V(RMS)", "患手屈肌",
                  "hand_tone", +1, "静息EMG高=肌张力高 → 预期 MAS 等级高"),
    BiomarkerSpec("wrist_co_contraction_index", "比值[0,1]", "腕屈肌 FCR vs 腕伸肌 ECU",
                  "hand_function", -1, "腕屈/伸共收缩高=痉挛/拮抗 → 预期手功能分级低"),
    BiomarkerSpec("emg_activation_rms", "V(RMS)", "患手肌肉",
                  "FMA_UE", +1, "主动激活幅度高=自主驱动强 → 预期 FMA 高"),
    BiomarkerSpec("movement_smoothness_sparc", "SPARC", "手部 IMU",
                  "FMA_UE", +1, "SPARC高=运动更平滑 → 预期 FMA 高"),
    BiomarkerSpec("range_of_motion_proxy", "deg/s(p2p)", "手部 IMU 陀螺仪",
                  "FMA_UE", +1, "活动范围大 → 预期 FMA 高"),
    BiomarkerSpec("tremor_index_3_6hz", "相对功率", "手部 IMU 加速度",
                  "FMA_UE", -1, "3-6Hz震颤功率高 → 预期 FMA 低"),
)

SPEC_BY_NAME = {s.name: s for s in SPECS}


def _label_to_ordinal(label_name: str, value) -> float:
    """把标签转为可排序数值（hand_tone 用 MAS 序数）。"""
    if label_name == "hand_tone":
        return float(MAS_ORDER.get(str(value), np.nan))
    return float(value)


def _percentile_rank(value: float, pool: Sequence[float]) -> float:
    """value 在 pool 中的百分位 [0,1]（含自身；并列取平均秩）。"""
    arr = np.asarray([p for p in pool if np.isfinite(p)], dtype=np.float64)
    if arr.size == 0 or not np.isfinite(value):
        return float("nan")
    less = np.sum(arr < value)
    equal = np.sum(arr == value)
    return float((less + 0.5 * equal) / arr.size)


def _verdict(consistency: float, n_valid: int) -> str:
    if n_valid < 3:
        return "数据不足"
    if not np.isfinite(consistency):
        return "数据不足"
    if consistency >= 0.70:
        return "合理"
    if consistency >= 0.40:
        return "部分合理"
    return "不合理"


def assess_subject(
    subject_id: str,
    subject_biomarkers: Dict[str, dict],          # {name: {value, n_valid}}
    subject_labels: Dict[str, object],            # {FMA_UE, BI, hand_tone, hand_function}
    cohort_biomarker_values: Dict[str, List[float]],  # {name: [所有受试者的值]}
    cohort_label_values: Dict[str, List[float]],      # {label: [所有受试者的序数值]}
) -> Dict[str, dict]:
    """对一个受试者的每个生物标志物给出评估字典。"""
    out: Dict[str, dict] = {}
    for spec in SPECS:
        bm = subject_biomarkers.get(spec.name, {"value": float("nan"), "n_valid": 0})
        value, n_valid = bm["value"], bm["n_valid"]

        bpe = _percentile_rank(value, cohort_biomarker_values.get(spec.name, []))
        lab_ord = _label_to_ordinal(spec.target_label, subject_labels.get(spec.target_label))
        lpe = _percentile_rank(lab_ord, cohort_label_values.get(spec.target_label, []))

        expected_bpe = lpe if spec.expected_dir > 0 else (1.0 - lpe)
        if np.isfinite(bpe) and np.isfinite(expected_bpe):
            consistency = 1.0 - abs(bpe - expected_bpe)
        else:
            consistency = float("nan")

        verdict = _verdict(consistency, n_valid)
        rationale = spec.rationale_zh
        if np.isfinite(bpe) and np.isfinite(lpe):
            rationale += (f"。实测百分位={bpe:.0%}, 标签百分位={lpe:.0%}, "
                          f"一致性={consistency:.0%}（n={n_valid}）")
        elif n_valid < 3:
            rationale += f"。有效 trial 不足（n={n_valid}），不作判定"

        out[spec.name] = {
            "value": value,
            "units": spec.units,
            "site": spec.site,
            "target_label": spec.target_label,
            "expected_direction": spec.expected_dir,
            "n_valid_trials": n_valid,
            "cohort_percentile": None if not np.isfinite(bpe) else round(bpe, 3),
            "consistency": None if not np.isfinite(consistency) else round(consistency, 3),
            "verdict": verdict,
            "rationale": rationale,
        }
    return out
