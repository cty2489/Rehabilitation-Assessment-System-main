"""生成生物标志物参考范围 JSON（严格文献依据，不做本队列经验统计）。

为 `biomarkers.BIOMARKER_NAMES` 的每一项给出参考范围条目，分层优先 Brunnstrom；
无 Brunnstrom 分层依据时给健康成人常模；多数项为本研究设备/协议特异量，文献无标准
范围，显式标注 reference_type="none"（不编造数值）。

凡有真实文献依据的范围，逐项标注 `source`（与顶层 `references` 一一对应）。

用法：
    python analysis/02_biomarkers/build_reference_ranges.py
输出：
    analysis/02_biomarkers/out/biomarker_reference_ranges.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import biomarkers as bm  # noqa: E402
from reasonableness import SPEC_BY_NAME  # noqa: E402

OUT_PATH = _HERE / "out" / "biomarker_reference_ranges.json"

# --- 文献库（URL/标识在检索中核实）------------------------------------------- #
REFERENCES = [
    {
        "id": "Saes2021",
        "citation": ("Saes M, Mohamed Refai MI, van Beijnum BJF, et al. Smoothness "
                     "metric during reach-to-grasp after stroke: part 2. longitudinal "
                     "association with motor impairment. J NeuroEng Rehabil. 2021;18:144."),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8461930/",
        "pmcid": "PMC8461930",
    },
    {
        "id": "Banks2017",
        "citation": ("Banks CL, Pai MM, McGuirk TE, et al. Electromyography Exposes "
                     "Heterogeneity in Muscle Co-Contraction following Stroke. "
                     "Front Neurol. 2017;8:699."),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5743661/",
        "pmcid": "PMC5743661",
    },
    {
        "id": "Yao2022CC",
        "citation": ("Yao J, et al. Upper Limbs Muscle Co-contraction Changes "
                     "Correlated With the Impairment of the Corticospinal Tract in "
                     "Stroke Survivors. Front Neurosci. 2022;16:886909."),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9198335/",
        "pmcid": "PMC9198335",
    },
    {
        "id": "CCISystReview2025",
        "citation": ("Assessing stroke-induced abnormal muscle coactivation in the "
                     "upper limb using the surface EMG co-contraction Index: A "
                     "systematic review. 2025."),
        "url": "https://pubmed.ncbi.nlm.nih.gov/39847816/",
        "pmid": "39847816",
    },
    {
        "id": "Liu2019CMC",
        "citation": ("Liu J, Sheng Y, Liu H. Corticomuscular Coherence and Its "
                     "Applications: A Review. Front Hum Neurosci. 2019;13:100."),
        "url": "https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00100/full",
        "doi": "10.3389/fnhum.2019.00100",
    },
    {
        "id": "AgiusAnastasi2017",
        "citation": ("Agius Anastasi A, Falzon O, Camilleri K, et al. Brain Symmetry "
                     "Index in Healthy and Stroke Patients for Assessment and "
                     "Prognosis. Stroke Res Treat. 2017;2017:8276136."),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5304313/",
        "pmcid": "PMC5304313",
    },
    {
        "id": "Angelidis2016TBR",
        "citation": ("Angelidis A, van der Does W, Schakel L, Putman P. Frontal EEG "
                     "theta/beta ratio as an electrophysiological marker for "
                     "attentional control and its test-retest reliability. Biol "
                     "Psychol. 2016;121:49-52."),
        "url": "https://pubmed.ncbi.nlm.nih.gov/27697551/",
        "pmid": "27697551",
    },
    {
        "id": "Wang2019Tremor",
        "citation": ("A Power Spectral Density-Based Method to Detect Tremor and "
                     "Tremor Intermittency in Movement Disorders. Sensors. 2019."),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6806079/",
        "pmcid": "PMC6806079",
    },
]
_REF_IDS = {r["id"] for r in REFERENCES}

# --- 逐标志物参考范围条目 ----------------------------------------------------- #
# reference_type: "healthy_norm"（有健康常模数值）/ "directional_trend"（仅方向）/
#                 "none"（设备/协议特异量，文献无标准范围）。
NONE_NOTE_EMG = ("本研究设备/协议特异的 EMG 量（原始电压/IEMG、共收缩指数、中位频率 MDF），"
                 "随设备增益、电极位置与肌对定义而变，文献无统一标准参考范围；仅供队列内方向比较。")
NONE_NOTE_IMU = "基于 IMU 陀螺角速度（deg/s）的设备特异量，且真实与模拟数据相差 2–3 个数量级，文献无标准参考范围。"
NONE_NOTE_EEGMOV = "运动相关 μ/β 功率变化以 EMG 包络划窗、EEG 与 EMG 跨模态未精同步，绝对值仅供队列内排名，文献无标准参考范围。"

ENTRIES = {
    # ---- 有健康常模数值 ----
    "movement_smoothness_sparc": {
        "reference_type": "healthy_norm",
        "healthy_reference": {"mean": -1.436, "sd": 0.038, "range": [-1.51, -1.36]},
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "increase",
        "source": ["Saes2021"],
        "confidence": "low",
        "note_zh": ("健康对照速度-SPARC ≈ -1.436(SD .038)；卒中急性期 ≈ -1.72→慢性期 -1.48，"
                    "随恢复升高趋近 0（Saes2021）。⚠️ 本模块 SPARC 计于加速度模值（取值 ~-2…-8），"
                    "与文献速度-SPARC 尺度不同，数值不可直接套用，仅供方向参考。"),
    },
    # ---- 仅方向性 ----
    "corticomuscular_coherence_beta": {
        "reference_type": "directional_trend",
        "healthy_reference": None,
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "increase",
        "source": ["Liu2019CMC"],
        "confidence": "low",
        "note_zh": ("β带(15–30Hz)皮层-肌肉相干在健康者高于卒中，且与 FMA-UE 正相关、随运动功能"
                    "恢复上升；文献无统一绝对阈值（Liu2019CMC）。本模块跨模态未精同步，绝对值仅队列内排名。"),
    },
    "pathological_asymmetry_index": {
        "reference_type": "directional_trend",
        "healthy_reference": {"mean": 0.0, "sd": None, "range": None},
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "decrease",
        "source": ["AgiusAnastasi2017"],
        "confidence": "low",
        "note_zh": ("对应脑对称性指数(BSI)：健康者两半球≈对称(趋近0)，卒中 BSI 升高且与 FMA 相关"
                    "（AgiusAnastasi2017）。本模块 PAI∈[-1,1]，健康预期≈0，随恢复绝对值趋近 0；无逐期阈值。"),
    },
    "interhemispheric_motor_coherence": {
        "reference_type": "directional_trend",
        "healthy_reference": None,
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "increase",
        "source": ["AgiusAnastasi2017"],
        "confidence": "low",
        "note_zh": ("半球间运动皮层耦合在康复中常随功能恢复上升（与半球间平衡/对称性恢复相关，"
                    "AgiusAnastasi2017）；文献无绝对阈值。"),
    },
    "prefrontal_theta_beta_ratio": {
        "reference_type": "directional_trend",
        "healthy_reference": None,
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "n/a",
        "source": ["Angelidis2016TBR"],
        "confidence": "low",
        "note_zh": ("额叶 θ(4–8)/β(13–30) 比率个体差异大、test-retest r≈.93，但无公认正常数值区间；"
                    "高 TBR 关联较弱的注意/执行控制（低觉醒）（Angelidis2016TBR）。仅方向参考，无逐期阈值。"),
    },
    "tremor_index_3_6hz": {
        "reference_type": "directional_trend",
        "healthy_reference": None,
        "by_brunnstrom": None,
        "expected_direction_with_recovery": "decrease",
        "source": ["Wang2019Tremor"],
        "confidence": "low",
        "note_zh": ("健康生理性震颤在 3–6 Hz 的相对功率低，病理性/异常震颤相对功率更高（Wang2019Tremor）；"
                    "文献无逐期绝对阈值，健康偏低、随损伤加重升高。"),
    },
}

# ---- 无文献标准范围的项（统一 none）---- #
_NONE_EMG = ("resting_emg_level", "wrist_co_contraction_index",
             "finger_co_contraction_index", "emg_activation_rms",
             "fcr_iemg", "fds_iemg", "ecu_iemg", "extensor_digitorum_iemg",
             "flexor_extensor_iemg_ratio", "emg_burst_duration",
             "fcr_mdf", "fds_mdf", "ecu_mdf", "extensor_digitorum_mdf")
_NONE_IMU = ("range_of_motion_proxy", "wrist_flexion_peak_velocity",
             "wrist_extension_peak_velocity", "finger_extension_peak_velocity")
_NONE_EEGMOV = ("movement_mu_power_change", "movement_beta_power_change")

for _names, _note in ((_NONE_EMG, NONE_NOTE_EMG),
                      (_NONE_IMU, NONE_NOTE_IMU),
                      (_NONE_EEGMOV, NONE_NOTE_EEGMOV)):
    for _n in _names:
        ENTRIES[_n] = {
            "reference_type": "none",
            "healthy_reference": None,
            "by_brunnstrom": None,
            # 无文献参考的项不臆断恢复方向（SPEC 的方向是指向各临床标签，非指向 Brunnstrom 升期）。
            "expected_direction_with_recovery": "n/a",
            "source": [],
            "confidence": "none",
            "note_zh": _note,
        }


def _units_for(name: str) -> str:
    """量纲优先复用 reasonableness.SPECS；缺失项给本模块约定量纲。"""
    if name in SPEC_BY_NAME:
        return SPEC_BY_NAME[name].units
    fallback = {
        "finger_co_contraction_index": "比值[0,1]",
        "fcr_iemg": "V·s", "fds_iemg": "V·s", "ecu_iemg": "V·s",
        "extensor_digitorum_iemg": "V·s",
        "flexor_extensor_iemg_ratio": "比值", "emg_burst_duration": "s",
        "fcr_mdf": "Hz", "fds_mdf": "Hz", "ecu_mdf": "Hz",
        "extensor_digitorum_mdf": "Hz",
        "prefrontal_theta_beta_ratio": "比值",
        "interhemispheric_motor_coherence": "相干[0,1]",
        "movement_mu_power_change": "相对变化", "movement_beta_power_change": "相对变化",
        "wrist_flexion_peak_velocity": "deg/s",
        "wrist_extension_peak_velocity": "deg/s",
        "finger_extension_peak_velocity": "deg/s",
    }
    return fallback.get(name, "—")


def build() -> dict:
    biomarkers = {}
    for name in bm.BIOMARKER_NAMES:
        if name not in ENTRIES:
            raise SystemExit(f"缺少 {name} 的参考范围条目，请补充 ENTRIES。")
        entry = {"units": _units_for(name), **ENTRIES[name]}
        biomarkers[name] = entry
    return {
        "meta": {
            "purpose": "为 02_biomarkers 的全部生物标志物提供参考范围，分层优先 Brunnstrom，"
                       "无分层依据时给健康成人常模。",
            "stratification": "Brunnstrom (fallback: healthy adult norm)",
            "disclaimer": "参考范围严格取自公开文献；多数标志物为本研究设备/协议特异量，"
                          "文献无标准范围，已显式标注 reference_type=none。绝非临床诊断阈值。",
            "scale_caveat": "本模块若干量纲（原始 EMG 电压、IMU 陀螺 deg/s、加速度 SPARC）与文献"
                            "测量方式不同，文献值即使存在也不可在绝对尺度直接套用，仅供方向参考。",
            "generated": "2026-06-17",
            "n_biomarkers": len(bm.BIOMARKER_NAMES),
        },
        "references": REFERENCES,
        "biomarkers": biomarkers,
    }


def _validate(doc: dict) -> None:
    assert set(doc["biomarkers"]) == set(bm.BIOMARKER_NAMES), "键名与 BIOMARKER_NAMES 不一致"
    for name, v in doc["biomarkers"].items():
        assert v["reference_type"] in ("healthy_norm", "directional_trend", "none"), name
        if v["reference_type"] == "none":
            assert v["healthy_reference"] is None and v["source"] == [], f"{name}: none 项需空"
        else:
            assert v["source"], f"{name}: 非 none 项必须有 source"
            for s in v["source"]:
                assert s in _REF_IDS, f"{name}: source {s} 未在 references 中"


def main() -> int:
    doc = build()
    _validate(doc)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    print(f"[已写出] {OUT_PATH}（{doc['meta']['n_biomarkers']} 项，校验通过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
