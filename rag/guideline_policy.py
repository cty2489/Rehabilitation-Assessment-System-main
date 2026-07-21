"""Scope guard for the test-only rehabilitation knowledge collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


_SENSOR_OR_BIOMARKER_TERMS = (
    "eeg",
    "emg",
    "imu",
    "脑电",
    "肌电",
    "生物标志物",
    "rms",
    "iemg",
    "mdf",
    "相干",
    "mu",
    "beta",
    "rom",
    "峰值速度",
    "平滑度",
)
_REFERENCE_RANGE_TERMS = (
    "参考范围",
    "正常范围",
    "正常值",
    "参考值",
    "阈值",
    "界值",
    "cutoff",
    "cut-off",
    "reference range",
    "normal range",
    "threshold",
)
_DOSE_TERMS = (
    "剂量",
    "分钟",
    "每次",
    "每天",
    "每日",
    "每周",
    "频次",
    "强度",
    "组数",
    "训练时长",
    "dose",
    "minute",
    "frequency",
    "intensity",
)
_DIRECT_PATIENT_TERMS = (
    "这个患者",
    "该患者",
    "此患者",
    "这位患者",
    "本患者",
    "当前患者",
    "本次患者",
    "这个病人",
    "该病人",
    "我的检查",
    "我的结果",
)
_GENERIC_PATIENT_TERMS = ("患者", "病人", "病例")
_CLINICAL_JUDGMENT_TERMS = (
    "异常",
    "偏高",
    "偏低",
    "诊断",
    "严重度",
    "严重程度",
    "分期",
    "几期",
    "属于哪一期",
    "brunnstrom",
    "布伦斯特罗姆",
    "布氏分期",
    "diagnosis",
    "severity",
    "stage",
)
_TREATMENT_PLAN_TERMS = ("训练方案", "康复方案", "治疗方案", "训练计划", "治疗计划", "处方")
_PRESCRIPTION_ACTION_TERMS = ("生成", "制定", "给出", "推荐", "安排", "开具", "自动")
_RESEARCH_CONTEXT_TERMS = (
    "论文",
    "研究",
    "综述",
    "荟萃",
    "相关性",
    "spearman",
    "fisher",
    "r²",
    "rmse",
    "准确率",
    "样本",
    "技术基准",
    "报告的",
)


@dataclass(frozen=True)
class GuidelineQueryDecision:
    action: str
    reason_code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def screen_guideline_test_query(query: str) -> GuidelineQueryDecision:
    """Keep research retrieval separate from patient-level clinical decisions."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    lower_query = clean_query.casefold()

    if _contains_any(lower_query, _REFERENCE_RANGE_TERMS):
        return GuidelineQueryDecision(
            action="block",
            reason_code="numeric_reference_out_of_scope",
            message=(
                "当前研究证据库不提供可迁移到本项目设备或单个患者的正常范围、"
                "参考值或数值阈值；不能根据研究相关性或模型性能推导这些数值。"
            ),
        )

    if _contains_any(lower_query, _DOSE_TERMS):
        return GuidelineQueryDecision(
            action="block",
            reason_code="training_dose_out_of_scope",
            message=(
                "当前研究证据库不输出具体分钟、次数、频次、强度或训练剂量；"
                "患者训练剂量需由临床人员结合标准评估和安全条件确认。"
            ),
        )

    direct_patient_context = _contains_any(lower_query, _DIRECT_PATIENT_TERMS)
    generic_patient_context = _contains_any(lower_query, _GENERIC_PATIENT_TERMS)
    research_context = _contains_any(lower_query, _RESEARCH_CONTEXT_TERMS)
    mentions_sensor = _contains_any(lower_query, _SENSOR_OR_BIOMARKER_TERMS)
    mentions_clinical_judgment = _contains_any(lower_query, _CLINICAL_JUDGMENT_TERMS)
    if mentions_clinical_judgment and (
        direct_patient_context
        or (generic_patient_context and not research_context)
        or (mentions_sensor and not research_context)
    ):
        return GuidelineQueryDecision(
            action="block",
            reason_code="patient_level_clinical_judgment_out_of_scope",
            message=(
                "当前研究证据库可以检索论文中的群体研究结果，但不能判断单个患者的"
                "异常、诊断、严重程度或 Brunnstrom 分期。"
            ),
        )

    mentions_treatment_plan = _contains_any(lower_query, _TREATMENT_PLAN_TERMS)
    requests_prescription = _contains_any(lower_query, _PRESCRIPTION_ACTION_TERMS)
    if mentions_treatment_plan and (
        direct_patient_context
        or (generic_patient_context and not research_context)
        or (requests_prescription and not research_context)
    ):
        return GuidelineQueryDecision(
            action="block",
            reason_code="automated_prescription_out_of_scope",
            message=(
                "当前研究证据库不为患者自动制定治疗方案、训练计划或处方；"
                "可改为检索研究方法、适用范围和证据局限。"
            ),
        )

    return GuidelineQueryDecision(action="retrieve", reason_code="in_scope", message="")


__all__ = ["GuidelineQueryDecision", "screen_guideline_test_query"]
