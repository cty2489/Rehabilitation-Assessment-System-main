"""Assemble the full multi-section Chinese rehab report (Markdown).

Mirrors ``大模型评估报告模板示例.docx``. Division of labour:

* **This module owns the deterministic skeleton**: section titles, the numeric
  columns of every table (指标 / 当前值), patient info, change-trend
  vs the previous visit, and the 26-gesture candidate space. Numbers come from
  the DL predictions + ``biomarkers.extract`` and are *never* delegated.
* **The LLM owns the clinical reasoning text** (see ``report.py::reason_clinical``):
  per-biomarker 解读/治疗建议, 综合亚型界定, 治疗策略, 手势组合 + 剂量, 预警, etc.
  Its output is a JSON ``clinical`` dict (schema in ``CLINICAL_SCHEMA_HINT``)
  which this module *back-fills* into the fixed skeleton. The LLM cannot alter
  any measured value because the value columns are rendered from code, not from
  its output.
* ``validate_clinical`` rejects incomplete or stage-inconsistent text. The caller
  may supply an explicitly labelled conservative fallback after model retries;
  generation provenance is stored outside this rendering module.

Public API:
    build_context(patient, predictions, biomarkers, history) -> dict
    validate_clinical(context, clinical) -> dict   # raises ClinicalUnavailable
    render_markdown(context, clinical) -> str
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from biomarker_refs import marker_ref, normalize_evidence_note
from schemas import PatientInfo, PredictionResult

import gestures

# Compact description of the JSON the LLM must return (kept here so prompt +
# parser share one source of truth).
CLINICAL_SCHEMA_HINT = """{
  "overall_interpretation": "总体分期及状态的临床解读（一句话，点明当前分期/过渡窗口与核心障碍）",
  "marker_text": { "<biomarker_key>": {"interpretation": "说明本次记录值、指标意义和同条件复测要求；没有有效常模时禁止判断偏高、偏低、正常或异常", "treatment_advice": "结合功能量表、动作表现和同条件复测给出训练/随访建议，不得仅凭单次设备特异量开具处方"}, ... },
  "overall_subtype": "综合亚型界定（一句话，必须含五要素：分期 + 优势运动模式 + 中枢驱动特征 + 协同分离程度 + 关节活动度状态。如：III期-屈肌优势伴中枢驱动不足亚型，协同开始解离，但关节活动度严重受限）",
  "treatment_strategy": ["每条都是一句高层策略，覆盖五维度：①策略名称（短语概括核心目标）②训练剂量（时间/频次/占比/辅助力度）③反馈标准（可量化阈值及奖励方式）④调整原则（需减少/替换/避免的训练）⑤安全注意（单次时长/疲劳程度/分次安排）；禁止输出具体方法字段，具体动作放在后续训练计划中", "..."],
  "gesture_plan": [ {"name": "必须取自候选手势库", "purpose": "训练目的", "force": "辅助力度", "reps": "重复次数"} ]，至少6个手势（仅当提供了候选手势库时才需要；未提供则省略本字段）,
  "weekly_plan": [ {"day": "周一", "content": "训练内容（只能用上面 gesture_plan 中的手势）", "duration": "预计时长"} ]，必须覆盖周一至周日共7天（仅当提供了候选手势库时才需要；未提供则省略本字段）,
  "warnings": ["预警与特殊建议1", "..."],
  "next_assessment": "固定输出：7天后执行下一次居家评估。"
}"""


_SIDE_PARALYSIS = {"左": "左", "右": "右", "L": "左", "R": "右"}


def _reference_context(key: str) -> Dict[str, Any]:
    """Return prompt/export metadata without presenting it as a clinical range."""
    ref = marker_ref(key)
    if ref is None:
        return {
            "type": "unknown",
            "absolute_comparison_applicable": False,
            "display_in_report": False,
        }
    return {
        "type": ref["reference_type"],
        "absolute_comparison_applicable": bool(ref["absolute_comparison_applicable"]),
        "expected_direction": ref["expected_direction"],
        "low": ref["lo"],
        "high": ref["hi"],
        "confidence": ref["confidence"],
        "source_ids": list(ref["source"]),
        "note": ref["note"],
        "display_in_report": False,
    }


def _decorate_biomarkers(biomarkers: Dict[str, Any]) -> Dict[str, Any]:
    """Attach evidence metadata to a copy of the prompt-facing marker rows."""
    groups = []
    for group in biomarkers.get("groups", []) or []:
        markers = []
        for marker in group.get("markers", []) or []:
            decorated = dict(marker)
            decorated.pop("ref_range", None)
            if decorated.get("note"):
                decorated["note"] = normalize_evidence_note(decorated["note"])
            decorated["reference"] = _reference_context(str(marker.get("key") or ""))
            markers.append(decorated)
        groups.append({**group, "markers": markers})
    return {**biomarkers, "groups": groups}


# --------------------------------------------------------------------------- #
# Deterministic context                                                        #
# --------------------------------------------------------------------------- #
def _trend(curr: Optional[float], prev: Optional[float], better: str = "up",
           unit: str = "", fmt: str = "{:.1f}") -> str:
    """Arrow + magnitude vs previous visit. ``better`` is informational only."""
    if prev is None or curr is None:
        return "首次评估"
    delta = curr - prev
    if abs(delta) < 1e-6:
        return "→ 持平"
    arrow = "↑" if delta > 0 else "↓"
    return f"{arrow} 较上次{'上升' if delta > 0 else '下降'}{fmt.format(abs(delta))}{unit}"


def build_context(
    patient: PatientInfo,
    predictions: PredictionResult,
    biomarkers: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    assessment_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the deterministic report context (numbers + structure, no prose).

    ``history`` (optional) is the previous assessment for the same patient:
    ``{"fma_ue","hand_tone","hand_function"}`` — used for the 变化趋势
    column. Pass ``None`` for a first visit.
    """
    stage = int(predictions.hand_function)
    side = _SIDE_PARALYSIS.get(patient.paralysis_side, patient.paralysis_side)

    prev = history or {}
    overall_rows = [
        {
            "metric": "Brunnstrom手部分期",
            "value": f"{_roman(stage)}期",
            "trend": _trend(stage, prev.get("hand_function"), fmt="{:.0f}", unit="期"),
        },
        {
            "metric": "手部肌张力（MAS）",
            "value": f"手部{predictions.hand_tone}级",
            "trend": _tone_trend(predictions.hand_tone, prev.get("hand_tone")),
        },
        {
            "metric": "FMA手部分数",
            "value": f"{int(round(predictions.FMA_UE))}/20",
            "trend": _trend(predictions.FMA_UE, prev.get("fma_ue"), fmt="{:.0f}", unit="分"),
        },
        # BI/改良 Barthel 指数偏向 ADL 独立性评估；当前系统聚焦上肢/手功能、
        # EEG/EMG/IMU biomarker 与训练处方，因此不再进入在线报告。
    ]

    return {
        "patient": {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "sex": patient.sex,
            "disease_days": patient.disease_days,
            "diagnosis": patient.diagnosis,
            "side": side,
        },
        "predictions": {
            "FMA_UE": int(round(predictions.FMA_UE)),
            # BI 已剔除：不传给大模型，故大模型评估不参考 Barthel 指数。
            "hand_tone": predictions.hand_tone,
            "hand_function": stage,
        },
        "stage": stage,
        "stage_roman": _roman(stage),
        "overall_rows": overall_rows,
        "biomarkers": _decorate_biomarkers(biomarkers),
        "gesture_library": gestures.gesture_names(),
        "assessment_context": dict(assessment_context or {}),
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
    }


def _roman(n: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}.get(int(n), str(n))


_TONE_ORDER = {"0": 0, "1": 1, "1+": 1.5, "2": 2, "3": 3, "4": 4}


def _tone_trend(curr: str, prev: Optional[str]) -> str:
    if prev is None:
        return "首次评估"
    c = _TONE_ORDER.get(str(curr))
    p = _TONE_ORDER.get(str(prev))
    if c is None or p is None:
        return "—"
    if abs(c - p) < 1e-6:
        return "→ 持平"
    return f"{'↑' if c > p else '↓'} 较上次{'上升' if c > p else '下降'}{abs(c - p):g}级"


NEXT_ASSESSMENT_TEXT = "7天后执行下一次居家评估。"


# --------------------------------------------------------------------------- #
# Markdown rendering                                                           #
# --------------------------------------------------------------------------- #
def _table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _cell(s: Any) -> str:
    return str(s).replace("\n", " ").replace("|", "／")


# Minimum number of recommended gestures (mirrors the f4844b template's 6).
MIN_GESTURES = 6

# Canonical week days, in order, for the weekly training plan.
_WEEK_DAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


class ClinicalUnavailable(Exception):
    """Raised when supplied clinical reasoning is missing or invalid."""


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _normalize_marker_text_entry(v: Any) -> Optional[Dict[str, str]]:
    """Accept the canonical marker object and compact LLM-friendly aliases.

    DeepSeek-R1-Distill is much more reliable when asked to emit compact JSON,
    so the prompt may use ``["解读", "建议"]`` per biomarker and this validator
    expands it back to the canonical shape used by rendering/export.
    """
    if isinstance(v, dict):
        interpretation = (
            v.get("interpretation")
            or v.get("interpretion")  # tolerate a common model typo
            or v.get("解读")
        )
        advice = (
            v.get("treatment_advice")
            or v.get("treatmentAdvice")
            or v.get("treatation_advice")  # tolerate a common model typo
            or v.get("治疗建议")
            or v.get("建议")
        )
        if _nonempty_str(interpretation) and _nonempty_str(advice):
            return {
                "interpretation": str(interpretation).strip(),
                "treatment_advice": str(advice).strip(),
            }
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        interpretation, advice = v[0], v[1]
        if _nonempty_str(interpretation) and _nonempty_str(advice):
            return {
                "interpretation": str(interpretation).strip(),
                "treatment_advice": str(advice).strip(),
            }
    if isinstance(v, (list, tuple)) and len(v) >= 1 and _nonempty_str(v[0]):
        combined = str(v[0]).strip()
        for delim in ("治疗建议：", "治疗建议:", "建议：", "建议:", "【建议】", "。治疗建议", "\n治疗建议"):
            idx = combined.find(delim)
            if idx > 0:
                return {
                    "interpretation": combined[:idx].strip().rstrip("。."),
                    "treatment_advice": combined[idx + len(delim):].strip(),
                }
        return {
            "interpretation": combined,
            "treatment_advice": "建议结合临床评估与复测趋势调整训练方案。",
        }
    return None


_UNSUPPORTED_ABSOLUTE_CLAIM = re.compile(
    r"偏高|偏低|较高|较低|过高|过低|高于|低于|正常范围|范围内|超标|异常值|募集不足"
)
_EVIDENCE_AWARE_WORDS = ("复测", "同设备", "同流程", "单次", "不能直接", "不作")
_CONDITIONAL_ADVICE_WORDS = ("复测", "结合", "若", "根据", "治疗师", "医师")
_GROUP_FOLLOW_UP = {
    "emg": "后续同条件复测应保持电极位置、设备增益和任务一致，并结合手部MAS、FMA手部分数及动作表现调整训练。",
    "eeg": "后续同条件复测应保持导联、任务和伪迹处理一致，并结合运动表现由康复治疗师调整训练。",
    "imu": "后续同条件复测应保持传感器位置和动作任务一致，并结合轨迹质量及功能量表调整训练。",
}


def _marker_value_text(marker: Dict[str, Any]) -> str:
    value = marker.get("value", "—")
    unit = str(marker.get("unit") or "").strip()
    return f"{value}{(' ' + unit) if unit else ''}"


def _enforce_marker_evidence_policy(
    marker: Dict[str, Any],
    group_key: str,
    text: Dict[str, str],
) -> Dict[str, str]:
    """Neutralise unsupported single-value judgements before report rendering.

    Prompt instructions improve model behaviour, but this deterministic gate is
    the final protection against wording such as "偏高" or "正常范围内" when
    the current device/protocol has no valid absolute comparator.
    """
    ref = marker.get("reference") or {}
    rtype = str(ref.get("type") or "unknown")
    if rtype == "unknown" or ref.get("absolute_comparison_applicable"):
        return text

    interpretation = text["interpretation"].strip()
    advice = text["treatment_advice"].strip()
    interpretation_ok = (
        not _UNSUPPORTED_ABSOLUTE_CLAIM.search(interpretation)
        and any(word in interpretation for word in _EVIDENCE_AWARE_WORDS)
    )
    advice_ok = any(word in advice for word in _CONDITIONAL_ADVICE_WORDS)
    if interpretation_ok and advice_ok:
        return text

    value_text = _marker_value_text(marker)
    if rtype == "directional_trend":
        direction = {"increase": "升高", "decrease": "下降"}.get(
            str(ref.get("expected_direction") or "")
        )
        direction_text = f"文献仅提示康复过程中通常{direction}" if direction else "文献仅提供方向性证据"
        safe_interpretation = (
            f"本次记录值为{value_text}；{direction_text}，单次结果不能判断变化趋势，需同条件复测。"
        )
    elif rtype == "healthy_norm":
        safe_interpretation = (
            f"本次记录值为{value_text}；当前算法与文献常模计算尺度不同，不能直接比较，需同条件复测。"
        )
    else:
        safe_interpretation = (
            f"本次记录值为{value_text}，仅作为同设备、同流程复测基线；单次结果不判断正常或异常。"
        )
    return {
        "interpretation": safe_interpretation,
        "treatment_advice": _GROUP_FOLLOW_UP.get(
            group_key,
            "后续在相同采集条件下复测，并结合临床量表与动作表现调整训练。",
        ),
    }


_SPECIFIC_METHOD_SEGMENT = re.compile(
    r"(?:^|[；;]\s*)具体方法(?:[（(][^）)]*[）)])?\s*[：:]\s*.*?"
    r"(?=(?:[；;]\s*)?(?:训练剂量|反馈标准|调整原则|安全注意)\s*[：:]|$)"
)


def _strip_specific_method(value: Any) -> str:
    """Remove the duplicated implementation field from a strategy sentence."""
    text = str(value or "").strip()
    text = _SPECIFIC_METHOD_SEGMENT.sub("", text)
    text = re.sub(r"^[；;\s]+|[；;\s]+$", "", text)
    text = re.sub(r"[；;]\s*[；;]", "；", text)
    return text.strip()


def _normalize_strategy_item(v: Any) -> Optional[str]:
    if _nonempty_str(v):
        return _strip_specific_method(v) or None
    if isinstance(v, dict):
        method_keys = {"method", "specific_method", "具体方法", "方法"}
        values = [
            str(item).strip()
            for key, item in v.items()
            if str(key).strip() not in method_keys and _nonempty_str(item)
        ]
        if values:
            return _strip_specific_method("；".join(values)) or None
    return None


def _normalize_weekly_plan(
    llm_plan: Optional[List[Dict[str, Any]]],
    gesture_plan: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Return a normalised 7-day weekly plan, or None if the LLM's is unusable.

    Requirements: every entry has day/content, the days span the whole week, and
    each training day's content only references gestures from the recommended
    combination (``gesture_plan``). Rest days (休息/自由训练) need no gesture.
    """
    if not isinstance(llm_plan, list) or not llm_plan:
        return None
    rows = [r for r in llm_plan if isinstance(r, dict) and r.get("day") and r.get("content")]
    if len(rows) < len(_WEEK_DAYS):  # must cover Mon–Sun
        return None
    days_text = "".join(str(r["day"]) for r in rows)
    if not all(d in days_text for d in _WEEK_DAYS):
        return None
    nameset = {g.get("name", "") for g in gesture_plan if g.get("name")}
    if nameset:
        for r in rows:
            content = str(r["content"])
            mentioned = any(g in content for g in nameset)
            only_rest = any(k in content for k in ("休息", "自由训练"))
            if not mentioned and not only_rest:
                return None
    return [
        {"day": r["day"], "content": r["content"], "duration": r.get("duration", "—")}
        for r in rows
    ]


def validate_clinical(context: Dict[str, Any], clinical: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the LLM clinical-reasoning JSON and assemble the render dict.

    Raises ``ClinicalUnavailable`` if the supplied output is missing required fields, leaves
    any biomarker without interpretation/advice, or produces an overall subtype whose 分期
    prefix disagrees with the measured Brunnstrom stage (the parrot-the-example
    failure mode). Gesture fields are only required/validated once the clinical
    team's 26-gesture library is ready; until then they render as a placeholder.
    """
    if not isinstance(clinical, dict):
        raise ClinicalUnavailable("大模型未返回结构化结果（JSON 解析为空）")

    roman = str(context.get("stage_roman", ""))
    prefix = f"{roman}期"

    # ── total interpretation ──
    if not _nonempty_str(clinical.get("overall_interpretation")):
        raise ClinicalUnavailable("缺少总体临床解读（overall_interpretation）")

    # ── per-biomarker interpretation + advice ──
    # Only markers that were actually measured (available=True) require an LLM
    # reading. Device-format bundles legitimately can't compute every marker; an
    # unavailable marker is back-filled with a fixed "数据不足" note rather than
    # forcing the model to fabricate (or failing the whole report). Markers with
    # no explicit "available" flag default to available (hospital path / older
    # extract output) so behaviour there is unchanged.
    src_mt = clinical.get("marker_text")
    if not isinstance(src_mt, dict):
        src_mt = {}
    marker_text: Dict[str, Dict[str, str]] = {}
    for group in context["biomarkers"].get("groups", []):
        for m in group["markers"]:
            key = m["key"]
            if m.get("available", True) is False:
                marker_text[key] = {
                    "interpretation": "本次数据不足，未予解读",
                    "treatment_advice": "—",
                }
                continue
            txt = _normalize_marker_text_entry(src_mt.get(key))
            if txt is None:
                raise ClinicalUnavailable(f"生物标志物 {m['name']}（{key}）缺少解读或治疗建议")
            marker_text[key] = _enforce_marker_evidence_policy(m, group["key"], {
                "interpretation": txt["interpretation"].strip(),
                "treatment_advice": txt["treatment_advice"].strip(),
            })

    # ── overall subtype: 五要素一句话，分期前缀必须一致 ──
    overall_subtype = clinical.get("overall_subtype")
    if not _nonempty_str(overall_subtype):
        raise ClinicalUnavailable("缺少综合亚型界定（overall_subtype）")
    if prefix and not overall_subtype.lstrip().startswith(prefix):
        raise ClinicalUnavailable(
            f"综合亚型分期（{overall_subtype.strip()[:8]}…）与实测 Brunnstrom {prefix} 不一致")

    # ── treatment strategy: non-empty list of non-empty strings ──
    strat_src = clinical.get("treatment_strategy")
    if not isinstance(strat_src, list):
        raise ClinicalUnavailable("缺少治疗策略要点（treatment_strategy）")
    treatment_strategy = [s for s in (_normalize_strategy_item(item) for item in strat_src) if s]
    if not treatment_strategy:
        raise ClinicalUnavailable("治疗策略要点为空")

    warnings = [w.strip() for w in (clinical.get("warnings") or []) if _nonempty_str(w)]

    c: Dict[str, Any] = {
        "overall_interpretation": clinical["overall_interpretation"].strip(),
        "marker_text": marker_text,
        "overall_subtype": overall_subtype.strip(),
        "treatment_strategy": treatment_strategy,
        "warnings": warnings,
        "not_recommended": [],
        "next_assessment": NEXT_ASSESSMENT_TEXT,
    }

    # ── gesture plan + weekly plan: only when the 26-gesture library is ready ──
    # Soft requirement: single-pass models (qwen3 等) can pick a 6+ gesture combo
    # and a 7-day plan from the library, but segmented models (baichuan2 /
    # deepseek / glm) don't emit gestures at all. Rather than fail the WHOLE
    # report into a conservative fallback when a model can't produce gestures,
    # skip only the gesture section (rendered with an explanation) and keep the
    # measured values + clinical reasoning the model DID produce.
    if gestures.library_ready():
        plan = [g for g in (clinical.get("gesture_plan") or [])
                if isinstance(g, dict) and g.get("name") in set(gestures.gesture_names())]
        weekly = (_normalize_weekly_plan(clinical.get("weekly_plan"), plan)
                  if len(plan) >= MIN_GESTURES else None)
        if len(plan) >= MIN_GESTURES and weekly is not None:
            c["gesture_plan"] = plan
            c["weekly_plan"] = weekly
            c["gesture_ready"] = True
        else:
            # Model (typically segmented) didn't return a usable gesture plan.
            c["gesture_plan"] = []
            c["weekly_plan"] = []
            c["gesture_ready"] = False
            c["gesture_skipped"] = True
    else:
        c["gesture_plan"] = []
        c["weekly_plan"] = []
        c["gesture_ready"] = False
    return c


def render_markdown(context: Dict[str, Any], clinical: Optional[Dict[str, Any]]) -> str:
    """Render the full report as Markdown.

    ``clinical`` is the selected reasoning JSON (LLM or explicitly labelled
    fallback); ``validate_clinical`` rejects missing or invalid text.
    """
    c = validate_clinical(context, clinical)
    p = context["patient"]
    out: List[str] = []

    out.append("# 智能康复评估报告")
    out.append("")
    assessment_context = context.get("assessment_context") or {}
    if assessment_context.get("validation_status") == "engineering_validation_only":
        out.append(
            "> **设备端工程验证提示：** 当前设备通道布局与医院训练数据不同，"
            "本次模型评分及生物标志物仅用于接口联调和同条件复测观察，"
            "尚不能替代经临床验证的量表评估或作为诊疗依据。"
        )
        out.append("")
    quality = assessment_context.get("quality") or {}
    if quality.get("status") == "needs_review":
        out.append(
            "> **信号质量复核：** 本次采集中存在同步回退、时长不足或采样率不一致的试次，"
            "请先复核原始信号与设备佩戴情况，再解读结果。"
        )
        out.append("")

    # 一、患者基本信息
    out.append("## 一、患者基本信息")
    out.append("")
    age = p["age"] if p["age"] is not None else "—"
    days = p["disease_days"] if p["disease_days"] is not None else "—"
    out.append(f"- 患者ID：{p['patient_id']}")
    out.append(f"- 姓名：{p['name']}")
    out.append(f"- 年龄/性别：{age}岁/{p['sex']}")
    out.append(f"- 病程：{days}天")
    out.append(f"- 卒中类型：{p['diagnosis']}，{p['side']}侧偏瘫")
    out.append("")

    # 二、本次评估结果
    out.append("## 二、本次评估结果（基于多模态数据）")
    out.append("")
    out.append("### 1. 总体分期及状态")
    out.append("")
    out.append(_table(
        ["指标", "评估结果", "变化趋势"],
        [[r["metric"], r["value"], r["trend"]] for r in context["overall_rows"]],
    ))
    out.append("")
    out.append(f"**临床解读：** {c['overall_interpretation']}")
    out.append("")

    # 关键生物标志物输出与解读
    out.append("### 2. 关键生物标志物输出与解读")
    out.append("")
    out.append(
        "> 说明：生物标志物受设备与采集流程影响。本报告仅用于同一患者在相同设备、"
        "相同采集流程下的连续变化观察，不作为正常或异常的诊断阈值。"
    )
    out.append("")
    group_titles = {"emg": "（1）肌电标志物（基于本次主动动作评估）",
                    "eeg": "（2）脑电标志物（基于本次主动动作评估）",
                    "imu": "（3）运动学标志物（IMU）"}
    _groups = context["biomarkers"].get("groups", [])
    if not _groups:
        # Extraction failed upstream (see inference.py) — never leave a silent
        # blank section; tell the reader where to look.
        out.append("> 本次生物标志物提取未成功，本段缺失；请检查后端日志中 [biomarkers] 的报错。")
        out.append("")
    for group in _groups:
        out.append(f"#### {group_titles.get(group['key'], group['label'])}")
        out.append("")
        rows = []
        for m in group["markers"]:
            txt = c["marker_text"].get(m["key"], {})
            value_disp = f"{m['value']}{(' ' + m['unit']) if m['unit'] else ''}"
            rows.append([
                m["name"],
                value_disp,
                txt.get("interpretation", "—"),
                txt.get("treatment_advice", "—"),
            ])
        out.append(_table(["标志物", "当前值", "解读", "训练/随访建议"], rows))
        # Per-group note (e.g. ROM is an estimate). Each marker's note
        # is itself a "；"-joined multi-fragment string, so dedupe at the FRAGMENT
        # level (preserving order) — otherwise shared fragments like "设备特异量"
        # / "Welch 谱…" repeat across markers in the footnote.
        seen_frag: set = set()
        frags: List[str] = []
        for m in group["markers"]:
            for frag in (m.get("note") or "").split("；"):
                frag = frag.strip()
                if frag and frag not in seen_frag:
                    seen_frag.add(frag)
                    frags.append(frag)
        if frags:
            out.append("")
            out.append("> " + "；".join(frags))
        out.append("")

    # 三、综合亚型界定与治疗策略
    out.append("## 三、综合亚型界定与治疗策略")
    out.append("")
    out.append(f"根据上述生物标志物，患者可归类为：**{c['overall_subtype']}**")
    out.append("")
    out.append("**治疗策略要点：**")
    out.append("")
    for i, s in enumerate(c["treatment_strategy"], 1):
        out.append(f"{i}. {s}")
    out.append("")

    # 四、下周具体训练参数
    out.append("## 四、下周具体训练参数")
    out.append("")
    if not c.get("gesture_ready"):
        if c.get("gesture_skipped"):
            # Library IS ready, but this model (typically a segmented model like
            # baichuan2 / deepseek / glm) did not emit a valid 6+ gesture combo.
            # Keep the rest of the report instead of failing into a fallback.
            out.append("> 当前报告模型本次未生成合规的手势处方（分段生成模型通常不输出手势组合）；"
                       "如需完整的推荐手势组合与每周训练计划，建议改用 Qwen3 / InternLM3 / Mistral 等单次生成模型重新生成。")
        else:
            # The clinical team's 26-gesture library is not configured yet, so the
            # LLM was not asked to pick gestures. Show a placeholder instead of an
            # invented (and potentially misleading) plan.
            out.append("> 手势库待补充：请先由康复团队审核 `backend/config/gestures_26.example.json`，"
                       "确认后复制为 `backend/config/gestures_26.json`，再由大模型从中动态选取手势组合与每周训练计划。")
        out.append("")
    else:
        out.append("> ⚠️ 以下手势组合与训练计划由系统从已配置手势库自动推荐，正式执行前请由康复治疗师核准手势选择、辅助力度与重复次数。")
        out.append("")
        out.append("### 1. 推荐手势组合（从26个中动态选取）")
        out.append("")
        g_rows = [
            [str(i), g["name"], g.get("purpose", ""), g.get("force", ""), g.get("reps", "")]
            for i, g in enumerate(c["gesture_plan"], 1)
        ]
        out.append(_table(["顺序", "手势名称", "训练目的", "辅助力度设置", "重复次数"], g_rows))
        out.append("")
        out.append("### 2. 每周训练计划")
        out.append("")
        out.append(_table(
            ["训练日", "训练内容", "预计时长"],
            [[w["day"], w["content"], w["duration"]] for w in c["weekly_plan"]],
        ))
        out.append("")

    # 五、预警与特殊建议
    out.append("## 五、预警与特殊建议")
    out.append("")
    for i, w in enumerate(c["warnings"], 1):
        out.append(f"{i}. {w}")
    out.append("")

    # 六、下次评估时间
    out.append("## 六、下次评估时间")
    out.append("")
    out.append(f"建议：{c['next_assessment']}")
    out.append("")

    return "\n".join(out).rstrip() + "\n"


__all__ = [
    "CLINICAL_SCHEMA_HINT",
    "ClinicalUnavailable",
    "build_context",
    "validate_clinical",
    "render_markdown",
]
