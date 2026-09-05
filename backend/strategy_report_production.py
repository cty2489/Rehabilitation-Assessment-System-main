"""Production assembly for the training-strategy report endpoint.

This module is deliberately limited to the report feature.  It calls the
existing package parser, DL/biomarker pipeline and ``planner_rag`` components;
it does not modify or replace any of those shared implementations.

No report, patient profile or score is selected by ZIP hash, filename, patient
identifier, CASE identifier or a score combination.  Public facts are assembled
deterministically from the current upload and current pipeline result.  The LLM
is used only for the narrative recommendation supplied by ``ReportGenerator``.
"""

from __future__ import annotations

import json
import math
import queue
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4

import knowledge_admin
import mysql_db
from clinical_pipeline import production_adapter
from clinical_pipeline.orchestrator import PipelineRunStatus
from clinical_pipeline.validator import ValidationStatus
from eval_package import EvalPackage, read_eval_package
from inference import run_pipeline
from report import llm_model_name
from schemas import PatientInfo


MAS_VALUES = {"0", "1", "1+", "2", "3", "4"}
GROUP_ORDER = ("emg", "eeg", "imu")
GROUP_LABELS = {
    "emg": "肌电标志物（基于本次主动动作评估）",
    "eeg": "脑电标志物（基于本次主动动作评估）",
    "imu": "运动学标志物（IMU）",
}
VOLTAGE_RMS_KEYS = {"resting_emg_level", "emg_activation_rms"}
REQUIRED_PLANNER_SCORES = ("fma_hand", "hand_mas", "brunnstrom_hand")


class StrategyPipelineError(RuntimeError):
    """The report-only production chain could not complete."""


@dataclass(frozen=True)
class PipelineArtifacts:
    package: EvalPackage
    patient: PatientInfo
    score_bundle: dict[str, Any]
    inference_result: dict[str, Any]
    orchestration_result: Any
    publication_status: str
    publication_warnings: list[str]
    profile: Optional[dict[str, Any]]
    stage_events: list[dict[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten manifest-attached clinical facts without any external lookup."""
    output: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return output
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output[_normal_key(key)] = item.get("value") if isinstance(item, Mapping) and "value" in item else item
        output[_normal_key(path)] = output[_normal_key(key)]
        if isinstance(item, Mapping):
            output.update(_flatten_mapping(item, path))
    return output


def _pick(flat: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        value = flat.get(_normal_key(alias))
        if value not in (None, "", "—", "-"):
            return value
    return None


def _integer(value: Any, minimum: int, maximum: int) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = int(round(number))
    return rounded if minimum <= rounded <= maximum else None


def _mas(value: Any) -> Optional[str]:
    text = str(value or "").strip().replace("＋", "+")
    return text if text in MAS_VALUES else None


def _manifest_clinical_scores(manifest: Mapping[str, Any]) -> dict[str, Any]:
    attached = {
        "clinical_mapping": manifest.get("clinical_mapping") or {},
        "clinical_profile": manifest.get("clinical_profile") or {},
        "clinical_scores": manifest.get("clinical_scores") or {},
    }
    flat = _flatten_mapping(attached)
    return {
        "fma_wrist": _integer(_pick(flat, ("fma_wrist", "fmawrist", "fma_wrist_score", "fma腕部", "fma腕")), 0, 10),
        "fma_hand": _integer(_pick(flat, ("fma_hand", "fmahand", "fma_ue", "fma_hand_score", "fma手部", "fma手")), 0, 20),
        "wrist_mas": _mas(_pick(flat, ("wrist_mas", "mas_wrist", "腕mas", "腕部mas"))),
        "hand_mas": _mas(_pick(flat, ("hand_mas", "mas_hand", "hand_tone", "手mas", "手部mas"))),
        "brunnstrom_hand": _integer(_pick(flat, ("brunnstrom_hand", "brunnstrom", "hand_function", "brunnstrom手功能", "布氏分期")), 1, 6),
    }


def _score_bundle(manifest: Mapping[str, Any], predictions: Mapping[str, Any]) -> dict[str, Any]:
    clinical = _manifest_clinical_scores(manifest)
    clinical_complete = all(clinical.get(key) is not None for key in REQUIRED_PLANNER_SCORES)
    if clinical_complete:
        primary = {
            "fma_hand": clinical["fma_hand"],
            "hand_mas": clinical["hand_mas"],
            "brunnstrom_hand": clinical["brunnstrom_hand"],
        }
        mode = "doctor_clinical_score"
        primary_source = "本次数据包随附的医生临床评分"
        dl_used_for_report_scores = False
    else:
        primary = {
            "fma_hand": _integer(predictions.get("FMA_UE"), 0, 20),
            "hand_mas": _mas(predictions.get("hand_tone")),
            "brunnstrom_hand": _integer(predictions.get("hand_function"), 1, 6),
        }
        mode = "dl_prediction"
        primary_source = "本次原始信号的DL评估结果"
        dl_used_for_report_scores = True
    if any(primary.get(key) is None for key in REQUIRED_PLANNER_SCORES):
        raise StrategyPipelineError("FMA手部、手MAS或Brunnstrom结果不完整，无法进入QualityGate")
    return {
        "mode": mode,
        "primary_source": primary_source,
        "deep_learning_score_prediction_used": dl_used_for_report_scores,
        "fma_wrist": clinical.get("fma_wrist"),
        "fma_hand": primary["fma_hand"],
        "wrist_mas": clinical.get("wrist_mas"),
        "hand_mas": primary["hand_mas"],
        "brunnstrom_hand": primary["brunnstrom_hand"],
        "field_sources": {
            "fma_wrist": "本次数据包随附的医生临床评分" if clinical.get("fma_wrist") is not None else "本次未提供",
            "fma_hand": primary_source,
            "wrist_mas": "本次数据包随附的医生临床评分" if clinical.get("wrist_mas") is not None else "本次未提供",
            "hand_mas": primary_source,
            "brunnstrom_hand": primary_source,
        },
    }


def _read_manifest(package_root: Path) -> dict[str, Any]:
    path = package_root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        raise StrategyPipelineError("manifest.json无法读取") from exc
    if not isinstance(value, dict):
        raise StrategyPipelineError("manifest.json必须为对象")
    return value


def _resolve_patient(package: EvalPackage) -> PatientInfo:
    """Reuse the current patient-master precedence without changing it."""
    values = dict(package.patient_prefill)
    patient_id = str(values.get("patient_id") or "").strip()
    if patient_id:
        try:
            enrolled = mysql_db.get_patient_by_business_id(patient_id)
        except Exception as exc:  # noqa: BLE001 - report can still use manifest facts
            enrolled = None
            print(f"[strategy-report][patient-master-warning] {type(exc).__name__}: {exc}")
        if enrolled:
            for key in ("name", "sex", "age", "diagnosis", "disease_days", "paralysis_side"):
                if enrolled.get(key) not in (None, ""):
                    values[key] = enrolled[key]
    missing = [
        label
        for key, label in (
            ("patient_id", "患者编号"),
            ("sex", "性别"),
            ("diagnosis", "诊断"),
            ("paralysis_side", "偏瘫侧"),
        )
        if values.get(key) in (None, "")
    ]
    if missing:
        raise StrategyPipelineError("患者基本信息缺失：" + "、".join(missing))
    return PatientInfo(
        patient_id=patient_id,
        name=str(values.get("name") or "匿名患者"),
        sex=str(values["sex"]),
        age=values.get("age"),
        diagnosis=str(values["diagnosis"]),
        disease_days=values.get("disease_days"),
        paralysis_side=str(values["paralysis_side"]),
    )


def _stage_event(stage: str, status: str, detail: str = "") -> dict[str, Any]:
    return {"at": _now(), "stage": stage, "status": status, "detail": detail[:500]}


def _display_number(value: Any, *, decimals: int = 8) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未获得"
    if not math.isfinite(number):
        return "未获得"
    text = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _biomarker_rows(biomarkers: Optional[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in (biomarkers or {}).get("groups", []) or []:
        group_key = str(group.get("key") or "").lower()
        for marker in group.get("markers", []) or []:
            key = str(marker.get("key") or "")
            available = bool(marker.get("available", False))
            value = marker.get("value") if available else None
            unit = str(marker.get("unit") or "")
            conversion = None
            if available and key in VOLTAGE_RMS_KEYS and unit.startswith("V"):
                value = float(value) * 1_000_000.0
                unit = "μV"
                conversion = "由本次V(RMS)计算结果换算，1 V=10^6 μV"
            rows.append(
                {
                    "group_key": group_key,
                    "group_label": str(group.get("label") or GROUP_LABELS.get(group_key, group_key.upper())),
                    "marker_key": key,
                    "marker_name": str(marker.get("name") or key),
                    "available": available,
                    "value_num": value,
                    "value_text": _display_number(value) if available else "未获得",
                    "unit": unit or "-",
                    "n_valid": int(marker.get("n_valid") or 0),
                    "unit_conversion": conversion,
                }
            )
    rows.sort(key=lambda item: (GROUP_ORDER.index(item["group_key"]) if item["group_key"] in GROUP_ORDER else 99))
    missing = [item["marker_key"] for item in rows if not item["available"]]
    coverage = {
        "available": sum(1 for item in rows if item["available"]),
        "total": len(rows),
        "missing_keys": missing,
        "interpretation_policy": "无统一常模的指标仅用于同设备、同流程、同任务条件下的纵向比较，不凭单次值判断正常或异常。",
    }
    return rows, coverage


def _score_text(value: Any, maximum: Optional[int] = None) -> str:
    if value is None:
        return "本次未提供"
    return f"{value}/{maximum}分" if maximum else str(value)


def _subtype(scores: Mapping[str, Any]) -> dict[str, str]:
    stage = int(scores["brunnstrom_hand"])
    names = {
        1: "随意运动尚未出现-被动维护与运动意图建立型",
        2: "共同运动萌发-主动启动不足型",
        3: "屈肌协同占优-选择性伸展受限型",
        4: "共同运动减弱-分离运动建立型",
        5: "分离运动改善-精细控制不足型",
        6: "接近正常-速度与精度优化型",
    }
    tone = scores.get("hand_mas")
    tone_text = f"手MAS为{tone}级" if tone is not None else "手MAS本次未提供"
    wrist = _score_text(scores.get("fma_wrist"), 10)
    wrist_mas = scores.get("wrist_mas") or "本次未提供"
    return {
        "name": f"Brunnstrom {stage}期-{names[stage]}",
        "summary": (
            f"本次结果为FMA腕部{wrist}、FMA手部{scores['fma_hand']}/20分、"
            f"腕MAS {wrist_mas}级、{tone_text}、Brunnstrom手功能{stage}期。"
        ),
        "boundary": "该亚型是对本次量表和动作信号结果的结构化归纳，不替代康复医学诊断；训练安排仍需结合关节活动度、感觉、疼痛、认知及心肺耐受由治疗师现场调整。",
    }


def _non_numeric_advice(text: str) -> str:
    """Keep only validated LLM advice clauses that cannot alter numeric facts."""
    clauses = [part.strip(" ，；。") for part in re.split(r"[。；\n]", str(text or ""))]
    safe = [part for part in clauses if part and not re.search(r"\d|%|％|/", part)]
    return "；".join(safe[:2])


def _strategy_scopes(scores: Mapping[str, Any]) -> list[str]:
    wrist = scores.get("fma_wrist")
    return ["wrist", "hand"] if wrist is not None and int(wrist) < 9 else ["hand"]


def _gesture_choice(stage: int, scope: str, fma_hand: int) -> Optional[str]:
    if stage == 1:
        return "SS-15" if scope == "wrist" else "SS-16"
    if stage == 2:
        return "SS-15" if scope == "wrist" else ("SS-10" if fma_hand <= 5 else "SS-18")
    if stage == 3:
        return "SS-15" if scope == "wrist" else ("SS-16" if fma_hand <= 6 else "SS-22")
    if stage == 4:
        return "SS-14" if scope == "wrist" else ("SS-1" if fma_hand <= 12 else "SS-19")
    if stage == 5:
        return "SS-19" if scope == "wrist" else ("SS-21" if fma_hand <= 16 else "SS-24")
    if stage == 6:
        return "SS-24" if scope == "hand" else "SS-19"
    return None


_GESTURE_DETAILS = {
    "SS-1": ("食指屈曲", "其余手指保持相对伸展，食指完成选择性屈曲。", "用于食指分离控制练习。"),
    "SS-10": ("拇指屈曲", "手掌保持稳定，拇指完成可控制的屈曲与回位。", "用于拇指主动控制练习。"),
    "SS-14": ("四指伸展", "拇指保持舒适位，其余四指尽量伸展。", "用于手指伸展与腕手协同练习。"),
    "SS-15": ("五指伸展", "手掌打开，拇指及其余四指尽量伸展。", "用于手掌打开及主动释放练习。"),
    "SS-16": ("五指屈曲", "五指逐步屈曲形成握拳，并在可控范围内回位。", "用于屈曲启动与握持准备。"),
    "SS-18": ("柱状抓握", "拇指与其余手指围绕柱状物形成稳定抓握。", "用于基础抓握与放松练习。"),
    "SS-19": ("棍状物抓握", "手指围绕棍状物完成抓握并保持腕部稳定。", "用于抓握中的腕手协同练习。"),
    "SS-21": ("环形抓握", "拇指与手指形成环形抓握，维持指腹接触。", "用于抓握精度与持续控制练习。"),
    "SS-22": ("球体抓握", "拇指展开对掌，其余手指屈曲形成球形抓握。", "用于拇指对掌和整体抓握练习。"),
    "SS-24": ("拇指指尖捏取", "拇指指尖与目标手指指尖对合，保持腕部稳定。", "用于精细捏取和放置练习。"),
}


def _gesture_action(resource_dir: Path, scores: Mapping[str, Any], scope: str) -> Optional[dict[str, Any]]:
    stage = int(scores["brunnstrom_hand"])
    code = _gesture_choice(stage, scope, int(scores["fma_hand"]))
    if not code:
        return None
    library = json.loads((resource_dir / "gesture_library.json").read_text(encoding="utf-8"))
    item = next((candidate for candidate in library.get("gestures", []) if candidate.get("code") == code), None)
    if not item:
        return None
    image_filename = Path(str(item.get("image") or "")).name
    image_path = resource_dir / "gesture_images" / image_filename
    if not image_filename or not image_path.is_file():
        return None
    name, key_points, purpose = _GESTURE_DETAILS.get(code, (str(item.get("name") or code), "按手势库图示在舒适范围内完成。", "作为本次训练的手型参考。"))
    return {
        "reference_role": "配合手型" if scope == "wrist" else "目标手型",
        "gesture_code": code,
        "gesture_name": name,
        "brunnstrom_stage": stage,
        "gesture_key_points": key_points,
        "gesture_purpose": purpose,
        "display_note": "图片仅作手型参考；实际训练应依据关节活动度、张力和主动控制能力由治疗师调整。",
        "image_filename": image_filename,
        "image_sha256": str(item.get("image_sha256") or ""),
        "scope": scope,
    }


def _marker_value(rows: list[dict[str, Any]], key: str) -> str:
    item = next((row for row in rows if row["marker_key"] == key), None)
    if not item or not item["available"]:
        return "未获得"
    return f"{item['value_text']}{item['unit']}"


def _fitt(stage: int, tone: str, scope: str) -> dict[str, str]:
    if stage <= 2:
        frequency, repetitions, duration = "每周训练5天，在治疗师指导下每日1次。", "每组6至8次，共2组，以能保持目标动作质量为上限。", "每次约12分钟，组间休息1至2分钟。"
    elif stage <= 4:
        frequency, repetitions, duration = "每周训练5天，每日1次；耐受良好时可增加短时家庭练习。", "每组8至12次，共2至3组，以动作质量稳定且不过度代偿为准。", "每次约15至20分钟，组间休息1至2分钟。"
    else:
        frequency, repetitions, duration = "每周训练4至5天，每日1次。", "每组10至15次，共2至3组，以速度提高时仍保持准确为准。", "每次约20分钟，任务间按疲劳情况休息。"
    assistance = "采用治疗师辅助或减重支持，从可完成的主动参与开始" if stage <= 2 else "采用主动训练，必要时给予最小限度的方向性提示"
    if scope == "wrist":
        training_type = f"{assistance}，练习腕背伸、屈伸转换及前臂稳定，并逐步过渡到腕手协同任务。"
    else:
        training_type = f"{assistance}，练习手掌打开、抓握、释放及与当前分期相符的精细操作。"
    tone_note = "；若张力在重复动作中明显增高，应降低速度或减少阻力" if tone not in {"0", "1"} else ""
    return {"F": frequency, "I": repetitions + tone_note, "T_type": training_type, "T_time": duration}


def _strategies(
    *,
    scores: Mapping[str, Any],
    biomarkers: list[dict[str, Any]],
    llm_recommendations: list[str],
    source_ids: list[str],
    resource_dir: Path,
) -> list[dict[str, Any]]:
    stage = int(scores["brunnstrom_hand"])
    hand_score = int(scores["fma_hand"])
    tone = str(scores["hand_mas"])
    scopes = _strategy_scopes(scores)
    output: list[dict[str, Any]] = []
    for index, scope in enumerate(scopes, start=1):
        llm_text = _non_numeric_advice(llm_recommendations[index - 1] if index - 1 < len(llm_recommendations) else "")
        if not llm_text:
            llm_text = "建议以可完成的主动参与为基础，在治疗师观察下逐步提高动作的选择性、稳定性和任务迁移能力"
        if scope == "wrist":
            title = "腕位控制与腕手协同训练"
            fact = f"本次FMA腕部为{scores['fma_wrist']}/10分，训练可围绕腕部稳定、屈伸转换及其在手部任务中的协同展开。"
            observation = f"腕屈伸肌共收缩指数为{_marker_value(biomarkers, 'wrist_co_contraction_index')}，腕伸方向峰值角速度为{_marker_value(biomarkers, 'wrist_extension_peak_velocity')}"
            precaution = "训练时保持前臂和腕部在舒适对线范围内，避免以肩部抬高或躯干侧倾代替腕部运动；如出现疼痛、麻木、张力持续增高或动作质量明显下降，应降低难度并由治疗师重新评估。"
        else:
            title = "患手抓握、释放与精细操作训练" if stage >= 4 else "患手主动开启、抓握与释放训练"
            fact = f"本次FMA手部为{hand_score}/20分、手MAS为{tone}级、Brunnstrom手功能为{stage}期，训练应与当前主动控制能力和肌张力表现相匹配。"
            observation = f"指屈伸肌共收缩指数为{_marker_value(biomarkers, 'finger_co_contraction_index')}，伸指峰值角速度为{_marker_value(biomarkers, 'finger_extension_peak_velocity')}"
            precaution = "训练中避免强行掰指、持续性紧握或长时间压迫掌面；若出现指间关节疼痛、皮肤受压、麻木、张力增高后难以放松或动作失控，应暂停并调整辅助方式和任务难度。"
        refs = "、".join(source_ids[:4]) or "本次未检索到可公开引用的来源UID"
        output.append(
            {
                "number": f"策略{'一' if index == 1 else '二'}",
                "scope": scope,
                "title": title,
                "recommendation": fact + llm_text.rstrip("。") + "。",
                "hand_rehabilitation_action": _gesture_action(resource_dir, scores, scope),
                "fitt": _fitt(stage, tone, scope),
                "basis": f"依据本次真实量表来源与原始信号计算结果：{observation}；上述无统一常模指标只用于同条件纵向观察。知识来源UID：{refs}。",
                "progression": "连续复测中如任务完成质量稳定、代偿不增加且训练后能在合理休息内恢复，可一次只调整一个变量，例如减少辅助、扩大活动范围、增加任务步骤或提高准确性要求；复测时继续记录相同量表与同条件biomarker趋势。",
                "precaution": precaution,
                "llm_recommendation_source": "planner_rag.ReportGenerator",
            }
        )
    return output


def _source_ids(orchestration_result: Any) -> list[str]:
    values: list[str] = []
    report = orchestration_result.report
    if report:
        candidates = list(report.citations)
        for finding in report.findings:
            candidates.extend(finding.citations)
        for value in candidates:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
    return values


def _references(orchestration_result: Any, source_ids: list[str]) -> list[dict[str, Any]]:
    try:
        snapshot = knowledge_admin.load_snapshot()
        by_id = {str(source.get("source_id") or ""): source for source in snapshot.sources}
    except Exception:  # noqa: BLE001
        by_id = {}
    retrieval_metadata: dict[str, dict[str, Any]] = {}
    if orchestration_result.retrieval is not None:
        for evidence in orchestration_result.retrieval.evidence:
            for source_id in evidence.source_ids:
                retrieval_metadata.setdefault(str(source_id), {}).update(evidence.metadata or {})
    output: list[dict[str, Any]] = []
    for index, source_id in enumerate(source_ids, start=1):
        source = by_id.get(source_id, {})
        metadata = retrieval_metadata.get(source_id, {})
        output.append(
            {
                "number": index,
                "uid": source_id,
                "title": str(source.get("title") or "来源标题不可用"),
                "page": str(metadata.get("page") or metadata.get("pages") or source.get("page") or source.get("pages") or "页码不可用"),
                "url": str(metadata.get("url") or source.get("url") or ""),
                "use": str(source.get("scope") or source.get("note") or "用于本次检索增强的医学解释与训练建议。"),
            }
        )
    return output


def _public_profile(
    *,
    patient: PatientInfo,
    scores: Mapping[str, Any],
    biomarkers: Optional[Mapping[str, Any]],
    orchestration_result: Any,
    publication_status: str,
    publication_warnings: list[str],
    resource_dir: Path,
) -> dict[str, Any]:
    generated_at = _now()
    report_token = uuid4().hex[:10].upper()
    rows, coverage = _biomarker_rows(biomarkers)
    source_ids = _source_ids(orchestration_result)
    report = orchestration_result.report
    recommendations = list(report.recommendations) if report else []
    identity = {
        "case_id": f"SR-{report_token}",
        "report_number": f"SR-{report_token}-R01",
        "patient_code": "匿名患者",
    }
    return {
        "schema_version": "rehab.production_strategy_report.v1",
        "template_version": "v11-production",
        "generated_at": generated_at,
        "publication": {"status": publication_status, "warnings": publication_warnings},
        "report_identity": identity,
        "patient": {
            "sex": patient.sex,
            "age": patient.age if patient.age is not None else "本次未提供",
            "diagnosis": patient.diagnosis,
            "disease_days": patient.disease_days if patient.disease_days is not None else "本次未提供",
            "paralysis_side": patient.paralysis_side,
        },
        "clinical_fact_card": {
            "scores": {
                "fma_wrist": {"value": scores.get("fma_wrist"), "maximum": 10, "source": scores["field_sources"]["fma_wrist"]},
                "fma_hand": {"value": scores["fma_hand"], "maximum": 20, "source": scores["field_sources"]["fma_hand"]},
                "wrist_mas": {"value": scores.get("wrist_mas"), "source": scores["field_sources"]["wrist_mas"]},
                "hand_mas": {"value": scores["hand_mas"], "source": scores["field_sources"]["hand_mas"]},
                "brunnstrom_hand": {"value": scores["brunnstrom_hand"], "source": scores["field_sources"]["brunnstrom_hand"]},
            },
            "score_mode": scores["mode"],
        },
        "score_prediction_policy": {
            "deep_learning_score_prediction_used": scores["deep_learning_score_prediction_used"],
            "displayed_score_source": scores["primary_source"],
            "note": "报告事实由程序从本次manifest、患者主数据和真实pipeline结果确定性装配，LLM无权修改。",
        },
        "subtype_assessment": _subtype(scores),
        "biomarker_coverage": coverage,
        "biomarkers": rows,
        "recommendations": _strategies(
            scores=scores,
            biomarkers=rows,
            llm_recommendations=recommendations,
            source_ids=source_ids,
            resource_dir=resource_dir,
        ),
        "knowledge_evidence": {
            "core_references": _references(orchestration_result, source_ids),
            "knowledge_graph_notice": "本节仅列出本次planner_rag链路实际引用的来源；UID来自知识库，页码和原文链接只在真实元数据存在时展示。",
        },
        "pipeline_provenance": {
            "pipeline_mode": "real_upload_pipeline",
            "quality_gate": orchestration_result.quality_gate.decision.value if orchestration_result.quality_gate else None,
            "planner_rag_status": orchestration_result.status.value,
            "retrieval_status": orchestration_result.retrieval.status.value if orchestration_result.retrieval else None,
            "knowledge_graph_mode": orchestration_result.trace.artifact_refs.get("knowledge_graph_mode"),
            "knowledge_graph_status": orchestration_result.trace.artifact_refs.get("knowledge_graph_status"),
            "validator_status": orchestration_result.validation.status.value if orchestration_result.validation else None,
            "report_generation_mode": report.generation_mode if report else None,
            "preset_case_fixture_used": False,
        },
        "privacy_policy": "公开报告不显示真实姓名、医院患者ID、上传文件名或原始信号路径。",
        "disclaimer": "本报告用于康复临床辅助与研究记录，不能替代医生面诊、体格检查和个体化训练方案。",
    }


def _publication(orchestration_result: Any, biomarkers: Optional[Mapping[str, Any]]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if orchestration_result.status != PipelineRunStatus.COMPLETED:
        if orchestration_result.failure:
            warnings.append(f"{orchestration_result.failure.module.value}：{orchestration_result.failure.message}")
        warnings.extend(orchestration_result.block_reasons)
        return "MANUAL_REVIEW", warnings or ["pipeline未完成"]
    validation = orchestration_result.validation
    if validation is None:
        return "MANUAL_REVIEW", ["Validator未返回结果"]
    warnings.extend(issue.message for issue in validation.issues)
    if validation.status == ValidationStatus.MANUAL_REVIEW:
        return "MANUAL_REVIEW", warnings or ["Validator要求人工复核"]
    coverage = (biomarkers or {}).get("coverage") or {}
    available = int(coverage.get("available") or 0)
    total = int(coverage.get("total") or 0)
    if validation.status == ValidationStatus.WARNING or available < total:
        if available < total:
            warnings.append(f"biomarker覆盖{available}/{total}，缺失项已明确标示")
        return "WARNING", warnings
    return "PASSED", warnings


def run_report_pipeline(
    *,
    package_root: Path,
    institution: str,
    registry: Any,
    resource_dir: Path,
) -> PipelineArtifacts:
    """Run the real report chain for one already-securely-extracted upload."""
    events = [_stage_event("data_parsing", "started")]
    package = read_eval_package(package_root, institution=institution)
    if not package.eeg_paths or not package.emg_paths:
        raise StrategyPipelineError("数据包没有可用的EEG/EMG/IMU主动任务")
    patient = _resolve_patient(package)
    manifest = _read_manifest(package_root)
    events.append(_stage_event("data_parsing", "completed", f"trials={package.n_trials}"))

    pipeline_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    events.append(_stage_event("biomarker_and_dl_pipeline", "started"))
    inference_result = run_pipeline(
        package.eeg_paths,
        package.emg_paths,
        registry,
        pipeline_queue,
        affected_side=patient.paralysis_side,
        institution=package.institution,
        trial_details=package.trial_details,
    )
    events.append(_stage_event("biomarker", "completed", json.dumps((inference_result.get("_biomarkers") or {}).get("coverage") or {}, ensure_ascii=False)))
    scores = _score_bundle(manifest, inference_result)
    events.append(_stage_event("clinical_score_mapping", "completed", scores["mode"]))

    predictions_for_report = {
        "FMA_UE": scores["fma_hand"],
        "hand_tone": scores["hand_mas"],
        "hand_function": scores["brunnstrom_hand"],
    }
    model_id = llm_model_name()
    if not model_id:
        raise StrategyPipelineError("当前未配置报告大模型")
    request = production_adapter.adapt_production_input(
        patient=patient,
        predictions_raw=predictions_for_report,
        biomarkers=inference_result.get("_biomarkers"),
        quality=inference_result.get("_quality") or {},
        assessment_id=str(package.manifest_summary.get("assessment_id") or "") or None,
        patient_id=patient.patient_id,
        report_model_id=model_id,
        context_id=f"strategy-{uuid4().hex}",
    )
    orchestrator = production_adapter.build_production_orchestrator(request.report_model_id)
    orchestration_result = orchestrator.run(request.assessment_input)
    if orchestration_result.status == PipelineRunStatus.FAILED:
        failure = orchestration_result.failure
        if failure is not None and failure.module.value == "ReportGenerator" and orchestration_result.report_input is not None:
            orchestration_result, _ = production_adapter.recover_report_generator_failure(
                orchestration_result,
                report_model_id=request.report_model_id,
            )
    for event in orchestration_result.module_events:
        events.append(_stage_event(event.module.value, event.status.value, event.detail or ""))
    publication_status, publication_warnings = _publication(orchestration_result, inference_result.get("_biomarkers"))
    events.append(_stage_event("publication_gate", "completed", publication_status))
    profile = None
    if publication_status in {"PASSED", "WARNING"}:
        profile = _public_profile(
            patient=patient,
            scores=scores,
            biomarkers=inference_result.get("_biomarkers"),
            orchestration_result=orchestration_result,
            publication_status=publication_status,
            publication_warnings=publication_warnings,
            resource_dir=resource_dir,
        )
    return PipelineArtifacts(
        package=package,
        patient=patient,
        score_bundle=scores,
        inference_result=inference_result,
        orchestration_result=orchestration_result,
        publication_status=publication_status,
        publication_warnings=publication_warnings,
        profile=profile,
        stage_events=events,
    )


def internal_result(artifacts: PipelineArtifacts) -> dict[str, Any]:
    """Serializable private result retained even when publication is blocked."""
    return {
        "schema_version": "rehab.strategy_report_internal_result.v1",
        "created_at": _now(),
        "publication_status": artifacts.publication_status,
        "publication_warnings": artifacts.publication_warnings,
        "patient": artifacts.patient.model_dump(mode="json"),
        "manifest_summary": artifacts.package.manifest_summary,
        "clinical_scores": artifacts.score_bundle,
        "predictions": {
            key: value
            for key, value in artifacts.inference_result.items()
            if not key.startswith("_")
        },
        "biomarkers": artifacts.inference_result.get("_biomarkers"),
        "quality": artifacts.inference_result.get("_quality"),
        "orchestration": artifacts.orchestration_result.model_dump(mode="json"),
        "stage_events": artifacts.stage_events,
        "preset_case_fixture_used": False,
    }


__all__ = [
    "PipelineArtifacts",
    "StrategyPipelineError",
    "internal_result",
    "run_report_pipeline",
]
