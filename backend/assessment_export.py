"""Persist machine-readable and human-readable assessment export files.

Each MySQL-backed assessment can be materialized as:

* result.json       stable structured payload for device/system integration
* report.pdf        clinician-facing PDF summary
* manifest.json     file metadata and checksums
* export.zip        bundle containing the three files above
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from biomarker_refs import marker_ref

SCHEMA_VERSION = "rehab.assessment_result.v2"

_GROUP_LABELS = {
    "emg": "肌电标志物",
    "eeg": "脑电标志物",
    "imu": "运动学标志物",
}

_GROUP_DESCRIPTIONS = {
    "emg": "基于本次评估包中的肌电信号分析结果。",
    "eeg": "基于本次评估包中的脑电信号分析结果。",
    "imu": "基于本次评估包中的运动学/IMU 信号分析结果。",
}

_GROUP_ORDER = {"emg": 0, "eeg": 1, "imu": 2}

_BIOMARKER_DISPLAY_NOTE = (
    "生物标志物受设备与采集流程影响，仅用于同一患者在相同设备、相同采集流程下的连续变化观察，"
    "不作为正常或异常的诊断阈值。"
)


def export_root() -> Path:
    return Path(os.environ.get("EXPORT_ROOT", str(Path(__file__).resolve().parents[1] / "exports")))


@dataclass(frozen=True)
class ExportBundle:
    root: Path
    result_json: Path
    report_pdf: Path
    manifest_json: Path
    export_zip: Path
    manifest: Dict[str, Any]


def _now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _safe_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text)
    return text.strip("._") or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(name: str, path: Path) -> Dict[str, Any]:
    return {
        "name": name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _model(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dl_model_version": a.get("model_version"),
        "llm_provider": a.get("llm_provider"),
        "llm_model": a.get("llm_model"),
    }


def _as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _roman(value: Any) -> Optional[str]:
    n = _as_int(value)
    if n is None:
        return None
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}.get(n, str(n))


def _md_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.replace("**", "")
    return text.strip()


def _numbered_text(line: str) -> Optional[str]:
    m = re.match(r"^\s*\d+[.、]\s*(.+?)\s*$", line)
    return m.group(1).strip() if m else None


_SPECIFIC_METHOD_SEGMENT = re.compile(
    r"(?:^|[；;]\s*)具体方法(?:[（(][^）)]*[）)])?\s*[：:]\s*.*?"
    r"(?=(?:[；;]\s*)?(?:训练剂量|反馈标准|调整原则|安全注意)\s*[：:]|$)"
)


def _strip_specific_method(value: Any) -> str:
    text = _SPECIFIC_METHOD_SEGMENT.sub("", str(value or "").strip())
    text = re.sub(r"^[；;\s]+|[；;\s]+$", "", text)
    return re.sub(r"[；;]\s*[；;]", "；", text).strip()


def _group_key_from_heading(text: str) -> Optional[str]:
    if "肌电" in text:
        return "emg"
    if "脑电" in text:
        return "eeg"
    if "运动学" in text or "IMU" in text.upper():
        return "imu"
    return None


def _md_cells(line: str) -> List[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    return [_md_text(cell.strip()) for cell in line.strip("|").split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells)


def _read_md_table(lines: List[str], start: int) -> Tuple[List[List[str]], int]:
    rows: List[List[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = _md_cells(lines[i])
        if cells and not _is_separator_row(cells):
            rows.append(cells)
        i += 1
    return rows, i


def _parse_report_markdown(report_text: str) -> Dict[str, Any]:
    """Extract the deterministic report sections from our own Markdown output.

    Older rows only persist Markdown, not the raw LLM JSON. The report skeleton is
    stable (``report_builder.render_markdown``), so export v2 reconstructs the
    structured fields from that Markdown rather than changing the database.
    """
    parsed: Dict[str, Any] = {
        "overall_interpretation": None,
        "biomarker_text": {},
        "overall_subtype": None,
        "treatment_strategy": [],
        "gesture_plan": [],
        "weekly_plan": [],
        "warnings": [],
        "next_assessment": None,
    }
    lines = (report_text or "").splitlines()
    current_group: Optional[str] = None
    section: Optional[str] = None
    collect_strategy = False

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        clean = _md_text(line)

        if line.startswith("## "):
            collect_strategy = False
            if "综合亚型" in clean:
                section = "strategy"
            elif "预警" in clean:
                section = "warnings"
            elif "下次评估" in clean:
                section = "next_assessment"
            else:
                section = None

        if line.startswith("####"):
            current_group = _group_key_from_heading(clean)

        if line.startswith("|"):
            table, i = _read_md_table(lines, i)
            if table:
                headers = table[0]
                rows = table[1:]
                if headers[:5] == ["标志物", "当前值", "参考范围", "解读", "治疗建议"] and current_group:
                    group_map = parsed["biomarker_text"].setdefault(current_group, {})
                    for row in rows:
                        if len(row) >= 5:
                            group_map[row[0]] = {
                                "current_value_text": row[1],
                                "reference_range_text": row[2],
                                "interpretation": None,
                                "treatment_advice": None,
                                "legacy_reference_rule": True,
                            }
                elif (
                    headers[:3] == ["标志物", "当前值", "解读"]
                    and len(headers) >= 4
                    and headers[3] in {"训练/随访建议", "治疗建议"}
                    and current_group
                ):
                    group_map = parsed["biomarker_text"].setdefault(current_group, {})
                    for row in rows:
                        if len(row) >= 4:
                            group_map[row[0]] = {
                                "current_value_text": row[1],
                                "interpretation": row[2],
                                "treatment_advice": row[3],
                            }
                elif "手势名称" in headers:
                    name_idx = headers.index("手势名称")
                    for row in rows:
                        if len(row) > name_idx:
                            parsed["gesture_plan"].append({
                                "name": row[name_idx],
                                "purpose": row[2] if len(row) > 2 else None,
                                "assistance": row[3] if len(row) > 3 else None,
                                "repetitions": row[4] if len(row) > 4 else None,
                            })
                elif headers[:3] == ["训练日", "训练内容", "预计时长"]:
                    for row in rows:
                        if len(row) >= 3:
                            parsed["weekly_plan"].append({
                                "day": row[0],
                                "content": row[1],
                                "duration": row[2],
                            })
            continue

        if line.startswith("**临床解读"):
            parsed["overall_interpretation"] = _md_text(line.split("：", 1)[-1])
        elif "患者可归类为" in clean:
            m = re.search(r"患者可归类为：(.+)$", clean)
            if m:
                parsed["overall_subtype"] = m.group(1).strip(" 。")
        elif clean.startswith("治疗策略要点"):
            collect_strategy = True
        elif collect_strategy:
            item = _numbered_text(clean)
            if item:
                item = _strip_specific_method(item)
                if item:
                    parsed["treatment_strategy"].append(item)
        elif section == "warnings":
            item = _numbered_text(clean)
            if item:
                parsed["warnings"].append(item)
        elif section == "next_assessment" and clean.startswith("建议："):
            parsed["next_assessment"] = clean.split("：", 1)[-1].strip()

        i += 1

    return parsed


def _value_text(value: Any, unit: Any = None) -> str:
    if value in (None, "", "—"):
        return "—"
    if isinstance(value, float) and value.is_integer():
        value_s = str(int(value))
    else:
        value_s = str(value).strip()
    unit_s = str(unit or "").strip()
    if unit_s and unit_s != "—" and unit_s not in value_s:
        return f"{value_s}{unit_s}"
    return value_s


def _is_available(marker: Dict[str, Any]) -> bool:
    if "available" not in marker:
        return True
    return bool(marker.get("available"))


def _biomarker_coverage(assessment: Dict[str, Any]) -> Dict[str, Any]:
    items = assessment.get("biomarker_items") or []
    raw = assessment.get("biomarkers") if isinstance(assessment.get("biomarkers"), dict) else {}
    raw_cov = raw.get("coverage") if isinstance(raw, dict) else None
    total = int(raw_cov.get("total")) if isinstance(raw_cov, dict) and raw_cov.get("total") is not None else len(items)
    available = (
        int(raw_cov.get("available"))
        if isinstance(raw_cov, dict) and raw_cov.get("available") is not None
        else sum(1 for item in items if _is_available(item))
    )
    missing = (
        list(raw_cov.get("missing_keys") or [])
        if isinstance(raw_cov, dict)
        else [item.get("marker_key") for item in items if not _is_available(item)]
    )
    missing = [str(k) for k in missing if k]
    return {
        "available_count": available,
        "total_count": total,
        "missing_keys": missing,
        "policy": "数据不足或当前采集格式暂不支持的指标不生成临床解读。",
    }


def _biomarker_sections(assessment: Dict[str, Any], parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    text_by_group = parsed.get("biomarker_text") or {}
    for marker in assessment.get("biomarker_items") or []:
        if not _is_available(marker):
            continue
        group_key = marker.get("group_key") or "other"
        marker_name = marker.get("marker_name") or marker.get("marker_key") or "未命名指标"
        row_text = (text_by_group.get(group_key) or {}).get(marker_name, {})
        current_value_text = _value_text(marker.get("value_text"), marker.get("unit"))
        ref = marker_ref(str(marker.get("marker_key") or ""))
        reference_metadata = {
            "display": False,
            "text": None,
            "type": ref["reference_type"] if ref else "unknown",
            "absolute_comparison_applicable": bool(ref and ref["absolute_comparison_applicable"]),
            "expected_direction": ref["expected_direction"] if ref else "n/a",
            "range": (
                {"low": ref["lo"], "high": ref["hi"]}
                if ref and (ref["lo"] is not None or ref["hi"] is not None)
                else None
            ),
            "confidence": ref["confidence"] if ref else "none",
            "source_ids": list(ref["source"]) if ref else [],
            "note": ref["note"] if ref else None,
            "stored_summary": marker.get("ref_range") or row_text.get("reference_range_text"),
        }
        indicator = {
            "indicator_key": marker.get("marker_key"),
            "indicator_name": marker_name,
            "current_value": {
                "value": marker.get("value_num") if marker.get("value_num") is not None else marker.get("value_text"),
                "unit": marker.get("unit"),
                "text": row_text.get("current_value_text") or current_value_text,
            },
            "reference_range": reference_metadata,
            "valid_trial_count": marker.get("n_valid"),
            "interpretation": row_text.get("interpretation"),
            "treatment_advice": row_text.get("treatment_advice"),
            "interpretation_status": (
                "legacy_hidden"
                if row_text.get("legacy_reference_rule")
                else "available"
                if row_text.get("interpretation") and row_text.get("treatment_advice")
                else "not_available"
            ),
        }
        if marker.get("note"):
            indicator["note"] = marker.get("note")
        grouped.setdefault(group_key, []).append(indicator)

    sections: List[Dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda g: _GROUP_ORDER.get(g, 99)):
        sections.append({
            "section_key": group_key,
            "section_name": _GROUP_LABELS.get(group_key) or group_key,
            "description": _GROUP_DESCRIPTIONS.get(group_key),
            "indicators": grouped[group_key],
        })
    return sections


def _clinical_scores(assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage = _as_int(assessment.get("hand_function"))
    stage_roman = _roman(stage)
    return [
        {
            "key": "FMA_UE",
            "indicator_name": "FMA-UE 手部分数",
            "value": assessment.get("fma_ue"),
            "unit": "分",
            "scale": "0-20",
            "display_value": _value_text(assessment.get("fma_ue"), "分"),
        },
        {
            "key": "hand_tone",
            "indicator_name": "手部肌张力（MAS）",
            "value": assessment.get("hand_tone"),
            "unit": "级",
            "display_value": _value_text(assessment.get("hand_tone"), "级"),
        },
        {
            "key": "hand_function",
            "indicator_name": "Brunnstrom 手功能分期",
            "value": stage,
            "stage_roman": stage_roman,
            "display_value": f"{stage_roman or stage or '—'}期" if (stage_roman or stage) else "—",
        },
    ]


def _patient_basic_info(assessment: Dict[str, Any]) -> Dict[str, Any]:
    disease_days = assessment.get("disease_days")
    diagnosis = assessment.get("diagnosis")
    side = assessment.get("paralysis_side")
    stroke_type = "，".join(str(x) for x in (diagnosis, f"{side}侧偏瘫" if side else None) if x)
    return {
        "patient_db_id": assessment.get("patient_db_id"),
        "patient_id": assessment.get("patient_id"),
        "name": assessment.get("name"),
        "age": assessment.get("age"),
        "sex": assessment.get("sex"),
        "diagnosis": diagnosis,
        "paralysis_side": side,
        "disease_course": {
            "value": disease_days,
            "unit": "天",
            "description": f"卒中后{disease_days}天" if disease_days is not None else None,
        },
        "stroke_type": stroke_type or None,
    }


def build_result_payload(assessment: Dict[str, Any]) -> Dict[str, Any]:
    parsed = _parse_report_markdown(assessment.get("report") or "")
    stage = _as_int(assessment.get("hand_function"))
    stage_roman = _roman(stage)
    model = _model(assessment)
    coverage = _biomarker_coverage(assessment)
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _now_iso(),
        "report_metadata": {
            "assessment_db_id": assessment.get("id"),
            "assessment_id": assessment.get("assessment_id"),
            "session_id": assessment.get("session_id"),
            "source": assessment.get("source"),
            "institution": assessment.get("institution"),
            "package_name": assessment.get("package_name"),
            "package_hash": assessment.get("package_hash"),
            "n_trials": assessment.get("n_trials"),
            "created_at": assessment.get("created_at"),
            "assessment_time": assessment.get("assessment_time"),
            "report_status": assessment.get("report_status"),
            "parse_warnings": assessment.get("parse_warnings") or [],
            "model": model,
        },
        "patient_basic_info": _patient_basic_info(assessment),
        "stage_assessment": {
            "brunnstrom_stage": {
                "stage": stage_roman,
                "stage_number": stage,
                "assessment_region": f"{assessment.get('paralysis_side')}手部" if assessment.get("paralysis_side") else None,
                "clinical_interpretation": parsed.get("overall_interpretation"),
            }
        },
        "clinical_scores": _clinical_scores(assessment),
        "biomarker_coverage": coverage,
        "biomarker_interpretation_policy": {
            "user_facing_reference_range": "hidden",
            "comparison_basis": "same_patient_same_device_same_protocol_longitudinal",
            "single_measurement_rule": "do_not_classify_normal_abnormal",
            "display_note": _BIOMARKER_DISPLAY_NOTE,
        },
        "biomarker_sections": _biomarker_sections(assessment, parsed),
        "subtype_classification_and_treatment_strategy": {
            "subtype_classification": {
                "main_stage": f"brunnstrom_stage_{stage_roman}" if stage_roman else None,
                "overall_subtype": parsed.get("overall_subtype"),
            },
            "treatment_strategy": {
                "overall_strategies": parsed.get("treatment_strategy") or [],
            },
        },
        "next_week_training_plan": {
            "recommended_gestures": parsed.get("gesture_plan") or [],
            "weekly_schedule": parsed.get("weekly_plan") or [],
            "not_recommended_gestures": [],
        },
        "warnings_and_recommendations": {
            "warnings": parsed.get("warnings") or [],
            "next_assessment": {
                "recommendation": parsed.get("next_assessment"),
            },
        },
        "natural_language_summary": {
            "for_doctor": " ".join(
                x for x in (parsed.get("overall_interpretation"), parsed.get("overall_subtype")) if x
            ) or None,
            "for_patient": (
                "请在康复师指导下按本报告训练计划执行，并按建议时间复评。"
                if parsed.get("next_assessment") else None
            ),
        },
    }


def _pdf_text(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def write_report_pdf(path: Path, payload: Dict[str, Any]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("缺少 reportlab，无法生成 PDF：pip install reportlab") from exc

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "BaseCN",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=10,
        leading=15,
        wordWrap="CJK",
    )
    title = ParagraphStyle(
        "TitleCN",
        parent=base,
        fontSize=18,
        leading=24,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=base,
        fontSize=13,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#0a6573"),
    )
    h3 = ParagraphStyle(
        "H3CN",
        parent=base,
        fontSize=11,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#26364a"),
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=base,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#566579"),
    )

    def p(text: Any, style: ParagraphStyle = base) -> Paragraph:
        escaped = (
            _pdf_text(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        return Paragraph(escaped, style)

    def kv_table(
        rows: List[List[Any]],
        widths: Optional[List[float]] = None,
        *,
        font_size: float = 9,
        header: bool = True,
    ) -> Table:
        table = Table(
            [[p(cell) for cell in row] for row in rows],
            colWidths=widths,
            repeatRows=1 if rows and header else 0,
        )
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 4),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe5ee")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style_cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")))
        if len(rows) > 2:
            style_cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]))
        table.setStyle(
            TableStyle(style_cmds)
        )
        return table

    meta = payload["report_metadata"]
    patient = payload["patient_basic_info"]
    stage = payload["stage_assessment"]["brunnstrom_stage"]
    model = meta.get("model") or {}
    coverage = payload.get("biomarker_coverage") or {}
    story: List[Any] = [
        p("智能康复评估结果报告", title),
        p(f"导出时间：{payload['exported_at']}"),
        Spacer(1, 4 * mm),
        p("患者与评估信息", h2),
        kv_table(
            [
                ["字段", "内容", "字段", "内容"],
                ["患者编号", _pdf_text(patient.get("patient_id")), "姓名", _pdf_text(patient.get("name"))],
                ["性别", _pdf_text(patient.get("sex")), "年龄", _pdf_text(patient.get("age"))],
                ["诊断", _pdf_text(patient.get("diagnosis")), "偏瘫侧", _pdf_text(patient.get("paralysis_side"))],
                ["病程", _value_text((patient.get("disease_course") or {}).get("value"), "天"),
                 "采集时间", _pdf_text(meta.get("assessment_time"))],
                ["Session", _pdf_text(meta.get("session_id")), "Assessment ID", _pdf_text(meta.get("assessment_id"))],
                ["数据包", _pdf_text(meta.get("package_name")), "Trial 数", _pdf_text(meta.get("n_trials"))],
            ],
            [28 * mm, 52 * mm, 28 * mm, 52 * mm],
        ),
        p("本次评估结果", h2),
        kv_table(
            [["指标", "结果", "量表/范围"]]
            + [
                [score.get("indicator_name"), score.get("display_value"), score.get("scale") or "—"]
                for score in payload.get("clinical_scores", [])
            ],
            [60 * mm, 62 * mm, 58 * mm],
        ),
        p("Brunnstrom 分期", h3),
        p(
            f"{_pdf_text(stage.get('stage'))}期"
            f"（{_pdf_text(stage.get('assessment_region'))}）："
            f"{_pdf_text(stage.get('clinical_interpretation'))}"
        ),
    ]

    story.extend([
        p("关键生物标志物输出与解读", h2),
        p(
            f"本次可计算 {coverage.get('available_count', 0)}/{coverage.get('total_count', 0)} 项；"
            "数据不足或当前采集格式暂不支持的指标不生成临床解读。",
            note,
        ),
        p(_BIOMARKER_DISPLAY_NOTE, note),
    ])
    for section in payload.get("biomarker_sections", []) or []:
        story.append(p(section.get("section_name"), h3))
        indicators = section.get("indicators", []) or []
        legacy_only = bool(indicators) and all(
            marker.get("interpretation_status") == "legacy_hidden" for marker in indicators
        )
        rows = [["指标", "当前值"]] if legacy_only else [["指标", "当前值", "解读", "训练/随访建议"]]
        for marker in indicators:
            current = marker.get("current_value") or {}
            if legacy_only:
                rows.append([marker.get("indicator_name"), current.get("text")])
            else:
                rows.append([
                    marker.get("indicator_name"),
                    current.get("text"),
                    marker.get("interpretation"),
                    marker.get("treatment_advice"),
                ])
        widths = [92 * mm, 90 * mm] if legacy_only else [42 * mm, 30 * mm, 54 * mm, 56 * mm]
        story.append(kv_table(rows, widths, font_size=8))
        if legacy_only:
            story.append(p("该历史报告采用旧版参考规则，单次高低判断已隐藏。", note))

    strategy = payload.get("subtype_classification_and_treatment_strategy") or {}
    subtype = (strategy.get("subtype_classification") or {}).get("overall_subtype")
    strategies = (strategy.get("treatment_strategy") or {}).get("overall_strategies") or []
    story.extend([p("综合亚型界定与治疗策略", h2)])
    if subtype:
        story.append(p(f"综合亚型：{subtype}"))
    for i, item in enumerate(strategies, 1):
        story.append(p(f"{i}. {item}"))

    plan = payload.get("next_week_training_plan") or {}
    gestures = plan.get("recommended_gestures") or []
    weekly = plan.get("weekly_schedule") or []
    story.extend([p("下周训练计划", h2)])
    if gestures:
        story.append(p("推荐手势组合", h3))
        rows = [["手势名称", "训练目的", "辅助力度", "重复次数"]]
        for item in gestures:
            rows.append([item.get("name"), item.get("purpose"), item.get("assistance"), item.get("repetitions")])
        story.append(kv_table(rows, [34 * mm, 62 * mm, 44 * mm, 42 * mm], font_size=8.5))
    if weekly:
        story.append(p("每周安排", h3))
        rows = [["训练日", "训练内容", "预计时长"]]
        for item in weekly:
            rows.append([item.get("day"), item.get("content"), item.get("duration")])
        story.append(kv_table(rows, [32 * mm, 110 * mm, 40 * mm], font_size=8.5))
    if not gestures and not weekly:
        story.append(p("本次未生成结构化手势计划；请结合康复师建议安排训练。", note))

    warn = payload.get("warnings_and_recommendations") or {}
    warnings = warn.get("warnings") or []
    next_assessment = (warn.get("next_assessment") or {}).get("recommendation")
    story.extend([p("预警与下次评估", h2)])
    for i, item in enumerate(warnings, 1):
        story.append(p(f"{i}. {item}"))
    if next_assessment:
        story.append(p(f"下次评估建议：{next_assessment}"))

    story.extend(
        [
            p("追溯信息", h2),
            kv_table(
                [
                    ["字段", "内容"],
                    ["数据包 SHA-256", _pdf_text(meta.get("package_hash"))],
                    ["DL 模型版本", _pdf_text(model.get("dl_model_version"))],
                    ["LLM", f"{_pdf_text(model.get('llm_provider'))} / {_pdf_text(model.get('llm_model'))}"],
                    ["JSON Schema", payload["schema_version"]],
                ],
                [38 * mm, 130 * mm],
            ),
        ]
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    doc = SimpleDocTemplate(
        str(tmp),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="智能康复评估结果报告",
    )
    doc.build(story)
    tmp.replace(path)


def export_filename(assessment: Dict[str, Any], suffix: str) -> str:
    patient_id = _safe_part(assessment.get("patient_id"), "patient")
    assessment_id = _safe_part(assessment.get("assessment_id") or assessment.get("id"), "assessment")
    return f"rehab_assessment_{patient_id}_{assessment_id}.{suffix}"


def ensure_assessment_export(assessment: Dict[str, Any], force: bool = False) -> ExportBundle:
    assessment_db_id = int(assessment["id"])
    root = export_root() / "assessments" / str(assessment_db_id)
    root.mkdir(parents=True, exist_ok=True)

    result_json = root / "result.json"
    report_pdf = root / "report.pdf"
    manifest_json = root / "manifest.json"
    export_zip = root / "export.zip"

    if force or not result_json.exists() or not report_pdf.exists():
        payload = build_result_payload(assessment)
        _write_json(result_json, payload)
        write_report_pdf(report_pdf, payload)

    file_entries = []
    for name, path in (("result.json", result_json), ("report.pdf", report_pdf)):
        file_entries.append(file_info(name, path))

    manifest = {
        "schema_version": "rehab.export_manifest.v1",
        "result_schema_version": SCHEMA_VERSION,
        "exported_at": _now_iso(),
        "assessment_db_id": assessment_db_id,
        "patient_id": assessment.get("patient_id"),
        "assessment_id": assessment.get("assessment_id"),
        "files": file_entries,
    }
    _write_json(manifest_json, manifest)

    tmp_zip = export_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_json, "result.json")
        zf.write(report_pdf, "report.pdf")
        zf.write(manifest_json, "manifest.json")
    tmp_zip.replace(export_zip)
    return ExportBundle(root, result_json, report_pdf, manifest_json, export_zip, manifest)
