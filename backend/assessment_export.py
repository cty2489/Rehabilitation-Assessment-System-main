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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "rehab.assessment_result.v1"


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


def _predictions(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "FMA_UE": a.get("fma_ue"),
        "BI": a.get("bi"),
        "hand_tone": a.get("hand_tone"),
        "hand_function": a.get("hand_function"),
    }


def _patient(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patient_db_id": a.get("patient_db_id"),
        "patient_id": a.get("patient_id"),
        "name": a.get("name"),
        "sex": a.get("sex"),
        "age": a.get("age"),
        "diagnosis": a.get("diagnosis"),
        "paralysis_side": a.get("paralysis_side"),
        "disease_days": a.get("disease_days"),
    }


def _assessment(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": a.get("id"),
        "source": a.get("source"),
        "institution": a.get("institution"),
        "assessment_id": a.get("assessment_id"),
        "session_id": a.get("session_id"),
        "package_name": a.get("package_name"),
        "package_hash": a.get("package_hash"),
        "n_trials": a.get("n_trials"),
        "created_at": a.get("created_at"),
        "assessment_time": a.get("assessment_time"),
        "report_status": a.get("report_status"),
        "parse_warnings": a.get("parse_warnings"),
    }


def _model(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dl_model_version": a.get("model_version"),
        "llm_provider": a.get("llm_provider"),
        "llm_model": a.get("llm_model"),
    }


def build_result_payload(assessment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _now_iso(),
        "assessment": _assessment(assessment),
        "patient": _patient(assessment),
        "predictions": _predictions(assessment),
        "biomarkers": assessment.get("biomarker_items") or [],
        "biomarkers_raw": assessment.get("biomarkers"),
        "trials": assessment.get("trials") or [],
        "report": {
            "format": "markdown",
            "content": assessment.get("report") or "",
        },
        "prediction_json": assessment.get("prediction_json"),
        "model": _model(assessment),
    }


def _plain_markdown(text: str) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+", "• ", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        line = line.replace("**", "")
        if re.match(r"^\|?\s*:?-{2,}", line):
            continue
        lines.append(line)
    return lines


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

    def p(text: Any, style: ParagraphStyle = base) -> Paragraph:
        escaped = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        return Paragraph(escaped, style)

    def kv_table(rows: List[List[Any]], widths: Optional[List[float]] = None) -> Table:
        table = Table(
            [[p(cell) for cell in row] for row in rows],
            colWidths=widths,
            repeatRows=1 if rows else 0,
        )
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 13),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe5ee")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    assessment = payload["assessment"]
    patient = payload["patient"]
    predictions = payload["predictions"]
    model = payload["model"]
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
                ["记录生成时间", _pdf_text(assessment.get("created_at")), "数据采集时间", _pdf_text(assessment.get("assessment_time"))],
                ["Session", _pdf_text(assessment.get("session_id")), "Assessment ID", _pdf_text(assessment.get("assessment_id"))],
                ["数据包", _pdf_text(assessment.get("package_name")), "Trial 数", _pdf_text(assessment.get("n_trials"))],
            ],
            [28 * mm, 52 * mm, 28 * mm, 52 * mm],
        ),
        p("核心评分结果", h2),
        kv_table(
            [
                ["指标", "结果"],
                ["FMA-UE 手部分数", _pdf_text(predictions.get("FMA_UE"))],
                ["Barthel 指数", _pdf_text(predictions.get("BI"))],
                ["手部肌张力（MAS）", _pdf_text(predictions.get("hand_tone"))],
                ["Brunnstrom 手功能分期", _pdf_text(predictions.get("hand_function"))],
            ],
            [55 * mm, 105 * mm],
        ),
    ]

    biomarkers = payload.get("biomarkers") or []
    if biomarkers:
        story.extend([p("关键生物标志物", h2)])
        rows = [["分组", "指标", "当前值", "单位", "参考范围", "有效试次"]]
        for marker in biomarkers:
            rows.append(
                [
                    marker.get("group_label") or marker.get("group_key") or "—",
                    marker.get("marker_name") or marker.get("marker_key") or "—",
                    marker.get("value_text") or "—",
                    marker.get("unit") or "—",
                    marker.get("ref_range") or "—",
                    marker.get("n_valid") if marker.get("n_valid") is not None else "—",
                ]
            )
        story.append(kv_table(rows, [24 * mm, 46 * mm, 32 * mm, 20 * mm, 40 * mm, 18 * mm]))

    report_text = (payload.get("report") or {}).get("content") or ""
    if report_text:
        story.extend([p("AI 康复评估报告", h2)])
        for line in _plain_markdown(report_text)[:220]:
            story.append(p(line))

    story.extend(
        [
            p("追溯信息", h2),
            kv_table(
                [
                    ["字段", "内容"],
                    ["数据包 SHA-256", _pdf_text(assessment.get("package_hash"))],
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
