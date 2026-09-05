"""Deterministic renderer for the production v11-style strategy report.

The renderer never infers or changes facts.  It receives the current upload's
validated pipeline profile and only controls the established visual template.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


FORBIDDEN_PUBLIC_TERMS = (
    "近期SMART",
    "远期SMART",
    "SMART目标",
    "有效任务",
    "比较规则/备注",
    "FMA上肢",
    "Barthel",
    "防护跌落",
    "项目事实卡",
    "项目口径",
    "医生复核与修订栏",
)


def register_fonts() -> tuple[str, str]:
    """Register an available Chinese font without adding a deployment package."""
    packaged_font = Path(__file__).with_name("strategy_report_resources") / "fonts" / "NotoSansSC-VF.ttf"
    candidates = (
        (packaged_font, packaged_font),
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")),
        (Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"), Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")),
        (Path("/usr/share/fonts/truetype/arphic/uming.ttc"), Path("/usr/share/fonts/truetype/arphic/uming.ttc")),
    )
    for regular, bold in candidates:
        if not regular.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ReportCN", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("ReportCN-Bold", str(bold if bold.exists() else regular), subfontIndex=0))
            return "ReportCN", "ReportCN-Bold"
        except Exception:
            continue
    # ReportLab ships this CID font mapping; it keeps Chinese output functional
    # on the server without copying a proprietary Windows font.
    if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light", "STSong-Light"


def make_styles(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "StrategyTitleCN", parent=base["Title"], fontName=bold_font,
            fontSize=19, leading=26, textColor=colors.HexColor("#123A45"),
            alignment=TA_CENTER, spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "StrategyH1CN", fontName=bold_font, fontSize=13.5, leading=19,
            textColor=colors.HexColor("#0F766E"), spaceBefore=3 * mm,
            spaceAfter=2.2 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "StrategyH2CN", fontName=bold_font, fontSize=11.2, leading=16,
            textColor=colors.HexColor("#155E75"), spaceBefore=2.5 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "StrategyBodyCN", fontName=font, fontSize=8.7, leading=14,
            textColor=colors.HexColor("#24333A"), spaceAfter=1.8 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "StrategySmallCN", fontName=font, fontSize=7.2, leading=10.5,
            textColor=colors.HexColor("#3F5057"), wordWrap="CJK",
        ),
        "small_bold": ParagraphStyle(
            "StrategySmallBoldCN", fontName=bold_font, fontSize=7.3, leading=10.5,
            textColor=colors.HexColor("#263B43"), wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "StrategyCalloutCN", fontName=font, fontSize=8.5, leading=13.5,
            textColor=colors.HexColor("#17434B"), borderColor=colors.HexColor("#99D5CF"),
            borderWidth=0.8, borderPadding=7, backColor=colors.HexColor("#EFFAF8"),
            spaceAfter=3 * mm, wordWrap="CJK",
        ),
        "rec_title": ParagraphStyle(
            "StrategyRecTitleCN", fontName=bold_font, fontSize=10.5, leading=15,
            textColor=colors.white, backColor=colors.HexColor("#0F766E"),
            borderPadding=(5, 7, 5, 7), spaceAfter=0, wordWrap="CJK",
        ),
    }


def P(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = str(text if text is not None else "-")
    safe = safe.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br/>")
    return Paragraph(safe, style)


def table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9F0ED")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#153D46")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8CED2")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFA")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]
    )


def count_label(count: int) -> str:
    return {1: "一", 2: "两"}.get(count, str(count))


def stage_label(stage: int) -> str:
    return {1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ", 4: "Ⅳ", 5: "Ⅴ", 6: "Ⅵ"}[stage]


def _score_result(score: dict[str, Any], maximum: int | None = None) -> str:
    value = score.get("value")
    if value is None:
        return "本次未提供"
    return f"{value}/{maximum}分" if maximum else f"{value}级"


def _with_unit(value: Any, unit: str) -> str:
    return f"{value}{unit}" if isinstance(value, (int, float)) else str(value)


def _gesture_cell(action: dict[str, Any] | None, styles: dict[str, Any], image_dir: Path) -> Table:
    if not action:
        return Table(
            [[P("暂无匹配动作，需治疗师复核", styles["small_bold"])]],
            colWidths=[128 * mm],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]),
        )
    image_path = image_dir / str(action["image_filename"])
    if not image_path.is_file():
        raise FileNotFoundError(f"手势库图片缺失：{image_path.name}")
    image = RLImage(str(image_path))
    scale = min((24 * mm) / image.imageWidth, (18 * mm) / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    description = [
        P(
            f"{action['reference_role']}：{action['gesture_code']} {action['gesture_name']}"
            f"（Brunnstrom {stage_label(int(action['brunnstrom_stage']))}期）",
            styles["small_bold"],
        ),
        P("手型要点：" + str(action["gesture_key_points"]), styles["small"]),
        P("主要用途：" + str(action["gesture_purpose"]), styles["small"]),
        P(action["display_note"], styles["small"]),
    ]
    table = Table([[image, description]], colWidths=[28 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return table


def markdown_report(data: dict[str, Any]) -> str:
    ident = data["report_identity"]
    patient = data["patient"]
    scores = data["clinical_fact_card"]["scores"]
    subtype = data["subtype_assessment"]
    lines = [
        "# 康复评估与训练策略报告", "",
        f"- 病例匿名编号：{ident['case_id']}",
        f"- 报告编号：{ident['report_number']}",
        f"- 患者匿名编号：{ident['patient_code']}",
        f"- 性别/年龄：{patient['sex']} / {_with_unit(patient['age'], '岁')}",
        f"- 诊断：{patient['diagnosis']}；病程：{_with_unit(patient['disease_days'], '天')}；偏瘫侧：{patient['paralysis_side']}侧",
        "- 评估医生：________；职称：________", "",
        "## 一、腕手专项评分与综合亚型", "",
        "| 临床指标 | 实测结果 | 临床含义 |", "|---|---:|---|",
        f"| FMA腕部 | {_score_result(scores['fma_wrist'], 10)} | 腕背伸稳定性、交替屈伸及环转任务；来源：{scores['fma_wrist']['source']} |",
        f"| FMA手部 | {_score_result(scores['fma_hand'], 20)} | 手指共同屈伸、钩状抓握、侧捏、对捏、柱状抓握及球形抓握；来源：{scores['fma_hand']['source']} |",
        f"| 腕/手MAS | {_score_result(scores['wrist_mas'])} / {_score_result(scores['hand_mas'])} | 被动活动阻力大小及阻力出现的关节活动范围 |",
        f"| Brunnstrom手功能 | {scores['brunnstrom_hand']['value']}期 | 脑卒中运动恢复阶段；来源：{scores['brunnstrom_hand']['source']} |", "",
        f"**亚型：{subtype['name']}**", "", subtype["summary"], "", subtype["boundary"], "",
        "## 二、生理与运动学指标计算结果", "",
        f"本次共获得{data['biomarker_coverage']['available']}/{data['biomarker_coverage']['total']}项指标。"
        + data["biomarker_coverage"]["interpretation_policy"], "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker in data["biomarkers"]:
        grouped[marker["group_key"]].append(marker)
    for key in ("emg", "eeg", "imu"):
        items = grouped.get(key, [])
        if not items:
            continue
        lines.extend([f"### {items[0]['group_label']}", "", "| 指标 | 数值 | 单位 |", "|---|---:|---|"])
        for item in items:
            lines.append(f"| {item['marker_name']} | {item['value_text']} | {item.get('unit') or '-'} |")
        lines.append("")
    count = len(data["recommendations"])
    lines.extend([
        f"## 三、详细康复训练策略（共{count_label(count)}条）", "",
        "> FITT框架：F为训练频率，I为训练强度，第一个T为训练类型，第二个T为训练时间。", "",
    ])
    for rec in data["recommendations"]:
        action = rec.get("hand_rehabilitation_action")
        lines.extend([
            f"### {rec['number']}｜{rec['title']}", "",
            f"- 策略概述：{rec['recommendation']}",
            *(([
                f"- {action['reference_role']}（手势库参考）：{action['gesture_code']} {action['gesture_name']}"
                f"（Brunnstrom {stage_label(int(action['brunnstrom_stage']))}期）",
                f"  - 手型要点：{action['gesture_key_points']}",
                f"  - 主要用途：{action['gesture_purpose']}",
                f"  - 说明：{action['display_note']}",
                f"  - 手型照片：手势库图片/{action['image_filename']}",
            ]) if action else ["- 手势库参考：暂无匹配动作，需治疗师复核"]),
            f"- F-训练频率：{rec['fitt']['F']}",
            f"- I-训练强度：{rec['fitt']['I']}",
            f"- T-训练类型：{rec['fitt']['T_type']}",
            f"- T-训练时间：{rec['fitt']['T_time']}",
            f"- 策略依据：{rec['basis']}",
            f"- 进阶与观察：{rec['progression']}",
            f"- 注意事项：{rec['precaution']}", "",
        ])
    lines.extend(["## 四、康复训练策略参考依据", "", data["knowledge_evidence"]["knowledge_graph_notice"], ""])
    for ref in data["knowledge_evidence"]["core_references"]:
        link = f"[查看原文]({ref['url']})" if ref.get("url") else "查看原文：链接不可用"
        lines.append(
            f"{ref['number']}. {ref['title']}；UID：{ref['uid']}；页码：{ref['page']}；{link}。用途：{ref['use']}"
        )
    lines.extend(["", "---", "", data["disclaimer"], ""])
    text = "\n".join(lines)
    found = [term for term in FORBIDDEN_PUBLIC_TERMS if term in text]
    if found:
        raise ValueError(f"报告出现禁用字段：{', '.join(found)}")
    return text


def build_pdf(data: dict[str, Any], path: Path, gesture_image_dir: Path) -> None:
    font, bold_font = register_fonts()
    styles = make_styles(font, bold_font)
    ident = data["report_identity"]
    patient = data["patient"]
    scores = data["clinical_fact_card"]["scores"]
    subtype = data["subtype_assessment"]
    width, height = A4
    left = right = 16 * mm
    top, bottom = 23 * mm, 15 * mm

    def header_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, height - 14 * mm, width, 14 * mm, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#B8D9D5"))
        canvas.setLineWidth(0.5)
        canvas.line(left, height - 11.5 * mm, width - right, height - 11.5 * mm)
        canvas.setFont(font, 7.2)
        canvas.setFillColor(colors.HexColor("#587078"))
        canvas.drawString(left, height - 9 * mm, f"康复评估与训练策略报告 · {ident['case_id']}")
        canvas.drawRightString(width - right, height - 9 * mm, "匿名康复评估报告")
        canvas.line(left, 10.5 * mm, width - right, 10.5 * mm)
        canvas.setFont(font, 6.6)
        canvas.drawString(left, 7.3 * mm, "用于康复临床辅助与研究记录，不能替代医生面诊、体格检查和个体化训练方案")
        canvas.drawRightString(width - right, 7.3 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(path), pagesize=A4, leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom,
        title=f"{ident['case_id']} 匿名康复评估与训练策略报告",
        author="Anonymous rehabilitation report builder",
        subject="Wrist-hand assessment and individualized rehabilitation training strategy",
    )
    frame = Frame(left, bottom, width - left - right, height - top - bottom, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPageEnd=header_footer)])
    story: list[Any] = [Spacer(1, 5 * mm), P("康复评估与训练策略报告", styles["title"])]
    identity_rows = [
        [P("病例匿名编号", styles["small_bold"]), P(ident["case_id"], styles["small"]), P("报告编号", styles["small_bold"]), P(ident["report_number"], styles["small"])],
        [P("患者匿名编号", styles["small_bold"]), P(ident["patient_code"], styles["small"]), P("性别 / 年龄", styles["small_bold"]), P(f"{patient['sex']} / {_with_unit(patient['age'], '岁')}", styles["small"])],
        [P("诊断", styles["small_bold"]), P(patient["diagnosis"], styles["small"]), P("病程 / 偏瘫侧", styles["small_bold"]), P(f"{_with_unit(patient['disease_days'], '天')} / {patient['paralysis_side']}侧", styles["small"])],
        [P("评估医生", styles["small_bold"]), P("________________", styles["small"]), P("职称", styles["small_bold"]), P("________________", styles["small"])],
    ]
    identity = Table(identity_rows, colWidths=[27 * mm, 55 * mm, 30 * mm, 55 * mm])
    identity.setStyle(table_style())
    story.append(identity)
    publication = data.get("publication") or {}
    if publication.get("status") == "WARNING":
        warning_text = "；".join(publication.get("warnings") or ["本报告带质量警告，请结合现场评估复核。"])
        story.extend([Spacer(1, 2 * mm), P("质量警告：" + warning_text, styles["callout"])])
    story.extend([Spacer(1, 4 * mm), P("一、腕手专项评分与综合亚型", styles["h1"])])
    fact_rows = [
        [P("临床指标", styles["small_bold"]), P("实测结果", styles["small_bold"]), P("临床含义", styles["small_bold"])],
        [P("FMA腕部", styles["small"]), P(_score_result(scores["fma_wrist"], 10), styles["small"]), P("腕背伸稳定性、交替屈伸及环转任务；来源：" + scores["fma_wrist"]["source"], styles["small"])],
        [P("FMA手部", styles["small"]), P(_score_result(scores["fma_hand"], 20), styles["small"]), P("手指共同屈伸、钩状抓握、侧捏、对捏、柱状抓握及球形抓握；来源：" + scores["fma_hand"]["source"], styles["small"])],
        [P("腕/手MAS", styles["small"]), P(f"{_score_result(scores['wrist_mas'])} / {_score_result(scores['hand_mas'])}", styles["small"]), P("被动活动阻力大小及阻力出现的关节活动范围", styles["small"])],
        [P("Brunnstrom手功能", styles["small"]), P(f"{scores['brunnstrom_hand']['value']}期", styles["small"]), P("脑卒中运动恢复阶段；来源：" + scores["brunnstrom_hand"]["source"], styles["small"])],
    ]
    fact = Table(fact_rows, colWidths=[38 * mm, 37 * mm, 92 * mm], repeatRows=1)
    fact.setStyle(table_style())
    story.extend([
        fact, Spacer(1, 2.5 * mm), P("亚型：" + subtype["name"], styles["callout"]),
        P(subtype["summary"], styles["body"]), P(subtype["boundary"], styles["small"]),
        PageBreak(), P("二、生理与运动学指标计算结果", styles["h1"]),
        P(
            f"本次共获得{data['biomarker_coverage']['available']}/{data['biomarker_coverage']['total']}项指标。"
            + data["biomarker_coverage"]["interpretation_policy"], styles["callout"],
        ),
    ])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker in data["biomarkers"]:
        grouped[marker["group_key"]].append(marker)
    for key in ("emg", "eeg", "imu"):
        items = grouped.get(key, [])
        if not items:
            continue
        story.append(P(items[0]["group_label"], styles["h2"]))
        rows: list[list[Any]] = [[P("指标", styles["small_bold"]), P("数值", styles["small_bold"]), P("单位", styles["small_bold"])]]
        for item in items:
            rows.append([P(item["marker_name"], styles["small"]), P(item["value_text"], styles["small"]), P(item.get("unit") or "-", styles["small"])])
        table = Table(rows, colWidths=[93 * mm, 37 * mm, 37 * mm], repeatRows=1)
        table.setStyle(table_style())
        story.extend([table, Spacer(1, 2.5 * mm)])
    count = len(data["recommendations"])
    story.extend([
        PageBreak(), P(f"三、详细康复训练策略（共{count_label(count)}条）", styles["h1"]),
        P("FITT框架：F-频率、I-强度、T-类型、T-时间。", styles["callout"]),
    ])
    for index, rec in enumerate(data["recommendations"]):
        if index:
            story.append(PageBreak())
        rows = [
            [P("策略概述", styles["small_bold"]), P(rec["recommendation"], styles["body"])],
            [P(("配合手型" if rec["scope"] == "wrist" else "目标手型") + "（手势库参考）", styles["small_bold"]), _gesture_cell(rec["hand_rehabilitation_action"], styles, gesture_image_dir)],
            [P("F-训练频率", styles["small_bold"]), P(rec["fitt"]["F"], styles["small"])],
            [P("I-训练强度", styles["small_bold"]), P(rec["fitt"]["I"], styles["small"])],
            [P("T-训练类型", styles["small_bold"]), P(rec["fitt"]["T_type"], styles["small"])],
            [P("T-训练时间", styles["small_bold"]), P(rec["fitt"]["T_time"], styles["small"])],
            [P("策略依据", styles["small_bold"]), P(rec["basis"], styles["small"])],
            [P("进阶与观察", styles["small_bold"]), P(rec["progression"], styles["small"])],
            [P("注意事项", styles["small_bold"]), P(rec["precaution"], styles["small"])],
        ]
        table = Table(rows, colWidths=[34 * mm, 133 * mm], style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8CED2")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF8F7")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([P(f"{rec['number']}｜{rec['title']}", styles["rec_title"]), table])
    story.extend([
        PageBreak(), P("四、康复训练策略参考依据", styles["h1"]),
        P(data["knowledge_evidence"]["knowledge_graph_notice"], styles["callout"]),
    ])
    ref_rows: list[list[Any]] = [[P("序号", styles["small_bold"]), P("来源与用途", styles["small_bold"])]]
    for ref in data["knowledge_evidence"]["core_references"]:
        title = html.escape(str(ref.get("title") or "来源标题不可用"))
        uid = html.escape(str(ref.get("uid") or "UID不可用"))
        page = html.escape(str(ref.get("page") or "页码不可用"))
        use = html.escape(str(ref.get("use") or ""))
        url = str(ref.get("url") or "").strip()
        link = f'<link href="{html.escape(url, quote=True)}" color="#0F766E"><u>查看原文</u></link>' if url else "查看原文：链接不可用"
        reference = Paragraph(
            f"{title}<br/>UID：{uid}；页码：{page}；{link}<br/>用途：{use}",
            styles["small"],
        )
        ref_rows.append([P(f"[{ref['number']}]", styles["small"]), reference])
    refs = Table(ref_rows, colWidths=[14 * mm, 153 * mm], repeatRows=1)
    refs.setStyle(table_style())
    story.append(refs)
    doc.build(story)

    if path.stat().st_size < 1024 or path.read_bytes()[:4] != b"%PDF":
        raise RuntimeError("PDF生成结果无效")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profile(data: dict[str, Any], resource_dir: Path) -> None:
    if data.get("schema_version") != "rehab.production_strategy_report.v1":
        raise ValueError("仅支持真实pipeline生成的生产报告档案")
    publication = data.get("publication") or {}
    if publication.get("status") not in {"PASSED", "WARNING"}:
        raise ValueError("MANUAL_REVIEW不得进入正式报告渲染")
    scores = data["clinical_fact_card"]["scores"]
    for key in ("fma_hand", "brunnstrom_hand"):
        value = scores[key]["value"]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key}不是整数评分结果")
    wrist_value = scores["fma_wrist"].get("value")
    if wrist_value is not None and (isinstance(wrist_value, bool) or int(wrist_value) != wrist_value):
        raise ValueError("fma_wrist不是整数评分结果")
    biomarkers = data.get("biomarkers") or []
    if len(biomarkers) != 26 or len({item.get("marker_key") for item in biomarkers}) != 26:
        raise ValueError("正式报告必须确定性装配全部26项biomarker行")
    library_path = resource_dir / "gesture_library.json"
    source_docx = resource_dir / "source" / "根据布氏分期的训练手势分类20260724.docx"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    if source_docx.is_file() and _sha256(source_docx) != library["source_sha256"]:
        raise ValueError("手势库源Word文件哈希校验失败")
    by_code = {item["code"]: item for item in library["gestures"]}
    patient_stage = int(scores["brunnstrom_hand"]["value"])
    for rec in data["recommendations"]:
        action = rec.get("hand_rehabilitation_action")
        if action is None:
            continue
        code = action["gesture_code"]
        item = by_code.get(code)
        if not item or item["name"] != action["gesture_name"]:
            raise ValueError(f"手势库代码与名称不匹配：{code}")
        if int(action["brunnstrom_stage"]) != patient_stage:
            raise ValueError(f"手势分期与病例分期不匹配：{code}")
        if patient_stage < 6 and patient_stage not in item["allowed_stages"]:
            raise ValueError(f"手势不适用于本例Brunnstrom分期：{code}")
        image_path = resource_dir / "gesture_images" / action["image_filename"]
        if not image_path.is_file():
            raise ValueError(f"缺少手势图片：{action['image_filename']}")
        if _sha256(image_path) != item["image_sha256"] or action["image_sha256"] != item["image_sha256"]:
            raise ValueError(f"手势图片哈希校验失败：{code}")
    provenance = data.get("pipeline_provenance") or {}
    if provenance.get("preset_case_fixture_used") is not False:
        raise ValueError("生产报告不得使用预置病例fixture")
    public_text = markdown_report(data)
    found = [term for term in FORBIDDEN_PUBLIC_TERMS if term in public_text]
    if found:
        raise ValueError(f"报告出现禁用字段：{', '.join(found)}")
