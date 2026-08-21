"""Deterministic import of the fixed clinician workbook for Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field, model_validator

DOCTOR_SHEET_NAME = "医生填写"
INPUT_SCHEMA_VERSION = "rehab.llm-benchmark-stage1-input.v1"


class Stage1WorkbookError(ValueError):
    """Raised when the fixed doctor workbook cannot be imported safely."""


class Stage1ClinicalInput(BaseModel):
    """The only nine fields allowed in the Stage 1 medical Prompt."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sex: str = Field(min_length=1, max_length=32)
    age: int = Field(ge=0, le=150)
    diagnosis: str = Field(min_length=1, max_length=255)
    disease_course: str = Field(min_length=1, max_length=255)
    paralysis_side: str = Field(min_length=1, max_length=32)
    fma_wrist: float = Field(ge=0.0, le=10.0)
    fma_hand: float = Field(ge=0.0, le=20.0)
    hand_mas: str
    brunnstrom_hand: int = Field(ge=1, le=6)

    @model_validator(mode="after")
    def validate_mas(self) -> "Stage1ClinicalInput":
        if self.hand_mas not in {"0", "1", "1+", "2", "3", "4"}:
            raise ValueError("hand_mas必须为0、1、1+、2、3或4")
        return self


@dataclass(frozen=True)
class Stage1Case:
    """Case metadata is kept outside the medical Prompt."""

    patient_id: str
    clinical_input: Stage1ClinicalInput
    doctor1_text: str = ""
    doctor2_text: str = ""
    source_row: int = 0


_EXPECTED_HEADERS = (
    "患者编号",
    "性别",
    "年龄",
    "诊断",
    "病程",
    "患侧",
    "FMA腕",
    "FMA手",
    "手MAS",
    "Brunnstrom",
    "康复师1：评估与建议",
    "康复师2：复核/修改",
)
_FORBIDDEN_HEADERS = {
    "BI",
    "FMA上肢",
    "FMA_UE",
    "腕肌张力",
    "biomarker",
    "EEG",
    "EMG",
    "IMU",
}


def _header_text(value: Any) -> str:
    text = str(value or "").strip().replace("：", ":").replace(" ", "")
    return text.replace("（按模板填）", "").replace("(按模板填)", "")


def _required_text(value: Any, field: str, row_number: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise Stage1WorkbookError(f"第{row_number}行{field}为空")
    return text


def _number(value: Any, field: str, row_number: int) -> float:
    if isinstance(value, bool):
        raise Stage1WorkbookError(f"第{row_number}行{field}必须是数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1WorkbookError(f"第{row_number}行{field}必须是数值") from exc
    if not math.isfinite(result):
        raise Stage1WorkbookError(f"第{row_number}行{field}必须是有限数值")
    return result


def _integer(value: Any, field: str, row_number: int) -> int:
    result = _number(value, field, row_number)
    if not result.is_integer():
        raise Stage1WorkbookError(f"第{row_number}行{field}必须是整数")
    return int(result)


def _mas(value: Any, field: str, row_number: int) -> str:
    text = str(value or "").strip()
    if text in {"0", "1", "1+", "2", "3", "4"}:
        return text
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) in {0, 1, 2, 3, 4}:
        return str(int(value))
    raise Stage1WorkbookError(f"第{row_number}行{field}必须为0、1、1+、2、3或4")


def _check_headers(raw_headers: tuple[Any, ...]) -> dict[str, int]:
    headers = [_header_text(value) for value in raw_headers]
    if len(headers) < len(_EXPECTED_HEADERS):
        raise Stage1WorkbookError("医生填写工作表必须包含A-L十二列")
    actual = tuple(headers[: len(_EXPECTED_HEADERS)])
    expected = tuple(_header_text(value) for value in _EXPECTED_HEADERS)
    if actual != expected:
        raise Stage1WorkbookError(
            "医生填写工作表A-L列不符合固定模板；实际="
            + "、".join(actual)
            + "；期望="
            + "、".join(expected)
        )
    extras = [header for header in headers[len(_EXPECTED_HEADERS) :] if header]
    forbidden = sorted(set(headers) & {_header_text(item) for item in _FORBIDDEN_HEADERS})
    if forbidden:
        raise Stage1WorkbookError("固定Stage1禁止额外字段：" + "、".join(forbidden))
    if extras:
        raise Stage1WorkbookError("医生填写工作表存在额外字段：" + "、".join(extras))
    return {name: index for index, name in enumerate(expected)}


def read_doctor_sheet(path: str | Path) -> tuple[Stage1Case, ...]:
    """Read sheet ``医生填写`` and only map A-J into clinical input.

    K/L are returned as audit text and are never passed to Prompt builders.
    """

    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise Stage1WorkbookError(f"未找到医生Excel：{workbook_path}")
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if DOCTOR_SHEET_NAME not in workbook.sheetnames:
            raise Stage1WorkbookError(f"Excel缺少工作表：{DOCTOR_SHEET_NAME}")
        sheet = workbook[DOCTOR_SHEET_NAME]
        rows = sheet.iter_rows(values_only=True)
        index: dict[str, int] | None = None
        header_row_number = 0
        data_rows = None
        for row_number, raw_row in enumerate(rows, start=1):
            if any(_header_text(value) == "患者编号" for value in raw_row):
                index = _check_headers(tuple(raw_row))
                header_row_number = row_number
                data_rows = rows
                break
        if index is None or data_rows is None:
            raise Stage1WorkbookError("医生填写工作表未找到固定A-L表头")
        cases: list[Stage1Case] = []
        seen: set[str] = set()
        for row_number, raw_row in enumerate(data_rows, start=header_row_number + 1):
            if not any(value not in (None, "") for value in raw_row):
                continue
            patient_id = _required_text(raw_row[index["患者编号"]], "患者编号", row_number)
            if patient_id in seen:
                raise Stage1WorkbookError(f"patient_id重复：{patient_id}")
            seen.add(patient_id)
            age = _integer(raw_row[index["年龄"]], "年龄", row_number)
            fma_wrist = _number(raw_row[index["FMA腕"]], "FMA腕", row_number)
            fma_hand = _number(raw_row[index["FMA手"]], "FMA手", row_number)
            if not 0 <= age <= 150:
                raise Stage1WorkbookError(f"第{row_number}行年龄必须在0–150")
            if not 0 <= fma_wrist <= 10:
                raise Stage1WorkbookError(f"第{row_number}行FMA腕必须在0–10")
            if not 0 <= fma_hand <= 20:
                raise Stage1WorkbookError(f"第{row_number}行FMA手必须在0–20")
            brunnstrom = _integer(raw_row[index["Brunnstrom"]], "Brunnstrom", row_number)
            if not 1 <= brunnstrom <= 6:
                raise Stage1WorkbookError(f"第{row_number}行Brunnstrom必须在1–6")
            clinical = Stage1ClinicalInput(
                sex=_required_text(raw_row[index["性别"]], "性别", row_number),
                age=age,
                diagnosis=_required_text(raw_row[index["诊断"]], "诊断", row_number),
                disease_course=_required_text(raw_row[index["病程"]], "病程", row_number),
                paralysis_side=_required_text(raw_row[index["患侧"]], "患侧", row_number),
                fma_wrist=fma_wrist,
                fma_hand=fma_hand,
                hand_mas=_mas(raw_row[index["手MAS"]], "手MAS", row_number),
                brunnstrom_hand=brunnstrom,
            )
            cases.append(Stage1Case(
                patient_id=patient_id,
                clinical_input=clinical,
                doctor1_text=str(raw_row[index["康复师1:评估与建议"]] or "").strip(),
                doctor2_text=str(raw_row[index["康复师2:复核/修改"]] or "").strip(),
                source_row=row_number,
            ))
        if not cases:
            raise Stage1WorkbookError("医生填写工作表没有有效患者行")
        return tuple(cases)
    finally:
        workbook.close()


__all__ = [
    "DOCTOR_SHEET_NAME",
    "INPUT_SCHEMA_VERSION",
    "Stage1Case",
    "Stage1ClinicalInput",
    "Stage1WorkbookError",
    "read_doctor_sheet",
]
