"""Patient identity policy shared by device assessment upload variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from schemas import PatientInfo


@dataclass(frozen=True)
class DevicePatientPolicyError(ValueError):
    status_code: int
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def _nonblank(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _valid_choice(value: Any, allowed: set[str], field: str) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise ValueError(f"缺少有效 {field}")
    return text


def resolve_device_patient(
    *,
    requested_patient_id: Optional[str],
    manifest_patient_id: Optional[str],
    enrolled: Optional[Mapping[str, Any]],
    require_registered: bool,
    request_profile: Mapping[str, Any],
    manifest_profile: Mapping[str, Any],
) -> PatientInfo:
    """Resolve one authoritative patient or raise a structured policy error."""
    requested = str(requested_patient_id or "").strip()
    manifested = str(manifest_patient_id or "").strip()
    if requested and manifested and requested != manifested:
        raise DevicePatientPolicyError(
            409,
            "PATIENT_ID_MISMATCH",
            "请求中的 patient_id 与 manifest.json 不一致",
        )
    patient_id = requested or manifested
    if not patient_id:
        raise DevicePatientPolicyError(
            422,
            "PATIENT_ID_REQUIRED",
            "表单、查询参数或 manifest.json 至少提供一个 patient_id",
        )

    stored = dict(enrolled or {})
    if require_registered and not stored:
        raise DevicePatientPolicyError(
            404,
            "PATIENT_NOT_FOUND",
            "该患者尚未在云端注册",
        )

    try:
        if require_registered:
            return PatientInfo(
                patient_id=patient_id,
                name=stored.get("name"),
                sex=_valid_choice(stored.get("sex"), {"男", "女"}, "sex（男/女）"),
                age=stored.get("age"),
                diagnosis=stored.get("diagnosis"),
                disease_days=stored.get("disease_days"),
                paralysis_side=_valid_choice(
                    stored.get("paralysis_side"), {"左", "右"}, "paralysis_side（左/右）"
                ),
            )
        return PatientInfo(
            patient_id=patient_id,
            name=_nonblank(
                stored.get("name"), request_profile.get("name"),
                manifest_profile.get("name"), patient_id,
            ),
            sex=_valid_choice(
                _nonblank(stored.get("sex"), request_profile.get("sex"), manifest_profile.get("sex")),
                {"男", "女"},
                "sex（男/女）",
            ),
            age=(
                stored.get("age")
                if stored.get("age") is not None
                else request_profile.get("age")
                if request_profile.get("age") is not None
                else manifest_profile.get("age")
            ),
            diagnosis=_nonblank(
                stored.get("diagnosis"), request_profile.get("diagnosis"),
                manifest_profile.get("diagnosis"), "未填写",
            ),
            disease_days=(
                stored.get("disease_days")
                if stored.get("disease_days") is not None
                else request_profile.get("disease_days")
                if request_profile.get("disease_days") is not None
                else manifest_profile.get("disease_days")
            ),
            paralysis_side=_valid_choice(
                _nonblank(
                    stored.get("paralysis_side"), request_profile.get("paralysis_side"),
                    manifest_profile.get("paralysis_side"),
                ),
                {"左", "右"},
                "paralysis_side（左/右）",
            ),
        )
    except DevicePatientPolicyError:
        raise
    except Exception as exc:
        if require_registered:
            raise DevicePatientPolicyError(
                409,
                "PATIENT_PROFILE_INCOMPLETE",
                f"云端患者档案不完整：{exc}",
            ) from exc
        raise DevicePatientPolicyError(
            422,
            "PATIENT_PROFILE_INVALID",
            f"患者信息无效：{exc}",
        ) from exc


__all__ = ["DevicePatientPolicyError", "resolve_device_patient"]
