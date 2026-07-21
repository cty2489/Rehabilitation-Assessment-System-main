"""Pydantic schemas for the rehabilitation assessment API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class PatientInfo(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=64, description="患者编号")
    name: str = Field(..., min_length=1, max_length=128, description="姓名")
    sex: Literal["男", "女"]
    age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis: str = Field(..., min_length=1, max_length=255, description="诊断")
    disease_days: Optional[int] = Field(None, ge=0)
    paralysis_side: Literal["左", "右"]

    @field_validator("patient_id", "name", "diagnosis", mode="before")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if any(ord(char) < 32 for char in text):
            raise ValueError("文本中不能包含控制字符")
        return text


class PredictionResult(BaseModel):
    FMA_UE: float = Field(..., ge=0.0, le=20.0, description="FMA手部分数")
    BI: float = Field(..., ge=0.0, le=100.0, description="旧记录兼容字段；当前在线报告不展示")
    hand_tone: str = Field(..., description='手部肌张力："0"/"1"/"1+"/"2"/"3"/"4"')
    hand_function: int = Field(..., ge=1, le=6, description="Brunnstrom手部分期 1–6")


class AssessSessionResponse(BaseModel):
    session_id: str
    n_trials: int


class AssessmentResult(BaseModel):
    session_id: str
    patient_info: PatientInfo
    predictions: PredictionResult
    report: Optional[str] = None
    quality: Dict[str, Any] = Field(default_factory=dict)
    validation_status: str = "research_assessment"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=512)


class AuthLoginResponse(BaseModel):
    user: str
    expires_in: int = Field(..., ge=1)


class LlmSettingsUpdate(BaseModel):
    active_model_id: str = Field(..., min_length=1, description="报告生成使用的大模型配置 ID")


class LlmModelSettingsUpdate(BaseModel):
    weight_path: Optional[str] = Field(None, description="本地 HF 权重路径")
    remote_url: Optional[str] = Field(None, description="远程 LLM 服务地址")
    enabled: Optional[bool] = Field(None, description="是否启用该候选")
    adapter_dir: Optional[str] = Field(None, description="可选 LoRA adapter 目录")
    use_adapter: Optional[bool] = Field(None, description="是否加载 LoRA adapter")


class DeviceCredentialCreate(BaseModel):
    device_id: str = Field(
        ...,
        min_length=2,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    label: str = Field(..., min_length=1, max_length=128)


class DeviceCredentialUpdate(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=128)
    status: Optional[Literal["active", "disabled"]] = None


# --------------------------------------------------------------------------- #
# Patient management + records + stats (MySQL-backed)                          #
# --------------------------------------------------------------------------- #
class PatientUpdate(BaseModel):
    """PATCH semantics — all fields optional; patient_id is NOT editable."""

    name: Optional[str] = None
    sex: Optional[Literal["男", "女"]] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis: Optional[str] = None
    disease_days: Optional[int] = Field(None, ge=0)
    paralysis_side: Optional[Literal["左", "右"]] = None
    hand_function: Optional[int] = Field(None, ge=1, le=6)
    birth_date: Optional[str] = None      # 出生年月日 yyyy-mm-dd
    id_number: Optional[str] = None       # 身份证号
    phone: Optional[str] = None           # 手机号
    onset_date: Optional[str] = None      # 发病日期 yyyy-mm-dd


class AssessmentRecord(BaseModel):
    id: int
    source: Optional[str] = None
    assessment_id: Optional[str] = None
    session_id: Optional[str] = None
    package_name: Optional[str] = None
    institution: Optional[str] = None
    n_trials: Optional[int] = None
    package_hash: Optional[str] = None
    created_at: str
    assessment_time: Optional[str] = None
    fma_ue: float
    hand_tone: str
    hand_function: int
    report: Optional[str] = None
    report_status: str
    biomarkers: Optional[Any] = None
    parse_warnings: Optional[Any] = None
    prediction_json: Optional[Any] = None
    model_version: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    quality_json: Optional[Any] = None
    validation_status: Optional[str] = None
    report_generation: Optional[str] = None
    trials: List[Dict[str, Any]] = Field(default_factory=list)
    biomarker_items: List[Dict[str, Any]] = Field(default_factory=list)


class PatientSummary(BaseModel):
    id: int
    patient_id: str
    name: str
    sex: str
    age: Optional[int] = None
    diagnosis: str
    disease_days: Optional[int] = None
    paralysis_side: str
    hand_function: Optional[int] = None
    birth_date: Optional[str] = None
    id_number: Optional[str] = None
    phone: Optional[str] = None
    onset_date: Optional[str] = None
    created_at: str
    updated_at: str
    assessment_count: int = 0
    last_assessed_at: Optional[str] = None


class PatientDetail(PatientSummary):
    assessments: List[AssessmentRecord] = Field(default_factory=list)


class AssessmentOverviewItem(BaseModel):
    id: int
    created_at: str
    patient_db_id: int
    patient_id: str
    name: str
    fma_ue: float
    hand_tone: str
    hand_function: int
    report_status: str


class AssessmentOverview(BaseModel):
    total: int
    items: List[AssessmentOverviewItem]


class StatsSummary(BaseModel):
    patient_count: int
    assessment_count: int
    report_failed_count: int
    diagnosis_distribution: Dict[str, int]
    hand_function_distribution: Dict[str, int]
    avg_fma_ue: Optional[float] = None
    assessments_by_day: List[Dict[str, Union[str, int]]]


# --------------------------------------------------------------------------- #
# Device-end (task-interface) MySQL store                                      #
# --------------------------------------------------------------------------- #
class EnrollmentRequest(BaseModel):
    """医院入组：患者基本信息（最小集）+ 可选的第一次上肢/手功能评估记录。"""

    # 基本信息（最小集，后续可扩展）
    patient_id: str = Field(..., description="患者编号（业务主键）")
    name: str = Field(..., description="姓名")
    sex: Literal["男", "女"] = "男"
    age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis: Optional[str] = None
    paralysis_side: Optional[Literal["左", "右"]] = None
    disease_days: Optional[int] = Field(None, ge=0)

    # 第一次评估记录（医院给出，手工录入；可全空表示仅入组基本信息）
    fma_ue: Optional[float] = Field(None, ge=0.0, le=20.0)
    hand_tone: Optional[Literal["0", "1", "1+", "2", "3", "4"]] = None
    hand_function: Optional[int] = Field(None, ge=1, le=6)
    assessment_time: Optional[str] = None
    report: Optional[str] = None


class DevicePatientRegistrationRequest(BaseModel):
    """Patient profile created by an authenticated training device."""

    schema_version: Literal["rehab.patient.v1"] = "rehab.patient.v1"
    patient_id: str = Field(
        ...,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Z][A-Z0-9]{2,31}_[0-9]{4,8}$",
        description="设备端生成的全局患者编号，例如 DEV001_0001",
    )
    name: str = Field(..., min_length=1, max_length=128)
    sex: Literal["男", "女"]
    age: int = Field(..., ge=0, le=120)
    diagnosis: str = Field(..., min_length=1, max_length=255)
    paralysis_side: Literal["左", "右"]
    disease_days: Optional[int] = Field(None, ge=0)
    hand_brunnstrom_stage: Optional[
        Literal["I", "II", "III", "IV", "V", "VI"]
    ] = Field(None, description="登记时的Brunnstrom手功能分期，映射为初始hand_function")

    @field_validator("patient_id", mode="before")
    @classmethod
    def normalize_patient_id(cls, value: Any) -> str:
        return str(value or "").strip().upper()

    @field_validator("name", "diagnosis", mode="before")
    @classmethod
    def validate_registration_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if any(ord(char) < 32 for char in text):
            raise ValueError("文本中不能包含控制字符")
        return text


class DevicePatientRegistrationResponse(BaseModel):
    schema_version: Literal["rehab.patient.v1"] = "rehab.patient.v1"
    patient_id: str
    hand_brunnstrom_stage: Optional[
        Literal["I", "II", "III", "IV", "V", "VI"]
    ] = None
    created: bool
    message: str
    created_at: str
    updated_at: str


class MysqlAssessmentItem(BaseModel):
    id: int
    created_at: str
    patient_db_id: int
    patient_id: str
    name: Optional[str] = None
    source: str
    assessment_id: Optional[str] = None
    session_id: Optional[str] = None
    package_name: Optional[str] = None
    institution: Optional[str] = None
    n_trials: Optional[int] = None
    package_hash: Optional[str] = None
    assessment_time: Optional[str] = None
    fma_ue: float
    hand_tone: str
    hand_function: int
    report_status: str
    model_version: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    validation_status: Optional[str] = None
    report_generation: Optional[str] = None


class MysqlAssessmentDetail(MysqlAssessmentItem):
    sex: Optional[str] = None
    age: Optional[int] = None
    diagnosis: Optional[str] = None
    paralysis_side: Optional[str] = None
    disease_days: Optional[int] = None
    quality_json: Optional[Any] = None
    report: Optional[str] = None
    biomarkers: Optional[Any] = None
    parse_warnings: Optional[Any] = None
    prediction_json: Optional[Any] = None
    trials: List[Dict[str, Any]] = Field(default_factory=list)
    biomarker_items: List[Dict[str, Any]] = Field(default_factory=list)


class MysqlAssessmentList(BaseModel):
    total: int
    items: List[MysqlAssessmentItem]
