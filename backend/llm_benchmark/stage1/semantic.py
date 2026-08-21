"""Transparent semantic contract checks for Stage 1 candidates.

These checks never edit model text.  They classify the parsed candidate and
persist error codes so first-pass model quality remains auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .clinical_input import Stage1ClinicalInput
from .output import Stage1CandidateOutput

SEMANTIC_VALIDATOR_VERSION = "stage1_semantic_contract_v1"
_CJK = re.compile(r"[\u3400-\u9fff]")

_FORBIDDEN = (
    ("UNSUPPORTED_ROM", r"\bROM\b|关节活动度"),
    ("UNSUPPORTED_STRENGTH", r"肌力|MMT|握力|肌肉力量"),
    ("UNSUPPORTED_SENSATION", r"感觉"),
    ("UNSUPPORTED_PAIN", r"疼痛|痛感"),
    ("UNSUPPORTED_ADL", r"\bADL\b|日常生活|生活自理|独立生活"),
    ("UNSUPPORTED_COGNITION", r"认知"),
    ("UNSUPPORTED_MENTAL", r"情绪|心理|抑郁|焦虑"),
    ("UNSUPPORTED_SHOULDER_RISK", r"肩关节脱位|肩胛骨脱位|肩峰撞击"),
    ("UNSUPPORTED_IMMOBILIZATION", r"长期制动"),
    ("UNSUPPORTED_PROGNOSIS", r"预后|恢复潜力"),
    ("UNSUPPORTED_COMPLICATION", r"并发症|肌肉萎缩"),
    ("UNSUPPORTED_MODALITY", r"电刺激|神经肌肉电刺激|关节松动|手法治疗|心理治疗"),
)


@dataclass(frozen=True)
class SemanticValidationResult:
    valid: bool
    errors: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "validator_version": SEMANTIC_VALIDATOR_VERSION,
            "valid": self.valid,
            "error_codes": [item["error_code"] for item in self.errors],
            "errors": [dict(item) for item in self.errors],
        }


def _all_text(candidate: Stage1CandidateOutput) -> str:
    parts = [candidate.integrated_assessment]
    for item in candidate.rehabilitation_plan:
        parts.extend([item.action, item.goal, item.reason, item.precaution])
    return "\n".join(parts)


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[。！？!?]", text))


def _add(errors: list[dict[str, str]], code: str, message: str) -> None:
    if not any(item["error_code"] == code for item in errors):
        errors.append({"error_code": code, "message": message})


def validate_stage1_semantics(
    candidate: Stage1CandidateOutput,
    clinical_input: Stage1ClinicalInput,
) -> SemanticValidationResult:
    """Classify a candidate without changing any generated field."""

    errors: list[dict[str, str]] = []
    assessment = candidate.integrated_assessment
    all_text = _all_text(candidate)

    if not _CJK.search(all_text):
        _add(errors, "NON_CHINESE_OUTPUT", "报告字段值没有简体中文内容")

    if len(candidate.rehabilitation_plan) != 3:
        _add(errors, "PLAN_COUNT_VIOLATION", "rehabilitation_plan必须恰好包含3条")

    sentences = _sentence_count(assessment)
    if sentences < 2 or sentences > 3:
        _add(errors, "ASSESSMENT_SENTENCE_COUNT", "integrated_assessment应为2至3句")
    if re.search(r"建议\s*[一二三123]?|建议\d|康复建议", assessment):
        _add(errors, "ASSESSMENT_REPEATS_PLAN", "综合评估不应重复建议或包含建议编号")

    for code, pattern in _FORBIDDEN:
        if re.search(pattern, all_text, flags=re.IGNORECASE):
            _add(errors, code, "出现输入中未采集或本阶段禁止生成的内容")

    # Score-direction contradictions are checked against the code-owned input.
    if clinical_input.fma_wrist <= 0 and re.search(r"腕[^。；，,]{0,12}(?:良好|正常|接近正常|功能好)", all_text):
        _add(errors, "FMA_WRIST_DIRECTION_CONTRADICTION", "FMA腕为0时不能表述为腕部功能良好")
    if clinical_input.fma_wrist >= 9 and re.search(r"腕[^。；，,]{0,12}(?:严重受限|完全丧失|完全受限|几乎完全)", all_text):
        _add(errors, "FMA_WRIST_DIRECTION_CONTRADICTION", "FMA腕高分时不能表述为严重丧失")
    if clinical_input.fma_hand <= 2 and re.search(r"手[^。；，,]{0,12}(?:良好|正常|接近正常|功能好)", all_text):
        _add(errors, "FMA_HAND_DIRECTION_CONTRADICTION", "FMA手低分时不能表述为手部功能良好")
    if clinical_input.fma_hand >= 18 and re.search(r"手[^。；，,]{0,12}(?:严重受限|完全丧失|完全受限|几乎完全)", all_text):
        _add(errors, "FMA_HAND_DIRECTION_CONTRADICTION", "FMA手高分时不能表述为严重丧失")

    if clinical_input.hand_mas == "0" and re.search(r"(?<!无)(?<!未见)(?<!没有)(?:肌张力增高|肌张力升高|痉挛|高张力)", all_text):
        _add(errors, "MAS_DIRECTION_CONTRADICTION", "MAS 0不能表述为肌张力增高")
    if clinical_input.hand_mas == "1+" and re.search(r"肌张力(?:正常|未见增高)|无肌张力增高", all_text):
        _add(errors, "MAS_DIRECTION_CONTRADICTION", "MAS 1+不能表述为无肌张力增高")
    if clinical_input.brunnstrom_hand == 3 and re.search(r"更分离|分离运动|接近协调|协调运动为主", all_text):
        _add(errors, "BRUNNSTROM_DIRECTION_CONTRADICTION", "Brunnstrom III期不能表述为更分离或接近协调")
    if clinical_input.brunnstrom_hand == 6 and re.search(r"共同运动为主|僵硬阶段|仅有主动运动尝试", all_text):
        _add(errors, "BRUNNSTROM_DIRECTION_CONTRADICTION", "Brunnstrom VI期不能表述为共同运动为主或僵硬")

    generic_actions = {"手部训练", "腕手训练", "综合训练", "综合康复训练", "加强治疗", "物理治疗", "职业治疗", "整体训练"}
    normalized_actions = [re.sub(r"\s+", "", item.action) for item in candidate.rehabilitation_plan]
    if any(item in generic_actions for item in normalized_actions):
        _add(errors, "GENERIC_ACTION", "训练动作过于宽泛，必须指明训练对象和运动任务")
    if len(set(normalized_actions)) != len(normalized_actions):
        _add(errors, "DUPLICATE_ACTION", "三条训练动作不能重复")
    for index, item in enumerate(candidate.rehabilitation_plan, start=1):
        if not re.search(r"腕|手指|拇指|抓|放|屈|伸|协调|控制|分离", item.action):
            _add(errors, f"ACTION_NOT_SPECIFIC_{index}", f"第{index}条动作缺少具体训练对象或运动任务")
        if not re.search(r"FMA|MAS|Brunnstrom|运动任务|肌张力|共同运动|分离", item.reason, flags=re.IGNORECASE):
            _add(errors, f"REASON_LACKS_FACT_ANCHOR_{index}", f"第{index}条原因缺少程序事实依据")

    return SemanticValidationResult(valid=not errors, errors=tuple(errors))


__all__ = ["SEMANTIC_VALIDATOR_VERSION", "SemanticValidationResult", "validate_stage1_semantics"]
