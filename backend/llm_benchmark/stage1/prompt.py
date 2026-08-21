"""Stage 1 clinical-only Prompt assembly."""

from __future__ import annotations

import json
from pathlib import Path

from .clinical_input import Stage1ClinicalInput
from .clinical_facts import render_stage1_fact_card

STAGE1_PROMPT_VERSION = "rehab_llm_benchmark_stage1_advice_v2"
STAGE1_PROTOCOL = "program_fact_layer_llm_advice_v2"
STAGE1_PROMPT_VARIANT = "stage1_clinical_baseline"
STAGE1_SERIALIZATION_RETRY_VERSION = "stage1_json_serialization_retry_v1"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rehab_llm_benchmark_stage1_advice_v2.txt"

# This suffix is deliberately limited to output serialization. It does not add
# clinical facts, change the nine-field input, or ask the model to repair its
# medical content. It is only used after the first response fails parsing.
STAGE1_SERIALIZATION_RETRY_SUFFIX = """

【格式重试】上一轮输出未通过程序的JSON契约校验。请基于完全相同的九项临床输入和相同医学任务重新生成；本轮只允许返回一个合法JSON对象，不要输出思维过程、Markdown或说明文字。顶层只能有 integrated_assessment 和 rehabilitation_plan；rehabilitation_plan 必须是恰好3个对象，每个对象只能有 action、goal、reason、precaution 四个字符串字段。不要修改、补充或编造临床事实，也不要增加其他字段。
""".strip()

STAGE1_ADVICE_SERIALIZATION_RETRY_SUFFIX = """

【格式重试】上一轮输出未通过程序的 advice-only JSON 契约校验。基于完全相同的九项临床输入和事实卡重新生成；本轮只允许返回一个合法JSON对象。顶层只能有 rehabilitation_plan，必须恰好包含3个对象，每个对象只能有 action、goal、reason、precaution 四个字符串字段。不要输出 integrated_assessment、思维过程、Markdown或其他文字。
""".strip()


def load_stage1_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if f"prompt_version = {STAGE1_PROMPT_VERSION}" not in text:
        raise RuntimeError("Stage1固定Prompt缺少版本标记")
    if f"stage = {STAGE1_PROMPT_VARIANT}" not in text:
        raise RuntimeError("Stage1固定Prompt缺少stage标记")
    return text


def build_stage1_messages(clinical_input: Stage1ClinicalInput) -> list[dict[str, str]]:
    """Build messages with nine raw fields plus a code-owned fact card."""

    payload = clinical_input.model_dump(mode="json")
    user_content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\n"
        + render_stage1_fact_card(clinical_input)
    )
    return [
        {"role": "system", "content": load_stage1_prompt()},
        {"role": "user", "content": user_content},
    ]


def build_stage1_serialization_retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy the fixed messages and append only the deterministic format retry."""

    if len(messages) < 2 or messages[0].get("role") != "system":
        raise ValueError("Stage1 messages缺少system消息")
    retry_messages = [dict(message) for message in messages]
    retry_messages[0]["content"] = (
        str(retry_messages[0].get("content") or "").rstrip()
        + "\n\n"
        + STAGE1_SERIALIZATION_RETRY_SUFFIX
    )
    return retry_messages


def build_stage1_advice_serialization_retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) < 2 or messages[0].get("role") != "system":
        raise ValueError("Stage1 messages缺少system消息")
    retry_messages = [dict(message) for message in messages]
    retry_messages[0]["content"] = (
        str(retry_messages[0].get("content") or "").rstrip()
        + "\n\n"
        + STAGE1_ADVICE_SERIALIZATION_RETRY_SUFFIX
    )
    return retry_messages


__all__ = [
    "PROMPT_PATH",
    "STAGE1_SERIALIZATION_RETRY_SUFFIX",
    "STAGE1_ADVICE_SERIALIZATION_RETRY_SUFFIX",
    "STAGE1_SERIALIZATION_RETRY_VERSION",
    "STAGE1_PROMPT_VARIANT",
    "STAGE1_PROMPT_VERSION",
    "STAGE1_PROTOCOL",
    "build_stage1_messages",
    "build_stage1_serialization_retry_messages",
    "build_stage1_advice_serialization_retry_messages",
    "load_stage1_prompt",
]
