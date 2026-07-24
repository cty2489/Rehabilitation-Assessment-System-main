"""Minimal LLM knowledge-query planner for ``planner_rag`` v0.1.

The planner consumes only interpreted observations and the existing fixed
knowledge package. It produces retrieval topics and queries; it does not make
clinical conclusions or enter the production report flow.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence

from pydantic import Field

from .contracts import (
    ContractModel,
    CoreKnowledgeBundle,
    InterpretationResult,
    KnowledgePlan,
    KnowledgeTopic,
    RetrievalQuery,
)


PlannerMessage = Dict[str, str]


class PlannerLlmClient(Protocol):
    """Small boundary around the project's already selected LLM."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        messages: Sequence[PlannerMessage],
        *,
        attempt: int,
    ) -> str: ...


class _PlannerPayload(ContractModel):
    topics: List[KnowledgeTopic] = Field(min_length=1)
    queries: List[RetrievalQuery] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1000)
    generation_mode: Literal["llm"]


_LEADING_THINK_BLOCK = re.compile(
    r"^\s*<think>.*?</think>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_KEYS = {
    "needs_retrieval",
    "diagnosis",
    "diagnoses",
    "pathological_mechanism",
    "mechanism_conclusion",
    "treatment",
    "treatment_advice",
    "rehabilitation_advice",
    "recommendation",
    "recommendations",
    "training_dose",
    "dosage",
}
_FORBIDDEN_ASSERTIONS = (
    re.compile(r"(?:诊断|确诊)\s*(?:为|是|：|:)", re.IGNORECASE),
    re.compile(r"病理机制\s*(?:为|是|：|:)", re.IGNORECASE),
    re.compile(r"(?:治疗方案|康复建议|训练方案)\s*(?:为|是|：|:)", re.IGNORECASE),
    re.compile(
        r"(?:建议|应当|需要).{0,16}(?:每日|每天|每周|每次)\s*"
        r"\d+(?:\.\d+)?\s*(?:分钟|小时|次|组)",
        re.IGNORECASE,
    ),
)
_FORBIDDEN_REASON_RECOMMENDATION = re.compile(
    r"(?:建议|应当|应该)\s*(?:进行|采用|给予|安排|接受)",
    re.IGNORECASE,
)


def _json_payload(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Planner LLM返回为空")
    normalized = _LEADING_THINK_BLOCK.sub("", text, count=1).strip()
    try:
        payload = json.loads(normalized)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Planner LLM未返回合法JSON对象") from exc
    if not isinstance(payload, dict):
        raise ValueError("Planner LLM返回的JSON顶层必须是对象")
    return payload


def _forbidden_key_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.strip().lower() in _FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, f"{path}[{index}]"))
    return found


def _validate_plan_scope(
    payload: _PlannerPayload,
    interpretation: InterpretationResult,
) -> None:
    finding_ids = {finding.finding_id for finding in interpretation.findings}
    topic_ids = [topic.topic_id for topic in payload.topics]
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError("Planner topic_id必须唯一")

    for topic in payload.topics:
        unknown = set(topic.finding_ids) - finding_ids
        if unknown:
            raise ValueError(
                "Planner topic引用了未知finding_id：" + "、".join(sorted(unknown))
            )

    query_ids = [query.query_id for query in payload.queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Planner query_id必须唯一")
    unknown_topics = {
        query.topic_id for query in payload.queries if query.topic_id not in topic_ids
    }
    if unknown_topics:
        raise ValueError(
            "Planner query引用了未知topic_id：" + "、".join(sorted(unknown_topics))
        )
    uncovered_topics = set(topic_ids) - {query.topic_id for query in payload.queries}
    if uncovered_topics:
        raise ValueError(
            "Planner topic缺少检索query：" + "、".join(sorted(uncovered_topics))
        )

    output_text = "\n".join(
        [payload.reason]
        + [topic.label for topic in payload.topics]
        + [query.text for query in payload.queries]
    )
    for pattern in _FORBIDDEN_ASSERTIONS:
        if pattern.search(output_text):
            raise ValueError("Planner输出包含诊断、机制、处方或训练剂量结论")
    if _FORBIDDEN_REASON_RECOMMENDATION.search(payload.reason):
        raise ValueError("Planner reason包含康复或治疗建议")


def _planner_messages(
    interpretation: InterpretationResult,
    core_knowledge: CoreKnowledgeBundle,
    *,
    retry: bool,
) -> list[PlannerMessage]:
    schema_example = {
        "topics": [
            {
                "topic_id": "topic-1",
                "label": "待检索的知识主题",
                "finding_ids": ["输入中真实存在的finding_id"],
                "priority": "medium",
            }
        ],
        "queries": [
            {
                "query_id": "query-1",
                "topic_id": "topic-1",
                "text": "面向知识库的完整检索查询",
            }
        ],
        "reason": "为何需要这些补充证据的简短说明",
        "generation_mode": "llm",
    }
    system = (
        "你是康复评估系统中的KnowledgePlanner LLM，不是诊断器、报告生成器或治疗决策器。"
        "你只负责把结构化观察结果规划成知识主题和检索查询。"
        "不得输出诊断、病理机制结论、康复建议、训练方案、训练剂量或needs_retrieval。"
        "不得回答查询本身，也不得把患者观察写成确定性临床结论。"
        "topic.finding_ids只能使用输入中已有的finding_id，query.topic_id只能引用本次topics。"
        "当输入包含临床量表finding时，除指标解释证据外，还要规划独立的康复干预适用条件查询。"
        "至少分别检索：任务特异/重复/渐进训练与CIMT适用条件；FES适用条件；"
        "镜像反馈与痉挛管理适用条件。这些查询只用于获取证据，不代表直接推荐治疗。"
        "只返回一个合法JSON对象，不要Markdown、代码块或额外文字。"
    )
    if retry:
        system += "上一次输出未通过结构或边界校验；请严格按本次JSON形状重新生成。"

    planner_input = {
        "interpretation": interpretation.model_dump(mode="json"),
        "core_knowledge": core_knowledge.model_dump(mode="json"),
    }
    user = (
        "【输入】\n"
        + json.dumps(planner_input, ensure_ascii=False, separators=(",", ":"))
        + "\n【唯一允许的输出形状】\n"
        + json.dumps(schema_example, ensure_ascii=False, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class ExistingLlmClient:
    """Use the in-process model already selected by the report subsystem."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        max_new_tokens: int = 768,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens必须大于0")
        self._model_id = (model_id or "").strip()
        self._max_new_tokens = max_new_tokens

    @property
    def model_id(self) -> str:
        if self._model_id:
            return self._model_id
        import report

        return report.llm_model_name().strip() or report.llm_provider()

    def generate(
        self,
        messages: Sequence[PlannerMessage],
        *,
        attempt: int,
    ) -> str:
        import report

        if report.llm_provider() != "local":
            raise RuntimeError(
                "最小版KnowledgePlanner默认适配器只复用当前本地LLM；"
                "其他provider需注入PlannerLlmClient"
            )
        model = report.REPORT_MODEL
        model.ensure_loaded()
        return report._generate_local_text(
            model,
            list(messages),
            sample=attempt > 1,
            max_new_tokens=self._max_new_tokens,
        )


class KnowledgePlanner:
    """Create one validated retrieval plan, with one retry and a safe fallback."""

    def __init__(self, llm_client: Optional[PlannerLlmClient] = None) -> None:
        self._llm = llm_client or ExistingLlmClient()
        self._model_id = str(self._llm.model_id).strip()
        if not self._model_id:
            raise ValueError("Planner LLM model_id不能为空")

    def plan(
        self,
        interpretation: InterpretationResult,
        core_knowledge: CoreKnowledgeBundle,
    ) -> KnowledgePlan:
        if not isinstance(interpretation, InterpretationResult):
            raise TypeError("interpretation必须是InterpretationResult")
        if not isinstance(core_knowledge, CoreKnowledgeBundle):
            raise TypeError("core_knowledge必须是CoreKnowledgeBundle")

        for attempt in (1, 2):
            try:
                raw = self._llm.generate(
                    _planner_messages(
                        interpretation,
                        core_knowledge,
                        retry=attempt > 1,
                    ),
                    attempt=attempt,
                )
                payload_dict = _json_payload(raw)
                forbidden = _forbidden_key_paths(payload_dict)
                if forbidden:
                    raise ValueError(
                        "Planner输出包含禁止字段：" + "、".join(forbidden)
                    )
                payload = _PlannerPayload.model_validate(payload_dict)
                _validate_plan_scope(payload, interpretation)
                return KnowledgePlan(
                    planner_model_id=self._model_id,
                    topics=payload.topics,
                    queries=payload.queries,
                    reason=payload.reason,
                    generation_mode="llm",
                )
            except Exception:  # noqa: BLE001 - two attempts share one safe fallback
                if attempt == 2:
                    break
        return self._fallback(interpretation=interpretation)

    def _fallback(self, *, interpretation: InterpretationResult) -> KnowledgePlan:
        names: list[str] = []
        finding_ids: list[str] = []
        for finding in interpretation.findings:
            finding_ids.append(finding.finding_id)
            if finding.name not in names:
                names.append(finding.name)

        query_prefix = "、".join(names)
        suffix = "相关康复评估研究证据"
        max_prefix_length = 4000 - len(suffix) - 1
        query_text = f"{query_prefix[:max_prefix_length]}：{suffix}"
        topic_id = "fallback-topic-1"
        return KnowledgePlan(
            planner_model_id=self._model_id,
            topics=[
                KnowledgeTopic(
                    topic_id=topic_id,
                    label="结构化观察结果相关知识",
                    finding_ids=finding_ids,
                    priority="medium",
                )
            ],
            queries=[
                RetrievalQuery(
                    query_id="fallback-query-1",
                    topic_id=topic_id,
                    text=query_text,
                )
            ],
            reason=(
                "Planner LLM连续两次未返回符合契约的检索计划，"
                "已按finding名称生成保守检索查询。"
            ),
            generation_mode="fallback",
        )


__all__ = [
    "ExistingLlmClient",
    "KnowledgePlanner",
    "PlannerLlmClient",
]
