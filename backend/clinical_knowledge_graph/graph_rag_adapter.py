"""Bridge graph evidence into the existing LLM Planner/RAG contracts."""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from clinical_pipeline.contracts import (
    CanonicalAssessmentContext,
    InterpretationResult,
    KnowledgePlan,
    KnowledgeTopic,
    RetrievalQuery,
)

from .graph_engine import GraphEngine
from .rule_engine import RuleEngine


_TEXT_PUNCTUATION = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_TEXT_ALIASES = (
    ("fmaue", "fma"),
    ("fugl-meyer", "fma"),
    ("handmas", "mas"),
    ("modifiedashworth", "mas"),
    ("brunnstrom手功能分期", "brunnstrom"),
    ("肌电", "emg"),
    ("脑电", "eeg"),
    ("皮层肌肉相干", "cmc"),
    ("皮层-肌肉相干", "cmc"),
)


def _normalized_text(value: Any) -> str:
    """Normalize only obvious spelling/format variants for safe topic dedup."""
    text = str(value or "").lower()
    for source, target in _TEXT_ALIASES:
        text = text.replace(source, target)
    return _TEXT_PUNCTUATION.sub("", text)


def _semantic_markers(label: Any, query: Any) -> set[str]:
    """Return coarse topic markers; unknown/mixed text is intentionally retained."""
    text = _normalized_text(f"{label} {query}")
    markers: set[str] = set()
    checks = {
        "fma": ("fma", "手功能量表", "fugl"),
        "tone": ("mas", "肌张力", "痉挛"),
        "brunnstrom": ("brunnstrom", "手功能分期", "布氏"),
        "emg": ("emg", "共收缩", "肌肉募集", "屈伸肌", "共同激活", "肌电", "iEMG", "中位频率", "爆发"),
        "eeg": ("eeg", "cmc", "半球", "μ", "β节律", "theta", "前额叶"),
        "measurement_context": ("测量条件", "有效试次", "采样率", "同步", "复测边界"),
    }
    for marker, words in checks.items():
        if any(_normalized_text(word) in text for word in words):
            markers.add(marker)
    return markers


def _is_obvious_duplicate(
    existing: KnowledgeTopic,
    existing_query: Optional[RetrievalQuery],
    candidate: Mapping[str, Any],
) -> bool:
    """Merge only exact-normalized or single-domain topic duplicates.

    A broad Planner topic covering several domains is not merged into a single
    graph topic, even if it mentions that domain.  This avoids deleting useful
    distinctions merely because Chinese wording is similar.
    """
    candidate_label = _normalized_text(candidate.get("label"))
    candidate_query = _normalized_text(candidate.get("query"))
    existing_label = _normalized_text(existing.label)
    existing_query_text = _normalized_text(existing_query.text if existing_query else "")
    if candidate_label and candidate_label == existing_label:
        return True
    if candidate_query and candidate_query == existing_query_text:
        return True
    existing_markers = _semantic_markers(existing.label, existing_query.text if existing_query else "")
    candidate_markers = _semantic_markers(candidate.get("label"), candidate.get("query"))
    return (
        len(existing_markers) == 1
        and existing_markers == candidate_markers
        and bool(existing_markers)
    )


def _higher_priority(left: str, right: str) -> str:
    ranks = {"high": 0, "medium": 1, "low": 2}
    return left if ranks.get(left, 1) <= ranks.get(right, 1) else right


class GraphRagAdapter:
    """Create graph evidence and preserve its topics when Planner supplements it."""

    def __init__(
        self,
        graph_engine: Optional[GraphEngine] = None,
        rule_engine: Optional[RuleEngine] = None,
    ) -> None:
        self._graph_engine = graph_engine or GraphEngine()
        self._rule_engine = rule_engine or RuleEngine()

    def build_evidence(
        self,
        context: CanonicalAssessmentContext,
        interpretation: InterpretationResult,
    ) -> Dict[str, Any]:
        evidence = self._graph_engine.analyze(context, interpretation)
        evidence["rule_results"] = self._rule_engine.evaluate(evidence)
        return evidence

    @staticmethod
    def planner_context(evidence: Mapping[str, Any]) -> Dict[str, Any]:
        """Return only non-identifying graph outputs for the LLM Planner prompt."""
        quality = (evidence.get("patient_summary") or {}).get("quality") or {}
        status = str(quality.get("status") or "").strip().lower()
        quality_context = {
            "status": status or "unknown",
            "message": (
                "当前质量状态通过；measurement_context_topics仅用于测量条件、"
                "解释边界和同条件复测检索，不表示当前信号质量差。"
                if status == "pass"
                else (
                    "当前质量状态需要复核；data_quality_warnings列出可能受影响的指标解释。"
                    if status == "needs_review"
                    else "未提供明确质量总状态；measurement_context_topics仅表示解释边界，不表示当前信号质量差。"
                )
            ),
        }
        return {
            "graph_status": evidence.get("graph_status"),
            "prototype_scope": evidence.get("prototype_scope"),
            "analysis_dimensions": evidence.get("analysis_dimensions") or [],
            "rag_topics": evidence.get("rag_topics") or [],
            "measurement_context_topics": evidence.get("measurement_context_topics")
            or [],
            "rule_results": [
                item for item in (evidence.get("rule_results") or []) if item.get("matched")
            ],
            "confounders": evidence.get("confounders") or [],
            "data_quality_warnings": evidence.get("data_quality_warnings") or [],
            "data_quality_context": quality_context,
            "expert_review_status": "pending",
        }

    @staticmethod
    def merge_with_plan(
        plan: KnowledgePlan,
        evidence: Mapping[str, Any],
    ) -> KnowledgePlan:
        """Keep graph seeds while merging only obvious Planner/graph duplicates."""
        topics = list(plan.topics)
        queries = list(plan.queries)
        used_topic_ids = {topic.topic_id for topic in topics}
        used_query_ids = {query.query_id for query in queries}
        topic_origins = {
            topic.topic_id: list(
                dict.fromkeys(plan.topic_origins.get(topic.topic_id) or ["planner"])
            )
            for topic in topics
        }
        query_by_topic = {query.topic_id: query for query in queries}

        graph_items = [
            *(evidence.get("rag_topics") or []),
            *(evidence.get("measurement_context_topics") or []),
        ]
        for item in graph_items:
            finding_ids = list(dict.fromkeys(item.get("finding_ids") or []))
            if not finding_ids:
                continue
            graph_origin = f"graph:{item['topic_id']}"
            duplicate_index = next(
                (
                    index
                    for index, topic in enumerate(topics)
                    if _is_obvious_duplicate(topic, query_by_topic.get(topic.topic_id), item)
                ),
                None,
            )
            if duplicate_index is not None:
                existing = topics[duplicate_index]
                topics[duplicate_index] = existing.model_copy(
                    update={
                        "finding_ids": list(
                            dict.fromkeys(existing.finding_ids + finding_ids)
                        ),
                        "priority": _higher_priority(
                            existing.priority,
                            str(item.get("priority", "medium")),
                        ),
                    }
                )
                topic_origins[existing.topic_id] = list(
                    dict.fromkeys(topic_origins[existing.topic_id] + [graph_origin])
                )
                continue
            suffix = str(item["topic_id"]).replace("topic:", "").replace("_", "-")
            is_context = item["topic_id"] == "topic:measurement_context"
            topic_prefix = "kg-context" if is_context else "kg-topic"
            query_prefix = "kg-query-context" if is_context else "kg-query"
            topic_id = f"{topic_prefix}-{suffix}"
            query_id = f"{query_prefix}-{suffix}"
            if topic_id in used_topic_ids:
                continue
            while query_id in used_query_ids:
                query_id += "-graph"
            topics.append(
                KnowledgeTopic(
                    topic_id=topic_id,
                    label=f"图谱：{item['label']}",
                    finding_ids=finding_ids,
                    priority=item.get("priority", "medium"),
                )
            )
            queries.append(
                RetrievalQuery(
                    query_id=query_id,
                    topic_id=topic_id,
                    text=str(item["query"]),
                )
            )
            used_topic_ids.add(topic_id)
            used_query_ids.add(query_id)
            topic_origins[topic_id] = [graph_origin]

        return KnowledgePlan(
            planner_model_id=plan.planner_model_id,
            topics=topics,
            queries=queries,
            topic_origins=topic_origins,
            reason=(
                "图谱增强原型已先生成待审核的结构化主题；已合并明显重复主题。"
                "；"
                + plan.reason
            )[:1000],
            generation_mode=plan.generation_mode,
        )


__all__ = ["GraphRagAdapter"]
