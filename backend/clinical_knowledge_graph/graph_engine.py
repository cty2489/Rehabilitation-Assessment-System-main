"""Runtime matching for the JSON non-IMU clinical knowledge graph.

The graph is deliberately a lightweight, immutable JSON resource.  It stores
stable *associative* links only; this engine turns a single canonical assessment
into a traceable evidence bundle without persisting patient data or assigning
clinical abnormality thresholds.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from clinical_pipeline.contracts import (
    CanonicalAssessmentContext,
    FindingModality,
    InterpretationResult,
)


_ROOT = Path(__file__).resolve().parent
_DATA_PATH = _ROOT / "knowledge_graph_data.json"
_EXCLUDED_MODALITY = FindingModality.IMU.value


class GraphDataError(ValueError):
    """Raised when the static prototype graph cannot be used safely."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphDataError(f"无法读取知识图谱数据：{path}") from exc
    if not isinstance(value, dict):
        raise GraphDataError("知识图谱顶层必须是对象")
    return value


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class GraphEngine:
    """Match real canonical inputs to static non-IMU graph paths.

    ``interpretation`` is intentionally an input: its existing finding status
    carries the project-approved distinction between missing, direction-only
    and not-classifiable measurements.  The graph does not recalculate or
    reinterpret any medical reference range.
    """

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self._data_path = data_path or _DATA_PATH
        self._data = _read_json(self._data_path)
        if self._data.get("schema_version") != "rehab.clinical-knowledge-graph.v1":
            raise GraphDataError("知识图谱schema_version不受支持")

        nodes = self._data.get("nodes")
        relations = self._data.get("relations")
        defaults = self._data.get("relation_attribute_defaults")
        if not isinstance(nodes, list) or not isinstance(relations, list):
            raise GraphDataError("知识图谱必须包含nodes和relations列表")
        if not isinstance(defaults, dict):
            raise GraphDataError("知识图谱缺少relation_attribute_defaults")

        self._nodes = {str(node.get("node_id")): dict(node) for node in nodes}
        if len(self._nodes) != len(nodes) or "" in self._nodes:
            raise GraphDataError("知识图谱node_id必须唯一且非空")
        self._nodes_by_source_field = {
            str(node["source_field"]): node_id
            for node_id, node in self._nodes.items()
            if node.get("source_field")
        }
        self._relations: list[Dict[str, Any]] = []
        for raw in relations:
            relation = dict(defaults)
            relation.update(raw)
            if not relation.get("relation_id") or not relation.get("from") or not relation.get("to"):
                raise GraphDataError("每条关系必须具有relation_id、from和to")
            if relation["from"] not in self._nodes or relation["to"] not in self._nodes:
                raise GraphDataError(f"关系引用未知节点：{relation['relation_id']}")
            self._relations.append(relation)
        self._outgoing: Dict[str, list[Dict[str, Any]]] = {}
        for relation in self._relations:
            self._outgoing.setdefault(str(relation["from"]), []).append(relation)

    @property
    def data_path(self) -> Path:
        return self._data_path

    def analyze(
        self,
        context: CanonicalAssessmentContext,
        interpretation: InterpretationResult,
    ) -> Dict[str, Any]:
        if not isinstance(context, CanonicalAssessmentContext):
            raise TypeError("context必须是CanonicalAssessmentContext")
        if not isinstance(interpretation, InterpretationResult):
            raise TypeError("interpretation必须是InterpretationResult")

        marker_by_key = {marker.metric_key: marker for marker in context.biomarkers}
        indicator_states = self._indicator_states(interpretation, marker_by_key)
        paths, dimensions, topics = self._match_paths(indicator_states)
        quality = dict(context.quality_metadata)
        confounders, warnings = self._confounders(indicator_states, quality)
        measurement_context_topics: Dict[str, Dict[str, Any]] = {}

        if confounders:
            self._add_measurement_context_topic(
                measurement_context_topics,
                paths,
                indicator_states,
            )
            # MVP：数据质量为独立审计分支，不生成临床维度节点。
            # 维度清单只来自生物标志物路径（_match_paths）。

        evidence_sources = self._evidence_sources(dimensions)
        patient = context.patient
        summary = {
            "patient_id": patient.patient_id,
            "age": patient.age,
            "sex": patient.sex,
            "diagnosis": patient.diagnosis,
            "disease_days": patient.disease_days,
            "paralysis_side": patient.paralysis_side,
            "quality": quality,
            "available_by_modality": {
                modality: sum(
                    1
                    for state in indicator_states
                    if state.get("source_modality") == modality and state["available"]
                )
                for modality in ("emg", "eeg")
            },
            "missing_non_imu_marker_count": sum(
                1
                for state in indicator_states
                if state.get("source_modality") in {"emg", "eeg"}
                and not state["available"]
            ),
            "excluded_modalities": ["imu"],
        }
        return {
            "schema_version": "rehab.clinical-knowledge-graph-evidence.v1",
            "graph_status": "matched",
            "prototype_scope": self._data.get("prototype_scope"),
            "patient_summary": summary,
            "analysis_dimensions": list(dimensions.values()),
            "matched_indicator_states": indicator_states,
            "matched_graph_paths": paths,
            "rule_results": [],
            "rag_topics": list(topics.values()),
            "measurement_context_topics": list(measurement_context_topics.values()),
            "confounders": confounders,
            "data_quality_warnings": warnings,
            "evidence_sources": evidence_sources,
            "expert_review_status": "pending",
        }

    def _indicator_states(
        self,
        interpretation: InterpretationResult,
        marker_by_key: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        states: list[Dict[str, Any]] = []
        for finding in interpretation.findings:
            if getattr(finding.modality, "value", finding.modality) == _EXCLUDED_MODALITY:
                continue
            source_field = str(finding.source_field)
            # MVP：模型预测结果（predictions.*）是运行时患者输出，
            # 不进入生物标志物链（避免原 ModelPrediction→FunctionalFinding→ClinicalDimension 循环）。
            if source_field.startswith("predictions."):
                continue
            node_id = self._nodes_by_source_field.get(source_field)
            if node_id is None:
                continue
            # 仅接受生物标志物节点；ModelPrediction/PredictionTarget 不生成 IndicatorState。
            node_type = self._nodes[node_id]["node_type"]
            if node_type not in {"EMGBiomarker", "EEGBiomarker"}:
                continue
            marker = marker_by_key.get(finding.metric_key)
            source_modality = (
                str(getattr(marker.modality, "value", marker.modality))
                if marker is not None
                else None
            )
            if source_modality == _EXCLUDED_MODALITY:
                continue
            available = bool(marker.available) if marker is not None else finding.value is not None
            n_valid = int(marker.n_valid) if marker is not None else None
            states.append(
                {
                    "state_id": f"state:{finding.finding_id}",
                    "node_type": "IndicatorState",
                    "indicator_node_id": node_id,
                    "indicator_node_type": self._nodes[node_id]["node_type"],
                    "finding_id": finding.finding_id,
                    "metric_key": finding.metric_key,
                    "label": finding.name,
                    "source_field": source_field,
                    "source_modality": source_modality,
                    "value": finding.value,
                    "unit": finding.unit,
                    "available": available,
                    "n_valid": n_valid,
                    "state": getattr(finding.status, "value", str(finding.status)),
                    "interpretation_basis": finding.basis.kind.value,
                }
            )
        return states

    def _match_paths(
        self,
        states: list[Dict[str, Any]],
    ) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        paths: list[Dict[str, Any]] = []
        dimensions: Dict[str, Dict[str, Any]] = {}
        topics: Dict[str, Dict[str, Any]] = {}
        for state in states:
            if not state["available"]:
                continue
            # MVP 防线：生物标志物路径只从 EMG/EEG 指标出发；模型输出/目标定义不进入该链。
            if state.get("indicator_node_type") not in {"EMGBiomarker", "EEGBiomarker"}:
                continue
            for measurement in self._outgoing.get(state["indicator_node_id"], []):
                if measurement["type"] not in {"MEASURES", "DESCRIBES"}:
                    continue
                functional_id = measurement["to"]
                for belongs in self._outgoing.get(functional_id, []):
                    if belongs["type"] != "BELONGS_TO":
                        continue
                    dimension_id = belongs["to"]
                    dimensions.setdefault(dimension_id, self._dimension_record(dimension_id))
                    for suggests in self._outgoing.get(dimension_id, []):
                        if suggests["type"] != "SUGGESTS_TOPIC":
                            continue
                        topic_id = suggests["to"]
                        path_id = (
                            f"path:{state['metric_key']}:{functional_id.split(':', 1)[1]}:"
                            f"{dimension_id.split(':', 1)[1]}:{topic_id.split(':', 1)[1]}"
                        )
                        path = {
                            "path_id": path_id,
                            "source_field": state["source_field"],
                            "node_path": [
                                {"node_type": "PatientContext", "source_field": state["source_field"]},
                                {"node_id": state["state_id"], "node_type": "IndicatorState", "state": state["state"]},
                                self._path_node(state["indicator_node_id"]),
                                self._path_node(functional_id),
                                self._path_node(dimension_id),
                                self._path_node(topic_id),
                            ],
                            "relation_path": [
                                {
                                    "relation_id": f"runtime:patient-context:{state['metric_key']}",
                                    "type": "ASSOCIATED_WITH",
                                    "source_type": "runtime_patient_context",
                                    "source_reference": state["source_field"],
                                    "evidence_level": "unverified",
                                    "applicable_population": "本系统测试数据；适用人群待专家确认",
                                    "applicable_task": "本系统主动上肢/手部评估任务；具体任务映射待确认",
                                    "causal_status": "associative",
                                    "expert_review_status": "pending",
                                    "notes": "运行时患者字段映射；仅表示本次数据来源，不表达医学因果。",
                                },
                                self._relation_view(measurement),
                                self._relation_view(belongs),
                                self._relation_view(suggests),
                            ],
                        }
                        paths.append(path)
                        topic = topics.setdefault(topic_id, self._topic_record(topic_id))
                        topic["finding_ids"] = _unique(topic["finding_ids"] + [state["finding_id"]])
                        topic["graph_path_ids"] = _unique(topic["graph_path_ids"] + [path_id])
        return paths, dimensions, topics

    def _confounders(
        self,
        states: list[Dict[str, Any]],
        quality: Mapping[str, Any],
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        values: list[Dict[str, Any]] = []
        warnings: list[Dict[str, Any]] = []

        def add(node_id: str, reason: str, affected: list[str]) -> None:
            node = self._nodes[node_id]
            values.append(
                {
                    "confounder_id": node_id,
                    "label": node["label"],
                    "reason": reason,
                    "affected_indicator_state_ids": affected,
                    "expert_review_status": "pending",
                }
            )

        def quality_warning(
            code: str,
            message: str,
            affected: list[str],
        ) -> None:
            warnings.append(
                {
                    "code": code,
                    "message": message,
                    "affected_indicator_state_ids": affected,
                    "expert_review_status": "pending",
                }
            )

        missing = [state["state_id"] for state in states if not state["available"]]
        if missing:
            add("confounder:missing_or_invalid_marker", "存在不可用指标或有效试次不足。", missing)
        protocol_limited = [
            state["state_id"]
            for state in states
            if state["state"] in {"not_classifiable", "direction_only"}
        ]
        if protocol_limited:
            add(
                "confounder:device_or_protocol_specific_metric",
                "部分指标当前仅可作为同条件复测记录，不能由本次单值形成绝对分类。",
                protocol_limited,
            )
        all_non_imu = [state["state_id"] for state in states]
        checks = (
            ("short_trial_count", "confounder:short_trial", "存在时长不足的试次。"),
            ("sync_fallback_count", "confounder:sync_fallback", "存在同步回退的试次。"),
            (
                "sampling_rate_mismatch_count",
                "confounder:sampling_rate_mismatch",
                "存在采样率与声明不一致的试次。",
            ),
        )
        active_checks: list[tuple[str, str, str]] = []
        for field, node_id, reason in checks:
            try:
                active = int(quality.get(field) or 0) > 0
            except (TypeError, ValueError):
                active = False
            if active:
                add(node_id, reason, [])
                active_checks.append((field, node_id, reason))

        status = str(quality.get("status") or "").strip().lower()
        if status == "needs_review":
            quality_warning(
                "quality_status_not_pass",
                "本次质量状态不是pass；相关指标解释需保留质量限制。",
                all_non_imu,
            )
            for field, _node_id, reason in active_checks:
                quality_warning(
                    field,
                    reason,
                    all_non_imu,
                )
        return values, warnings

    def _add_measurement_context_topic(
        self,
        topics: Dict[str, Dict[str, Any]],
        paths: list[Dict[str, Any]],
        states: list[Dict[str, Any]],
    ) -> None:
        topic_id = "topic:measurement_context"
        topic = topics.setdefault(topic_id, self._topic_record(topic_id))
        topic["finding_ids"] = _unique(topic["finding_ids"] + [state["finding_id"] for state in states])
        topic["graph_path_ids"] = _unique(topic["graph_path_ids"] + [path["path_id"] for path in paths])

    def _dimension_record(self, node_id: str) -> Dict[str, Any]:
        node = self._nodes[node_id]
        return {"dimension_id": node_id, "label": node["label"], "expert_review_status": "pending"}

    def _topic_record(self, node_id: str) -> Dict[str, Any]:
        node = self._nodes[node_id]
        priority = "high" if node_id in {
            "topic:FMA_hand_interpretation",
            "topic:MAS_hand_interpretation",
            "topic:Brunnstrom_hand_interpretation",
        } else "medium"
        return {
            "topic_id": node_id,
            "label": node["label"],
            "query": node["query_template"],
            "priority": priority,
            "finding_ids": [],
            "graph_path_ids": [],
            "expert_review_status": "pending",
        }

    def _evidence_sources(self, dimensions: Mapping[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
        source_ids: list[str] = []
        for dimension_id in dimensions:
            source_ids.extend(
                relation["to"]
                for relation in self._outgoing.get(dimension_id, [])
                if relation["type"] == "IMPLEMENTED_FROM"
            )
        return [
            {
                "source_id": source_id,
                "label": self._nodes[source_id]["label"],
                "source_reference": self._nodes[source_id].get("reference"),
                "source_relation_type": "IMPLEMENTED_FROM",
                "evidence_level": "unverified",
                "expert_review_status": "pending",
            }
            for source_id in _unique(source_ids)
        ]

    def _path_node(self, node_id: str) -> Dict[str, Any]:
        node = self._nodes[node_id]
        return {"node_id": node_id, "node_type": node["node_type"], "label": node["label"]}

    @staticmethod
    def _relation_view(relation: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: deepcopy(relation[key])
            for key in (
                "relation_id",
                "type",
                "source_type",
                "source_reference",
                "evidence_level",
                "applicable_population",
                "applicable_task",
                "causal_status",
                "expert_review_status",
                "notes",
            )
        }


__all__ = ["GraphDataError", "GraphEngine"]
