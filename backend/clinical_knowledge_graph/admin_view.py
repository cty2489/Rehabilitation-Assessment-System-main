"""Read-only admin projection of the static clinical knowledge graph.

This module exposes the existing JSON graph as a bounded, UI-friendly payload.
It does not add nodes, infer patient state, or alter any medical relationship.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


_ROOT = Path(__file__).resolve().parent
_GRAPH_PATH = _ROOT / "knowledge_graph_data.json"
_RULES_PATH = _ROOT / "clinical_rules.json"


class KnowledgeGraphAdminError(ValueError):
    """Raised when the static graph cannot be projected safely."""


def _read_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeGraphAdminError(f"无法读取知识图谱管理数据：{path.name}") from exc
    if not isinstance(value, dict):
        raise KnowledgeGraphAdminError(f"{path.name} 顶层必须是对象")
    return value


def knowledge_graph_admin_payload(
    graph_path: Path | None = None,
    rules_path: Path | None = None,
) -> Dict[str, Any]:
    """Return the existing static graph with materialized relation attributes."""

    graph = _read_object(graph_path or _GRAPH_PATH)
    rules_data = _read_object(rules_path or _RULES_PATH)
    if graph.get("schema_version") != "rehab.clinical-knowledge-graph.v1":
        raise KnowledgeGraphAdminError("知识图谱 schema_version 不受支持")

    raw_nodes = graph.get("nodes")
    raw_relations = graph.get("relations")
    defaults = graph.get("relation_attribute_defaults")
    raw_rules = rules_data.get("rules")
    if not isinstance(raw_nodes, list) or not isinstance(raw_relations, list):
        raise KnowledgeGraphAdminError("知识图谱缺少 nodes 或 relations 列表")
    if not isinstance(defaults, dict):
        raise KnowledgeGraphAdminError("知识图谱缺少关系默认属性")
    if not isinstance(raw_rules, list):
        raise KnowledgeGraphAdminError("知识图谱规则文件缺少 rules 列表")

    nodes = [dict(node) for node in raw_nodes if isinstance(node, dict)]
    node_ids = {str(node.get("node_id") or "") for node in nodes}
    if len(nodes) != len(raw_nodes) or "" in node_ids or len(node_ids) != len(nodes):
        raise KnowledgeGraphAdminError("知识图谱节点必须是具有唯一 node_id 的对象")

    relations: list[Dict[str, Any]] = []
    for raw in raw_relations:
        if not isinstance(raw, dict):
            raise KnowledgeGraphAdminError("知识图谱关系必须是对象")
        relation = {**defaults, **raw}
        if (
            not relation.get("relation_id")
            or relation.get("from") not in node_ids
            or relation.get("to") not in node_ids
        ):
            raise KnowledgeGraphAdminError("知识图谱关系缺少 ID 或引用了未知节点")
        relations.append(relation)

    rules = [dict(rule) for rule in raw_rules if isinstance(rule, dict)]
    if len(rules) != len(raw_rules):
        raise KnowledgeGraphAdminError("知识图谱规则必须是对象")

    node_type_counts = Counter(str(node.get("node_type") or "Unknown") for node in nodes)
    relation_type_counts = Counter(str(relation.get("type") or "UNKNOWN") for relation in relations)
    return {
        "schema_version": "rehab.knowledge-graph-admin.v1",
        "graph_schema_version": graph["schema_version"],
        "prototype_scope": graph.get("prototype_scope"),
        "node_types": list(graph.get("node_types") or []),
        "relation_types": list(graph.get("relation_types") or []),
        "summary": {
            "node_count": len(nodes),
            "relation_count": len(relations),
            "rule_count": len(rules),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
            "pending_relation_count": sum(
                relation.get("expert_review_status") == "pending"
                for relation in relations
            ),
            "unverified_relation_count": sum(
                relation.get("evidence_level") == "unverified"
                for relation in relations
            ),
            "implemented_from_count": relation_type_counts.get("IMPLEMENTED_FROM", 0),
            "supported_by_count": relation_type_counts.get("SUPPORTED_BY", 0),
            "pending_rule_count": sum(
                rule.get("expert_review_status") == "pending"
                for rule in rules
            ),
        },
        "nodes": nodes,
        "relations": relations,
        "rules": rules,
        "expert_review_status": "pending",
        "prototype_notice": (
            "当前为测试环境中的研究原型。关系用于解释系统如何组织指标、功能表现、"
            "临床维度和检索主题，不表示诊断或确定因果。"
        ),
    }


__all__ = [
    "KnowledgeGraphAdminError",
    "knowledge_graph_admin_payload",
]
