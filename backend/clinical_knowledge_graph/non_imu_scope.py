"""Single non-IMU scope boundary for graph-enhanced pipeline runs.

The legacy pipeline still supports IMU.  This module creates a transient view
for ``graph_enhanced`` only, so the graph, Planner, Retriever and report input
share the same non-IMU boundary without changing ``llm_only`` behaviour.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Tuple

from clinical_pipeline.contracts import (
    CanonicalAssessmentContext,
    FindingModality,
    InterpretationResult,
)


_IMU = FindingModality.IMU.value


def _modality_value(value: Any) -> str:
    return str(getattr(value, "value", value))


class NonImuScope:
    """Build and verify the transient non-IMU view used by graph enhancement."""

    @staticmethod
    def assessment_view(
        context: CanonicalAssessmentContext,
        interpretation: InterpretationResult,
    ) -> Tuple[CanonicalAssessmentContext, InterpretationResult]:
        """Return matching context/findings with every IMU item removed once."""
        scoped_context = context.model_copy(
            update={
                "biomarkers": [
                    marker
                    for marker in context.biomarkers
                    if _modality_value(marker.modality) != _IMU
                ]
            }
        )
        scoped_findings = [
            finding
            for finding in interpretation.findings
            if _modality_value(finding.modality) != _IMU
        ]
        allowed_finding_ids = {finding.finding_id for finding in scoped_findings}
        scoped_interpretation = interpretation.model_copy(
            update={
                "findings": scoped_findings,
                "known_combinations": [
                    item
                    for item in interpretation.known_combinations
                    if set(item.finding_ids).issubset(allowed_finding_ids)
                ],
            }
        )
        return scoped_context, scoped_interpretation

    @staticmethod
    def evidence_view(
        evidence: Mapping[str, Any],
        interpretation: InterpretationResult,
    ) -> Dict[str, Any]:
        """Defensively remove IMU references from graph evidence/context output.

        GraphEngine already excludes IMU.  Keeping this final sanitization here
        makes the scope boundary explicit and protects Planner context if a
        future graph relation accidentally references a filtered finding.
        """
        result = deepcopy(dict(evidence))
        allowed_finding_ids = {finding.finding_id for finding in interpretation.findings}
        states = [
            state
            for state in result.get("matched_indicator_states") or []
            if state.get("finding_id") in allowed_finding_ids
            and state.get("source_modality") != _IMU
        ]
        allowed_state_ids = {state.get("state_id") for state in states}
        paths = [
            path
            for path in result.get("matched_graph_paths") or []
            if len(path.get("node_path") or []) > 1
            and path["node_path"][1].get("node_id") in allowed_state_ids
        ]
        allowed_path_ids = {path.get("path_id") for path in paths}

        def scoped_topics(values: Any) -> list[Dict[str, Any]]:
            output: list[Dict[str, Any]] = []
            for raw in values or []:
                item = dict(raw)
                item["finding_ids"] = [
                    finding_id
                    for finding_id in item.get("finding_ids") or []
                    if finding_id in allowed_finding_ids
                ]
                item["graph_path_ids"] = [
                    path_id
                    for path_id in item.get("graph_path_ids") or []
                    if path_id in allowed_path_ids
                ]
                if item["finding_ids"] or item.get("topic_id") == "topic:measurement_context":
                    output.append(item)
            return output

        result["matched_indicator_states"] = states
        result["matched_graph_paths"] = paths
        result["rag_topics"] = scoped_topics(result.get("rag_topics"))
        result["measurement_context_topics"] = scoped_topics(
            result.get("measurement_context_topics")
        )
        for confounder in result.get("confounders") or []:
            confounder["affected_indicator_state_ids"] = [
                state_id
                for state_id in confounder.get("affected_indicator_state_ids") or []
                if state_id in allowed_state_ids
            ]
        for warning in result.get("data_quality_warnings") or []:
            if isinstance(warning, dict):
                warning["affected_indicator_state_ids"] = [
                    state_id
                    for state_id in warning.get("affected_indicator_state_ids") or []
                    if state_id in allowed_state_ids
                ]
        return result


__all__ = ["NonImuScope"]
