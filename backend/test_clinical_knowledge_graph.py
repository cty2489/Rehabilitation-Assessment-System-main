"""Contract tests for the lightweight non-IMU clinical knowledge graph (MVP-aligned)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from clinical_knowledge_graph.admin_view import knowledge_graph_admin_payload
from clinical_knowledge_graph.demo_non_imu import sample_context
from clinical_knowledge_graph.graph_engine import GraphEngine
from clinical_knowledge_graph.graph_rag_adapter import GraphRagAdapter
from clinical_pipeline.contracts import KnowledgePlan, KnowledgeTopic, RetrievalQuery
from clinical_pipeline.interpreter import Interpreter


_GRAPH_ROOT = Path(__file__).resolve().parent / "clinical_knowledge_graph"
_RELATION_FIELDS = {
    "relation_id",
    "source_type",
    "source_reference",
    "evidence_level",
    "applicable_population",
    "applicable_task",
    "causal_status",
    "expert_review_status",
    "notes",
}

# MVP：正式纳入的指标类型与数量（暂缓 EEG 3 项不入正式图谱）
_EXPECTED_EMG_COUNT = 14
_EXPECTED_EEG_COUNT = 3
_EXPECTED_NODE_COUNT = 51
_EXPECTED_RELATION_COUNT = 44

# 循环链相关节点（v2/v3 已删除，禁止出现）
_LOOP_NODE_IDS = {
    "prediction:FMA_UE",
    "prediction:hand_tone",
    "prediction:hand_function",
    "functional:hand_action_performance",
    "functional:muscle_tone_observation",
    "functional:hand_motor_recovery_stage",
    "dimension:hand_function",
    "dimension:muscle_tone",
    "dimension:hand_motor_recovery",
    "dimension:multimodal_coordination",
    "functional:eeg_corticomuscular_coordination",
    "functional:eeg_sensorimotor_rhythm",
}

# 暂缓 EEG（跨模态未精同步，不入正式图谱）
_DEFERRED_KEYS = {
    "corticomuscular_coherence_beta",
    "movement_mu_power_change",
    "movement_beta_power_change",
}


class ClinicalKnowledgeGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = sample_context()
        self.interpretation = Interpreter().interpret(self.context)

    def test_static_graph_contains_expected_non_imu_biomarker_types(self) -> None:
        data = json.loads((_GRAPH_ROOT / "knowledge_graph_data.json").read_text())
        nodes = data["nodes"]
        self.assertEqual(
            len([node for node in nodes if node["node_type"] == "EMGBiomarker"]),
            _EXPECTED_EMG_COUNT,
        )
        self.assertEqual(
            len([node for node in nodes if node["node_type"] == "EEGBiomarker"]),
            _EXPECTED_EEG_COUNT,
        )
        self.assertEqual(
            len([node for node in nodes if node["node_type"] == "ModelPrediction"]),
            3,
        )
        # PredictionTarget 静态节点存在（MVP 新增）
        self.assertEqual(
            len([node for node in nodes if node["node_type"] == "PredictionTarget"]),
            3,
        )
        # 三个量表解释 RAGTopic 存在
        topic_ids = {node["node_id"] for node in nodes if node["node_type"] == "EvidenceTopic"}
        self.assertIn("topic:FMA_hand_interpretation", topic_ids)
        self.assertIn("topic:MAS_hand_interpretation", topic_ids)
        self.assertIn("topic:Brunnstrom_hand_interpretation", topic_ids)
        # 循环链节点不存在
        node_ids = {node["node_id"] for node in nodes}
        self.assertFalse(_LOOP_NODE_IDS & node_ids)
        # 暂缓 EEG 不入正式图谱
        deferred_in_nodes = {
            node["node_id"]
            for node in nodes
            if any(key in node.get("node_id", "") for key in _DEFERRED_KEYS)
        }
        self.assertFalse(deferred_in_nodes)
        # 无 IMU 指标
        source_fields = [str(node.get("source_field", "")) for node in nodes]
        self.assertFalse(any(field.startswith("biomarkers.imu") for field in source_fields))

        defaults = data["relation_attribute_defaults"]
        for relation in data["relations"]:
            materialized = {**defaults, **relation}
            self.assertTrue(_RELATION_FIELDS.issubset(materialized))
            self.assertEqual(materialized["evidence_level"], "unverified")
            self.assertEqual(materialized["expert_review_status"], "pending")
            self.assertEqual(materialized["causal_status"], "associative")
        # MVP 新增关系类型
        self.assertIn("PREDICTS_TARGET", data["relation_types"])
        self.assertIn("HAS_INTERPRETATION_TOPIC", data["relation_types"])
        self.assertIn("IMPLEMENTED_FROM", data["relation_types"])
        self.assertFalse(
            any(relation["type"] == "SUPPORTED_BY" for relation in data["relations"])
        )
        self.assertTrue(
            any(relation["type"] == "IMPLEMENTED_FROM" for relation in data["relations"])
        )

    def test_admin_projection_exposes_real_reviewable_graph_without_imu(self) -> None:
        payload = knowledge_graph_admin_payload()

        self.assertEqual(payload["schema_version"], "rehab.knowledge-graph-admin.v1")
        self.assertEqual(payload["summary"]["node_count"], _EXPECTED_NODE_COUNT)
        self.assertEqual(payload["summary"]["relation_count"], _EXPECTED_RELATION_COUNT)
        self.assertEqual(payload["summary"]["rule_count"], 5)
        self.assertEqual(payload["summary"]["supported_by_count"], 0)
        self.assertEqual(len(payload["nodes"]), _EXPECTED_NODE_COUNT)
        self.assertEqual(len(payload["relations"]), _EXPECTED_RELATION_COUNT)
        self.assertTrue(
            all(_RELATION_FIELDS.issubset(relation) for relation in payload["relations"])
        )
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        self.assertNotIn('"node_type": "imu', serialized)
        self.assertNotIn("movement_smoothness_sparc", serialized)
        # PredictionResult 是运行时对象：不出现为节点类型，不承载患者值
        self.assertNotIn("predictionresult", payload["summary"]["node_type_counts"])
        self.assertNotIn("patient_id", json.dumps(payload["nodes"], ensure_ascii=False))

    def test_evidence_bundle_excludes_imu_and_has_complete_trace_paths(self) -> None:
        evidence = GraphRagAdapter().build_evidence(self.context, self.interpretation)

        self.assertEqual(evidence["expert_review_status"], "pending")
        self.assertEqual(evidence["patient_summary"]["excluded_modalities"], ["imu"])
        self.assertEqual(evidence["data_quality_warnings"], [])
        self.assertEqual(
            [item["topic_id"] for item in evidence["measurement_context_topics"]],
            ["topic:measurement_context"],
        )
        self.assertTrue(evidence["matched_graph_paths"])
        self.assertFalse(
            any(
                state["source_modality"] == "imu"
                for state in evidence["matched_indicator_states"]
            )
        )
        self.assertFalse(
            any("movement_smoothness_sparc" in json.dumps(path) for path in evidence["matched_graph_paths"])
        )
        for path in evidence["matched_graph_paths"]:
            self.assertEqual(path["node_path"][0]["node_type"], "PatientContext")
            self.assertEqual(path["node_path"][1]["node_type"], "IndicatorState")
            self.assertEqual(path["node_path"][-1]["node_type"], "EvidenceTopic")
            for relation in path["relation_path"]:
                self.assertTrue(_RELATION_FIELDS.issubset(relation))

    def test_deferred_eeg_indicators_do_not_enter_graph_paths(self) -> None:
        """MVP：暂缓 EEG 即使运行时有值，也不进入正式图谱路径。"""
        evidence = GraphRagAdapter().build_evidence(self.context, self.interpretation)
        path_blob = json.dumps(evidence["matched_graph_paths"], ensure_ascii=False)
        state_blob = json.dumps(evidence["matched_indicator_states"], ensure_ascii=False)
        for key in _DEFERRED_KEYS:
            self.assertNotIn(key, path_blob)
            self.assertNotIn(key, state_blob)

    def test_prediction_outputs_do_not_enter_biomarker_chain(self) -> None:
        """MVP：模型输出不进入 FunctionalFinding / ClinicalDimension 链。"""
        evidence = GraphRagAdapter().build_evidence(self.context, self.interpretation)
        path_blob = json.dumps(evidence["matched_graph_paths"], ensure_ascii=False)
        # 无 predictions.* 路径（模型输出独立为 PredictionResult 运行时对象）
        self.assertNotIn("predictions.fma_ue", path_blob.lower())
        self.assertNotIn("predictions.hand_tone", path_blob.lower())
        self.assertNotIn("predictions.hand_function", path_blob.lower())
        # 生物标志物路径只从 EMG/EEG 出发
        node_types = set()
        for path in evidence["matched_graph_paths"]:
            for node in path["node_path"]:
                node_types.add(node.get("node_type"))
        self.assertTrue(node_types & {"EMGBiomarker", "EEGBiomarker"})
        self.assertNotIn("ModelPrediction", node_types)

    def test_static_graph_contains_no_patient_prediction_values(self) -> None:
        """MVP：静态图谱不包含患者具体预测值/CanonicalPredictions。"""
        raw = ( _GRAPH_ROOT / "knowledge_graph_data.json").read_text(encoding="utf-8")
        self.assertNotIn("patient_id", raw)
        self.assertNotIn("assessment_id", raw)
        self.assertNotIn("CanonicalPredictions", raw)
        self.assertNotIn("PredictionResult", [n.get("node_type","") for n in json.loads(raw)["nodes"]])

    def test_prefrontal_theta_beta_remains_descriptive_without_clinical_confirmation(self) -> None:
        """MVP：前额叶 θ/β 未获临床确认前，保留指标+描述性功能表征，不建立维度/RAG主题。"""
        data = json.loads((_GRAPH_ROOT / "knowledge_graph_data.json").read_text())
        nodes = data["nodes"]
        rels = data["relations"]
        node_ids = {n["node_id"] for n in nodes}

        # 1. 指标存在
        self.assertIn("eeg:prefrontal_theta_beta_ratio", node_ids)
        # 2. FunctionalFinding 存在且标记描述性
        ff = next(n for n in nodes if n["node_id"] == "functional:eeg_prefrontal_theta_beta")
        self.assertEqual(ff["node_type"], "FunctionalFinding")
        self.assertTrue(ff.get("descriptive_only", False))
        # 3. 不连接 ClinicalDimension
        self.assertFalse(
            any(
                r["from"] == "functional:eeg_prefrontal_theta_beta"
                and r["type"] == "BELONGS_TO"
                for r in rels
            )
        )
        # 4. 不生成对应临床 RAGTopic
        self.assertNotIn("dimension:prefrontal_spectral", node_ids)
        self.assertNotIn("topic:prefrontal_spectral", node_ids)
        self.assertFalse(
            any(r["from"] == "dimension:prefrontal_spectral" for r in rels)
        )

    def test_quality_rule_remains_non_clinical_and_reviewable(self) -> None:
        review_context = self.context.model_copy(
            update={
                "quality_metadata": {
                    **self.context.quality_metadata,
                    "status": "needs_review",
                    "sync_fallback_count": 1,
                }
            }
        )
        evidence = GraphRagAdapter().build_evidence(
            review_context,
            Interpreter().interpret(review_context),
        )
        quality_rule = next(
            item
            for item in evidence["rule_results"]
            if item["rule_id"] == "rule:quality-needs-review"
        )
        self.assertTrue(quality_rule["matched"])
        self.assertEqual(quality_rule["result"]["status"], "caution")
        self.assertEqual(quality_rule["evidence_level"], "unverified")
        self.assertEqual(quality_rule["expert_review_status"], "pending")
        self.assertTrue(evidence["data_quality_warnings"])
        self.assertEqual(
            evidence["data_quality_warnings"][0]["code"],
            "quality_status_not_pass",
        )
        self.assertTrue(
            evidence["data_quality_warnings"][0]["affected_indicator_state_ids"]
        )
        # MVP：数据质量不作为临床维度出现在维度列表
        dim_ids = [d["dimension_id"] for d in evidence["analysis_dimensions"]]
        self.assertNotIn("dimension:data_quality_repeatability", dim_ids)

    def test_planner_adapter_strips_patient_identity_and_preserves_graph_topics(self) -> None:
        adapter = GraphRagAdapter()
        evidence = adapter.build_evidence(self.context, self.interpretation)
        planner_context = adapter.planner_context(evidence)
        self.assertNotIn("KG-DEMO-001", json.dumps(planner_context, ensure_ascii=False))
        self.assertEqual(planner_context["data_quality_context"]["status"], "pass")
        self.assertIn("当前质量状态通过", planner_context["data_quality_context"]["message"])
        self.assertEqual(planner_context["data_quality_warnings"], [])

        base_plan = KnowledgePlan(
            planner_model_id="test-model",
            topics=[
                KnowledgeTopic(
                    topic_id="planner-topic-1",
                    label="Planner补充主题",
                    finding_ids=["prediction:FMA_UE"],
                    priority="medium",
                )
            ],
            queries=[
                RetrievalQuery(
                    query_id="planner-query-1",
                    topic_id="planner-topic-1",
                    text="FMA模型预测解释边界",
                )
            ],
            reason="测试Planner补充主题。",
            generation_mode="llm",
        )
        merged = adapter.merge_with_plan(base_plan, evidence)
        merged_ids = {topic.topic_id for topic in merged.topics}
        self.assertIn("planner-topic-1", merged_ids)
        self.assertTrue(any(topic_id.startswith("kg-topic-") for topic_id in merged_ids))
        self.assertEqual(len(merged.topics), len(merged.queries))

    def test_obvious_emg_topic_duplicate_is_merged_with_origin_trace(self) -> None:
        evidence = GraphRagAdapter().build_evidence(self.context, self.interpretation)
        base_plan = KnowledgePlan(
            planner_model_id="test-model",
            topics=[
                KnowledgeTopic(
                    topic_id="planner-emg",
                    label="肌电协同与募集记录解释",
                    finding_ids=["biomarker:wrist_co_contraction_index"],
                    priority="medium",
                )
            ],
            queries=[
                RetrievalQuery(
                    query_id="planner-query-emg",
                    topic_id="planner-emg",
                    text="上肢康复中肌电共收缩、肌肉募集和同条件复测的解释边界",
                )
            ],
            reason="测试肌电语义去重。",
            generation_mode="llm",
        )

        merged = GraphRagAdapter().merge_with_plan(base_plan, evidence)

        self.assertNotIn("kg-topic-emg", {topic.topic_id for topic in merged.topics})
        # MVP：EMG 主题更名为肌电活动与共同激活
        self.assertIn("graph:topic:emg_activation", merged.topic_origins["planner-emg"])
        emg_topic = next(topic for topic in merged.topics if topic.topic_id == "planner-emg")
        self.assertIn("biomarker:emg_activation_rms", emg_topic.finding_ids)


if __name__ == "__main__":
    unittest.main()
