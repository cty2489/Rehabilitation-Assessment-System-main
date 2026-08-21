import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_benchmark import (
    AssessmentInputMode,
    BenchmarkClinicalScores,
    BenchmarkEvidenceCard,
    BenchmarkGraphContext,
    BenchmarkKnowledgeContext,
    BenchmarkLlmOutput,
    BenchmarkPatientInfo,
    BenchmarkRunConfig,
    DoctorReviewScore,
    anonymous_report_id,
    build_blind_review_packet,
    build_evaluation_input,
    build_experiment_log,
    build_messages,
    build_registry_mapping,
    generate_benchmark_report,
    materialize_batch_manifest,
    materialize_benchmark_generation,
    parse_model_output,
    render_report,
)
from llm_benchmark.storage import append_blind_review_packet, append_experiment_log


class LlmBenchmarkInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.patient = BenchmarkPatientInfo(
            patient_id="P-001",
            name="测试患者",
            sex="男",
            age=58,
            diagnosis="脑卒中后偏瘫",
            disease_days=120,
            paralysis_side="左",
        )
        self.scores = BenchmarkClinicalScores(
            fma_wrist=6,
            FMA_UE=12,
            hand_tone="1+",
            hand_function=4,
        )
        markers = []
        for index in range(26):
            modality = ("eeg", "emg", "imu")[index % 3]
            markers.append(
                {
                    "key": f"marker_{index + 1}",
                    "name": f"原始指标{index + 1}",
                    "value_num": float(index + 1),
                    "unit": "u",
                    "available": True,
                    "n_valid": 3,
                }
            )
        self.biomarkers = {
            "groups": [
                {"key": "eeg", "markers": markers[0:9]},
                {"key": "emg", "markers": markers[9:18]},
                {"key": "imu", "markers": markers[18:26]},
            ]
        }
        self.evaluation_input = build_evaluation_input(
            patient=self.patient,
            clinical_scores=self.scores,
            biomarkers=self.biomarkers,
        )
        self.model_output = BenchmarkLlmOutput(
            biomarker_interpretation="仅解释有意义的指标，并注明单次测量的边界。",
            integrated_assessment="结合临床评分和关键指标，当前主要问题需要治疗师进一步观察。",
            rehabilitation_plan=[
                {
                    "action": "在治疗师保护下进行抓握后主动放开练习",
                    "goal": "改善抓握—放开衔接",
                    "reason": "与当前手功能阶段和现场观察目标一致",
                    "precaution": "出现明显代偿或疼痛时停止并由治疗师调整",
                },
                {
                    "action": "进行轻柔的手指主动伸展与屈曲转换",
                    "goal": "观察手指分离运动",
                    "reason": "与当前手部功能评定的动作质量观察一致",
                    "precaution": "出现阻力明显增加或疼痛时停止",
                },
                {
                    "action": "练习拿取、放置并主动松开容易控制的物品",
                    "goal": "提升任务中的抓放衔接",
                    "reason": "把可观察的手部动作转化为简单任务",
                    "precaution": "先由治疗师确认安全并避免代偿",
                },
            ],
        )

    def test_input_schema_preserves_manual_scores_and_all_26_markers(self):
        self.assertEqual(AssessmentInputMode.MANUAL_CLINICAL_SCORES, self.evaluation_input.config.assessment_input_mode)
        self.assertEqual(12, self.evaluation_input.clinical_scores.FMA_UE)
        self.assertEqual(6, self.evaluation_input.clinical_scores.fma_wrist)
        self.assertEqual("1+", self.evaluation_input.clinical_scores.hand_tone)
        self.assertEqual(4, self.evaluation_input.clinical_scores.hand_function)
        self.assertEqual(26, len(self.evaluation_input.biomarkers))
        self.assertEqual("原始指标1", self.evaluation_input.biomarkers[0].name)
        self.assertEqual(1.0, self.evaluation_input.biomarkers[0].value)

    def test_all_four_switch_combinations_are_independent(self):
        configs = [
            BenchmarkRunConfig(),
            BenchmarkRunConfig(rag_enabled=True, rag_version="rehab_knowledge_v1_candidate"),
            BenchmarkRunConfig(knowledge_graph_enabled=True, kg_version="kg_v1"),
            BenchmarkRunConfig(
                rag_enabled=True,
                rag_version="rehab_knowledge_v1_candidate",
                knowledge_graph_enabled=True,
                kg_version="kg_v1",
            ),
        ]
        self.assertEqual(
            [(False, False), (True, False), (False, True), (True, True)],
            [(item.rag_enabled, item.knowledge_graph_enabled) for item in configs],
        )
        with self.assertRaises(ValidationError):
            BenchmarkRunConfig(rag_enabled=True)
        with self.assertRaises(ValidationError):
            BenchmarkRunConfig(rag_version="old")
        with self.assertRaises(ValidationError):
            BenchmarkRunConfig(knowledge_graph_enabled=True)

    def test_prompt_injects_only_enabled_context(self):
        rag = BenchmarkKnowledgeContext(
            rag_version="rehab_knowledge_v1_candidate",
            evidence_cards=[],
            retrieved_text=["来源文本"],
        )
        kg = BenchmarkGraphContext(kg_version="kg_v1", findings=[{"key": "x"}])
        messages = build_messages(self.evaluation_input)
        self.assertEqual(2, len(messages))
        self.assertIn("fixed_assessment_input", messages[1]["content"])
        self.assertIn("patient_context", messages[1]["content"])
        self.assertIn("clinical_scores", messages[1]["content"])
        self.assertIn("biomarkers", messages[1]["content"])
        self.assertNotIn("测试患者", messages[1]["content"])
        self.assertNotIn("rag_context", messages[1]["content"])
        self.assertNotIn("knowledge_graph_context", messages[1]["content"])
        with self.assertRaises(ValueError):
            build_messages(self.evaluation_input, knowledge_context=rag)

        rag_input = self.evaluation_input.model_copy(
            update={
                "config": BenchmarkRunConfig(
                    rag_enabled=True,
                    rag_version="rehab_knowledge_v1_candidate",
                )
            }
        )
        rag_messages = build_messages(rag_input, knowledge_context=rag)
        self.assertIn("rag_context", rag_messages[1]["content"])
        self.assertNotIn("knowledge_graph_context", rag_messages[1]["content"])
        with self.assertRaises(ValueError):
            build_messages(rag_input)

        kg_input = self.evaluation_input.model_copy(
            update={
                "config": BenchmarkRunConfig(
                    knowledge_graph_enabled=True,
                    kg_version="kg_v1",
                )
            }
        )
        kg_messages = build_messages(kg_input, graph_context=kg)
        self.assertIn("knowledge_graph_context", kg_messages[1]["content"])
        self.assertNotIn("rag_context", kg_messages[1]["content"])

    def test_parser_accepts_json_fence_and_rejects_extra_top_level_fields(self):
        raw = "<think>hidden</think>\n```json\n" + json.dumps(
            self.model_output.model_dump(mode="json"), ensure_ascii=False
        ) + "\n```"
        parsed = parse_model_output(raw)
        self.assertEqual(self.model_output, parsed)
        invalid = self.model_output.model_dump(mode="json")
        invalid["patient_name"] = "不允许"
        with self.assertRaises(ValueError):
            parse_model_output(json.dumps(invalid, ensure_ascii=False))

    def test_report_renderer_keeps_four_sections_and_program_owned_facts(self):
        report = render_report(self.evaluation_input, self.model_output)
        self.assertEqual(4, sum(line.startswith("## ") for line in report.splitlines()))
        self.assertIn("P-001", report)
        self.assertIn("FMA手（fma_hand） | 12/20", report)
        self.assertIn("FMA腕（fma_wrist） | 6", report)
        self.assertIn("原始指标26", report)
        self.assertIn("## 四、康复建议", report)
        self.assertNotIn("综合亚型", report)
        self.assertNotIn("BI", report)
        self.assertIn("- 训练目的：改善抓握—放开衔接", report)
        self.assertIn("数据质量：通过", report)
        self.assertNotIn("依据来源卡片", report)
        prompt = build_messages(self.evaluation_input)[1]["content"]
        self.assertIn('"fma_wrist": 6.0', prompt)
        self.assertIn('"fma_hand": 12.0', prompt)
        self.assertNotIn('"BI"', prompt)

        rag_input = self.evaluation_input.model_copy(
            update={
                "config": BenchmarkRunConfig(
                    rag_enabled=True,
                    rag_version="rehab_knowledge_v1_candidate",
                )
            }
        )
        card = BenchmarkEvidenceCard(
            uid="U179-P100-C01",
            title="康复评定学",
            source_type="textbook",
            page_start=100,
            page_end=100,
            source_file="/srv/source/U179.pdf",
        )
        rag_report = render_report(rag_input, self.model_output, evidence_cards=[card])
        self.assertIn("U179-P100-C01", rag_report)
        self.assertIn("第100页", rag_report)
        self.assertIn("查看原文", rag_report)

    def test_log_and_blind_review_keep_audit_boundary(self):
        report_id = anonymous_report_id("case001", 1)
        self.assertEqual("CASE001-R01", report_id)
        log = build_experiment_log(
            evaluation_input=self.evaluation_input,
            anonymous_report_id=report_id,
            llm_model_id="model.internal.1",
            llm_model_name="内部测试模型",
            raw_model_output="原始模型返回内容",
            parsed_model_output=self.model_output,
            temperature=0.0,
            max_new_tokens=1024,
            generation_time=1234,
        )
        self.assertEqual("原始模型返回内容", log.raw_model_output)
        self.assertTrue(log.parse_success)
        self.assertEqual(1234, log.generation_time_ms)
        packet = build_blind_review_packet(
            report_id=report_id,
            evaluation_input=self.evaluation_input,
            report_markdown="# 匿名报告",
        )
        packet_text = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False)
        self.assertNotIn("model.internal.1", packet_text)
        self.assertNotIn("rag_enabled", packet_text)
        mapping = build_registry_mapping(
            report_id=report_id,
            evaluation_input=self.evaluation_input,
            llm_model_id="model.internal.1",
        )
        self.assertEqual("model.internal.1", mapping.llm_model_id)
        with self.assertRaises(ValidationError):
            DoctorReviewScore(
                clinical_correctness=6,
                completeness=4,
                clinical_usefulness=4,
                safety=4,
                hallucination_or_factual_error=False,
                serious_clinical_error=False,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "logs" / "experiment.jsonl"
            packet_path = root / "review" / "packets.jsonl"
            append_experiment_log(log_path, log)
            append_blind_review_packet(packet_path, packet)
            self.assertEqual(1, len(log_path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(1, len(packet_path.read_text(encoding="utf-8").splitlines()))

    def test_generation_materializes_report_log_and_blind_packet(self):
        raw = json.dumps(self.model_output.model_dump(mode="json"), ensure_ascii=False)
        result = generate_benchmark_report(
            evaluation_input=self.evaluation_input,
            model_id="model.internal.1",
            model_generate=lambda *_args, **_kwargs: raw,
            case_id="CASE001",
            report_index=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_root = materialize_batch_manifest(
                root,
                batch_id="batch_001",
                manifest={"batch_id": "batch_001"},
                validation_report={"status": "passed"},
            )
            report_path = materialize_benchmark_generation(batch_root, result)
            self.assertTrue(report_path.is_file())
            self.assertTrue((batch_root / "patients" / "P-001" / "reports" / "model.internal.1.md").is_file())
            self.assertEqual(1, len((batch_root / "logs" / "experiment.jsonl").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(1, len((batch_root / "logs" / "registry.jsonl").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(1, len((batch_root / "logs" / "blind_review.jsonl").read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    unittest.main()
