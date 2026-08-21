import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from openpyxl import Workbook

from llm_benchmark import (
    BenchmarkGraphContext,
    BenchmarkKnowledgeContext,
    BenchmarkRunConfig,
    generate_benchmark_report,
    generate_benchmark_batch_reports,
    prepare_batch_inputs,
    read_benchmark_batch,
)


def _fake_biomarkers(*_args, **_kwargs):
    groups = []
    for modality, start, end in (("eeg", 1, 7), ("emg", 7, 21), ("imu", 21, 27)):
        groups.append({
            "key": modality,
            "markers": [
                {
                    "key": f"marker_{index}",
                    "name": f"指标{index}",
                    "value": float(index),
                    "unit": "u",
                    "available": True,
                    "n_valid": 1,
                }
                for index in range(start, end)
            ],
        })
    return {"groups": groups, "coverage": {"available": 26, "total": 26}}


class BenchmarkBatchTests(unittest.TestCase):
    def _patient(self, root: Path, patient_id: str) -> None:
        patient_root = root / patient_id
        (patient_root / "active").mkdir(parents=True)
        manifest = {
            "patient_id": patient_id,
            "patient_name": patient_id,
            "patient_gender": "男",
            "patient_age": 55,
            "diagnosis": "脑卒中后偏瘫",
            "disease_days": 120,
            "paralysis_side": "左",
            "assessments": [
                {
                    "assessment_type": "active",
                    "action_id": "action_1",
                    "action_name": "伸指",
                    "trials": [
                        {"trial_index": 1, "eeg_file": "active/eeg.bdf", "emg_imu_file": "active/emg.csv"}
                    ],
                }
            ],
        }
        (patient_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        (patient_root / "active/eeg.bdf").write_bytes(b"eeg")
        (patient_root / "active/emg.csv").write_bytes(b"emg")

    def _batch_root(self, root: Path) -> None:
        self._patient(root, "P001")
        self._patient(root, "P002")
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["patient_id", "fma_wrist", "fma_hand", "hand_mas", "brunnstrom_hand"])
        sheet.append(["P001", 6, 15, "2", 4])
        sheet.append(["P002", 3, 8, "1+", 3])
        workbook.save(root / "clinical_scores.xlsx")

    def test_two_patient_batch_matches_scores_and_reuses_extractor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._batch_root(root)
            batch = read_benchmark_batch(root, "device")
            inputs = prepare_batch_inputs(
                batch,
                biomarker_extractor=_fake_biomarkers,
            )
        self.assertEqual(["P001", "P002"], [item.patient.patient_id for item in batch.patients])
        self.assertEqual([6.0, 3.0], [item.clinical_scores.fma_wrist for item in inputs])
        self.assertEqual([15.0, 8.0], [item.clinical_scores.fma_hand for item in inputs])
        self.assertEqual(26, len(inputs[0].biomarkers))
        self.assertEqual("clinician_provided", inputs[0].clinical_score_source.value)

    def test_missing_or_extra_patient_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._patient(root, "P001")
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["patient_id", "fma_wrist", "fma_hand", "hand_mas", "brunnstrom_hand"])
            sheet.append(["P999", 1, 1, "0", 1])
            workbook.save(root / "clinical_scores.xlsx")
            with self.assertRaisesRegex(ValueError, "没有clinical_scores"):
                read_benchmark_batch(root, "device")


class BenchmarkGenerationSwitchTests(unittest.TestCase):
    def _input(self):
        from llm_benchmark import BenchmarkClinicalScores, BenchmarkPatientInfo, build_evaluation_input

        return build_evaluation_input(
            patient=BenchmarkPatientInfo(
                patient_id="P001", name="测试", sex="男", age=50,
                diagnosis="卒中", disease_days=100, paralysis_side="左",
            ),
            clinical_scores=BenchmarkClinicalScores(
                fma_wrist=6, fma_hand=15, hand_mas="2", brunnstrom_hand=4,
            ),
            biomarkers=_fake_biomarkers(),
        )

    def test_all_four_switches_control_provider_calls_and_prompt_context(self):
        calls = {"rag": 0, "kg": 0}

        def rag_provider(value):
            calls["rag"] += 1
            return BenchmarkKnowledgeContext(rag_version=value.config.rag_version, retrieved_text=["RAG"])

        def kg_provider(value):
            calls["kg"] += 1
            return BenchmarkGraphContext(kg_version=value.config.kg_version, findings=[{"rule": "x"}])

        output = json.dumps({
            "biomarker_interpretation": "解释关键指标及复测边界",
            "integrated_assessment": "综合临床评分和指标",
            "rehabilitation_plan": [{
                "action": "治疗师指导下练习抓握后放开",
                "goal": "观察抓放衔接",
                "reason": "与当前手功能评定一致",
                "precaution": "疼痛或代偿时停止",
            }, {
                "action": "练习手指主动打开和闭合",
                "goal": "观察分离运动",
                "reason": "与当前手功能评定一致",
                "precaution": "阻力增加时停止",
            }, {
                "action": "练习拿取和放置轻便物品",
                "goal": "观察任务中的抓放",
                "reason": "与当前手功能评定一致",
                "precaution": "治疗师确认安全后进行",
            }],
        }, ensure_ascii=False)

        for config in (
            BenchmarkRunConfig(),
            BenchmarkRunConfig(rag_enabled=True, rag_version="rag_v1"),
            BenchmarkRunConfig(knowledge_graph_enabled=True, kg_version="kg_v1"),
            BenchmarkRunConfig(rag_enabled=True, rag_version="rag_v1", knowledge_graph_enabled=True, kg_version="kg_v1"),
        ):
            result = generate_benchmark_report(
                evaluation_input=self._input().model_copy(update={"config": config}),
                model_id="mock-model",
                model_generate=lambda *_args, **_kwargs: output,
                retrieval_provider=rag_provider,
                graph_provider=kg_provider,
                case_id="CASE001",
                report_index=1,
            )
            prompt = result.messages[1]["content"]
            self.assertEqual(config.rag_enabled, "rag_context" in prompt)
            self.assertEqual(config.knowledge_graph_enabled, "knowledge_graph_context" in prompt)
            if config.assessment_input_mode.value == "manual_clinical_scores":
                self.assertIn('"clinical_score_source": "clinician_provided"', prompt)
                self.assertNotIn("模型预测", prompt)
                self.assertNotIn("模型预测", result.report_markdown)
        self.assertEqual(2, calls["rag"])
        self.assertEqual(2, calls["kg"])

    def test_single_and_batch_generation_share_the_same_core(self):
        raw = json.dumps({
            "biomarker_interpretation": "解释关键指标及复测边界",
            "integrated_assessment": "综合临床评分和指标",
            "rehabilitation_plan": [
                {
                    "action": "练习抓握后放开",
                    "goal": "观察抓放衔接",
                    "reason": "与当前评定一致",
                    "precaution": "疼痛时停止",
                },
                {
                    "action": "练习手指打开",
                    "goal": "观察分离运动",
                    "reason": "与当前评定一致",
                    "precaution": "阻力增加时停止",
                },
                {
                    "action": "练习拿取并放置物品",
                    "goal": "观察任务动作",
                    "reason": "与当前评定一致",
                    "precaution": "治疗师确认安全",
                },
            ],
        }, ensure_ascii=False)
        first = self._input()
        second = first.model_copy(update={
            "patient": first.patient.model_copy(update={"patient_id": "P002"})
        })
        single = generate_benchmark_report(
            evaluation_input=first,
            model_id="mock-model",
            model_generate=lambda *_args, **_kwargs: raw,
            case_id="CASE001",
            report_index=1,
        )
        batch = generate_benchmark_batch_reports(
            [first, second],
            model_id="mock-model",
            model_generate=lambda *_args, **_kwargs: raw,
            report_index=1,
        )
        self.assertEqual("CASE001-R01", single.anonymous_report_id)
        self.assertEqual(["CASE001-R01", "CASE002-R01"], [item.anonymous_report_id for item in batch])


if __name__ == "__main__":
    unittest.main()
