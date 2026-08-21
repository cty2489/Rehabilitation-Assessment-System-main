import json
import tempfile
import unittest
from pathlib import Path

from llm_benchmark.evaluation.v2.aggregate import evaluate_batch, evaluate_single_case, write_evaluation_outputs
from llm_benchmark.evaluation.v2.candidate_reader import read_candidate_report
from llm_benchmark.evaluation.v2.reference_schema import validate_reference_payload
from llm_benchmark.v2 import build_evaluation_input, build_messages, generate_benchmark_report, parse_model_output
from llm_benchmark.v2.schemas import BenchmarkLlmOutput, BenchmarkPatientInfo, BenchmarkClinicalScores


PLAN = [
    {"action": "练习抓握后主动放开", "goal": "观察抓放衔接", "reason": "与当前手功能阶段一致", "precaution": "疼痛或明显代偿时停止"},
    {"action": "练习手指主动打开和闭合", "goal": "观察分离运动", "reason": "用于训练手部控制", "precaution": "阻力增加时停止"},
    {"action": "练习拿取并放置轻便物品", "goal": "改善任务抓放", "reason": "转化为简单任务", "precaution": "治疗师确认安全后进行"},
]


def _output():
    return {
        "integrated_assessment": "结合FMA腕、FMA手、MAS和Brunnstrom，当前右侧手部主动控制受限，训练重点是抓握后主动放开。",
        "rehabilitation_plan": PLAN,
    }


def _reference(patient_id: str):
    return {
        "schema_version": "rehab.llm-benchmark-reference.v2",
        "reference_id": f"REF-{patient_id}-V2",
        "patient_id": patient_id,
        "reference_version": "gold_v2",
        "review_status": "approved",
        "sections": _output(),
        "audit": {"created_at": "2026-08-16T00:00:00+08:00", "reviewer_ids": ["TEST-V2"], "revision": 1, "previous_reference_id": None, "change_summary": "TEST ONLY"},
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _v2_report(patient_id: str, payload: dict, *, prompt_version="rehab_llm_benchmark_v2", schema_version="rehab.llm-benchmark-report.v2"):
    return {
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "output_schema_version": "rehab.llm-benchmark-output.v2",
        "patient_id": patient_id,
        "parsed_model_output": payload,
        "report_markdown": "固定FMA=15、MAS=1+、biomarker RMS=999、综合亚型=测试文本；这些内容不能进入评分。",
    }


def _make_synthetic_v2(root: Path):
    gold_root = root / "gold_v2"
    batch_root = root / "benchmark_run"
    for patient_id in ("P001", "P002", "P003"):
        _write_json(gold_root / f"{patient_id}.json", _reference(patient_id))
        (batch_root / "patients" / patient_id / "reports").mkdir(parents=True, exist_ok=True)

    for patient_id in ("P001", "P002", "P003"):
        _write_json(batch_root / "patients" / patient_id / "reports" / "ModelPerfect.json", _v2_report(patient_id, _output()))

    partial_p1 = json.loads(json.dumps(_output()))
    partial_p1["integrated_assessment"] = "结合临床评分，当前手部主动控制受限。"
    _write_json(batch_root / "patients" / "P001" / "reports" / "ModelPartial.json", _v2_report("P001", partial_p1))
    # P002 deliberately has no ModelPartial file: it is a missing Candidate.
    partial_p3 = json.loads(json.dumps(_output()))
    partial_p3["biomarker_interpretation"] = "单次绝对值不能形成独立临床结论。"
    partial_p3["rehabilitation_plan"] = PLAN + [PLAN[0], PLAN[1], PLAN[2]]
    _write_json(batch_root / "patients" / "P003" / "reports" / "ModelPartial.json", _v2_report("P003", partial_p3))

    (batch_root / "patients" / "P001" / "reports" / "ModelFailed.json").write_text("not-json", encoding="utf-8")
    _write_json(batch_root / "patients" / "P003" / "reports" / "ModelFailed.json", _v2_report("P003", _output(), prompt_version="rehab_llm_benchmark_v1", schema_version="rehab.llm-benchmark-report.v1"))
    return gold_root, batch_root


def _input():
    patient = BenchmarkPatientInfo(patient_id="P001", name="测试患者", sex="男", age=58, diagnosis="脑卒中后偏瘫", disease_days=120, paralysis_side="右")
    scores = BenchmarkClinicalScores(fma_wrist=6, fma_hand=15, hand_mas="1+", brunnstrom_hand=4)
    biomarkers = {"groups": [{"key": "eeg", "markers": [{"key": "eeg_1", "name": "β功率", "value_num": 1.5, "unit": "u", "available": True, "n_valid": 3}]}]}
    return build_evaluation_input(patient=patient, clinical_scores=scores, biomarkers=biomarkers)


class BenchmarkV2Tests(unittest.TestCase):
    def test_v2_schema_has_exactly_two_generated_fields(self):
        output = BenchmarkLlmOutput(**_output())
        self.assertEqual({"integrated_assessment", "rehabilitation_plan"}, set(output.model_dump()))
        with self.assertRaises(ValueError):
            parse_model_output(json.dumps({**_output(), "biomarker_interpretation": "不允许"}, ensure_ascii=False))
        parsed = parse_model_output(json.dumps(_output(), ensure_ascii=False))
        self.assertNotIn("biomarker_interpretation", parsed.model_dump())
        self.assertNotIn("clinical_subtype", parsed.model_dump())

    def test_v2_prompt_contains_single_visit_limits_and_no_v1_output_contract(self):
        messages = build_messages(_input())
        self.assertIn("rehab_llm_benchmark_v2", messages[0]["content"])
        self.assertIn("不得仅根据某个 biomarker 的单次绝对数值", messages[0]["content"])
        self.assertIn("不得输出 clinical_subtype 或 subtype", messages[0]["content"])
        self.assertIn("integrated_assessment", messages[0]["content"])
        self.assertIn("rehabilitation_plan", messages[0]["content"])
        self.assertNotIn('"biomarker_interpretation": "', messages[0]["content"])

    def test_v2_service_uses_v2_prompt_log_and_two_section_report(self):
        raw = json.dumps(_output(), ensure_ascii=False)
        result = generate_benchmark_report(
            evaluation_input=_input(),
            model_id="mock-v2",
            model_generate=lambda *_args, **_kwargs: raw,
            case_id="CASE001",
            report_index=1,
        )
        self.assertEqual("rehab_llm_benchmark_v2", result.experiment_log.prompt_version)
        self.assertEqual("rehab.llm-benchmark-log.v2", result.experiment_log.schema_version)
        self.assertIn("## 一、综合康复评估", result.report_markdown)
        self.assertIn("## 二、康复建议", result.report_markdown)
        self.assertNotIn("生物标志物分析", result.report_markdown)
        self.assertNotIn("综合亚型", result.report_markdown)
        self.assertNotIn("clinical_subtype", result.report_markdown)

    def test_v2_gold_rejects_v1_and_forbidden_sections(self):
        valid = validate_reference_payload(_reference("P001"))
        self.assertTrue(valid.valid)
        v1 = json.loads(json.dumps(_reference("P001")))
        v1["schema_version"] = "rehab.llm-benchmark-reference.v1"
        self.assertFalse(validate_reference_payload(v1).valid)
        forbidden = json.loads(json.dumps(_reference("P001")))
        forbidden["sections"]["biomarker_interpretation"] = "不要进入Gold"
        result = validate_reference_payload(forbidden)
        self.assertFalse(result.valid)
        self.assertIn("forbidden_section:biomarker_interpretation", result.errors)

    def test_v1_candidate_is_rejected_by_v2_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            _write_json(path, _v2_report("P001", _output(), prompt_version="rehab_llm_benchmark_v1", schema_version="rehab.llm-benchmark-report.v1"))
            record = read_candidate_report(path, patient_id="P001", model_id="Old")
            self.assertTrue(record.invalid_candidate)
            self.assertEqual("VERSION_MISMATCH", record.errors[0]["error_code"])

    def test_v2_overall_ignores_biomarker_and_subtype_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            payload = _output()
            payload["biomarker_interpretation"] = "单次 RMS=999，属于严重异常。"
            payload["clinical_subtype"] = "不存在的亚型"
            _write_json(path, _v2_report("P001", payload))
            candidate = read_candidate_report(path, patient_id="P001", model_id="M")
            case = evaluate_single_case(validate_reference_payload(_reference("P001")).reference, candidate)
            overall = next(row for row in case.rows if row["section"] == "overall_generated_content")
            self.assertEqual(1.0, overall["rouge1_f1"])
            self.assertIn("biomarker_interpretation", candidate.unexpected_sections)
            self.assertIn("clinical_subtype", candidate.unexpected_sections)

    def test_v2_synthetic_fixed_denominator_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_root, batch_root = _make_synthetic_v2(root)
            output = root / "output"
            run = evaluate_batch(batch_root=batch_root, reference_root=gold_root, output_root=output, model_ids=["ModelPerfect", "ModelPartial", "ModelFailed"], evaluation_id="synthetic-v2")
            write_evaluation_outputs(run)
            summary = {row["model_id"]: row for row in run.summaries}
            self.assertEqual((3, 3, 3, 0, 0), (summary["ModelPerfect"]["n_expected"], summary["ModelPerfect"]["n_scored"], summary["ModelPerfect"]["n_schema_valid"], summary["ModelPerfect"]["n_invalid"], summary["ModelPerfect"]["n_missing"]))
            self.assertEqual((3, 3, 1, 0, 1), (summary["ModelPartial"]["n_expected"], summary["ModelPartial"]["n_scored"], summary["ModelPartial"]["n_schema_valid"], summary["ModelPartial"]["n_invalid"], summary["ModelPartial"]["n_missing"]))
            self.assertEqual((3, 3, 0, 2, 1), (summary["ModelFailed"]["n_expected"], summary["ModelFailed"]["n_scored"], summary["ModelFailed"]["n_schema_valid"], summary["ModelFailed"]["n_invalid"], summary["ModelFailed"]["n_missing"]))
            self.assertEqual(0.0, summary["ModelPerfect"]["generation_failure_rate"])
            self.assertEqual(1.0, summary["ModelFailed"]["generation_failure_rate"])
            self.assertEqual(1.0, summary["ModelPerfect"]["overall_rouge1_mean"])
            self.assertAlmostEqual(100.0, summary["ModelPerfect"]["overall_corpus_bleu4"], places=5)
            self.assertGreater(summary["ModelPerfect"]["overall_rouge1_mean"], summary["ModelPartial"]["overall_rouge1_mean"])
            self.assertGreater(summary["ModelPartial"]["overall_rouge1_mean"], summary["ModelFailed"]["overall_rouge1_mean"])
            self.assertEqual(3 * 3, len([row for row in run.rows if row["section"] == "overall_generated_content"]))
            self.assertEqual(3 * 3 * 3, len(run.rows))
            manifest = run.manifest
            self.assertEqual("rehab_llm_metrics_v2", manifest["metrics_version"])
            self.assertEqual(["integrated_assessment", "rehabilitation_plan", "overall_generated_content"], manifest["included_sections"])
            self.assertIn("biomarker_interpretation", manifest["excluded_sections"])
            self.assertFalse(manifest["biomarker_policy"]["absolute_normality_inference_allowed"])

    def test_v2_plan_count_and_medical_direction_rules_remain(self):
        from llm_benchmark.evaluation.canonicalize import canonicalize_rehabilitation_plan
        from llm_benchmark.evaluation.metrics import score_rouge
        for count in (0, 1, 2, 3, 5, 6):
            plan = PLAN[:count] if count <= 3 else PLAN + [PLAN[0]] * (count - 3)
            result = canonicalize_rehabilitation_plan(plan)
            self.assertEqual(count == 0, result.missing)
            self.assertEqual(count not in (0, 3, 5), result.plan_count_violation)
        tokenizer_text = "右侧功能受限，肌张力升高，未见明显异常。"
        tokenizer = __import__("llm_benchmark.evaluation.tokenizer", fromlist=["ChineseMedicalTokenizer"]).ChineseMedicalTokenizer()
        for changed in ("左侧功能受限，肌张力升高，未见明显异常。", "右侧功能受限，肌张力降低，未见明显异常。", "右侧功能受限，肌张力升高，见明显异常。"):
            self.assertLess(score_rouge(tokenizer_text, changed, tokenizer=tokenizer)["rouge1"].f1, 1.0)

    def test_v1_history_file_still_declares_v1(self):
        path = Path("benchmark_evaluations/synthetic_v1_cli_output/metric_manifest.json")
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("rehab_llm_metrics_v1", manifest["metrics_version"])
            self.assertIn("biomarker_interpretation", manifest["included_sections"])


if __name__ == "__main__":
    unittest.main()
