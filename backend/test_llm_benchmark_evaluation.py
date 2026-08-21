import csv
import json
import tempfile
import unittest
from pathlib import Path

from llm_benchmark.evaluation.aggregate import evaluate_batch, evaluate_single_case, write_evaluation_outputs
from llm_benchmark.evaluation.candidate_reader import read_candidate_report
from llm_benchmark.evaluation.canonicalize import canonicalize_rehabilitation_plan
from llm_benchmark.evaluation.metrics import score_rouge
from llm_benchmark.evaluation.reference_schema import validate_reference_payload
from llm_benchmark.evaluation.tokenizer import ChineseMedicalTokenizer, tokenizer_rules_hash


PLAN = [
    {
        "action": "练习抓握后主动放开",
        "goal": "观察抓放衔接",
        "reason": "与当前手功能阶段观察一致",
        "precaution": "出现疼痛或明显代偿时停止",
    },
    {
        "action": "练习手指主动打开和闭合",
        "goal": "观察手指分离运动",
        "reason": "用于训练可观察的手部控制",
        "precaution": "阻力明显增加时停止",
    },
    {
        "action": "练习拿取并放置轻便物品",
        "goal": "改善任务中的抓放",
        "reason": "把动作控制转化为简单任务",
        "precaution": "由治疗师确认安全后进行",
    },
]


def _generated_text():
    return {
        "biomarker_interpretation": "FMA手评分15分，MAS为1+，EEG β活动稳定。",
        "integrated_assessment": "右侧手功能受限，当前重点是抓握后主动放开。",
        "rehabilitation_plan": PLAN,
    }


def _reference(patient_id: str) -> dict:
    return {
        "schema_version": "rehab.llm-benchmark-reference.v1",
        "reference_id": f"REF-{patient_id}-V1",
        "patient_id": patient_id,
        "reference_version": "gold_v1",
        "review_status": "approved",
        "sections": _generated_text(),
        "audit": {
            "created_at": "2026-08-16T00:00:00+08:00",
            "reviewer_ids": ["TEST-REVIEWER-1"],
            "revision": 1,
            "previous_reference_id": None,
            "change_summary": "TEST ONLY",
        },
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_synthetic_batch(root: Path) -> Path:
    reference_root = root / "gold_v1"
    batch_root = root / "benchmark_run"
    for patient_id in ("P001", "P002", "P003"):
        _write_json(reference_root / f"{patient_id}.json", _reference(patient_id))
        (batch_root / "patients" / patient_id / "reports").mkdir(parents=True, exist_ok=True)

    perfect = _generated_text()
    for patient_id in ("P001", "P002", "P003"):
        _write_json(
            batch_root / "patients" / patient_id / "reports" / "ModelPerfect.json",
            {
                "schema_version": "rehab.llm-benchmark-report.v1",
                "patient_id": patient_id,
                "parsed_model_output": perfect,
                "report_markdown": "固定患者事实 FMA=15、MAS=1+；这些内容不参与评分。",
            },
        )

    partial = _generated_text()
    partial["biomarker_interpretation"] = "FMA手评分14分，MAS为1+，EEG β活动稳定。"
    partial["rehabilitation_plan"] = PLAN[:2]
    _write_json(
        batch_root / "patients" / "P001" / "reports" / "ModelPartial.json",
        {"parsed_model_output": partial, "report_markdown": "固定事实和标题不参与评分。"},
    )
    _write_json(
        batch_root / "patients" / "P002" / "reports" / "ModelPartial.json",
        {
            "parsed_model_output": {
                "biomarker_interpretation": "未见明显异常。",
                "integrated_assessment": "右侧功能受限。",
            },
            "report_markdown": "这段 Markdown 不能补齐缺失 rehabilitation_plan。",
        },
    )
    six_plan = PLAN + [PLAN[0], PLAN[1], PLAN[2]]
    partial_three = _generated_text()
    partial_three["rehabilitation_plan"] = six_plan
    _write_json(
        batch_root / "patients" / "P003" / "reports" / "ModelPartial.json",
        {"parsed_model_output": partial_three},
    )

    (batch_root / "patients" / "P001" / "reports" / "ModelFailed.json").write_text("not-json", encoding="utf-8")
    # P002 deliberately has no ModelFailed report: it must remain in the
    # fixed denominator as an empty Candidate.
    _write_json(
        batch_root / "patients" / "P003" / "reports" / "ModelFailed.json",
        {
            "parsed_model_output": {
                "biomarker_interpretation": "完全不同的测试文本。",
                "integrated_assessment": "完全不同的测试判断。",
                "rehabilitation_plan": [
                    {"action": "测试动作", "goal": "测试目标", "reason": "测试理由", "precaution": "测试注意事项"}
                ],
            }
        },
    )
    return root


class EvaluationContractTests(unittest.TestCase):
    def test_tokenizer_preserves_medical_atoms_and_excludes_punctuation(self):
        text = "FMA手评分15分，MAS为1+，EEG β活动异常。20% 0–20 2.5"
        tokens = ChineseMedicalTokenizer().tokenize(text)
        self.assertEqual(
            ["FMA", "手", "评", "分", "15", "分", "MAS", "为", "1+", "EEG", "β", "活", "动", "异", "常", "20%", "0–20", "2.5"],
            tokens,
        )
        self.assertEqual(tokenizer_rules_hash(), tokenizer_rules_hash())
        self.assertNotIn("，", tokens)
        self.assertNotIn("。", tokens)

    def test_reference_requires_approved_complete_gold(self):
        valid = validate_reference_payload(_reference("P001"))
        self.assertTrue(valid.valid)
        invalid = _reference("P001")
        invalid["review_status"] = "draft"
        invalid["sections"].pop("integrated_assessment")
        result = validate_reference_payload(invalid)
        self.assertFalse(result.valid)
        self.assertIn("review_status_not_approved", result.errors)
        self.assertIn("missing_or_empty_section:integrated_assessment", result.errors)

    def test_normalization_removes_markdown_but_keeps_medical_semantics(self):
        from llm_benchmark.evaluation.normalization import normalize_medical_text

        self.assertEqual(
            "患者右侧上肢肌张力增高。",
            normalize_medical_text("### 综合评估\n\n- 患者右侧上肢肌张力增高。"),
        )
        self.assertEqual("FMA-UE 20% 不降低。", normalize_medical_text("FMA-UE 20% 不降低。"))

    def test_plan_canonicalization_never_inserts_field_labels(self):
        result = canonicalize_rehabilitation_plan(PLAN)
        self.assertEqual(3, result.count)
        self.assertFalse(result.plan_count_violation)
        self.assertNotIn("action:", result.text)
        self.assertNotIn("goal:", result.text)
        missing = canonicalize_rehabilitation_plan([{"action": "动作", "goal": "目标", "reason": "理由"}])
        self.assertIn("1:precaution", missing.missing_fields)
        self.assertEqual(1, missing.count)
        self.assertTrue(missing.plan_count_violation)

    def test_plan_count_policy_is_scoreable_but_flagged(self):
        for count in (0, 1, 2, 3, 5, 6):
            plan = PLAN[:count] if count <= 3 else PLAN + [PLAN[0]] * (count - 3)
            result = canonicalize_rehabilitation_plan(plan)
            self.assertEqual(count, result.count)
            self.assertEqual(count == 0, result.missing)
            self.assertEqual(count not in (0, 3, 5), result.plan_count_violation)

    def test_exact_text_and_semantic_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _reference("P001")
            _write_json(root / "gold.json", reference)
            exact_path = root / "exact.json"
            _write_json(exact_path, {"parsed_model_output": _generated_text()})
            exact = read_candidate_report(exact_path, patient_id="P001", model_id="Exact")
            exact_case = evaluate_single_case(
                validate_reference_payload(reference).reference,
                exact,
            )
            overall = next(row for row in exact_case.rows if row["section"] == "overall_generated_content")
            self.assertEqual(1.0, overall["rouge1_f1"])
            self.assertEqual(1.0, overall["rouge2_f1"])
            self.assertEqual(1.0, overall["rougeL_f1"])
            self.assertEqual(100.0, round(overall["sentence_bleu4"], 6))

            changed = _generated_text()
            changed["biomarker_interpretation"] = "FMA手评分14分，MAS为1+，EEG β活动稳定。"
            _write_json(root / "changed.json", {"parsed_model_output": changed})
            changed_case = evaluate_single_case(
                validate_reference_payload(reference).reference,
                read_candidate_report(root / "changed.json", patient_id="P001", model_id="Changed"),
            )
            changed_row = next(row for row in changed_case.rows if row["section"] == "biomarker_interpretation")
            self.assertLess(changed_row["rouge1_f1"], 1.0)

    def test_negation_direction_and_side_changes_lower_scores(self):
        tokenizer = ChineseMedicalTokenizer()
        reference = "右侧功能受限，肌张力升高，未见明显异常。"
        for candidate in (
            "右侧功能受限，肌张力升高，见明显异常。",
            "左侧功能受限，肌张力升高，未见明显异常。",
            "右侧功能受限，肌张力降低，未见明显异常。",
        ):
            score = score_rouge(reference, candidate, tokenizer=tokenizer)["rouge1"].f1
            self.assertLess(score, 1.0)

    def test_markdown_and_fixed_facts_never_affect_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _reference("P001")
            first = root / "first.json"
            second = root / "second.json"
            _write_json(first, {"parsed_model_output": _generated_text(), "report_markdown": "FMA=0 MAS=4 RAG UID=SECRET"})
            _write_json(second, {"parsed_model_output": _generated_text(), "report_markdown": "完全不同的标题和来源 URL"})
            first_case = evaluate_single_case(validate_reference_payload(reference).reference, read_candidate_report(first, patient_id="P001", model_id="M"))
            second_case = evaluate_single_case(validate_reference_payload(reference).reference, read_candidate_report(second, patient_id="P001", model_id="M"))
            first_scores = [(row["section"], row["rouge1_f1"], row["sentence_bleu4"]) for row in first_case.rows]
            second_scores = [(row["section"], row["rouge1_f1"], row["sentence_bleu4"]) for row in second_case.rows]
            self.assertEqual(first_scores, second_scores)

    def test_candidate_failures_and_extra_sections_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text("not-json", encoding="utf-8")
            record = read_candidate_report(invalid, patient_id="P001", model_id="M")
            self.assertTrue(record.invalid_candidate)
            self.assertFalse(record.parse_success)
            self.assertEqual(3, len(record.missing_sections))

            extra = root / "extra.json"
            payload = _generated_text()
            payload["diagnosis"] = "不能参与评价"
            _write_json(extra, {"parsed_model_output": payload})
            record = read_candidate_report(extra, patient_id="P001", model_id="M")
            self.assertFalse(record.invalid_candidate)
            self.assertEqual(("diagnosis",), record.unexpected_sections)

            mismatch = root / "mismatch.json"
            _write_json(mismatch, {"patient_id": "P002", "parsed_model_output": _generated_text()})
            mismatch_record = read_candidate_report(mismatch, patient_id="P001", model_id="M")
            self.assertTrue(mismatch_record.invalid_candidate)
            self.assertEqual("PATIENT_ID_MISMATCH", mismatch_record.errors[0]["error_code"])

    def test_synthetic_three_model_batch_uses_fixed_denominator_and_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_synthetic_batch(root)
            output = root / "evaluation"
            run = evaluate_batch(
                batch_root=root / "benchmark_run",
                reference_root=root / "gold_v1",
                output_root=output,
                model_ids=["ModelPerfect", "ModelPartial", "ModelFailed"],
                evaluation_id="synthetic-v1",
            )
            write_evaluation_outputs(run)
            summary = {row["model_id"]: row for row in run.summaries}
            self.assertEqual(3, summary["ModelPerfect"]["n_expected"])
            self.assertEqual(3, summary["ModelPartial"]["n_expected"])
            self.assertEqual(3, summary["ModelFailed"]["n_expected"])
            self.assertEqual(0, summary["ModelPerfect"]["n_invalid"])
            self.assertEqual(2, summary["ModelFailed"]["n_invalid"])
            self.assertGreater(summary["ModelPerfect"]["overall_rouge1_mean"], summary["ModelPartial"]["overall_rouge1_mean"])
            self.assertGreater(summary["ModelPartial"]["overall_rouge1_mean"], summary["ModelFailed"]["overall_rouge1_mean"])
            self.assertTrue((output / "per_case_metrics.csv").exists())
            self.assertTrue((output / "per_case_metrics.jsonl").exists())
            self.assertTrue((output / "per_model_summary.csv").exists())
            self.assertTrue((output / "corpus_bleu.json").exists())
            self.assertTrue((output / "evaluation_errors.jsonl").exists())
            self.assertTrue((output / "metric_manifest.json").exists())
            manifest = json.loads((output / "metric_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(["P001", "P002", "P003"], manifest["expected_patient_ids"])
            self.assertEqual("zh", manifest["bleu"]["tokenizer"])
            self.assertEqual(3 * 3 * 4, sum(1 for _ in (output / "per_case_metrics.jsonl").read_text(encoding="utf-8").splitlines()))

            with (output / "per_model_summary.csv").open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual({"ModelPerfect", "ModelPartial", "ModelFailed"}, {row["model_id"] for row in summary_rows})

    def test_invalid_gold_is_not_converted_to_model_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_synthetic_batch(root)
            invalid = _reference("P004")
            invalid["review_status"] = "draft"
            _write_json(root / "gold_v1" / "P004.json", invalid)
            run = evaluate_batch(
                batch_root=root / "benchmark_run",
                reference_root=root / "gold_v1",
                output_root=root / "evaluation",
                model_ids=["ModelPerfect"],
                evaluation_id="invalid-gold",
            )
            self.assertEqual(3, run.summaries[0]["n_expected"])
            self.assertTrue(any(error["error_code"] == "REFERENCE_INVALID" for error in run.errors))

    def test_batch_repeated_run_has_stable_case_and_tokenizer_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_synthetic_batch(root)
            first = evaluate_batch(
                batch_root=root / "benchmark_run",
                reference_root=root / "gold_v1",
                output_root=root / "first",
                model_ids=["ModelPerfect", "ModelPartial"],
                evaluation_id="one",
            )
            second = evaluate_batch(
                batch_root=root / "benchmark_run",
                reference_root=root / "gold_v1",
                output_root=root / "second",
                model_ids=["ModelPerfect", "ModelPartial"],
                evaluation_id="two",
            )
            self.assertEqual(first.manifest["case_order_hash"], second.manifest["case_order_hash"])
            self.assertEqual(first.manifest["candidate_manifest_hash"], second.manifest["candidate_manifest_hash"])
            self.assertEqual(first.manifest["normalization"], second.manifest["normalization"])
            self.assertEqual(first.manifest["rouge"]["tokenizer_rules_hash"], second.manifest["rouge"]["tokenizer_rules_hash"])
            self.assertEqual(first.rows, second.rows)


if __name__ == "__main__":
    unittest.main()
