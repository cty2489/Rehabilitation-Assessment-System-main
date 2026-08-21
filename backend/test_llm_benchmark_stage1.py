import csv
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from llm_benchmark.evaluation.v2.candidate_reader import read_candidate_report
from llm_benchmark.evaluation.v2.reference_schema import validate_reference_payload
from llm_benchmark.stage1.adapters import (
    ExistingModelAdapter,
    ExistingModelCatalog,
    Stage1ModelRouter,
    _extract_stage1_json,
    build_existing_stage1_config,
)
from llm_benchmark.stage1.clinical_input import Stage1ClinicalInput, read_doctor_sheet
from llm_benchmark.stage1.clinical_facts import build_stage1_fact_layer, render_stage1_fact_card
from llm_benchmark.stage1.doctor_reference import (
    parse_doctor_review,
    validate_doctor_sheet_progress,
    write_gold_references,
    write_progress_csv,
)
from llm_benchmark.stage1.output import Stage1CandidateOutput, parse_stage1_model_output
from llm_benchmark.stage1.prompt import build_stage1_messages, load_stage1_prompt
from llm_benchmark.stage1.semantic import validate_stage1_semantics
from llm_benchmark.stage1.runner import (
    Stage1GenerationError,
    Stage1RunConfig,
    build_stage1_manifest,
    generate_stage1_batch_reports,
    generate_stage1_report,
    run_stage1_experiment,
    write_stage1_result,
)


HEADERS = [
    "患者编号", "性别", "年龄", "诊断", "病程", "患侧", "FMA腕", "FMA手", "手MAS", "Brunnstrom",
    "康复师1：评估与建议（按模板填）", "康复师2：复核/修改（按模板填）",
]
REVIEW = (
    "患者男性，56岁，脑梗死，病程37天，左侧受累。"
    "当前腕手功能主要表现为腕手主动控制不足；主要功能受限为抓握后主动放开困难。"
    "建议1：进行腕背伸主动训练，目标是改善腕背伸控制，原因是当前腕主动控制不足，注意避免明显诱发屈肌协同。"
    "建议2：进行手指主动打开训练，目标是改善手指分离运动，原因是当前手指控制受限，注意出现明显代偿时暂停。"
    "建议3：进行轻便物品抓放训练，目标是练习抓放衔接，原因是需要改善任务中的手部控制，注意治疗师确认安全后进行。"
)


def _workbook(path: Path, *, review=REVIEW, draft="康复师1初稿") -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "医生填写"
    sheet.append(["只需填写黄色两列中的【填写】内容；其余患者资料已预填。"])
    sheet.append([])
    sheet.append(HEADERS)
    sheet.append(["S1", "男", 56, "脑梗死", "37天", "左", 0, 6, "0", 4, draft, review])
    workbook.save(path)


def _case():
    cases = read_doctor_sheet(_make_temp_workbook())
    return cases[0]


def _make_temp_workbook() -> Path:
    path = Path(tempfile.mkdtemp()) / "康复医生_极简完形填空表_32例.xlsx"
    _workbook(path)
    return path


PLAN = [
    {"action": "腕背伸主动", "goal": "改善腕背伸控制", "reason": "当前腕主动控制不足", "precaution": "疼痛时停止"},
    {"action": "手指主动打开", "goal": "改善分离运动", "reason": "当前手指控制受限", "precaution": "出现代偿时暂停"},
    {"action": "轻便物品抓放", "goal": "练习抓放衔接", "reason": "改善任务中的手部控制", "precaution": "治疗师确认安全"},
]
ADVICE_RAW = json.dumps({"rehabilitation_plan": PLAN}, ensure_ascii=False)


class Stage1BenchmarkTests(unittest.TestCase):
    def test_excel_a_to_j_read_and_kl_are_audit_only(self):
        path = _make_temp_workbook()
        cases = read_doctor_sheet(path)
        self.assertEqual(1, len(cases))
        case = cases[0]
        self.assertEqual("S1", case.patient_id)
        self.assertEqual("37天", case.clinical_input.disease_course)
        self.assertEqual(6, case.clinical_input.fma_hand)
        self.assertEqual("康复师1初稿", case.doctor1_text)
        self.assertEqual(REVIEW, case.doctor2_text)

    def test_prompt_contains_exactly_nine_fields_and_no_leakage(self):
        clinical = Stage1ClinicalInput(
            sex="男", age=56, diagnosis="脑梗死", disease_course="37天", paralysis_side="左",
            fma_wrist=0, fma_hand=6, hand_mas="0", brunnstrom_hand=4,
        )
        messages = build_stage1_messages(clinical)
        payload_text = messages[1]["content"].split("\n\n【程序固定临床事实卡】", 1)[0]
        payload = json.loads(payload_text)
        self.assertEqual({
            "sex", "age", "diagnosis", "disease_course", "paralysis_side",
            "fma_wrist", "fma_hand", "hand_mas", "brunnstrom_hand",
        }, set(payload))
        for forbidden in ("patient_id", "BI", "FMA上肢", "EEG", "EMG", "IMU", "康复师1", "康复师2", "医生2"):
            self.assertNotIn(forbidden, messages[1]["content"])
        self.assertIn("程序固定临床事实卡", messages[1]["content"])

    def test_prompt_has_stage1_fixed_contract(self):
        prompt = load_stage1_prompt()
        self.assertIn("stage = stage1_clinical_baseline", prompt)
        self.assertIn("恰好3条具体", prompt)
        self.assertIn("FMA腕和FMA手是功能任务评分", prompt)
        self.assertIn("Brunnstrom III期只能表述为", prompt)
        self.assertIn("Brunnstrom VI期只能表述为", prompt)
        self.assertIn("RAG、知识图谱、DL预测和biomarker均关闭", prompt)
        self.assertNotIn("clinical_subtype", prompt)

    def test_stage1_json_extractor_accepts_literal_control_chars_only_for_parsing(self):
        raw = '{"integrated_assessment":"第一句\n第二句","extra":"保留后续原文"}无关续写'
        parsed = _extract_stage1_json(raw)
        self.assertEqual("第一句\n第二句", parsed["integrated_assessment"])
        self.assertEqual("保留后续原文", parsed["extra"])

    def test_output_requires_exactly_three_plans(self):
        valid = {"integrated_assessment": "腕手主动控制受限。", "rehabilitation_plan": PLAN}
        parsed = parse_stage1_model_output(json.dumps(valid, ensure_ascii=False))
        self.assertIsInstance(parsed, Stage1CandidateOutput)
        self.assertFalse(parsed.plan_count_violation)
        for count in (2, 4):
            invalid = dict(valid, rehabilitation_plan=PLAN[:count] if count == 2 else PLAN + [PLAN[0]])
            abnormal = parse_stage1_model_output(json.dumps(invalid, ensure_ascii=False))
            self.assertTrue(abnormal.plan_count_violation)

    def test_l_placeholder_is_not_ready(self):
        result = parse_doctor_review("当前腕手功能主要表现为【填写】；主要功能受限为【填写】。", patient_id="S1")
        self.assertFalse(result.reference_ready)
        self.assertIn("WAITING_FOR_REVIEW", result.errors)

    def test_reference_parser_extracts_only_filled_content(self):
        result = parse_doctor_review(REVIEW, patient_id="S1")
        self.assertTrue(result.reference_ready)
        assert result.sections is not None
        self.assertEqual("腕手主动控制不足。抓握后主动放开困难", result.sections["integrated_assessment"])
        self.assertEqual(3, len(result.sections["rehabilitation_plan"]))
        self.assertEqual("腕背伸主动", result.sections["rehabilitation_plan"][0]["action"])
        self.assertNotIn("当前腕手功能主要表现为", result.sections["integrated_assessment"])
        self.assertNotIn("建议1", json.dumps(result.sections, ensure_ascii=False))

    def test_doctor2_incomplete_cannot_create_gold(self):
        path = Path(tempfile.mkdtemp()) / "input.xlsx"
        _workbook(path, review="当前腕手功能主要表现为【填写】；主要功能受限为【填写】。")
        cases = read_doctor_sheet(path)
        progress = validate_doctor_sheet_progress(cases)
        self.assertFalse(progress[0]["reference_ready"])
        output = Path(tempfile.mkdtemp()) / "gold"
        self.assertEqual((), write_gold_references(path, output))

    def test_approved_l_creates_v2_gold(self):
        path = _make_temp_workbook()
        output = Path(tempfile.mkdtemp()) / "benchmark_references" / "gold_v2"
        written = write_gold_references(path, output, generated_at="2026-08-16T00:00:00+08:00")
        self.assertEqual(1, len(written))
        payload = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual("rehab.llm-benchmark-reference.v2", payload["schema_version"])
        self.assertEqual("approved", payload["review_status"])
        self.assertEqual(3, len(payload["sections"]["rehabilitation_plan"]))
        self.assertNotIn("患者男性", payload["sections"]["integrated_assessment"])
        self.assertNotIn("FMA腕", json.dumps(payload["sections"], ensure_ascii=False))
        self.assertEqual("doctor_reference_parser_v1", payload["audit"]["importer_version"])

    def test_progress_csv_has_only_required_columns(self):
        cases = read_doctor_sheet(_make_temp_workbook())
        path = Path(tempfile.mkdtemp()) / "doctor_progress.csv"
        write_progress_csv(cases, path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual({"patient_id", "doctor1_complete", "doctor2_complete", "reference_ready", "error"}, set(row))
        self.assertEqual("true", row["reference_ready"])

    def test_stage1_single_and_batch_runner_use_same_input(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        raw = ADVICE_RAW

        def fake_model(messages, **kwargs):
            payload_text = messages[1]["content"].split("\n\n【程序固定临床事实卡】", 1)[0]
            self.assertEqual({"age", "brunnstrom_hand", "diagnosis", "disease_course", "fma_hand", "fma_wrist", "hand_mas", "paralysis_side", "sex"}, set(json.loads(payload_text)))
            self.assertFalse(any("rag" in key.lower() or "biomarker" in key.lower() for key in kwargs))
            return raw

        config = Stage1RunConfig(model_ids=["model_a", "model_b"], temperature=0, top_p=1, max_tokens=128)
        single = generate_stage1_report(case=case, model_id="model_a", model_generate=fake_model, config=config, case_id="CASE001")
        self.assertEqual(3, len(single.parsed_model_output.rehabilitation_plan))
        batch = generate_stage1_batch_reports([case], model_ids=config.model_ids, model_generate=fake_model, config=config)
        self.assertEqual(2, len(batch))
        self.assertTrue(all(item.candidate_document["prompt_version"] == "rehab_llm_benchmark_stage1_advice_v2" for item in batch))
        self.assertTrue(all("clinical_fact_layer" in item.candidate_document for item in batch))

    def test_candidate_is_directly_compatible_with_v2_evaluator(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        raw = ADVICE_RAW
        result = generate_stage1_report(case=case, model_id="model_a", model_generate=lambda *_args, **_kwargs: raw, case_id="CASE001")
        path = write_stage1_result(Path(tempfile.mkdtemp()), result)
        candidate = read_candidate_report(path, patient_id="S1", model_id="model_a")
        self.assertTrue(candidate.schema_valid)
        self.assertFalse(candidate.invalid_candidate)
        self.assertFalse(candidate.candidate_missing)

    def test_v2_records_two_or_four_plans_without_crashing(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        for count in (2, 4):
            output = dict(
                integrated_assessment="腕手功能受限。",
                rehabilitation_plan=PLAN[:count] if count == 2 else PLAN + [PLAN[0]],
            )
            document = {
                "schema_version": "rehab.llm-benchmark-report.v2",
                "prompt_version": "rehab_llm_benchmark_stage1_advice_v2",
                "parsed_model_output": output,
                "patient_id": "S1",
            }
            path = Path(tempfile.mkdtemp()) / f"candidate_{count}.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            candidate = read_candidate_report(path, patient_id="S1", model_id="m")
            self.assertFalse(candidate.invalid_candidate)
            self.assertTrue(candidate.plan_count_violation)
            self.assertTrue(any(error["error_code"] == "PLAN_COUNT_VIOLATION" for error in candidate.errors))

        reference = {
            "schema_version": "rehab.llm-benchmark-reference.v2",
            "reference_id": "REF-S1-V2",
            "patient_id": "S1",
            "reference_version": "gold_v2",
            "review_status": "approved",
            "sections": {"integrated_assessment": "腕手功能受限。", "rehabilitation_plan": PLAN[:2]},
        }
        self.assertFalse(validate_reference_payload(reference).valid)

    def test_manifest_has_all_stage1_switches_and_model_parameters(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        config = Stage1RunConfig(model_ids=["a", "b"], temperature=0.1, top_p=0.9, max_tokens=512)
        manifest = build_stage1_manifest(experiment_id="exp1", cases=[case], config=config, timestamp="2026-08-16T00:00:00+08:00")
        self.assertEqual("stage1_clinical_baseline", manifest["stage"])
        self.assertEqual(["S1"], manifest["patient_ids"])
        self.assertEqual(["a", "b"], manifest["model_ids"])
        for switch in ("rag_enabled", "knowledge_graph_enabled", "dl_prediction_enabled", "biomarker_injected"):
            self.assertFalse(manifest[switch])
        self.assertEqual((0.1, 0.9, 512), (manifest["temperature"], manifest["top_p"], manifest["max_tokens"]))

    def test_experiment_runner_persists_manifest_candidates_and_raw_logs(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        raw = ADVICE_RAW
        output = Path(tempfile.mkdtemp()) / "stage1_run"
        run = run_stage1_experiment(
            cases=[case],
            config=Stage1RunConfig(model_ids=["model_a", "model_b"], max_tokens=128),
            model_generate=lambda *_args, **_kwargs: raw,
            output_root=output,
            experiment_id="exp-stage1-test",
        )
        self.assertEqual(2, len(run.results))
        self.assertTrue((output / "stage1_manifest.json").is_file())
        self.assertEqual(2, len(list((output / "patients" / "S1" / "reports").glob("*.json"))))
        log_lines = (output / "logs" / "experiment_stage1.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(2, len(log_lines))
        self.assertIn("raw_model_output", json.loads(log_lines[0]))

    def test_experiment_runner_keeps_abnormal_plan_count_as_candidate(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        raw = json.dumps({"rehabilitation_plan": PLAN[:2]}, ensure_ascii=False)
        output = Path(tempfile.mkdtemp()) / "stage1_abnormal_plan"
        run = run_stage1_experiment(
            cases=[case],
            config=Stage1RunConfig(model_ids=["model_a"], max_tokens=128),
            model_generate=lambda *_args, **_kwargs: raw,
            output_root=output,
            experiment_id="exp-stage1-abnormal-plan",
        )
        self.assertEqual(1, len(run.results))
        self.assertTrue(run.results[0].parsed_model_output.plan_count_violation)
        log = json.loads((output / "logs" / "experiment_stage1.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(log["plan_count_violation"])
        self.assertTrue((output / "patients" / "S1" / "reports" / "model_a.json").is_file())

    def test_schema_failure_gets_one_serialization_retry_and_keeps_raw_attempts(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        invalid = json.dumps({
            "action": "错误层级的动作",
            "goal": "错误层级的目标",
            "reason": "错误层级的原因",
            "precaution": "错误层级的注意事项",
        }, ensure_ascii=False)
        valid = ADVICE_RAW
        calls = []

        def fake_model(messages, **kwargs):
            calls.append(messages)
            return invalid if len(calls) == 1 else valid

        result = generate_stage1_report(
            case=case,
            model_id="model_a",
            model_generate=fake_model,
            config=Stage1RunConfig(model_ids=["model_a"], max_tokens=128),
            case_id="CASE001",
        )
        self.assertEqual(2, len(calls))
        self.assertEqual(invalid, result.raw_model_outputs[0])
        self.assertEqual(valid, result.raw_model_outputs[1])
        self.assertEqual(1, result.candidate_document["format_retry_count"])
        self.assertEqual(2, len(result.candidate_document["raw_model_outputs"]))
        self.assertNotEqual(calls[0][0]["content"], calls[1][0]["content"])
        self.assertEqual(calls[0][1]["content"], calls[1][1]["content"])
        self.assertIn("格式重试", calls[1][0]["content"])

    def test_failed_serialization_retry_keeps_both_raw_outputs_in_error_log(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        invalid_one = '{"action":"第一轮错误输出"}'
        invalid_two = '{"action":"第二轮错误输出"}'
        calls = []

        def fake_model(_messages, **_kwargs):
            calls.append(True)
            return invalid_one if len(calls) == 1 else invalid_two

        output = Path(tempfile.mkdtemp()) / "stage1_retry_failure"
        run = run_stage1_experiment(
            cases=[case],
            config=Stage1RunConfig(model_ids=["model_a"], max_tokens=128),
            model_generate=fake_model,
            output_root=output,
            experiment_id="stage1-retry-failure",
        )
        self.assertEqual((), run.results)
        log = json.loads((output / "logs" / "experiment_stage1.jsonl").read_text(encoding="utf-8"))
        self.assertFalse(log["parse_success"])
        self.assertEqual([invalid_one, invalid_two], log["raw_model_outputs"])
        self.assertEqual(1, log["format_retry_count"])

    def test_existing_catalog_discovers_the_configured_six_models(self):
        catalog = ExistingModelCatalog.from_existing_settings()
        self.assertEqual({
            "qwen3_8b_hf",
            "deepseek_r1_distill_qwen7b",
            "baichuan2_7b_chat",
            "glm4_9b",
            "mistral7b_v03",
            "internlm3_8b",
        }, set(catalog.model_ids))
        self.assertEqual(6, len(catalog.specs))
        self.assertTrue(all(spec.provider == "local" for spec in catalog.specs))
        self.assertTrue(all(all(getattr(spec.capabilities, name) for name in ("temperature", "top_p", "max_tokens")) for spec in catalog.specs))

    def test_six_model_mock_router_isolated_and_failure_does_not_stop_others(self):
        case = read_doctor_sheet(_make_temp_workbook())[0]
        config = build_existing_stage1_config()
        calls = []
        shared_messages = []
        failed_id = "glm4_9b"

        def mock_generate(spec, messages, **kwargs):
            calls.append((spec.model_id, messages, kwargs))
            shared_messages.append(messages)
            if spec.model_id == failed_id:
                raise RuntimeError("mock failure")
            payload = {
                "rehabilitation_plan": [
                    {"action": f"{spec.model_id}-动作1", "goal": "目标1", "reason": "原因1", "precaution": "注意1"},
                    {"action": f"{spec.model_id}-动作2", "goal": "目标2", "reason": "原因2", "precaution": "注意2"},
                    {"action": f"{spec.model_id}-动作3", "goal": "目标3", "reason": "原因3", "precaution": "注意3"},
                ],
            }
            return json.dumps(payload, ensure_ascii=False)

        adapter = ExistingModelAdapter(mock_generate=mock_generate)
        router = Stage1ModelRouter(adapter)
        output = Path(tempfile.mkdtemp()) / "six_model_mock"
        run = run_stage1_experiment(
            cases=[case],
            config=config,
            model_generate=router,
            output_root=output,
            experiment_id="stage1-six-model-mock",
        )
        self.assertEqual(6, len(calls))
        self.assertEqual(set(config.model_ids), {item[0] for item in calls})
        self.assertEqual(5, len(run.results))
        self.assertEqual(5, len(list((output / "patients" / "S1" / "reports").glob("*.json"))))
        logs = [json.loads(line) for line in (output / "logs" / "experiment_stage1.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(6, len(logs))
        failed = next(item for item in logs if item["llm_model_id"] == failed_id)
        self.assertFalse(failed["parse_success"])
        self.assertIn("mock failure", failed["error"])
        self.assertEqual("local", next(item for item in logs if item["llm_model_id"] == "qwen3_8b_hf")["provider"])
        self.assertTrue(all(messages == shared_messages[0] for messages in shared_messages))
        self.assertEqual("rehab_llm_benchmark_stage1_advice_v2", run.manifest["prompt_version"])
        self.assertEqual(set(config.model_ids), set(run.manifest["model_ids"]))


if __name__ == "__main__":
    unittest.main()
