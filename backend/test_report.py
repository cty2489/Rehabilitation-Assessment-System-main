"""Tests for the local QLoRA report path (backend/report.py).

These do NOT load the 6B base model — they only cover the cheap, deterministic
glue: PatientInfo → demographics adaptation and the clear error raised when the
LoRA adapter directory is missing.

Run from backend/:
    python -m unittest test_report -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import report
from schemas import PatientInfo


def _patient() -> PatientInfo:
    return PatientInfo(
        patient_id="7",
        name="张三",
        sex="男",
        age=65,
        diagnosis="脑梗死",
        disease_days=90,
        paralysis_side="右",
    )


class AdaptationTests(unittest.TestCase):
    def test_to_demographics_maps_backend_fields(self) -> None:
        d = report.to_demographics(_patient())
        self.assertEqual(d["gender"], "男")
        self.assertEqual(d["age"], 65)
        self.assertEqual(d["disease"], "脑梗死")
        self.assertEqual(d["days_post"], 90)
        self.assertEqual(d["affected_side"], "右")

    def test_parse_clinical_json_strips_think_block(self) -> None:
        text = '<think>{"draft":"不要解析这里"}</think>\n{"overall_interpretation":"ok"}'
        self.assertEqual(
            report._parse_clinical_json(text),
            {"overall_interpretation": "ok"},
        )

    def test_parse_clinical_json_strips_orphan_think_end(self) -> None:
        text = '推理过程里可能有 {"draft":"不要解析"}\n</think>\n```json\n{"overall_interpretation":"ok"}\n```'
        self.assertEqual(
            report._parse_clinical_json(text),
            {"overall_interpretation": "ok"},
        )

    def test_parse_clinical_json_uses_first_complete_schema_object(self) -> None:
        text = '```json\n{"overall_interpretation":"first"}\n```\n```json\n{"overall_interpretation":"second"}\n```'
        self.assertEqual(
            report._parse_clinical_json(text),
            {"overall_interpretation": "first"},
        )

    def test_parse_clinical_json_ignores_nested_partial_objects(self) -> None:
        text = '{"overall_interpretation":"truncated", "marker_text": {"m1": {"interpretation":"i", "treatment_advice":"t"}'
        self.assertIsNone(report._parse_clinical_json(text))

    def test_parse_clinical_json_accepts_deepseek_closed_think_prefill(self) -> None:
        text = '</think>\n{"overall_interpretation":"ok"}'
        self.assertEqual(
            report._parse_clinical_json(text),
            {"overall_interpretation": "ok"},
        )

    def test_coerce_ordered_marker_text_list_to_key_map(self) -> None:
        markers = [{"key": "m1"}, {"key": "m2"}]
        raw = [["解读1", "建议1"], ["解读2", "建议2"]]
        self.assertEqual(
            report._coerce_marker_text_payload(raw, markers),
            {"m1": ["解读1", "建议1"], "m2": ["解读2", "建议2"]},
        )

    def test_rag_citation_must_come_from_retrieved_sources(self) -> None:
        context = {
            "rag_evidence": {
                "used_in_prompt": True,
                "sources": [{"knowledge_id": "KB-EMG-002"}],
            }
        }
        report._validate_rag_citations(
            context,
            {
                "overall_interpretation": "仅作同条件复测 [KB-EMG-002]",
                "rag_citations": ["KB-EMG-002"],
            },
        )
        with self.assertRaisesRegex(ValueError, "未检索到"):
            report._validate_rag_citations(
                context,
                {
                    "overall_interpretation": "编造引用 [KB-NOT-FOUND]",
                    "rag_citations": ["KB-NOT-FOUND"],
                },
            )

        with self.assertRaisesRegex(ValueError, "rag_citations"):
            report._validate_rag_citations(
                context,
                {"overall_interpretation": "没有声明引用"},
            )

    def test_prediction_mentions_must_match_measured_values(self) -> None:
        context = {
            "predictions": {"FMA_UE": 15, "hand_function": 6, "hand_tone": "0"},
            "stage_roman": "VI",
        }
        report._validate_prediction_mentions(
            context,
            {
                "overall_interpretation": "本次FMA手部分数15分，Brunnstrom VI期。"
            },
        )
        with self.assertRaisesRegex(ValueError, "FMA 分数 6"):
            report._validate_prediction_mentions(
                context,
                {"treatment_advice": "结合FMA手部分数6分调整训练。"},
            )
        with self.assertRaisesRegex(ValueError, "FMA 分数 6"):
            report._validate_prediction_mentions(
                context,
                {"overall_interpretation": "FMA 手部分数 6，手部 MAS 0级。"},
            )
        with self.assertRaisesRegex(ValueError, "Brunnstrom 分期 V"):
            report._validate_prediction_mentions(
                context,
                {"overall_interpretation": "当前Brunnstrom V期。"},
            )
        with self.assertRaisesRegex(ValueError, "Brunnstrom 分期 V"):
            report._validate_prediction_mentions(
                context,
                {"overall_interpretation": "当前 Brunnstrom 手分期 V 期。"},
            )
        with self.assertRaisesRegex(ValueError, "手部 MAS 2级"):
            report._validate_prediction_mentions(
                {**context, "predictions": {**context["predictions"], "hand_tone": "0"}},
                {"overall_interpretation": "当前手部 MAS 2级。"},
            )

    def test_coerce_wrong_marker_keys_by_order(self) -> None:
        markers = [{"key": "movement_mu_power_change"}, {"key": "movement_beta_power_change"}]
        raw = {
            "mu_power": ["解读1", "建议1"],
            "beta_power": ["解读2", "建议2"],
        }
        self.assertEqual(
            report._coerce_marker_text_payload(raw, markers),
            {
                "movement_mu_power_change": ["解读1", "建议1"],
                "movement_beta_power_change": ["解读2", "建议2"],
            },
        )

    def test_marker_payload_has_all_required_keys(self) -> None:
        self.assertTrue(report._marker_payload_has_keys({"m1": [], "m2": []}, ["m1", "m2"]))
        self.assertFalse(report._marker_payload_has_keys({"m1": []}, ["m1", "m2"]))
        self.assertTrue(report._marker_payload_has_keys([[], []], ["m1", "m2"]))
        self.assertFalse(report._marker_payload_has_keys([[]], ["m1", "m2"]))

    def test_segment_json_repairs_missing_outer_brace(self) -> None:
        text = '</think>\n{"marker_text":{"m1":["解读","建议"]}'
        obj = report._parse_segment_json(text, required_marker_keys=["m1"])
        self.assertEqual(obj, {"marker_text": {"m1": ["解读", "建议"]}})
        self.assertIsNone(report._parse_clinical_json(text))

    def test_segment_json_recovers_glm_extra_array_bracket(self) -> None:
        text = (
            '{"marker_text":{"m1":["解读1","建议1"]"],'
            '"m2":["解读2","建议2"]"]}}'
            '\n```json\n{"marker_text":{"m1":["重复","重复"]"]}}\n```'
        )
        obj = report._parse_segment_json(text, required_marker_keys=["m1", "m2"])
        self.assertEqual(
            obj,
            {"marker_text": {"m1": ["解读1", "建议1"], "m2": ["解读2", "建议2"]}},
        )

    def test_segment_json_repairs_baichuan_semicolon_arrays(self) -> None:
        text = '{"marker_text":{"m1":["解读1"; "建议1"],"m2":["解读2"; "建议2"]}}'
        obj = report._parse_segment_json(text, required_marker_keys=["m1", "m2"])
        self.assertEqual(
            obj,
            {"marker_text": {"m1": ["解读1", "建议1"], "m2": ["解读2", "建议2"]}},
        )

    def test_segment_json_repairs_unterminated_single_marker_array(self) -> None:
        text = '{"marker_text":{"emg_burst_duration":["肌电爆发持续时间偏长，提示募集控制不足。'
        obj = report._parse_segment_json(text, required_marker_keys=["emg_burst_duration"])
        self.assertEqual(
            obj,
            {"marker_text": {"emg_burst_duration": ["肌电爆发持续时间偏长，提示募集控制不足。"]}},
        )

    def test_complete_marker_grounding_skips_segmented_marker_generation(self) -> None:
        context = {
            "patient": {},
            "predictions": {"FMA_UE": 15, "hand_tone": "0", "hand_function": 6},
            "stage": 6,
            "stage_roman": "VI",
            "biomarkers": {
                "groups": [
                    {"key": "emg", "markers": [{"key": "m1", "available": True}]}
                ]
            },
            "rag_evidence": {
                "used_in_prompt": False,
                "marker_grounding_used": True,
                "marker_grounding_complete": True,
                "sources": [],
            },
        }

        class FakeReportModel:
            cfg = {
                "segment_summary_prefill": "",
                "segment_summary_max_new_tokens": 512,
                "segment_stop_on_json": True,
            }

        generated = json.dumps({
            "overall_interpretation": "Brunnstrom VI期，FMA手部分数15分，手部MAS 0级。",
            "overall_subtype": "VI期-运动模式待检查，中枢驱动待验证，协同分离及活动度待确认。",
            "treatment_strategy": ["条件性训练建议。"],
            "warnings": [],
        }, ensure_ascii=False)
        with patch.object(report, "_generate_local_text", return_value=generated) as generate:
            result = json.loads(
                report._reason_local_segmented_clinical_json(context, FakeReportModel())
            )

        generate.assert_called_once()
        self.assertNotIn("marker_text", result)

    def test_complete_marker_grounding_reduces_generation_budget(self) -> None:
        context = {
            "rag_evidence": {
                "marker_grounding_used": True,
                "marker_grounding_complete": True,
            }
        }
        with patch.dict("os.environ", {"LLM_MAX_NEW_TOKENS": ""}, clear=False):
            self.assertEqual(
                report._dynamic_report_max_new_tokens(
                    context, {"max_new_tokens": 4096}
                ),
                1400,
            )


class LoadErrorTests(unittest.TestCase):
    def test_missing_adapter_dir_raises_clear_error(self) -> None:
        rm = report.ReportModel()
        rm.adapter_dir = Path("/nonexistent/checkpoints_llm/yi15_6b")
        with patch.dict("os.environ", {"LLM_USE_ADAPTER": "1"}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                rm.load()
        self.assertIn("adapter", str(ctx.exception).lower())


class DeepSeekTests(unittest.TestCase):
    def test_missing_key_raises_clear_error(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                report._reason_deepseek({})
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_deepseek_payload_and_response_parsing(self) -> None:
        calls = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "{\"overall_interpretation\":\"ok\"}"}}]}

        class FakeClient:
            def __init__(self, timeout) -> None:
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def post(self, url, headers, json):
                calls.append({"url": url, "headers": headers, "json": json})
                return FakeResponse()

        env = {
            "DEEPSEEK_API_KEY": "sk-test",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
            "DEEPSEEK_MAX_TOKENS": "256",
            "DEEPSEEK_TEMPERATURE": "0",
        }
        with patch.dict("os.environ", env, clear=False), patch("httpx.Client", FakeClient):
            text = report._reason_deepseek({"stage_roman": "III", "biomarkers": {"groups": []}})

        self.assertEqual(text, "{\"overall_interpretation\":\"ok\"}")
        self.assertEqual(calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(calls[0]["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(calls[0]["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0]["json"]["max_tokens"], 256)
        self.assertFalse(calls[0]["json"]["stream"])


if __name__ == "__main__":
    unittest.main()
