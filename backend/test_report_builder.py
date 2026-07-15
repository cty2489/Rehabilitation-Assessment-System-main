"""Tests for the deterministic report assembly + LLM-output validation.

These cover the cheap, GPU-free glue:
  * ``report_builder.validate_clinical`` — the gate that replaces the old
    rule-engine fallback (missing fields / stage-mismatch → raise).
  * footnote fragment de-duplication in ``render_markdown``.
  * the rebuilt clinical prompt is few-shot-free and stage-grounded.

Run from backend/:
    python -m unittest test_report_builder -v
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import report_builder
from report_builder import ClinicalUnavailable
from schemas import PatientInfo, PredictionResult

# Project root so the `llm` package imports (mirrors report.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _patient() -> PatientInfo:
    return PatientInfo(
        patient_id="S4", name="S4", sex="男", age=72,
        diagnosis="脑梗死", disease_days=30, paralysis_side="左",
    )


def _predictions(stage: int = 6) -> PredictionResult:
    # Stage 6 → 罗马数字 "VI"; this is the patient whose text used to be
    # parroted as "III期" from the few-shot example.
    return PredictionResult(FMA_UE=18.0, BI=80.0, hand_tone="0", hand_function=stage)


def _biomarkers() -> dict:
    # Two EMG markers share the "DEDUPFRAG" note fragment so the footnote dedup
    # (the real bug) is exercised intra-group.
    return {
        "flat": {"m_emg": 1.0, "m_emg2": 0.5, "m_eeg": 0.2},
        "groups": [
            {"key": "emg", "label": "肌电", "markers": [
                {"key": "m_emg", "name": "屈/伸肌 IEMG 比", "value": 1.2,
                 "unit": "比值", "ref_range": "比值", "note": "DEDUPFRAG；私有A"},
                {"key": "m_emg2", "name": "共收缩指数", "value": 0.5,
                 "unit": "比值", "ref_range": "比值", "note": "DEDUPFRAG；私有B"}]},
            {"key": "eeg", "label": "脑电", "markers": [
                {"key": "m_eeg", "name": "皮层-肌肉相干", "value": 0.2,
                 "unit": "相干", "ref_range": "无绝对阈值", "note": "跨模态未精同步"}]},
        ],
    }


def _context(stage: int = 6) -> dict:
    return report_builder.build_context(_patient(), _predictions(stage), _biomarkers())


def _valid_clinical(roman: str = "VI") -> dict:
    return {
        "overall_interpretation": f"患者手功能处于Brunnstrom {roman}期，恢复良好。",
        "marker_text": {
            "m_emg": {"interpretation": "屈/伸肌比偏高，提示屈肌主导。",
                      "treatment_advice": "增强伸指训练。"},
            "m_emg2": {"interpretation": "共收缩偏高，分离不足。",
                       "treatment_advice": "抗协同分离训练。"},
            "m_eeg": {"interpretation": "相干偏低，中枢-外周耦合偏弱。",
                      "treatment_advice": "加运动想象与本体反馈。"},
        },
        "overall_subtype": f"{roman}期-伸肌渐参与伴中枢驱动尚可亚型，协同接近正常，且活动度接近目标。",
        "treatment_strategy": [
            "策略名称：分离控制；具体方法：每日健侧镜像配合患侧主动；训练剂量：单次30分钟；"
            "反馈标准：动作完成质量；调整原则：代偿时降低难度；安全注意：避免疲劳。"
        ],
        "warnings": [],
    }


class ValidateClinicalTests(unittest.TestCase):
    def test_valid_passes_and_assembles(self) -> None:
        c = report_builder.validate_clinical(_context(6), _valid_clinical("VI"))
        self.assertEqual(c["overall_subtype"][:2], "VI")
        self.assertIn("m_emg", c["marker_text"])
        self.assertFalse(c["gesture_ready"])      # no library configured
        self.assertEqual(c["gesture_plan"], [])
        self.assertEqual(c["next_assessment"], report_builder.NEXT_ASSESSMENT_TEXT)

    def test_compact_marker_arrays_are_normalized(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"] = {
            "m_emg": ["屈伸肌比偏高，提示屈肌参与偏多。", "增加伸肌主动募集与慢速回中训练。"],
            "m_emg2": ["共收缩指数偏高，分离控制不足。", "降低快速抓放比例，先做分离控制。"],
            "m_eeg": ["相干偏低，中枢-外周耦合需观察。", "加入运动想象和视觉反馈配对训练。"],
        }
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["marker_text"]["m_emg"]["treatment_advice"], "增加伸肌主动募集与慢速回中训练。")

    def test_single_marker_array_is_normalized_with_default_advice(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["m_emg"] = ["屈伸肌比偏高，提示屈肌参与偏多。"]
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["marker_text"]["m_emg"]["interpretation"], "屈伸肌比偏高，提示屈肌参与偏多。")
        self.assertIn("复测趋势", c["marker_text"]["m_emg"]["treatment_advice"])

    def test_single_marker_array_splits_baichuan_advice_label(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["m_emg"] = ["【解释】屈伸肌比偏高，提示屈肌参与偏多。【建议】增加伸肌主动募集训练。"]
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["marker_text"]["m_emg"]["interpretation"], "【解释】屈伸肌比偏高，提示屈肌参与偏多")
        self.assertEqual(c["marker_text"]["m_emg"]["treatment_advice"], "增加伸肌主动募集训练。")

    def test_marker_array_with_copied_input_row_uses_default_advice(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["m_emg"] = [
            "肌电标志物（基于患侧被动评估）",
            ["屈/伸肌 IEMG 比", 1.2, "比值", "设备特异量"],
        ]
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["marker_text"]["m_emg"]["interpretation"], "肌电标志物（基于患侧被动评估）")
        self.assertIn("复测趋势", c["marker_text"]["m_emg"]["treatment_advice"])

    def test_common_treatment_advice_typo_is_normalized(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["m_emg"] = {
            "interpretation": "屈伸肌比偏高，提示屈肌参与偏多。",
            "treatation_advice": "增加伸肌主动募集训练。",
        }
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["marker_text"]["m_emg"]["treatment_advice"], "增加伸肌主动募集训练。")

    def test_strategy_dict_items_are_normalized(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["treatment_strategy"] = [
            {
                "strategy": "分离控制优先",
                "specific_method": "镜像配合患侧主动",
                "dose": "单次20分钟",
                "safety": "疲劳时停止",
            },
        ]
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertEqual(c["treatment_strategy"], ["分离控制优先；单次20分钟；疲劳时停止"])

    def test_group_subtypes_are_not_required_or_returned(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["group_subtypes"] = {"emg": "III期-旧肌电亚型"}
        c = report_builder.validate_clinical(_context(6), clinical)
        self.assertNotIn("group_subtypes", c)

    def test_specific_method_segment_is_removed(self) -> None:
        c = report_builder.validate_clinical(_context(6), _valid_clinical("VI"))
        strategy = c["treatment_strategy"][0]
        self.assertNotIn("具体方法", strategy)
        self.assertNotIn("健侧镜像", strategy)
        self.assertIn("训练剂量", strategy)

    def test_none_raises(self) -> None:
        with self.assertRaises(ClinicalUnavailable):
            report_builder.validate_clinical(_context(6), None)

    def test_stage_prefix_mismatch_raises(self) -> None:
        """The exact parrot-the-example bug: VI 期 patient, III 期 subtype text."""
        bad = _valid_clinical("III")   # subtypes prefixed "III期" but stage is VI
        with self.assertRaises(ClinicalUnavailable):
            report_builder.validate_clinical(_context(6), bad)

    def test_missing_marker_raises(self) -> None:
        bad = copy.deepcopy(_valid_clinical("VI"))
        del bad["marker_text"]["m_eeg"]
        with self.assertRaises(ClinicalUnavailable):
            report_builder.validate_clinical(_context(6), bad)

    def test_empty_strategy_raises(self) -> None:
        bad = copy.deepcopy(_valid_clinical("VI"))
        bad["treatment_strategy"] = []
        with self.assertRaises(ClinicalUnavailable):
            report_builder.validate_clinical(_context(6), bad)

    def test_device_specific_marker_cannot_keep_unsupported_high_low_claim(self) -> None:
        biomarkers = _biomarkers()
        biomarkers["groups"][0]["markers"][0]["key"] = "fds_iemg"
        context = report_builder.build_context(_patient(), _predictions(6), biomarkers)
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["fds_iemg"] = clinical["marker_text"].pop("m_emg")
        result = report_builder.validate_clinical(context, clinical)
        text = result["marker_text"]["fds_iemg"]
        self.assertNotIn("偏高", text["interpretation"])
        self.assertIn("单次结果不判断正常或异常", text["interpretation"])
        self.assertIn("同条件", text["treatment_advice"])


class RenderTests(unittest.TestCase):
    def test_footnote_fragments_deduped(self) -> None:
        md = report_builder.render_markdown(_context(6), _valid_clinical("VI"))
        # Two EMG markers share "DEDUPFRAG"; it must appear ONCE in the footnote,
        # not once per marker. Nothing else mentions it (no table column).
        self.assertEqual(md.count("DEDUPFRAG"), 1)
        self.assertIn("私有A", md)
        self.assertIn("私有B", md)
        # Gesture library not ready → placeholder, not a fabricated plan.
        self.assertIn("手势库待补充", md)

    def test_biomarker_table_hides_reference_range_and_labels_hand_scales(self) -> None:
        md = report_builder.render_markdown(_context(6), _valid_clinical("VI"))
        self.assertNotIn("| 标志物 | 当前值 | 参考范围 |", md)
        self.assertIn("| 标志物 | 当前值 | 解读 | 训练/随访建议 |", md)
        self.assertIn("同一患者在相同设备、相同采集流程下", md)
        self.assertIn("Brunnstrom手部分期", md)
        self.assertIn("手部肌张力（MAS）", md)
        self.assertNotIn("**亚型界定：**", md)
        self.assertNotIn("具体方法", md)
        self.assertNotIn("健侧镜像", md)

    def test_device_engineering_and_quality_warnings_are_prominent(self) -> None:
        context = report_builder.build_context(
            _patient(),
            _predictions(6),
            _biomarkers(),
            assessment_context={
                "validation_status": "engineering_validation_only",
                "quality": {"status": "needs_review"},
            },
        )
        md = report_builder.render_markdown(context, _valid_clinical("VI"))
        self.assertIn("设备端工程验证提示", md)
        self.assertIn("不能替代", md)
        self.assertIn("采样率不一致", md)


class PromptTests(unittest.TestCase):
    def test_prompt_is_fewshot_free_and_stage_grounded(self) -> None:
        from llm.prompts import build_clinical_reasoning_messages
        ctx = dict(_context(6))
        ctx["schema_hint"] = report_builder.CLINICAL_SCHEMA_HINT
        ctx["gesture_ready"] = False
        msgs = build_clinical_reasoning_messages(ctx)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["system", "user"])           # no assistant few-shot turn
        sys_txt = msgs[0]["content"]
        self.assertIn("防套模板", sys_txt)
        self.assertIn("VI期", sys_txt)                         # stage_roman injected
        # The old verbatim exemplar must not be present to be copied.
        self.assertNotIn("III期-屈肌优势伴中枢驱动不足亚型，协同开始解离", sys_txt)
        # Gesture library not ready → prompt tells the model to omit gesture fields.
        self.assertIn("gesture_plan", sys_txt)
        self.assertIn("不得", sys_txt)
        self.assertIn("队列排名", sys_txt)
        self.assertNotIn("group_subtypes", "\n".join(m["content"] for m in msgs))
        self.assertIn("禁止输出“具体方法”字段", sys_txt)

    def test_compact_prompt_lists_marker_keys_and_schema(self) -> None:
        from llm.prompts import build_compact_clinical_reasoning_messages
        ctx = dict(_context(6))
        ctx["gesture_ready"] = False
        msgs = build_compact_clinical_reasoning_messages(ctx)
        joined = "\n".join(m["content"] for m in msgs)
        self.assertIn('"marker_keys":["m_emg","m_emg2","m_eeg"]', joined)
        self.assertIn('"m_emg":["interpretation","treatment_advice"]', joined)
        self.assertIn("每个值必须是 [解读, 治疗建议] 二元数组", joined)
        self.assertIn("VI期-", joined)
        self.assertNotIn("group_subtypes", joined)
        self.assertIn("禁止输出具体方法", joined)


class BiomarkerReferenceTests(unittest.TestCase):
    def test_sparc_literature_range_is_not_treated_as_directly_comparable(self) -> None:
        from biomarker_refs import judge, marker_ref

        ref = marker_ref("movement_smoothness_sparc")
        self.assertIsNotNone(ref)
        self.assertFalse(ref["absolute_comparison_applicable"])
        self.assertNotIn("参考范围内", judge("movement_smoothness_sparc", -1.44))

    def test_device_specific_value_has_no_user_facing_absolute_range(self) -> None:
        from biomarker_refs import ref_display

        self.assertEqual(ref_display("fds_iemg"), "不适用；仅支持同设备、同流程复测")


if __name__ == "__main__":
    unittest.main()
