"""Tests for rag.guideline_policy scope guard."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.guideline_policy import GuidelineQueryDecision, screen_guideline_test_query


class GuidelinePolicyTests(unittest.TestCase):
    """Tests for research retrieval and patient-level safety boundaries."""

    def test_in_scope_query_passes(self):
        result = screen_guideline_test_query("卒中后上肢康复训练原则")
        self.assertEqual(result.action, "retrieve")
        self.assertEqual(result.reason_code, "in_scope")

    def test_numeric_reference_blocked(self):
        result = screen_guideline_test_query("EMG 肌电异常阈值是多少")
        self.assertEqual(result.action, "block")
        self.assertEqual(result.reason_code, "numeric_reference_out_of_scope")

    def test_specific_dose_blocked(self):
        result = screen_guideline_test_query("EEG 训练方案剂量")
        self.assertEqual(result.action, "block")
        self.assertEqual(result.reason_code, "training_dose_out_of_scope")

    def test_imu_research_framework_allowed(self):
        result = screen_guideline_test_query("IMU 如何研究能力与真实生活表现的区别？")
        self.assertEqual(result.action, "retrieve")

    def test_reference_range_without_biomarker_is_still_blocked(self):
        result = screen_guideline_test_query("正常范围参考值")
        self.assertEqual(result.action, "block")
        self.assertEqual(result.reason_code, "numeric_reference_out_of_scope")

    def test_paper_brunnstrom_result_allowed(self):
        result = screen_guideline_test_query("论文报告的 Brunnstrom 手部 Spearman 相关性是多少？")
        self.assertEqual(result.action, "retrieve")

    def test_patient_brunnstrom_stage_blocked(self):
        result = screen_guideline_test_query("根据 IMU 结果，这位患者属于 Brunnstrom 几期？")
        self.assertEqual(result.action, "block")
        self.assertEqual(
            result.reason_code,
            "patient_level_clinical_judgment_out_of_scope",
        )

    def test_patient_treatment_plan_blocked(self):
        result = screen_guideline_test_query("请根据结果为该患者制定康复方案。")
        self.assertEqual(result.action, "block")
        self.assertEqual(result.reason_code, "automated_prescription_out_of_scope")

    def test_research_treatment_description_allowed(self):
        result = screen_guideline_test_query("请总结论文研究了哪些训练方案。")
        self.assertEqual(result.action, "retrieve")

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            screen_guideline_test_query("")

    def test_whitespace_query_raises(self):
        with self.assertRaises(ValueError):
            screen_guideline_test_query("   ")

    def test_decision_to_dict(self):
        result = screen_guideline_test_query("EMG 阈值异常")
        d = result.to_dict()
        self.assertIn("action", d)
        self.assertIn("reason_code", d)
        self.assertIn("message", d)


if __name__ == "__main__":
    unittest.main()
