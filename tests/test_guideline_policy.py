from __future__ import annotations

import unittest

from rag.guideline_policy import screen_guideline_test_query


class GuidelinePolicyTests(unittest.TestCase):
    def test_blocks_numeric_reference_request(self):
        decision = screen_guideline_test_query("请给出 IMU 指标的正常范围和异常阈值。")
        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.reason_code, "numeric_reference_out_of_scope")

    def test_blocks_specific_training_dose_request(self):
        decision = screen_guideline_test_query("请直接给出上肢训练每天应做多少分钟、每周多少次。")
        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.reason_code, "training_dose_out_of_scope")

    def test_allows_paper_brunnstrom_correlation_result(self):
        decision = screen_guideline_test_query("论文报告的 Brunnstrom 预测相关性是多少？")
        self.assertEqual(decision.action, "retrieve")

    def test_blocks_patient_brunnstrom_stage(self):
        decision = screen_guideline_test_query("这个患者属于 Brunnstrom 几期？")
        self.assertEqual(decision.action, "block")
        self.assertEqual(
            decision.reason_code,
            "patient_level_clinical_judgment_out_of_scope",
        )

    def test_rejects_empty_query(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            screen_guideline_test_query("  ")


if __name__ == "__main__":
    unittest.main()
