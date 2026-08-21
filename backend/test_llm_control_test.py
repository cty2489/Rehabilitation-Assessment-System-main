import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_control_test import ManualPredictionValues, input_fingerprint, quality_metadata
from schemas import PatientInfo


PATIENT = PatientInfo(
    patient_id="LLMTEST-001",
    name="测试病例",
    sex="男",
    age=60,
    diagnosis="脑梗死",
    disease_days=30,
    paralysis_side="左",
)


class LlmControlTestContractTests(unittest.TestCase):
    def test_manual_scores_follow_existing_ranges(self):
        values = ManualPredictionValues(FMA_UE=12, hand_tone="1+", hand_function=4)
        self.assertEqual(12, values.FMA_UE)
        with self.assertRaises(ValidationError):
            ManualPredictionValues(FMA_UE=21, hand_tone="1+", hand_function=4)
        with self.assertRaises(ValidationError):
            ManualPredictionValues(FMA_UE=12, hand_tone="5", hand_function=4)

    def test_fingerprint_includes_scores_and_signal_files(self):
        values = ManualPredictionValues(FMA_UE=12, hand_tone="1+", hand_function=4)
        sample = Path(__file__)
        first = input_fingerprint(PATIENT, values, [sample], [sample])
        second = input_fingerprint(PATIENT, values, [sample], [sample])
        changed = input_fingerprint(
            PATIENT,
            ManualPredictionValues(FMA_UE=13, hand_tone="1+", hand_function=4),
            [sample],
            [sample],
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_quality_distinguishes_manual_scores_from_signal_biomarkers(self):
        quality = quality_metadata(
            "abc",
            {"available": 25, "total": 26, "missing_keys": ["one"]},
            n_trials=3,
        )
        self.assertEqual("manual_ground_truth", quality["prediction_source"])
        self.assertEqual(
            "computed_from_uploaded_signals", quality["biomarker_source"]
        )
        self.assertTrue(quality["dl_inference_skipped"])
        self.assertTrue(quality["biomarkers_computed_from_signal"])
        self.assertFalse(quality["persisted"])


if __name__ == "__main__":
    unittest.main()
