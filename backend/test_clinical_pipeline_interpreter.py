from __future__ import annotations

import unittest
from collections import Counter

from biomarker_refs import REF_META
from inference_readings import brunnstrom_reading, hand_tone_reading

from clinical_pipeline.contracts import (
    CanonicalAssessmentContext,
    CanonicalBiomarker,
    CanonicalPatientInfo,
    CanonicalPredictions,
    FindingModality,
    FindingSeverity,
    FindingStatus,
    InterpretationResult,
    QualityDecision,
)
from clinical_pipeline.interpreter import Interpreter


def _marker(
    metric_key: str = "synthetic_reference_marker",
    *,
    value: float | None = 15.0,
    available: bool = True,
    modality: FindingModality = FindingModality.EMG,
) -> CanonicalBiomarker:
    return CanonicalBiomarker(
        metric_key=metric_key,
        name="测试参考指标",
        value=value,
        unit="test-unit",
        modality=modality,
        available=available,
        n_valid=1 if available else 0,
    )


def _context(*markers: CanonicalBiomarker) -> CanonicalAssessmentContext:
    return CanonicalAssessmentContext(
        context_id="context-1",
        assessment_id="assessment-1",
        quality_decision=QualityDecision.PASS,
        patient=CanonicalPatientInfo(patient_id="P001"),
        predictions=CanonicalPredictions(
            FMA_UE=8,
            hand_tone="2",
            hand_function=3,
        ),
        biomarkers=list(markers),
    )


def _synthetic_absolute_reference(_: str) -> dict:
    # Test-only engineering fixture. These are not medical thresholds.
    return {
        "units": "test-unit",
        "reference_type": "healthy_norm",
        "expected_direction": "n/a",
        "lo": 10.0,
        "hi": 20.0,
        "source": ["TEST-SOURCE"],
        "absolute_comparison_applicable": True,
    }


def _finding(result: InterpretationResult, metric_key: str):
    return next(item for item in result.findings if item.metric_key == metric_key)


class InterpreterTests(unittest.TestCase):
    def test_input_must_be_canonical_assessment_context(self) -> None:
        with self.assertRaisesRegex(TypeError, "CanonicalAssessmentContext"):
            Interpreter().interpret({"predictions": {}})  # type: ignore[arg-type]

    def test_within_reference(self) -> None:
        result = Interpreter(_synthetic_absolute_reference).interpret(
            _context(_marker(value=15.0))
        )
        finding = _finding(result, "synthetic_reference_marker")
        self.assertEqual(finding.status, FindingStatus.WITHIN_REFERENCE)
        self.assertEqual(finding.basis.lower_bound, 10.0)
        self.assertEqual(finding.basis.upper_bound, 20.0)

    def test_above_reference(self) -> None:
        result = Interpreter(_synthetic_absolute_reference).interpret(
            _context(_marker(value=21.0))
        )
        self.assertEqual(
            _finding(result, "synthetic_reference_marker").status,
            FindingStatus.ABOVE_REFERENCE,
        )

    def test_below_reference(self) -> None:
        result = Interpreter(_synthetic_absolute_reference).interpret(
            _context(_marker(value=9.0))
        )
        self.assertEqual(
            _finding(result, "synthetic_reference_marker").status,
            FindingStatus.BELOW_REFERENCE,
        )

    def test_device_specific_marker_is_not_classifiable(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("resting_emg_level", value=0.0002))
        )
        finding = _finding(result, "resting_emg_level")
        self.assertEqual(finding.status, FindingStatus.NOT_CLASSIFIABLE)
        self.assertIn("不支持单次正常或异常分类", finding.basis.description)

    def test_directional_marker_is_direction_only(self) -> None:
        result = Interpreter().interpret(
            _context(
                _marker(
                    "corticomuscular_coherence_beta",
                    value=0.31,
                    modality=FindingModality.EEG,
                )
            )
        )
        finding = _finding(result, "corticomuscular_coherence_beta")
        self.assertEqual(finding.status, FindingStatus.DIRECTION_ONLY)
        self.assertEqual(finding.modality, FindingModality.MULTIMODAL)
        self.assertIn("纵向复测比较", finding.description)
        self.assertIn("当前输入未提供历史测量", finding.description)
        self.assertIn("不作变化方向判断", finding.description)
        self.assertIsNone(finding.basis.expected_direction)

    def test_direction_only_does_not_claim_change_without_history(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("pathological_asymmetry_index", value=0.31))
        )
        finding = _finding(result, "pathological_asymmetry_index")
        self.assertEqual(finding.status, FindingStatus.DIRECTION_ONLY)
        for unsupported_claim in ("本次升高", "本次降低", "已经恢复", "有所恢复"):
            self.assertNotIn(unsupported_claim, finding.description)
        self.assertIsNone(finding.basis.expected_direction)

    def test_noncomparable_sparc_is_not_classifiable(self) -> None:
        result = Interpreter().interpret(
            _context(
                _marker(
                    "movement_smoothness_sparc",
                    value=-1.44,
                    modality=FindingModality.IMU,
                )
            )
        )
        finding = _finding(result, "movement_smoothness_sparc")
        self.assertEqual(finding.status, FindingStatus.NOT_CLASSIFIABLE)
        self.assertFalse(finding.basis.absolute_comparison_applicable)

    def test_missing_marker(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("range_of_motion_proxy", value=None, available=False))
        )
        finding = _finding(result, "range_of_motion_proxy")
        self.assertEqual(finding.status, FindingStatus.MISSING)
        self.assertIsNone(finding.value)

    def test_fma_is_explicitly_the_zero_to_twenty_hand_subscale(self) -> None:
        finding = _finding(Interpreter().interpret(_context()), "FMA_UE")
        self.assertEqual(finding.name, "FMA手部子量表，范围0–20")
        self.assertNotIn("总分", finding.name + finding.description)
        self.assertEqual(finding.source_field, "predictions.FMA_UE")

    def test_existing_scale_readings_are_reused(self) -> None:
        result = Interpreter().interpret(_context())
        self.assertIn(
            hand_tone_reading("2"),
            _finding(result, "hand_tone").description,
        )
        self.assertIn(
            brunnstrom_reading(3),
            _finding(result, "hand_function").description,
        )

    def test_all_scale_findings_are_explicitly_model_predictions(self) -> None:
        result = Interpreter().interpret(_context())
        for metric_key in ("FMA_UE", "hand_tone", "hand_function"):
            finding = _finding(result, metric_key)
            self.assertIn("模型预测结果", finding.description)
            self.assertIn("不是医生实测结论", finding.description)
            self.assertIn("深度模型预测字段", finding.basis.description)

    def test_all_26_biomarkers_have_one_consistent_classification_count(self) -> None:
        self.assertEqual(len(REF_META), 26)
        self.assertEqual(
            sum(
                1
                for reference in REF_META.values()
                if reference.get("reference_type") == "none"
            ),
            20,
        )
        markers = [
            _marker(metric_key, value=0.5)
            for metric_key in REF_META
        ]
        result = Interpreter().interpret(_context(*markers))
        biomarker_findings = [
            finding
            for finding in result.findings
            if finding.finding_id.startswith("biomarker:")
        ]
        counts = Counter(finding.status for finding in biomarker_findings)
        self.assertEqual(len(biomarker_findings), 26)
        # 20 reference_type=none markers plus non-comparable acceleration SPARC.
        self.assertEqual(counts[FindingStatus.NOT_CLASSIFIABLE], 21)
        self.assertEqual(counts[FindingStatus.DIRECTION_ONLY], 5)
        self.assertEqual(
            _finding(result, "movement_smoothness_sparc").status,
            FindingStatus.NOT_CLASSIFIABLE,
        )

    def test_every_severity_is_unknown_without_verified_mapping(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("resting_emg_level", value=0.0002))
        )
        self.assertTrue(result.findings)
        self.assertTrue(
            all(item.severity == FindingSeverity.UNKNOWN for item in result.findings)
        )

    def test_output_has_no_diagnosis_treatment_or_retrieval_planning(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("resting_emg_level", value=0.0002))
        )
        text = result.model_dump_json()
        for forbidden in ("诊断", "病理机制", "康复建议", "治疗建议", "训练剂量", "需要检索"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(result.known_combinations, [])

    def test_source_fields_are_traceable(self) -> None:
        result = Interpreter().interpret(
            _context(_marker("resting_emg_level", value=0.0002))
        )
        expected = {
            "FMA_UE": "predictions.FMA_UE",
            "hand_tone": "predictions.hand_tone",
            "hand_function": "predictions.hand_function",
            "resting_emg_level": "biomarkers.resting_emg_level.value",
        }
        self.assertEqual(
            {item.metric_key: item.source_field for item in result.findings},
            expected,
        )
        restored = InterpretationResult.model_validate_json(result.model_dump_json())
        self.assertEqual(len(restored.findings), 4)


if __name__ == "__main__":
    unittest.main()
