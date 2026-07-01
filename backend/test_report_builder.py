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
        "group_subtypes": {"emg": f"{roman}期-屈肌优势型", "eeg": f"{roman}期-中枢驱动可型"},
        "overall_subtype": f"{roman}期-伸肌渐参与伴中枢驱动尚可亚型，协同接近正常，且活动度接近目标。",
        "treatment_strategy": ["伸指训练：每日健侧镜像+患侧主动，单次30分钟。"],
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


if __name__ == "__main__":
    unittest.main()
