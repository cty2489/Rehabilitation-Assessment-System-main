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


def _grounded_context() -> dict:
    context = _context(6)
    context["rag_evidence"] = {
        "mode": "assist",
        "used_in_prompt": True,
        "used_in_report": True,
        "marker_grounding_used": True,
        "sources": [],
        "marker_sources": {
            "m_emg": {
                "knowledge_id": "KB-EMG-009",
                "system_key": "m_emg",
                "title": "屈伸肌积分肌电比",
                "knowledge_status": "blocked_current_implementation",
                "knowledge_status_label": "阻断当前实现",
                "proposed_claim": "该比值描述所选通道累计电活动的相对构成。",
                "allowed_interpretation": "通道核实后可探索屈伸活动偏向。",
                "prohibited_interpretation": "不得因比值大于1诊断屈肌痉挛。",
                "implementation_action": "核实通道并增加分母质量控制。",
                "clinical_ready": False,
                "references": ["[SRC-001] 测试文献"],
            }
        },
    }
    return context


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

    def test_exact_marker_knowledge_replaces_generic_llm_text(self) -> None:
        clinical = _valid_clinical("VI")
        clinical["marker_text"]["m_emg"] = {
            "interpretation": "该值为设备特异量，仅用于同设备同流程复测比较。",
            "treatment_advice": "建议复测。",
        }
        result = report_builder.validate_clinical(_grounded_context(), clinical)
        grounded = result["marker_text"]["m_emg"]
        self.assertIn("累计电活动的相对构成", grounded["interpretation"])
        self.assertIn("不得因比值大于1诊断屈肌痉挛", grounded["interpretation"])
        self.assertIn("[KB-EMG-009]", grounded["interpretation"])
        self.assertIn("核实通道并增加分母质量控制", grounded["treatment_advice"])
        self.assertIn("KB-EMG-009", result["rag_citations"])

    def test_complete_grounding_owns_numeric_prefix_and_safe_subtype(self) -> None:
        context = _grounded_context()
        base_source = context["rag_evidence"]["marker_sources"]["m_emg"]
        for key, knowledge_id in (("m_emg2", "KB-EMG-002"), ("m_eeg", "KB-EEG-002")):
            source = copy.deepcopy(base_source)
            source["system_key"] = key
            source["knowledge_id"] = knowledge_id
            context["rag_evidence"]["marker_sources"][key] = source
        context["rag_evidence"]["marker_grounding_complete"] = True

        clinical = _valid_clinical("VI")
        clinical["overall_interpretation"] = "中枢驱动不足，仍需进一步训练。"
        clinical["treatment_strategy"] = [
            "任务难度分级；每日2次；根据动作完成质量调整；疲劳时停止。",
            "神经兴奋性调节；根据EMG变化优化刺激参数。",
        ]
        clinical.pop("marker_text")
        clinical.pop("overall_subtype")
        result = report_builder.validate_clinical(context, clinical)

        self.assertIn("Brunnstrom手部分期为VI期", result["overall_interpretation"])
        self.assertIn("FMA手部分数为18/20", result["overall_interpretation"])
        self.assertIn("手部肌张力（MAS）为0级", result["overall_interpretation"])
        self.assertNotIn("中枢驱动不足", result["overall_interpretation"])
        self.assertIn("专业复核", result["overall_interpretation"])
        self.assertEqual(
            result["overall_subtype"],
            "VI期-运动模式待动作检查确认，中枢驱动特征待同步协议验证，"
            "协同分离程度待动作检查确认，关节活动度待角度测量确认",
        )
        self.assertEqual(set(result["marker_text"]), {"m_emg", "m_emg2", "m_eeg"})
        self.assertEqual(
            result["treatment_strategy"],
            ["任务难度分级；每日2次；根据动作完成质量调整；疲劳时停止。"],
        )
        self.assertEqual(len(result["warnings"]), 3)
        self.assertIn("未完成正式专家审核", result["warnings"][0])

    def test_marker_knowledge_does_not_change_shadow_report(self) -> None:
        context = _grounded_context()
        context["rag_evidence"]["mode"] = "shadow"
        context["rag_evidence"]["marker_grounding_used"] = False
        result = report_builder.validate_clinical(context, _valid_clinical("VI"))
        self.assertNotIn("KB-EMG-009", result["marker_text"]["m_emg"]["interpretation"])


class RenderTests(unittest.TestCase):
    def test_exact_marker_source_is_rendered_with_its_reference(self) -> None:
        md = report_builder.render_markdown(_grounded_context(), _valid_clinical("VI"))
        self.assertIn("[KB-EMG-009]", md)
        self.assertIn("辅助知识证据来源", md)
        self.assertIn("[SRC-001] 测试文献", md)

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

    def test_reviewed_rag_sources_are_rendered_only_when_used(self) -> None:
        context = dict(_context(6))
        context["rag_evidence"] = {
            "used_in_prompt": True,
            "sources": [
                {
                    "knowledge_id": "KB-001",
                    "title": "审核知识",
                    "knowledge_status_label": "指南候选",
                    "clinical_ready": True,
                    "source_document_id": "doc-1",
                    "source_entry_number": 2,
                    "references": ["指南A，第2章"],
                    "reviewed_by": "王医生",
                    "reviewed_at": "2026-07-16",
                }
            ],
        }
        clinical = _valid_clinical("VI")
        clinical["rag_citations"] = ["KB-001"]
        md = report_builder.render_markdown(context, clinical)
        self.assertIn("辅助知识证据来源", md)
        self.assertIn("KB-001", md)
        self.assertIn("指南候选", md)
        self.assertIn("王医生 / 2026-07-16", md)

        context["rag_evidence"]["used_in_prompt"] = False
        without_rag = report_builder.render_markdown(context, clinical)
        self.assertNotIn("辅助知识证据来源", without_rag)


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

    def test_complete_marker_grounding_uses_summary_only_prompt(self) -> None:
        from llm.prompts import build_clinical_reasoning_messages

        context = _grounded_context()
        context["rag_evidence"]["marker_grounding_complete"] = True
        context["schema_hint"] = report_builder.CLINICAL_SCHEMA_HINT
        context["gesture_ready"] = False
        messages = build_clinical_reasoning_messages(context)
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]

        self.assertIn("只生成不重复数值的定性摘要", system_text)
        self.assertIn("不得从单次 EMG/EEG/IMU 数值推导屈肌优势", system_text)
        self.assertNotIn('"marker_text"', user_text)
        self.assertNotIn('"overall_subtype"', user_text)
        self.assertNotIn('"biomarkers"', user_text)
        self.assertIn('"marker_grounding":{"complete":true', user_text)
        self.assertIn("禁止输出 marker_text", user_text)

    def test_rag_evidence_enters_prompt_only_after_governance_gate(self) -> None:
        from llm.prompts import build_clinical_reasoning_messages

        context = dict(_context(6))
        context["schema_hint"] = report_builder.CLINICAL_SCHEMA_HINT
        context["gesture_ready"] = False
        context["rag_evidence"] = {
            "used_in_prompt": False,
            "sources": [{"knowledge_id": "KB-001", "text": "不应进入 Prompt"}],
        }
        without_evidence = "\n".join(
            item["content"] for item in build_clinical_reasoning_messages(context)
        )
        self.assertNotIn("KB-001", without_evidence)

        context["rag_evidence"] = {
            "used_in_prompt": True,
            "sources": [
                {
                    "knowledge_id": "KB-001",
                    "title": "审核知识",
                    "text": "只支持同设备同流程复测。",
                    "source_document_id": "doc-1",
                    "source_entry_number": 1,
                    "references": ["来源A"],
                    "reviewed_by": "专家",
                    "reviewed_at": "2026-07-16",
                }
            ],
        }
        with_evidence = "\n".join(
            item["content"] for item in build_clinical_reasoning_messages(context)
        )
        self.assertIn("knowledge_evidence", with_evidence)
        self.assertIn("KB-001", with_evidence)
        self.assertIn("rag_citations", with_evidence)
        self.assertIn("患者实测数值和临床量表始终优先", with_evidence)

    def test_segmented_summary_receives_governed_rag_evidence(self) -> None:
        import report

        context = dict(_context(6))
        context["rag_evidence"] = {
            "used_in_prompt": True,
            "sources": [
                {
                    "knowledge_id": "KB-001",
                    "title": "审核知识",
                    "text": "证据正文",
                }
            ],
        }
        joined = "\n".join(item["content"] for item in report._segment_summary_messages(context))
        self.assertIn("knowledge_evidence", joined)
        self.assertIn("KB-001", joined)
        self.assertIn("rag_citations", joined)


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
