"""Tests for device-facing assessment export payloads.

Run from backend/:
    python -m unittest test_assessment_export -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import assessment_export


_REPORT = """# 智能康复评估报告

## 一、患者基本信息

## 二、本次评估结果（基于多模态数据）

### 1. 总体分期及状态

**临床解读：** VI期，主动运动基础较好，需继续巩固分离控制。

### 2. 关键生物标志物输出与解读

#### （1）肌电标志物（基于患侧被动评估）

| 标志物 | 当前值 | 参考范围 | 解读 | 治疗建议 |
| --- | --- | --- | --- | --- |
| 指浅屈肌积分肌电 | 1.2 uV.s | 设备特异量 | 屈肌激活偏高。 | 控制屈肌代偿。 |
| 缺失指标 | — | 设备特异量 | 本次数据不足，未予解读 | — |

**亚型界定：** VI期-肌电分离控制需巩固型

## 三、综合亚型界定与治疗策略

根据上述生物标志物，患者可归类为：**VI期-主动运动可量化伴分离控制需巩固亚型**

**治疗策略要点：**

1. 策略名称：分离控制优先；具体方法：每日进行腕伸与伸指训练；训练剂量：每日2组；反馈标准：动作完成质量；调整原则：代偿时降低难度；安全注意：避免疲劳。

## 四、下周具体训练参数

### 1. 推荐手势组合（从26个中动态选取）

| 顺序 | 手势名称 | 训练目的 | 辅助力度设置 | 重复次数 |
| --- | --- | --- | --- | --- |
| 1 | 伸食指 | 促进伸指启动 | 70% | 10次/组x3组 |

### 2. 每周训练计划

| 训练日 | 训练内容 | 预计时长 |
| --- | --- | --- |
| 周一 | 伸食指 | 30分钟 |

## 五、预警与特殊建议

1. 注意训练疲劳。

## 六、下次评估时间

建议：7天后执行下一次居家评估。
"""


def _assessment() -> dict:
    return {
        "id": 42,
        "source": "device",
        "institution": "device",
        "assessment_id": "EVAL_001",
        "session_id": "sess_001",
        "package_name": "patient.zip",
        "package_hash": "abc123",
        "n_trials": 9,
        "created_at": "2026-07-06 10:00:00",
        "assessment_time": "2026-07-06 09:58:00",
        "report_status": "generated",
        "parse_warnings": [],
        "patient_db_id": 7,
        "patient_id": "P001",
        "name": "张三",
        "sex": "男",
        "age": 62,
        "diagnosis": "脑梗死",
        "paralysis_side": "左",
        "disease_days": 120,
        "fma_ue": 18.0,
        "bi": 80.0,
        "hand_tone": "0",
        "hand_function": 6,
        "model_version": "test-dl",
        "llm_provider": "test",
        "llm_model": "test-llm",
        "report": _REPORT,
        "prediction_json": {"FMA_UE": 18.0},
        "biomarkers": {
            "coverage": {
                "available": 1,
                "total": 2,
                "missing_keys": ["missing_marker"],
            }
        },
        "biomarker_items": [
            {
                "group_key": "emg",
                "group_label": "肌电标志物",
                "marker_key": "fds_iemg",
                "marker_name": "指浅屈肌积分肌电",
                "value_text": "1.2",
                "value_num": 1.2,
                "unit": "uV.s",
                "ref_range": "设备特异量",
                "n_valid": 9,
                "available": True,
                "note": "设备特异量",
            },
            {
                "group_key": "emg",
                "group_label": "肌电标志物",
                "marker_key": "missing_marker",
                "marker_name": "缺失指标",
                "value_text": None,
                "value_num": None,
                "unit": "uV.s",
                "ref_range": "设备特异量",
                "n_valid": 0,
                "available": False,
                "note": "本次数据不足",
            },
        ],
        "trials": [{"trial_index": 1}],
    }


class AssessmentExportPayloadTests(unittest.TestCase):
    def test_pdf_markup_marks_latin_runs(self) -> None:
        markup = assessment_export._pdf_markup("患者 CLOUD_E2E_2026 V·s & <ok>")

        self.assertEqual(
            markup,
            '患者<font face="Helvetica"> CLOUD_E2E_2026 V·s &amp; &lt;ok&gt;</font>',
        )

    def test_result_payload_v2_removes_raw_duplicate_blocks(self) -> None:
        payload = assessment_export.build_result_payload(_assessment())

        self.assertEqual(payload["schema_version"], "rehab.assessment_result.v2")
        for duplicate_key in ("report", "prediction_json", "biomarkers_raw", "trials", "predictions"):
            self.assertNotIn(duplicate_key, payload)

        self.assertEqual(payload["stage_assessment"]["brunnstrom_stage"]["stage"], "VI")
        self.assertEqual(payload["biomarker_coverage"]["available_count"], 1)
        self.assertEqual(payload["biomarker_coverage"]["missing_keys"], ["missing_marker"])

        sections = payload["biomarker_sections"]
        self.assertEqual(len(sections), 1)
        self.assertNotIn("subtype", sections[0])
        indicators = sections[0]["indicators"]
        self.assertEqual([m["indicator_key"] for m in indicators], ["fds_iemg"])
        self.assertIsNone(indicators[0]["interpretation"])
        self.assertIsNone(indicators[0]["treatment_advice"])
        self.assertEqual(indicators[0]["interpretation_status"], "legacy_hidden")
        self.assertFalse(indicators[0]["reference_range"]["display"])
        self.assertIsNone(indicators[0]["reference_range"]["text"])
        self.assertEqual(indicators[0]["reference_range"]["type"], "none")
        self.assertFalse(indicators[0]["reference_range"]["absolute_comparison_applicable"])
        self.assertNotIn("队列", indicators[0]["reference_range"]["note"])
        self.assertIn("同一患者", indicators[0]["reference_range"]["note"])
        policy = payload["biomarker_interpretation_policy"]
        self.assertEqual(policy["user_facing_reference_range"], "hidden")
        self.assertEqual(policy["single_measurement_rule"], "do_not_classify_normal_abnormal")
        strategies = payload["subtype_classification_and_treatment_strategy"]["treatment_strategy"]["overall_strategies"]
        self.assertEqual(len(strategies), 1)
        self.assertNotIn("具体方法", strategies[0])
        self.assertNotIn("腕伸与伸指", strategies[0])
        self.assertIn("训练剂量", strategies[0])

    def test_new_four_column_report_preserves_evidence_aware_text(self) -> None:
        assessment = _assessment()
        assessment["report"] = _REPORT.replace(
            "| 标志物 | 当前值 | 参考范围 | 解读 | 治疗建议 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| 指浅屈肌积分肌电 | 1.2 uV.s | 设备特异量 | 屈肌激活偏高。 | 控制屈肌代偿。 |\n"
            "| 缺失指标 | — | 设备特异量 | 本次数据不足，未予解读 | — |",
            "| 标志物 | 当前值 | 解读 | 训练/随访建议 |\n"
            "| --- | --- | --- | --- |\n"
            "| 指浅屈肌积分肌电 | 1.2 uV.s | 本次值用于同条件复测。 | 结合功能量表调整训练。 |\n"
            "| 缺失指标 | — | 本次数据不足，未予解读 | — |",
        )
        payload = assessment_export.build_result_payload(assessment)
        marker = payload["biomarker_sections"][0]["indicators"][0]
        self.assertEqual(marker["interpretation"], "本次值用于同条件复测。")
        self.assertEqual(marker["treatment_advice"], "结合功能量表调整训练。")
        self.assertEqual(marker["interpretation_status"], "available")

    def test_knowledge_evidence_is_preserved_in_json_and_pdf(self) -> None:
        assessment = _assessment()
        evidence_block = """### 辅助知识证据来源

> 实验提示：本次显式启用了未完成临床审核的 Demo 知识。

| 知识ID | 标题 | 知识状态 | 来源 | 审核状态 |
| --- | --- | --- | --- | --- |
| KB-EMG-009 | 屈/伸肌 IEMG 比 | 研究用途 | [SRC-001] 测试文献 https://example.test/1；[SRC-002] 第二篇文献 https://example.test/2 | 内部试运行 / 未完成正式专家审核 |
| KB-EEG-001 | 功率不对称比 | 条件候选 | [SRC-001] 测试文献 https://example.test/1 | 内部试运行 / 未完成正式专家审核 |

"""
        assessment["report"] = assessment["report"].replace(
            "## 四、下周具体训练参数",
            evidence_block + "## 四、下周具体训练参数",
        )
        payload = assessment_export.build_result_payload(assessment)
        evidence = payload["knowledge_evidence"]
        self.assertTrue(evidence["used_in_report"])
        self.assertEqual(evidence["clinical_review_status"], "demo_unreviewed")
        self.assertEqual(len(evidence["entries"]), 2)
        self.assertEqual(evidence["entries"][0]["source_ids"], ["SRC-001", "SRC-002"])
        self.assertEqual(len(evidence["references"]), 2)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.pdf"
            assessment_export.write_report_pdf(out, payload)
            self.assertGreater(out.stat().st_size, 1000)

    def test_numeric_citations_are_preserved_per_claim_and_in_catalog(self) -> None:
        assessment = _assessment()
        assessment["report"] = _REPORT.replace(
            "**临床解读：** VI期，主动运动基础较好，需继续巩固分离控制。",
            "**临床解读：** VI期，主动运动基础较好，需继续巩固分离控制。【1】",
        ).replace(
            "| 标志物 | 当前值 | 参考范围 | 解读 | 治疗建议 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| 指浅屈肌积分肌电 | 1.2 uV.s | 设备特异量 | 屈肌激活偏高。 | 控制屈肌代偿。 |\n"
            "| 缺失指标 | — | 设备特异量 | 本次数据不足，未予解读 | — |",
            "| 标志物 | 当前值 | 解读 | 训练/随访建议 |\n"
            "| --- | --- | --- | --- |\n"
            "| 指浅屈肌积分肌电 | 1.2 uV.s | 仅用于同条件复测。【1】 | 结合量表调整训练。【1】【2】 |\n"
            "| 缺失指标 | — | 本次数据不足，未予解读 | — |",
        )
        assessment["report"] += """

## 七、依据来源与参考文献

### 1. 知识库证据审计

| 引用 | 知识ID | 知识条目 | 知识状态 | 来源ID | 审核状态 |
| --- | --- | --- | --- | --- | --- |
| 【1】【2】 | KB-EMG-009 | 屈伸肌积分肌电比 | 研究用途 | SRC-001、SRC-002 | 已审核 |

### 2. 参考文献

【1】第一篇测试文献

【2】第二篇测试文献
"""

        payload = assessment_export.build_result_payload(assessment)
        evidence = payload["knowledge_evidence"]
        self.assertEqual(evidence["citation_style"], "numeric_square_brackets_zh")
        self.assertEqual(evidence["entries"][0]["citation_numbers"], [1, 2])
        self.assertEqual(evidence["entries"][0]["source_ids"], ["SRC-001", "SRC-002"])
        self.assertEqual([item["number"] for item in evidence["references"]], [1, 2])
        self.assertEqual(evidence["references"][0]["source_id"], "SRC-001")

        indicator = payload["biomarker_sections"][0]["indicators"][0]
        self.assertEqual(indicator["interpretation_citation_numbers"], [1])
        self.assertEqual(indicator["treatment_advice_citation_numbers"], [1, 2])
        self.assertEqual(
            payload["stage_assessment"]["brunnstrom_stage"]["citation_numbers"],
            [1],
        )

    def test_report_pdf_can_be_written_from_v2_payload(self) -> None:
        payload = assessment_export.build_result_payload(_assessment())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.pdf"
            assessment_export.write_report_pdf(out, payload)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
