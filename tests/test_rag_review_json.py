from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.ingest.review_json import prepare_review_json_knowledge_base


def _document() -> dict:
    return {
        "schema_version": "rehab_rag_expert_review_draft/0.1",
        "version": "v0.1",
        "status": "pending_expert_review",
        "clinical_ready": False,
        "scope": {"population": "成人脑卒中后上肢康复"},
        "trial_release": {
            "release_id": "internal-rag-trial-v0.1",
            "expert_verified": False,
            "clinical_ready": False,
            "warning": "仅供内部测试。",
        },
        "sources": [
            {
                "source_id": "SRC-001",
                "title": "测试来源",
                "year": 2025,
                "evidence_tier": "A",
                "url": "https://example.test/source",
            }
        ],
        "entries": [
            {
                "knowledge_id": "KB-EMG-001",
                "domain": "EMG",
                "display_name": "测试指标",
                "aliases": ["RMS"],
                "system_key": "test_metric",
                "status": "blocked_current_implementation",
                "status_label": "阻断",
                "clinical_ready": False,
                "proposed_claim": "这是核心结论。",
                "allowed_interpretation": "仅可同条件复测。",
                "prohibited_interpretation": "不得自动诊断。",
                "acquisition_and_algorithm_requirements": "固定采集协议。",
                "reference_range_policy": "不显示通用范围。",
                "source_ids": ["SRC-001"],
                "expert_decision": "pending",
                "expert_reviewer": "",
                "expert_review_date": "",
            }
        ],
        "evaluation_questions": [
            {
                "question_id": "Q001",
                "category": "直接检索",
                "question": "测试指标如何解释？",
                "expected_knowledge_ids": ["KB-EMG-001"],
            }
        ],
    }


class RagReviewJsonTests(unittest.TestCase):
    def test_unreviewed_json_requires_explicit_trial_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.json"
            source.write_text(
                json.dumps(_document(), ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "allow-internal-trial"):
                prepare_review_json_knowledge_base(source, root / "out")

    def test_trial_conversion_preserves_governance_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.json"
            source.write_text(
                json.dumps(_document(), ensure_ascii=False), encoding="utf-8"
            )
            output = root / "out"
            result = prepare_review_json_knowledge_base(
                source,
                output,
                allow_internal_trial=True,
            )
            entry = json.loads((output / "entries.jsonl").read_text(encoding="utf-8"))
            chunk = json.loads((output / "chunks.jsonl").read_text(encoding="utf-8"))
            question = json.loads(
                (output / "evaluation_queries.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(result["manifest"]["counts"]["total_entries"], 1)
        self.assertTrue(entry["status"]["demo_ready"])
        self.assertFalse(entry["status"]["clinical_ready"])
        self.assertFalse(chunk["metadata"]["expert_verified"])
        self.assertEqual(
            chunk["metadata"]["knowledge_status"],
            "blocked_current_implementation",
        )
        self.assertIn("不得自动诊断", chunk["text"])
        self.assertEqual(question["expected_knowledge_ids"], ["KB-EMG-001"])


if __name__ == "__main__":
    unittest.main()
