from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_admin import KnowledgeSnapshot, KnowledgeUnavailable

from clinical_pipeline.core_knowledge import (
    CoreKnowledgeLoadError,
    CoreKnowledgeProvider,
)


def _entry(
    system_key: str,
    *,
    knowledge_id: str,
    allowed: str,
    prohibited: str,
    source_ids: list[str],
) -> dict:
    return {
        "knowledge_id": knowledge_id,
        "system_key": system_key,
        "title": "仅用于确认额外字段不会进入核心知识契约",
        "content": "测试正文不应被CoreKnowledgeProvider复制",
        "allowed_interpretation": allowed,
        "prohibited_interpretation": prohibited,
        "source": {"source_ids": source_ids},
    }


def _snapshot(*entries: dict) -> KnowledgeSnapshot:
    return KnowledgeSnapshot(
        root=Path("/test/existing-runtime"),
        manifest={
            "collection_id": "rehab_knowledge_trial_v0_2",
            "trial_release": {"release_id": "existing-core-v0.2"},
        },
        quality_report={},
        entries=tuple(entries),
        sources=(),
        validation_issues=(),
    )


class CoreKnowledgeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = _snapshot(
            _entry(
                "metric_a",
                knowledge_id="KB-A",
                allowed="测试允许解释A",
                prohibited="测试禁止解释A",
                source_ids=["SRC-A"],
            ),
            _entry(
                "metric_b",
                knowledge_id="KB-B",
                allowed="测试允许解释B",
                prohibited="测试禁止解释B",
                source_ids=["SRC-B1", "SRC-B2"],
            ),
        )

    def test_uses_existing_snapshot_loader_and_exact_system_keys(self) -> None:
        with patch(
            "clinical_pipeline.core_knowledge.load_snapshot",
            return_value=self.snapshot,
        ) as existing_loader:
            result = CoreKnowledgeProvider().provide(
                ["metric_b", "metric_a", "metric_b"]
            )

        existing_loader.assert_called_once_with()
        self.assertEqual(result.version, "existing-core-v0.2")
        self.assertEqual(
            [entry.system_key for entry in result.entries],
            ["metric_b", "metric_a"],
        )
        self.assertEqual(result.entries[0].knowledge_id, "KB-B")
        self.assertEqual(result.entries[0].allowed_interpretation, "测试允许解释B")
        self.assertEqual(result.entries[0].prohibited_interpretation, "测试禁止解释B")
        self.assertEqual(result.entries[0].source_ids, ["SRC-B1", "SRC-B2"])

    def test_output_does_not_copy_other_medical_knowledge_fields(self) -> None:
        result = CoreKnowledgeProvider(lambda: self.snapshot).provide(["metric_a"])
        dumped = result.entries[0].model_dump()
        self.assertEqual(
            set(dumped),
            {
                "knowledge_id",
                "system_key",
                "allowed_interpretation",
                "prohibited_interpretation",
                "source_ids",
            },
        )
        self.assertNotIn("content", dumped)
        self.assertNotIn("title", dumped)

    def test_missing_system_key_is_an_explicit_error(self) -> None:
        with self.assertRaisesRegex(
            CoreKnowledgeLoadError,
            "固定核心知识缺少system_key：metric_missing",
        ):
            CoreKnowledgeProvider(lambda: self.snapshot).provide(["metric_missing"])

    def test_snapshot_failure_is_an_explicit_error(self) -> None:
        def unavailable() -> KnowledgeSnapshot:
            raise KnowledgeUnavailable("缺少知识发布文件：entries.jsonl")

        with self.assertRaisesRegex(
            CoreKnowledgeLoadError,
            "固定核心知识加载失败：缺少知识发布文件",
        ):
            CoreKnowledgeProvider(unavailable).provide(["metric_a"])

    def test_empty_or_string_system_keys_are_rejected_before_loading(self) -> None:
        loader_called = False

        def loader() -> KnowledgeSnapshot:
            nonlocal loader_called
            loader_called = True
            return self.snapshot

        provider = CoreKnowledgeProvider(loader)
        with self.assertRaisesRegex(CoreKnowledgeLoadError, "至少需要一个system_key"):
            provider.provide([])
        with self.assertRaisesRegex(CoreKnowledgeLoadError, "必须是键列表"):
            provider.provide("metric_a")  # type: ignore[arg-type]
        self.assertFalse(loader_called)

    def test_malformed_existing_entry_is_an_explicit_error(self) -> None:
        malformed = _entry(
            "metric_a",
            knowledge_id="KB-A",
            allowed="",
            prohibited="测试禁止解释A",
            source_ids=["SRC-A"],
        )
        with self.assertRaisesRegex(
            CoreKnowledgeLoadError,
            "固定核心知识格式错误：metric_a",
        ):
            CoreKnowledgeProvider(lambda: _snapshot(malformed)).provide(["metric_a"])


if __name__ == "__main__":
    unittest.main()
