from __future__ import annotations

import unittest

from report_citations import (
    build_reference_catalog,
    extract_numeric_citations,
    render_numeric_citations,
)


class ReportCitationTests(unittest.TestCase):
    def test_numbers_follow_first_body_use_and_deduplicate_sources(self) -> None:
        evidence = {
            "sources": [
                {
                    "knowledge_id": "KB-A",
                    "title": "条目A",
                    "references": ["[SRC-001] 文献A", "[SRC-002] 文献B"],
                },
                {
                    "knowledge_id": "KB-B",
                    "title": "条目B",
                    "references": ["[SRC-002] 文献B", "[SRC-003] 文献C"],
                },
            ]
        }
        catalog = build_reference_catalog(
            evidence,
            ["KB-A", "KB-B"],
            body=["先使用[KB-B]。", "再使用[KB-A]。"],
        )

        self.assertEqual(
            [item["citation"] for item in catalog["references"]],
            ["文献B", "文献C", "文献A"],
        )
        self.assertEqual(
            render_numeric_citations("解释[KB-A]；建议[KB-B]", catalog),
            "解释【1】【3】；建议【1】【2】",
        )

    def test_missing_external_reference_is_labelled_as_knowledge_entry(self) -> None:
        catalog = build_reference_catalog(
            {
                "marker_sources": {
                    "m1": {
                        "knowledge_id": "KB-M1",
                        "title": "内部知识",
                        "source_document_id": "doc-1",
                        "source_entry_number": 7,
                    }
                }
            },
            ["KB-M1"],
        )
        reference = catalog["references"][0]
        self.assertIn("非外部文献", reference["citation"])
        self.assertEqual(reference["source_id"], "doc-1")

    def test_numeric_citations_are_extracted_once_in_order(self) -> None:
        self.assertEqual(extract_numeric_citations("依据【2】【1】，复述【2】"), [2, 1])


if __name__ == "__main__":
    unittest.main()
