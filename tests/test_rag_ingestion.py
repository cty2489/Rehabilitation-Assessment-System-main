from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rag.ingest.docx_parser import DocxBlock, parse_docx
from rag.ingest.pipeline import parse_entries, prepare_knowledge_base


def _write_minimal_docx(path: Path) -> None:
    document = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>条目1：测试知识</w:t></w:r></w:p>
    <w:p><w:r><w:t>· 知识标题：测试知识</w:t></w:r></w:p>
    <w:p><w:r><w:t>· 知识类别：测试分类</w:t></w:r></w:p>
    <w:p><w:r><w:t>· 正文内容：</w:t></w:r></w:p>
    <w:p><w:r><w:t>这是测试正文。</w:t></w:r></w:p>
    <w:p><w:r><w:t>· 关键术语：测试、正文</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


class RagIngestionTests(unittest.TestCase):
    def test_parse_entries_tracks_multiline_fields(self) -> None:
        entries = parse_entries(
            [
                DocxBlock("paragraph", "条目1：EMG"),
                DocxBlock("paragraph", "· 正文内容："),
                DocxBlock("paragraph", "第一段"),
                DocxBlock("paragraph", "第二段"),
                DocxBlock("paragraph", "· 关键术语：RMS、MDF"),
            ]
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["fields"]["content"], "第一段\n第二段")
        self.assertEqual(entries[0]["fields"]["keywords"], "RMS、MDF")

    def test_prepare_marks_unreviewed_entry_demo_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            source = tmp_path / "source.docx"
            _write_minimal_docx(source)
            config = tmp_path / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "collection_id": "demo",
                        "document_id": "doc-1",
                        "source_status": "pending",
                        "expert_review_status": "pending",
                        "entry_overrides": {"1": {"knowledge_id": "KB-001"}},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = tmp_path / "out"

            result = prepare_knowledge_base(source, config, output)
            entry = json.loads((output / "entries.jsonl").read_text(encoding="utf-8"))

            self.assertEqual(parse_docx(source).sha256, result["manifest"]["source"]["sha256"])
            self.assertTrue(entry["status"]["demo_ready"])
            self.assertFalse(entry["status"]["clinical_ready"])
            self.assertIn("missing_reference_source", entry["status"]["issues"])
            self.assertEqual(result["quality_report"]["counts"]["chunks"], 1)


if __name__ == "__main__":
    unittest.main()
