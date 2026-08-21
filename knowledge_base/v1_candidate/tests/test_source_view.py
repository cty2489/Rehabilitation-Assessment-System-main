import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "knowledge_base" / "v1_candidate" / "scripts"))
from source_view import source_detail  # noqa: E402


@unittest.skipUnless(
    (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").is_file(),
    "受控运行时知识包未随公开代码仓库发布",
)
class SourceViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [json.loads(line) for line in (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_textbook_click_has_pdf_and_page(self):
        chunk = next(c for c in self.chunks if c["metadata"].get("source_type") == "textbook" and c["metadata"].get("page_start"))
        detail = source_detail(chunk["chunk_id"])
        self.assertTrue(detail["source_pdf"])
        self.assertTrue(detail["uid"])
        self.assertTrue(detail["source_file_id"])
        self.assertTrue(detail["source_page_url"])
        self.assertTrue(detail["original_pdf"])
        self.assertIsInstance(detail["page"], int)
        self.assertIn("/pages/", detail["source_view_route"])
        self.assertEqual(detail["evidence_level"], "textbook_reference")

    def test_paper_click_has_page_locator(self):
        chunk = next(c for c in self.chunks if c["metadata"].get("source_type") in {"paper", "guideline"} and c["metadata"].get("page_start"))
        detail = source_detail(chunk["chunk_id"])
        self.assertTrue(detail["source_pdf"])
        self.assertTrue(detail["uid"])
        self.assertTrue(detail["source_file_id"])
        self.assertIsInstance(detail["page"], int)
        self.assertEqual(detail["evidence_level"], "research")


if __name__ == "__main__":
    unittest.main()
