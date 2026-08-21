import json
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "knowledge_base" / "v1_candidate" / "scripts"))
from version_selector import resolve_version  # noqa: E402


class CandidateContractTest(unittest.TestCase):
    def test_default_points_to_current_candidate_release(self):
        selected = resolve_version()
        self.assertEqual(selected["version"], "v1_candidate")
        self.assertFalse(selected["mutable"])

    def test_candidate_alias_resolves_to_immutable_runtime_release(self):
        selected = resolve_version("v1_candidate")
        self.assertEqual(selected["collection"], "rehab_knowledge_v1_candidate_next")
        self.assertFalse(selected["mutable"])

    def test_manifest_contract(self):
        manifest = json.loads((ROOT / "knowledge_base/v1_candidate/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["knowledge_base"], "rehab_knowledge_v1")
        self.assertEqual(manifest["release"], "v1_candidate_next")
        self.assertEqual(manifest["architecture"]["embedding_dimensions"], 1024)
        self.assertEqual(manifest["architecture"]["distance"], "Cosine")
        self.assertTrue(manifest["ingestion_policy"]["raw_ocr_overwritten"] is False)

    @unittest.skipUnless(
        (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").is_file(),
        "受控运行时知识包未随公开代码仓库发布",
    )
    def test_textbook_metadata_and_source_routes(self):
        chunks = [json.loads(line) for line in (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        textbook = [c for c in chunks if c["metadata"].get("source_type") == "textbook"]
        self.assertTrue(textbook)
        for chunk in chunks:
            meta = chunk["metadata"]
            for field in ("uid", "source_file", "page_start", "page_end", "source_type", "evidence_level", "knowledge_role"):
                self.assertIn(field, meta)
                self.assertTrue(meta[field] or field in {"page_start", "page_end"})
            self.assertTrue(meta["uid"])
            if meta["source_type"] == "textbook":
                self.assertEqual(meta["evidence_level"], "textbook_reference")
                self.assertIn("page_start", meta)
                self.assertIn("source_pdf", meta)
                self.assertIn("/pages/", meta["source_view_route"])

    @unittest.skipUnless(
        (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").is_file(),
        "受控运行时知识包未随公开代码仓库发布",
    )
    def test_uid_is_explicit_for_every_chunk(self):
        chunks = [json.loads(line) for line in (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(chunks)
        for chunk in chunks:
            meta = chunk["metadata"]
            self.assertTrue(meta.get("uid"), chunk["chunk_id"])
            self.assertTrue(meta.get("source_file"), chunk["chunk_id"])


if __name__ == "__main__":
    unittest.main()
