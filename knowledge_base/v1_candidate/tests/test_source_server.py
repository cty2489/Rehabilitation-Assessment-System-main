import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "knowledge_base" / "v1_candidate" / "scripts"))
from serve_source_files import SourceHandler  # noqa: E402


@unittest.skipUnless(
    (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").is_file(),
    "受控运行时知识包未随公开代码仓库发布",
)
class SourceServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = [json.loads(line) for line in (ROOT / "knowledge_base/v1_candidate/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.chunk = next(c for c in chunks if c["metadata"].get("source_file_id") and c["metadata"].get("page_number"))
        cls.meta = cls.chunk["metadata"]
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_page_route_redirects_to_pdf_fragment(self):
        url = f"{self.base}{self.meta['source_page_url']}"
        response = urlopen(Request(url, method="GET"))
        self.assertEqual(response.status, 200)
        self.assertIn("application/pdf", response.headers.get("Content-Type", ""))
        self.assertIn(f"#page={self.meta['page_number']}", response.geturl())

    def test_pdf_route_serves_staged_bytes(self):
        url = f"{self.base}{self.meta['source_pdf_url']}"
        response = urlopen(url)
        self.assertEqual(response.status, 200)
        self.assertGreater(int(response.headers["Content-Length"]), 100)


if __name__ == "__main__":
    unittest.main()
