#!/usr/bin/env python3
"""Minimal source-file HTTP adapter for the v1 candidate release.

The production backend can mount the same route contract. This standalone
adapter is also useful for an isolated click-through test without touching the
old RAG service.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "knowledge_base" / "v1_candidate"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_records() -> dict[str, dict]:
    return {row["source_file_id"]: row for row in jsonl(RELEASE / "source_file_manifest.jsonl")}


def ocr_images() -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for chunk in jsonl(RELEASE / "chunks.jsonl"):
        meta = chunk["metadata"]
        if meta.get("page_image") and meta.get("source_id") and meta.get("page_start"):
            result[(meta["source_id"], int(meta["page_start"]))] = ROOT / meta["page_image"]
    return result


def safe_file(file_id: str) -> Path | None:
    record = file_records().get(file_id)
    if not record:
        return None
    library_root = (RELEASE / "source_files").resolve()
    candidates = [
        library_root / f"{file_id}.pdf",
        (RELEASE / record["source_file"]).resolve(),
        (ROOT / record["source_file"]).resolve(),
    ]
    for path in candidates:
        if library_root in path.parents and path.is_file():
            return path
    return None


class SourceHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = unquote(urlparse(self.path).path)
        match = re.fullmatch(r"/api/rag/source-files/(FILE-[0-9a-f]+)/pdf", path)
        if match:
            return self._send_file(safe_file(match.group(1)))
        match = re.fullmatch(r"/api/rag/source-files/(FILE-[0-9a-f]+)/pages/([0-9]+)", path)
        if match:
            file_id, page = match.group(1), int(match.group(2))
            record = file_records().get(file_id)
            if not record or not 1 <= page <= (record.get("page_count") or 0):
                return self._send_error(404, "PDF page not found")
            self.send_response(302)
            self.send_header("Location", f"/api/rag/source-files/{file_id}/pdf#page={page}")
            self.end_headers()
            return
        match = re.fullmatch(r"/api/rag/ocr-scan-pages/([^/]+)/([0-9]+)", path)
        if match:
            image = ocr_images().get((match.group(1), int(match.group(2))))
            return self._send_file(image, content_type="image/png")
        return self._send_error(404, "source route not found")

    def _send_file(self, path: Path | None, content_type: str | None = None):
        if path is None or not path.is_file():
            return self._send_error(404, "source file not found")
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        try:
            with path.open("rb") as f:
                while block := f.read(1024 * 1024):
                    self.wfile.write(block)
        except (BrokenPipeError, ConnectionResetError):
            # A browser/test client may close after receiving headers; the
            # source route has still served a valid PDF response.
            return

    def _send_error(self, status: int, message: str):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def main() -> int:
    global RELEASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--release-dir", default=str(RELEASE))
    args = parser.parse_args()
    RELEASE = Path(args.release_dir).resolve()
    server = ThreadingHTTPServer((args.host, args.port), SourceHandler)
    print(f"source file server: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
