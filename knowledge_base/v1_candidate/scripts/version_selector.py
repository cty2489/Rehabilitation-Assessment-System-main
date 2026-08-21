#!/usr/bin/env python3
"""Small, side-effect-free version selector for RAG adapters and tests."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "knowledge_base" / "rag_version_config.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def resolve_version(version: str | None = None) -> dict:
    config = load_config()
    selected = version or config["rag_version"]
    if selected not in config["versions"]:
        raise ValueError(f"Unsupported rag_version: {selected}")
    record = dict(config["versions"][selected])
    record["version"] = selected
    record["runtime_path"] = str((ROOT / record["runtime_path"]).resolve())
    return record


if __name__ == "__main__":
    print(json.dumps(resolve_version(), ensure_ascii=False, indent=2))
