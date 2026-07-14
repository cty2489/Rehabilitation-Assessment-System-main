#!/usr/bin/env python3
"""Download only the dense SentenceTransformer files for BAAI/bge-m3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


_ALLOW_PATTERNS = [
    "1_Pooling/*",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the BGE-M3 dense embedding files")
    parser.add_argument("--repo-id", default="BAAI/bge-m3")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/rag_models/BAAI/bge-m3")
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    output = Path(args.output_dir)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=output,
        allow_patterns=_ALLOW_PATTERNS,
    )
    required = ["config.json", "modules.json", "pytorch_model.bin", "tokenizer.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"download incomplete; missing: {', '.join(missing)}")
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "output_dir": str(output),
                "size_bytes": size,
                "dense_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
