"""Cloud-GPU LLM inference service for rehab report generation.

Runs ONLY on a CUDA GPU machine (e.g. an AutoDL/seetacloud instance). Loads the
QLoRA-fine-tuned Yi-1.5-6B-Chat (base ``01-ai/Yi-1.5-6B-Chat`` + the LoRA adapter
in ``checkpoints_llm/yi15_6b/``) once at startup and exposes report-generation
endpoints that the local backend (``backend/report.py`` in *remote* mode) calls
over the public network.

The local Mac backend handles the 4 DL predictions on CPU and only delegates the
heavy report generation here.

Endpoints:
  POST /generate_messages  full-text generation from a raw chat-message list —
                           used by the structured clinical-reasoning report
                           (returns {"text": "...full..."}); the local backend
                           builds the prompt + parses the JSON
  GET  /health             model load status

Reuses the same loading plumbing as the local path so both modes produce
identical output:
  - model load   → ``backend.report.ReportModel`` (wraps llm.generate._load_model)
  - decoding     → ``backend.report._decoding_kwargs``
  - tag cleanup  → ``backend.report._strip_trailing_chat_tags``

Run on the GPU box. AutoDL "自定义服务" maps the instance's internal port 6006
to a public https://...:8443 address, so bind 0.0.0.0:6006:
    LLM_ADAPTER_DIR=./checkpoints_llm/yi15_6b LLM_LOAD_4BIT=1 \
        uvicorn llm_server:app --host 0.0.0.0 --port 6006

Then put the AutoDL custom-service URL (https://...:8443) into the local
backend's LLM_REMOTE_URL.

Environment knobs: same LLM_* as backend/.env (LLM_ADAPTER_DIR, LLM_MODEL_ID,
LLM_BASE_ID, LLM_LOAD_4BIT, LLM_MAX_NEW_TOKENS, LLM_NUM_BEAMS,
LLM_REPETITION_PENALTY). Listen port: LLM_SERVER_PORT (default 6006).
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Make both the project root and backend/ importable so we can reuse           #
# backend/report.py (which itself imports the `llm` package + `schemas`).      #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import report  # noqa: E402  (backend/report.py — reused loader + helpers)


class MessagesRequest(BaseModel):
    """Raw chat messages for the structured clinical-reasoning report.

    The local backend builds the full clinical-reasoning prompt (it owns the
    report skeleton + biomarkers) and sends only the chat messages here, so the
    GPU box needs no report_builder/biomarkers code.
    """
    messages: list


REPORT_MODEL = report.ReportModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    use_adapter = os.environ.get("LLM_USE_ADAPTER", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if use_adapter:
        print(f"[llm_server] loading Yi-1.5-6B + LoRA adapter from {REPORT_MODEL.adapter_dir}...")
    else:
        print("[llm_server] loading un-fine-tuned Yi-1.5-6B-Chat base (no adapter)...")
    REPORT_MODEL.load()  # fail loud at startup if GPU / deps (/ adapter) missing
    print("[llm_server] model loaded, ready to serve /generate_messages")
    yield


app = FastAPI(title="Rehab Report LLM Service", lifespan=lifespan)


def _generate_messages_text(messages: list) -> str:
    """Run greedy generation on a raw chat-message list, return the full text."""
    import torch

    rm = REPORT_MODEL
    assert rm.model is not None and rm.tok is not None
    tok, model = rm.tok, rm.model
    device = next(model.parameters()).device

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    eos = rm.eos_ids
    gen_kwargs = dict(
        **inputs,
        **report._decoding_kwargs(),
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=(eos if len(eos) > 1 else (eos[0] if eos else None)),
    )
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return report._strip_trailing_chat_tags(text)


@app.post("/generate_messages")
def generate_messages(req: MessagesRequest) -> Dict[str, Any]:
    """Generate full text from raw chat messages (used by the clinical-reasoning
    report path; returns ``{"text": "..."}`` or ``{"error": "..."}``)."""
    try:
        return {"text": _generate_messages_text(req.messages)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.get("/health")
def health() -> Dict[str, Any]:
    rm = REPORT_MODEL
    device = None
    if rm.model is not None:
        try:
            device = str(next(rm.model.parameters()).device)
        except Exception:  # noqa: BLE001
            device = "unknown"
    return {
        "status": "ok" if rm.loaded else "loading",
        "loaded": rm.loaded,
        "adapter_dir": str(rm.adapter_dir),
        "model_id": rm.model_id,
        "device": device,
    }


if __name__ == "__main__":
    import os

    import uvicorn

    # AutoDL "自定义服务" maps the instance's internal port 6006 to the public
    # https://...:8443 address, so bind 0.0.0.0:6006 by default.
    port = int(os.environ.get("LLM_SERVER_PORT", "6006"))
    uvicorn.run("llm_server:app", host="0.0.0.0", port=port, reload=False)
