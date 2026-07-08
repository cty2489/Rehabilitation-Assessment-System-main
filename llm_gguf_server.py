"""Local GGUF LLM service for rehab report generation.

This service is intentionally API-compatible with ``llm_server.py``'s
``POST /generate_messages`` endpoint, so the backend can use it through the
existing ``LLM_PROVIDER=remote`` path:

    LLM_REMOTE_URL=http://127.0.0.1:6008

It loads a split GGUF model with llama-cpp-python. Pass the first split file
(``...-00001-of-00002.gguf``); llama.cpp discovers the remaining split files in
the same directory.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _add_dll_path(path: Path) -> None:
    """Make a DLL folder visible to ctypes on Windows.

    The CUDA llama-cpp-python wheel depends on CUDA DLLs. In this project those
    are already present via the CUDA PyTorch wheel under ``torch/lib``.
    """
    if not path.exists():
        return
    path_s = str(path)
    os.environ["PATH"] = f"{path_s};{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path_s)
        except OSError:
            pass


def _prepare_windows_dll_paths() -> None:
    site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    _add_dll_path(site_packages / "torch" / "lib")
    _add_dll_path(site_packages / "llama_cpp" / "lib")
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        _add_dll_path(Path(cuda_path) / "bin")


def _default_model_path() -> str:
    candidate = (
        Path.home()
        / "Downloads"
        / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    )
    return str(candidate)


def _resolve_model_path() -> Path:
    raw = os.environ.get("LLM_GGUF_MODEL_PATH", "").strip() or _default_model_path()
    model_path = Path(raw).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model file not found: {model_path}. "
            "Set LLM_GGUF_MODEL_PATH to the first split GGUF file."
        )
    sibling = str(model_path).replace("-00001-of-00002.gguf", "-00002-of-00002.gguf")
    if sibling != str(model_path) and not Path(sibling).exists():
        raise FileNotFoundError(
            f"Missing second GGUF split: {sibling}. Keep both split files together "
            "and do not rename them."
        )
    return model_path


class MessagesRequest(BaseModel):
    messages: list[Dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


MODEL = None
MODEL_PATH: Optional[Path] = None
GEN_LOCK = threading.Lock()


def _load_model():
    global MODEL, MODEL_PATH

    _prepare_windows_dll_paths()
    from llama_cpp import Llama

    MODEL_PATH = _resolve_model_path()
    n_ctx = _env_int("LLM_GGUF_N_CTX", 12288)
    n_gpu_layers = _env_int("LLM_GGUF_N_GPU_LAYERS", -1)
    n_threads = _env_int("LLM_GGUF_N_THREADS", max(1, (os.cpu_count() or 8) - 2))
    n_batch = _env_int("LLM_GGUF_N_BATCH", 512)
    n_ubatch = _env_int("LLM_GGUF_N_UBATCH", 512)
    verbose = _env_flag("LLM_GGUF_VERBOSE", False)

    print(
        "[gguf_server] loading model "
        f"path={MODEL_PATH} n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} "
        f"n_threads={n_threads} n_batch={n_batch} n_ubatch={n_ubatch}"
    )
    MODEL = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        offload_kqv=True,
        use_mmap=True,
        verbose=verbose,
    )
    print("[gguf_server] model loaded, ready to serve /generate_messages")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Rehab Report GGUF LLM Service", lifespan=lifespan)


def _default_max_tokens() -> int:
    return _env_int("LLM_GGUF_MAX_TOKENS", 4096)


def _default_temperature() -> float:
    return _env_float("LLM_GGUF_TEMPERATURE", 0.0)


def _default_top_p() -> float:
    return _env_float("LLM_GGUF_TOP_P", 0.9)


def _default_repeat_penalty() -> float:
    return _env_float("LLM_GGUF_REPEAT_PENALTY", 1.05)


def _clean_text(text: str) -> str:
    for marker in ("<|im_end|>", "<|endoftext|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


@app.post("/generate_messages")
def generate_messages(req: MessagesRequest) -> Dict[str, Any]:
    """Generate text from raw chat messages.

    This matches the existing backend remote contract: return {"text": "..."}.
    On failure return {"error": "..."} so backend/report.py can surface it.
    """
    if MODEL is None:
        return {"error": "model is not loaded"}
    try:
        with GEN_LOCK:
            out = MODEL.create_chat_completion(
                messages=req.messages,
                max_tokens=req.max_tokens or _default_max_tokens(),
                temperature=(
                    _default_temperature()
                    if req.temperature is None
                    else float(req.temperature)
                ),
                top_p=_default_top_p() if req.top_p is None else float(req.top_p),
                repeat_penalty=(
                    _default_repeat_penalty()
                    if req.repeat_penalty is None
                    else float(req.repeat_penalty)
                ),
                stop=["<|im_end|>"],
            )
        text = out["choices"][0]["message"]["content"]
        return {"text": _clean_text(str(text))}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.post("/generate")
def generate(req: PromptRequest) -> Dict[str, Any]:
    """Small smoke-test endpoint for plain prompts."""
    return generate_messages(
        MessagesRequest(
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        import llama_cpp

        llama_cpp_version = llama_cpp.__version__
    except Exception:  # noqa: BLE001
        llama_cpp_version = "unknown"
    return {
        "status": "ok" if MODEL is not None else "loading",
        "loaded": MODEL is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "llama_cpp_python": llama_cpp_version,
        "n_ctx": _env_int("LLM_GGUF_N_CTX", 12288),
        "n_gpu_layers": _env_int("LLM_GGUF_N_GPU_LAYERS", -1),
        "max_tokens": _default_max_tokens(),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LLM_GGUF_SERVER_PORT", "6008"))
    host = os.environ.get("LLM_GGUF_SERVER_HOST", "127.0.0.1")
    uvicorn.run("llm_gguf_server:app", host=host, port=port, reload=False)
