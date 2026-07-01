"""Full multi-section rehab report: deterministic skeleton + LLM clinical reasoning.

The report mirrors ``大模型评估报告模板示例.docx`` (总体分期 / 生物标志物 /
亚型界定 / 治疗策略 / 手势组合 / 预警 / 下次评估). Division of labour:

  - ``report_builder`` owns the deterministic numeric skeleton (tables, the
    4 indicators, biomarker measured values + per-stage reference ranges, the
    26-gesture candidate space, change-trend vs the previous visit).
  - The LLM (this module) does the *clinical reasoning* over those numbers —
    per-biomarker 解读/治疗建议, subtype, 治疗策略, 手势组合 + 剂量, 预警 — and
    returns a JSON ``clinical`` dict that ``report_builder`` back-fills into the
    fixed skeleton. The LLM cannot alter any measured value (value columns are
    rendered from code). Works WITHOUT fine-tuning (constrained structured
    reasoning); fine-tuned weights simply improve the prose.
  - There is NO rule-engine fallback: if the LLM is unavailable / returns invalid
    text, ``validate_clinical`` raises and the caller surfaces a "大模型不可用"
    error instead of rendering a misleading deterministic template.

LLM provider selection:
  - ``LLM_PROVIDER=deepseek``: call DeepSeek's OpenAI-compatible
    ``/chat/completions`` API directly. This is the simplest no-GPU path.
  - **Remote** (set ``LLM_REMOTE_URL``): the local backend POSTs the chat
    messages to the cloud GPU service ``llm_server.py`` (``/generate_messages``)
    and parses the returned text. No torch/transformers needed locally.
  - **Local** (no ``LLM_REMOTE_URL``): load base+adapter in-process — needs a
    CUDA GPU + transformers/peft/bnb.

Reuses the existing LLM plumbing:
  - prompt          → ``llm.prompts.build_clinical_reasoning_messages``
  - base+adapter    → ``llm.generate._load_model`` / ``_resolve_eos_ids``
  - model config    → ``llm.model_registry.resolve``

Environment knobs (see backend/.env):
  LLM_PROVIDER        "deepseek" | "remote" | "local" (empty = legacy auto mode)
  DEEPSEEK_API_KEY    API key for LLM_PROVIDER=deepseek
  DEEPSEEK_MODEL      DeepSeek model name (default: deepseek-v4-flash)
  DEEPSEEK_BASE_URL   API base URL (default: https://api.deepseek.com)
  LLM_REMOTE_URL      cloud LLM service base URL; empty = local mode
  LLM_REMOTE_TIMEOUT  remote request timeout seconds (default 180)
  LLM_ADAPTER_DIR     LoRA adapter dir   (default <root>/checkpoints_llm/yi15_6b)
  LLM_MODEL_ID        model_registry id  (default yi15_6b)
  LLM_BASE_ID         override base HF id (default: from adapter model_id.txt / registry)
  LLM_LOAD_4BIT       "1"/"0" 4-bit NF4   (default 1)
  LLM_MAX_NEW_TOKENS / LLM_NUM_BEAMS / LLM_REPETITION_PENALTY  decoding knobs
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from schemas import PatientInfo, PredictionResult

import report_builder

# --------------------------------------------------------------------------- #
# Wire up the project root so the `llm` package (relative-import module) can   #
# be imported and its loading / prompt helpers reused.                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints_llm" / "yi15_6b"
DEFAULT_MODEL_ID = "yi15_6b"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# PatientInfo + PredictionResult → (demographics, labels) for llm.prompts.     #
# --------------------------------------------------------------------------- #
def to_demographics(patient: PatientInfo) -> Dict[str, Any]:
    """Adapt the backend PatientInfo to the demographics dict prompts expect."""
    return {
        "gender": patient.sex,            # 男 / 女
        "age": int(patient.age or 0),
        "disease": patient.diagnosis,
        "days_post": int(patient.disease_days or 0),
        "affected_side": patient.paralysis_side,  # 左 / 右 (prompts._SIDE_ZH ok)
    }


def build_clinical_messages(context: Dict[str, Any]):
    """Build the structured clinical-reasoning ChatML messages from a report context.

    ``context`` is ``report_builder.build_context(...)`` augmented with the JSON
    schema hint under ``schema_hint``; the model must reply with a single JSON
    object of clinical text only (numbers stay code-owned).
    """
    from llm.prompts import build_clinical_reasoning_messages  # local import
    import gestures  # local import (backend on sys.path)

    ctx = dict(context)
    ctx["schema_hint"] = report_builder.CLINICAL_SCHEMA_HINT
    # Tell the prompt whether to ask for gestures: only once the clinical team's
    # 26-gesture library is configured (avoids a base model inventing names).
    ctx["gesture_ready"] = gestures.library_ready()
    return build_clinical_reasoning_messages(ctx)


def _parse_clinical_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the single JSON object from the model output."""
    if not text:
        return None
    s = text.strip()
    # Strip accidental code fences.
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
    # Grab the outermost {...}.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Lazy-loaded singleton model holder.                                         #
# --------------------------------------------------------------------------- #
class ReportModel:
    """Holds the base Yi-1.5-6B model + LoRA adapter, loaded once."""

    def __init__(self) -> None:
        self.model = None
        self.tok = None
        self.cfg: Dict[str, Any] = {}
        self.eos_ids: list[int] = []
        self.adapter_dir: Path = Path(
            os.environ.get("LLM_ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR))
        )
        self.model_id: str = os.environ.get("LLM_MODEL_ID", DEFAULT_MODEL_ID)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the report LLM.

        By default (``LLM_USE_ADAPTER=0``) this loads the *un-fine-tuned base*
        ``Yi-1.5-6B-Chat`` — the QLoRA adapter is no longer attached. Set
        ``LLM_USE_ADAPTER=1`` to restore the old behaviour (base + LoRA adapter).
        Raises a clear RuntimeError if it can't.
        """
        use_adapter = _env_flag("LLM_USE_ADAPTER", False)

        # 1) When attaching the adapter, its directory must exist. Done before
        #    any heavy import so a misconfigured path gives a clear message
        #    rather than a confusing torch/transformers ImportError. In base-only
        #    mode the adapter dir is irrelevant, so we skip this check.
        if use_adapter and not self.adapter_dir.exists():
            raise RuntimeError(
                f"LoRA adapter 目录不存在：{self.adapter_dir}。"
                f"请将 yi15_6b adapter 放到该路径，或设置环境变量 LLM_ADAPTER_DIR；"
                f"或设置 LLM_USE_ADAPTER=0 使用未微调基座模型。"
            )

        # 2) Resolve base HF id + per-model config (extra_eos_tokens,
        #    trust_remote_code). model_registry is stdlib-only → safe to import.
        from llm.model_registry import resolve

        base_override = os.environ.get("LLM_BASE_ID")
        if base_override:
            _, cfg = resolve(base_override)
            hf_id = cfg["hf_id"]
        elif use_adapter:
            pin = self.adapter_dir / "model_id.txt"
            key = pin.read_text(encoding="utf-8").strip() if pin.exists() else self.model_id
            _, cfg = resolve(key)
            hf_id = cfg["hf_id"]
        else:
            # Base-only: resolve directly from the configured model_id (yi15_6b).
            _, cfg = resolve(self.model_id)
            hf_id = cfg["hf_id"]

        # 3) Heavy import + load. llm.generate pulls in torch/transformers/peft;
        #    on a non-CUDA host or without these deps this is where it fails, so
        #    wrap it with a clear, actionable error.
        load_4bit = _env_flag("LLM_LOAD_4BIT", True)
        adapter_arg = self.adapter_dir if use_adapter else None
        try:
            from llm.generate import _load_model, _resolve_eos_ids

            model, tok = _load_model(
                hf_id,
                adapter_arg,
                load_4bit=load_4bit,
                bf16=True,
                trust_remote_code=cfg["trust_remote_code"],
            )
            eos_ids = _resolve_eos_ids(tok, cfg)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"加载康复报告大模型失败（base={hf_id}, "
                f"adapter={adapter_arg or '（未微调基座）'}, 4bit={load_4bit}）："
                f"{exc}. 该模型需要 CUDA GPU 与 transformers/peft/bitsandbytes 依赖。"
            ) from exc

        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.eos_ids = eos_ids
        print(
            f"[startup] report LLM loaded: base={hf_id} "
            f"adapter={adapter_arg or '（未微调基座）'} "
            f"4bit={load_4bit} eos_ids={self.eos_ids}"
        )

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if not self.loaded:
                self.load()


# Module-level singleton; main.py calls .load() at startup, but generation also
# lazily ensures it's loaded so the module is usable standalone.
REPORT_MODEL = ReportModel()


# --------------------------------------------------------------------------- #
# Streaming generation                                                        #
# --------------------------------------------------------------------------- #
def _decoding_kwargs() -> Dict[str, Any]:
    # Clinical-reasoning JSON is larger than the legacy one-paragraph report, so
    # default to a higher new-token budget (override via LLM_MAX_NEW_TOKENS).
    max_new = int(os.environ.get("LLM_MAX_NEW_TOKENS", "1536"))
    num_beams = int(os.environ.get("LLM_NUM_BEAMS", "1"))
    rep = float(os.environ.get("LLM_REPETITION_PENALTY", "1.05"))
    return {"max_new_tokens": max_new, "num_beams": num_beams, "repetition_penalty": rep}


def llm_provider() -> str:
    """Return the selected report-generation provider.

    Empty ``LLM_PROVIDER`` preserves the original behaviour: use the custom
    remote service when ``LLM_REMOTE_URL`` is set, otherwise use the local
    Yi/LoRA loading path.
    """
    raw = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if raw:
        return raw
    return "remote" if remote_url() else "local"


def remote_url() -> str:
    """Return the configured remote LLM service base URL ("" if local mode)."""
    return os.environ.get("LLM_REMOTE_URL", "").strip().rstrip("/")


# Torch-free copy of llm.generate._strip_trailing_chat_tags so the *remote*
# path (local Mac backend, no torch/transformers) can strip phantom chat
# markers without importing the heavy llm.generate module.
_CHAT_TAG_SUFFIXES = (
    "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|user|>", "<|assistant|>",
    "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</s>", "[INST]", "[/INST]",
)


def _strip_trailing_chat_tags(text: str) -> str:
    cut = len(text)
    for tag in _CHAT_TAG_SUFFIXES:
        idx = text.find(tag)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].rstrip()


def stream_report(
    patient: PatientInfo,
    predictions: PredictionResult,
    q: "queue.Queue[Dict[str, Any]]",
    biomarkers: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, Any]] = None,
    report_model: Optional[ReportModel] = None,
) -> str:
    """Build + stream the full multi-section Chinese rehab report (Markdown).

    Flow: assemble a deterministic numeric skeleton (``report_builder``) → ask
    the LLM for the clinical-reasoning JSON (interpretations / treatment advice /
    subtype / strategy / gesture plan + dosing / warnings) → back-fill that text
    into the skeleton → render Markdown → stream it.

    The LLM call dispatches to the remote cloud-GPU service when
    ``LLM_REMOTE_URL`` is set, otherwise runs the QLoRA model in-process. There is
    no rule-engine fallback: if the LLM is unavailable / returns invalid text,
    ``_reason_clinical`` retries once then raises, and an ``error`` event ("大模型
    不可用…") is emitted instead of a misleading deterministic report.

    Pushes ``step_start``/``report_chunk``/``step_done`` (or ``error``) events
    onto ``q`` and returns the full Markdown string. ``biomarkers`` is the dict
    from ``biomarkers.extract``; ``history`` is the previous assessment (4
    indicators) for the change-trend column.
    """
    q.put({"type": "step_start", "step": "report", "label": "AI 报告生成"})
    try:
        bm = biomarkers or {"stage": int(predictions.hand_function), "groups": [], "flat": {}}
        context = report_builder.build_context(patient, predictions, bm, history)

        clinical = _reason_clinical(context, q, report_model)

        markdown = report_builder.render_markdown(context, clinical)
        # Stream the assembled report so the frontend's typewriter UX still works.
        _emit_markdown(markdown, q)
        q.put({"type": "step_done", "step": "report"})
        return markdown
    except Exception as exc:  # noqa: BLE001
        q.put({"type": "error", "message": f"AI 报告生成失败：{exc}"})
        raise


def _emit_markdown(markdown: str, q: "queue.Queue[Dict[str, Any]]", chunk: int = 80) -> None:
    """Emit the final Markdown in small chunks as ``report_chunk`` events."""
    for i in range(0, len(markdown), chunk):
        q.put({"type": "report_chunk", "chunk": markdown[i:i + chunk]})


def _reason_clinical(
    context: Dict[str, Any],
    q: "queue.Queue[Dict[str, Any]]",
    report_model: Optional[ReportModel] = None,
) -> Dict[str, Any]:
    """Get + validate the clinical-reasoning JSON from the LLM (remote or local).

    There is NO rule-engine fallback. The LLM owns all clinical text; if its
    output is missing/invalid (parse failure, missing fields, or a subtype whose
    分期 disagrees with the measured Brunnstrom stage — see
    ``report_builder.validate_clinical``), this **retries once** (local path
    re-samples for a different draft) and then **raises** so the caller surfaces
    a "大模型不可用" error instead of a misleading deterministic template.

    Returns the validated raw clinical dict on success.
    """
    provider = llm_provider()
    url = remote_url()
    attempts = 2
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            if provider == "deepseek":
                text = _reason_deepseek(context)
            elif provider == "remote":
                if not url:
                    raise RuntimeError("LLM_PROVIDER=remote 但 LLM_REMOTE_URL 为空")
                text = _reason_remote(url, context)
            elif provider == "local":
                text = _reason_local(context, report_model, sample=(i > 0))
            else:
                raise RuntimeError(
                    f"未知 LLM_PROVIDER={provider!r}，请使用 deepseek / remote / local"
                )
            clinical = _parse_clinical_json(text)
            report_builder.validate_clinical(context, clinical)  # raises if invalid
            return clinical  # type: ignore[return-value]  # validated non-None
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if i + 1 < attempts:
                q.put({"type": "step_detail", "step": "report",
                       "detail": f"大模型输出不合规（{exc}），正在重试…"})
    raise RuntimeError(f"大模型不可用或未返回有效的个体化报告：{last_err}")


def _remote_timeout() -> "Any":
    """httpx timeout: generous read budget (report can take tens of seconds)."""
    import httpx  # light dep, local backend only

    total = float(os.environ.get("LLM_REMOTE_TIMEOUT", "180"))
    return httpx.Timeout(total, connect=15.0)


def _deepseek_timeout() -> "Any":
    """httpx timeout for DeepSeek chat completions."""
    import httpx

    total = float(os.environ.get("DEEPSEEK_TIMEOUT", os.environ.get("LLM_REMOTE_TIMEOUT", "180")))
    return httpx.Timeout(total, connect=15.0)


def _deepseek_model() -> str:
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"


def _deepseek_base_url() -> str:
    return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")


def _reason_deepseek(context: Dict[str, Any]) -> str:
    """Call DeepSeek's OpenAI-compatible chat completions API.

    The rest of the report pipeline still owns prompt construction, JSON parsing,
    and schema validation. DeepSeek only supplies the clinical JSON text.
    """
    import httpx

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 为空，无法调用 DeepSeek API")

    messages = build_clinical_messages(context)
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", os.environ.get("LLM_MAX_NEW_TOKENS", "1536")))
    temperature = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0"))
    payload: Dict[str, Any] = {
        "model": _deepseek_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    # DeepSeek reasoner-style models may expose a thinking mode; keep it off for
    # this endpoint because downstream parsing expects a single JSON object.
    if _env_flag("DEEPSEEK_DISABLE_THINKING", True):
        payload["thinking"] = {"type": "disabled"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_deepseek_timeout()) as client:
        resp = client.post(
            f"{_deepseek_base_url()}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"DeepSeek 返回格式异常：{data}") from exc
    return _strip_trailing_chat_tags(str(text))


def _reason_remote(base_url: str, context: Dict[str, Any]) -> str:
    """POST the clinical-reasoning chat messages to the cloud-GPU service.

    Uses ``/generate_messages`` (full text, non-streaming — we need the whole
    JSON to parse it). Returns the raw model text.
    """
    import httpx  # light dep, local backend only

    messages = build_clinical_messages(context)
    with httpx.Client(timeout=_remote_timeout()) as client:
        resp = client.post(f"{base_url}/generate_messages", json={"messages": messages})
        resp.raise_for_status()
        data = resp.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return _strip_trailing_chat_tags(str(data.get("text", "")))


def _reason_local(
    context: Dict[str, Any],
    report_model: Optional[ReportModel] = None,
    sample: bool = False,
) -> str:
    """Run the QLoRA model in-process to get the clinical-reasoning text (GPU).

    ``sample=True`` (used on the retry) switches greedy decoding to low-temp
    sampling so the second attempt produces a different draft.
    """
    import torch  # local import: heavy dep

    rm = report_model or REPORT_MODEL
    rm.ensure_loaded()
    assert rm.model is not None and rm.tok is not None
    tok, model = rm.tok, rm.model
    device = next(model.parameters()).device

    messages = build_clinical_messages(context)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    eos = rm.eos_ids
    sampling = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample \
        else {"do_sample": False}
    gen_kwargs = dict(
        **inputs,
        **_decoding_kwargs(),
        **sampling,
        pad_token_id=tok.pad_token_id,
        eos_token_id=(eos if len(eos) > 1 else (eos[0] if eos else None)),
    )
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    # Decode only the newly generated tokens (skip the prompt).
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return _strip_trailing_chat_tags(text)
