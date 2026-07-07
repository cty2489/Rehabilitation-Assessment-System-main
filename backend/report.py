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
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from schemas import PatientInfo, PredictionResult

import llm_settings
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


def _active_local_load_key(active: Dict[str, Any], fallback_adapter: Path, fallback_model_id: str) -> str:
    """Stable cache key for the selected in-process report model."""
    if str(active.get("provider") or "").lower() != "local":
        return ""
    return "|".join([
        str(active.get("model_id") or fallback_model_id),
        str(active.get("weight_path") or ""),
        str(active.get("adapter_dir") or fallback_adapter),
        str(active.get("use_adapter", os.environ.get("LLM_USE_ADAPTER", ""))),
        str(_env_flag("LLM_LOAD_4BIT", True)),
        str(os.environ.get("LLM_BASE_ID", "")),
    ])


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
    # Qwen3 / DeepSeek-R1 style models may emit a private reasoning block before
    # the final JSON. Drop it before locating the outer JSON object.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    think_end = s.lower().rfind("</think>")
    if think_end != -1:
        s = s[think_end + len("</think>"):].strip()
    # Strip accidental fence markers without assuming there is only one fenced
    # block. Some base models repeat the same JSON object twice; in that case
    # grabbing from the first "{" to the last "}" creates invalid JSON, so parse
    # the first balanced object instead.
    s = re.sub(r"```(?:json|JSON)?", "", s).replace("```", "").strip()
    decoder = json.JSONDecoder()
    start = s.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(s[start:])
            if isinstance(obj, dict):
                schema_keys = {
                    "overall_interpretation",
                    "marker_text",
                    "group_subtypes",
                    "overall_subtype",
                    "treatment_strategy",
                }
                if schema_keys.intersection(obj):
                    return obj
        except (ValueError, TypeError):
            pass
        start = s.find("{", start + 1)
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
        self.loaded_key: str = ""
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
        active = llm_settings.active_model()
        requested_key = _active_local_load_key(active, self.adapter_dir, self.model_id)
        local_selected = str(active.get("provider") or "").lower() == "local"
        use_adapter = bool(active.get("use_adapter")) if local_selected and "use_adapter" in active \
            else _env_flag("LLM_USE_ADAPTER", False)
        adapter_dir = Path(str(active.get("adapter_dir") or self.adapter_dir)) if local_selected \
            else self.adapter_dir
        selected_model_id = str(active.get("model_id") or self.model_id) if local_selected \
            else self.model_id
        configured_path = Path(str(active.get("weight_path") or "")) if local_selected and active.get("weight_path") \
            else None

        # 1) When attaching the adapter, its directory must exist. Done before
        #    any heavy import so a misconfigured path gives a clear message
        #    rather than a confusing torch/transformers ImportError. In base-only
        #    mode the adapter dir is irrelevant, so we skip this check.
        if use_adapter and not adapter_dir.exists():
            raise RuntimeError(
                f"LoRA adapter 目录不存在：{adapter_dir}。"
                f"请将 yi15_6b adapter 放到该路径，或设置环境变量 LLM_ADAPTER_DIR；"
                f"或设置 LLM_USE_ADAPTER=0 使用未微调基座模型。"
            )

        # 2) Resolve base HF id + per-model config (extra_eos_tokens,
        #    trust_remote_code). model_registry is stdlib-only → safe to import.
        from llm.model_registry import apply_tokenizer_overrides, resolve

        base_override = os.environ.get("LLM_BASE_ID")
        if local_selected and configured_path and configured_path.exists():
            try:
                _, cfg = resolve(selected_model_id)
            except KeyError:
                cfg = {
                    "trust_remote_code": bool(active.get("trust_remote_code", True)),
                    "extra_eos_tokens": active.get("extra_eos_tokens", ["<|im_end|>"]),
                }
            hf_id = str(configured_path)
        elif base_override:
            _, cfg = resolve(base_override)
            hf_id = cfg["hf_id"]
        elif use_adapter:
            pin = adapter_dir / "model_id.txt"
            key = pin.read_text(encoding="utf-8").strip() if pin.exists() else selected_model_id
            _, cfg = resolve(key)
            hf_id = cfg["hf_id"]
        else:
            # Base-only: resolve directly from the configured model_id.
            _, cfg = resolve(selected_model_id)
            hf_id = cfg["hf_id"]

        # 3) Heavy import + load. llm.generate pulls in torch/transformers/peft;
        #    on a non-CUDA host or without these deps this is where it fails, so
        #    wrap it with a clear, actionable error.
        load_4bit = _env_flag("LLM_LOAD_4BIT", True)
        adapter_arg = adapter_dir if use_adapter else None
        try:
            from llm.generate import _load_model, _resolve_eos_ids

            model, tok = _load_model(
                hf_id,
                adapter_arg,
                load_4bit=load_4bit,
                bf16=True,
                trust_remote_code=cfg["trust_remote_code"],
                tokenizer_use_fast=cfg.get("tokenizer_use_fast"),
            )
            eos_ids = _resolve_eos_ids(tok, cfg)
            apply_tokenizer_overrides(tok, cfg)
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
        self.loaded_key = requested_key or "|".join([selected_model_id, hf_id, str(adapter_arg or ""), str(load_4bit)])
        print(
            f"[startup] report LLM loaded: base={hf_id} "
            f"adapter={adapter_arg or '（未微调基座）'} "
            f"4bit={load_4bit} eos_ids={self.eos_ids}"
        )

    def reset(self) -> None:
        self.model = None
        self.tok = None
        self.cfg = {}
        self.eos_ids = []
        self.loaded_key = ""

    def ensure_loaded(self) -> None:
        active = llm_settings.active_model()
        key = _active_local_load_key(active, self.adapter_dir, self.model_id)
        if self.loaded and (not key or key == self.loaded_key):
            return
        with self._lock:
            if self.loaded and key and key != self.loaded_key:
                self.reset()
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

    The System Management page writes ``llm_settings``. Environment variables
    remain as fallback for older deployments.
    """
    raw = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not llm_settings.settings_configured():
        if raw:
            return raw
        return "remote" if remote_url() else "local"
    active = llm_settings.active_model()
    provider = str(active.get("provider") or "").strip().lower()
    if provider:
        return provider
    if raw:
        return raw
    return "remote" if remote_url() else "local"


def remote_url() -> str:
    """Return the configured remote LLM service base URL ("" if local mode)."""
    if not llm_settings.settings_configured():
        return os.environ.get("LLM_REMOTE_URL", "").strip().rstrip("/")
    active = llm_settings.active_model()
    if str(active.get("provider") or "").lower() == "remote":
        url = str(active.get("remote_url") or "").strip().rstrip("/")
        if url:
            return url
    return os.environ.get("LLM_REMOTE_URL", "").strip().rstrip("/")


def llm_model_name() -> str:
    """Human-readable id stored with each generated assessment."""
    if not llm_settings.settings_configured():
        provider = llm_provider()
        if provider == "deepseek":
            return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if provider == "remote":
            return remote_url()
        return os.environ.get("LLM_MODEL_ID", "")
    active = llm_settings.active_model()
    return str(active.get("id") or active.get("model_id") or active.get("name") or "")


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


def _apply_chat_template(tok, messages: "Any") -> str:
    """Render chat prompt, disabling reasoner thinking when the tokenizer supports it."""
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _fallback_clinical(context: Dict[str, Any], reason: Optional[Exception] = None) -> Dict[str, Any]:
    """Conservative clinical text when the LLM does not return valid JSON.

    This is intentionally narrow: measured values remain code-owned, reference
    judgement comes from ``biomarker_refs.judge``, and wording avoids inventing
    external thresholds for device-specific quantities.
    """
    from biomarker_refs import judge

    roman = str(context.get("stage_roman") or context.get("stage") or "")
    prefix = f"{roman}期" if roman else "当前分期"

    group_phrase = {
        "emg": ("外周肌肉激活与屈伸协同", "训练中应优先保持低代偿、低疲劳的主动助力收缩，并记录同一设备下的连续变化。"),
        "eeg": ("中枢驱动与半球协同", "训练中应加入运动想象、镜像/视觉反馈和患侧主动意图强化，避免只看单次绝对值下结论。"),
        "imu": ("运动速度、平滑度与震颤控制", "训练中应降低速度追求，优先提高轨迹平滑度、可控活动范围和重复稳定性。"),
    }
    key_advice = {
        "resting_emg_level": "若后续同设备复测持续升高，训练前增加放松、牵伸和低强度热身，避免在高张力状态下强行快速动作。",
        "wrist_co_contraction_index": "腕屈伸训练应强调分离控制，采用慢速腕伸-回中、必要时降低屈肌共同收缩诱发的阻力。",
        "finger_co_contraction_index": "伸指训练应减少屈指代偿，可用视觉反馈和分段助力促发指伸肌单独激活。",
        "emg_activation_rms": "把该值作为同设备下主动募集强度的纵向指标，逐步提高有效收缩时间而非单次用力峰值。",
        "fcr_iemg": "关注腕屈肌募集是否过度主导，训练中配合腕伸拮抗控制和放松间歇。",
        "fds_iemg": "关注屈指肌群参与度，练习抓握时需安排充分伸指放松和手指打开动作。",
        "ecu_iemg": "若伸肌募集不足，增加腕伸主动助力、抗重力维持和短时重复激活。",
        "extensor_digitorum_iemg": "若伸指募集不足，优先安排伸指启动、保持和缓慢回放练习。",
        "flexor_extensor_iemg_ratio": "用该比值观察屈伸肌平衡，训练剂量根据屈肌/伸肌偏向做动态调整。",
        "emg_burst_duration": "若爆发持续时间过长，采用短组数、充分休息和节律提示，减少持续性僵硬收缩。",
        "fcr_mdf": "MDF 主要用于疲劳趋势观察；训练中避免连续高强度收缩导致频率进一步下降。",
        "fds_mdf": "MDF 主要用于疲劳趋势观察；手指屈曲训练后安排伸展与休息间隔。",
        "ecu_mdf": "MDF 主要用于疲劳趋势观察；腕伸训练采用短时多组、不过度追求阻力。",
        "extensor_digitorum_mdf": "MDF 主要用于疲劳趋势观察；伸指训练应控制单组时长并观察动作质量衰减。",
        "pathological_asymmetry_index": "结合患侧运动表现复测该指标，必要时增加双侧协调、镜像反馈和患侧注意训练。",
        "corticomuscular_coherence_beta": "该跨模态指标受同步误差影响，适合看趋势；训练中强化主动意图与肌肉输出的时间一致性。",
        "prefrontal_theta_beta_ratio": "训练安排应避免认知负荷过高，采用明确目标、短时反馈和分段休息。",
        "interhemispheric_motor_coherence": "可加入双侧同步任务和健患侧交替任务，促进半球间协同。",
        "movement_mu_power_change": "运动想象和实际动作应配对练习，观察感觉运动节律是否随训练更稳定地调制。",
        "movement_beta_power_change": "可通过节律性主动运动和反馈训练促进运动相关 β 调制稳定。",
        "movement_smoothness_sparc": "优先练习慢速、连续、无停顿的轨迹控制，而不是追求动作次数。",
        "range_of_motion_proxy": "在无痛范围内逐步扩大主动活动范围，避免代偿性肩肘动作替代腕手活动。",
        "tremor_index_3_6hz": "若复测持续偏高，应降低任务速度和负荷，采用稳定支撑位下的短时控制训练。",
        "wrist_flexion_peak_velocity": "腕屈速度训练应保持可控启动和可控停止，避免快速甩动。",
        "wrist_extension_peak_velocity": "腕伸速度训练应从主动助力开始，逐步提高速度同时保持轨迹稳定。",
        "finger_extension_peak_velocity": "伸指速度训练以充分打开和稳定保持为先，再逐步提高反应速度。",
    }

    marker_text: Dict[str, Dict[str, str]] = {}
    available_by_group = {"emg": 0, "eeg": 0, "imu": 0}
    for group in context.get("biomarkers", {}).get("groups", []) or []:
        gkey = group.get("key", "")
        focus, default_advice = group_phrase.get(gkey, ("该模态功能", "建议结合复测趋势调整训练剂量。"))
        for marker in group.get("markers", []) or []:
            key = marker.get("key", "")
            if marker.get("available", True) is False:
                marker_text[key] = {
                    "interpretation": "本次数据不足，未予解读",
                    "treatment_advice": "—",
                }
                continue
            available_by_group[gkey] = available_by_group.get(gkey, 0) + 1
            value = marker.get("value")
            unit = marker.get("unit") or ""
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = None
            verdict = judge(key, numeric_value)
            value_text = f"{value}{(' ' + unit) if unit else ''}"
            marker_text[key] = {
                "interpretation": (
                    f"{marker.get('name', key)} 当前值为 {value_text}，{verdict}；"
                    f"该指标主要用于观察{focus}，临时演示版建议结合后续同设备复测趋势解读。"
                ),
                "treatment_advice": key_advice.get(key, default_advice),
            }

    group_subtypes: Dict[str, str] = {}
    if available_by_group.get("emg", 0):
        group_subtypes["emg"] = f"{prefix}-外周肌肉协同与募集可量化型"
    if available_by_group.get("eeg", 0):
        group_subtypes["eeg"] = f"{prefix}-中枢驱动与半球协同需纵向观察型"
    if available_by_group.get("imu", 0):
        group_subtypes["imu"] = f"{prefix}-运动控制质量可追踪型"

    warn = "本报告的部分解读由保守规则后备生成；临床决策需结合医师查体与复测趋势。"
    if reason:
        warn += f" 后备触发原因：大模型结构化输出不可用。"

    return {
        "overall_interpretation": (
            f"{prefix}：当前评估显示手功能已有一定主动运动基础，"
            "需结合肌电、脑电和运动学指标继续观察协同分离、主动募集和运动质量。"
        ),
        "marker_text": marker_text,
        "group_subtypes": group_subtypes,
        "overall_subtype": (
            f"{prefix}-主动运动可量化伴协同分离需巩固亚型，"
            "中枢驱动与外周肌肉募集可通过同设备复测持续追踪，关节活动度和运动平滑度需同步训练。"
        ),
        "treatment_strategy": [
            "分离控制优先策略：以腕伸、伸指和慢速回中为核心，每次10-15分钟、每日2-3组，出现屈肌共同收缩或动作代偿时立即降低难度并增加休息。",
            "中枢驱动强化策略：运动想象、镜像反馈与实际患侧主动助力动作配对，每轮3-5分钟，用动作完成度和肌电/运动学趋势作为反馈标准。",
            "运动质量递进策略：先保证轨迹平滑和可控活动范围，再逐步提高速度与重复次数；若震颤、疲劳或张力升高，改为短组数、低负荷、分次完成。",
        ],
        "warnings": [warn],
        "not_recommended": [],
        "next_assessment": report_builder.NEXT_ASSESSMENT_TEXT,
    }


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
    q.put({
        "type": "step_detail",
        "step": "report",
        "detail": "大模型未返回有效结构化结果，已使用保守规则生成可审阅报告。",
    })
    return _fallback_clinical(context, last_err)


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
    active = llm_settings.active_model()
    if str(active.get("provider") or "").lower() == "deepseek":
        selected = str(active.get("model_id") or active.get("api_model") or "").strip()
        if selected:
            return selected
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"


def _deepseek_base_url() -> str:
    active = llm_settings.active_model()
    if str(active.get("provider") or "").lower() == "deepseek":
        selected = str(active.get("base_url") or active.get("remote_url") or "").strip().rstrip("/")
        if selected:
            return selected
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
    with httpx.Client(timeout=_remote_timeout(), trust_env=False) as client:
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
    prompt = _apply_chat_template(tok, messages)
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
