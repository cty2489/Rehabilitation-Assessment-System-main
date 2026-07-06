"""Per-model fine-tuning config for the LLM comparison pipeline.

Each entry maps a short ``model_id`` (used in checkpoint / output paths)
to everything that differs across model families:

  - ``hf_id``               HuggingFace repo id (or local path) for the base model
  - ``response_template``   String marker for ``DataCollatorForCompletionOnlyLM``
                            so loss is computed only on the assistant turn.
                            MUST appear verbatim once in the tokenized prompt
                            produced by ``tokenizer.apply_chat_template(...,
                            add_generation_prompt=False)``.
  - ``target_modules``      LoRA injection points. Llama-style models share
                            q/k/v/o_proj + gate/up/down_proj (Qwen2.5,
                            Mistral-7B, Yi-1.5, and GLM-4-0414 which uses
                            ``Glm4ForCausalLM`` since transformers 4.52);
                            the older ChatGLM-based GLM-4-Chat used a fused
                            ``query_key_value`` linear.
  - ``max_seq_length``      Per-model default. GLM-4-9B is capped at 768 to
                            fit on a 24 GB RTX 4090D under QLoRA 4-bit.
  - ``trust_remote_code``   Required by GLM-4 (custom modeling).
  - ``extra_eos_tokens``    Optional list of additional token strings that
                            should terminate generation. Needed when the
                            tokenizer's ``eos_token`` does not match the
                            chat-template's turn boundary (e.g. Yi-1.5's
                            ``<|im_end|>``), otherwise generate.py runs to
                            ``max_new_tokens`` and fabricates a next turn.

The report settings page uses the same short ids for baseline selection, so
adding a candidate here keeps training, local inference, and the web settings
page aligned.
"""
from __future__ import annotations

from typing import Tuple

# Llama-style projection names shared by Qwen2.5, Mistral, Yi-1.5, GLM-4-0414.
_LLAMA_STYLE_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_BAICHUAN2_TARGETS = [
    "W_pack", "o_proj", "gate_proj", "up_proj", "down_proj",
]

_BAICHUAN2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}<reserved_106>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}<reserved_107>{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<reserved_107>{% endif %}"
)


MODEL_REGISTRY: dict[str, dict] = {
    "qwen25_3b": {
        # Pre-quantized bnb-NF4 variant (~1.8 GB) — keeps tokenizer & module
        # names identical to Qwen/Qwen2.5-3B-Instruct so response_template
        # and target_modules below remain valid, but cuts HF cache footprint
        # to fit the 25 GB cloud disk budget.
        "hf_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        "extra_eos_tokens": [],
    },
    "qwen25_7b": {
        # 32K-context, ChatML (<|im_start|>) with native system-role support, so
        # it slots into the existing report prompt (system/user/assistant/user)
        # with no special casing. Used by the clinical-reasoning report path
        # because Yi-1.5-6B-Chat's 4K window cannot hold the ~6.9K-token prompt
        # (26 biomarkers + 26-gesture library) plus the JSON output.
        # Official fp16 repo + load-time bitsandbytes 4-bit quant — mirrors the
        # proven Yi-1.5-6B load path (avoids double-quantizing a pre-quant repo).
        # Disk-tight alternative: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" with
        # LLM_LOAD_4BIT=0 (weights already 4-bit).
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,          # training-only; inference context is 32K
        "trust_remote_code": True,
        "extra_eos_tokens": ["<|im_end|>"],  # stop at turn boundary, no phantom turn
    },
    "qwen3_8b": {
        # Main Chinese baseline selected by the current project plan. Qwen3
        # supports Qwen ChatML-style templates; for clinical report JSON we keep
        # thinking disabled at prompt level and stop on <|im_end|>.
        "hf_id": "Qwen/Qwen3-8B",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        "extra_eos_tokens": ["<|im_end|>"],
    },
    "deepseek_r1_distill_qwen7b": {
        # Kept as a report-model candidate for the professor's baseline sweep.
        # It is reasoner-style and can spend output budget on <think>; use a
        # stricter prompt / post-processing before treating it as final clinical
        # prose. Architecture follows Qwen/Llama-style projection names.
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "response_template": "<｜Assistant｜>",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        "extra_eos_tokens": ["<｜end▁of▁sentence｜>", "<|im_end|>"],
    },
    "baichuan2_7b_chat": {
        # Some Baichuan2 checkpoints do not expose tokenizer.chat_template.
        # apply_tokenizer_overrides injects a minimal user/assistant template so
        # SFT and local report inference fail less mysteriously.
        "hf_id": "baichuan-inc/Baichuan2-7B-Chat",
        "response_template": "<reserved_107>",
        "target_modules": list(_BAICHUAN2_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        "extra_eos_tokens": ["</s>"],
        "chat_template": _BAICHUAN2_CHAT_TEMPLATE,
    },
    "mistral7b_v03": {
        # Replaces deepseek_r1_distill_qwen_7b: R1-Distill is a think-then-answer
        # model that burns its entire max_new_tokens budget on chain-of-thought
        # before emitting the actual reply, which is incompatible with our
        # fixed-句法骨架 SFT setup. Mistral-7B-Instruct-v0.3 is a clean
        # LlamaForCausalLM, Apache 2.0 (no HF gating), and ~5 GB at bnb-NF4.
        # Chat template renders the assistant boundary as `[/INST]` (then a
        # space, then the response, terminated with </s>). The SentencePiece
        # boundary at `[/INST]` is handled by train_lora.py's id-sequence
        # search the same way Yi's <|im_start|>assistant\n marker is.
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "response_template": "[/INST]",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": [],
    },
    "glm4_9b": {
        # Pre-quantized bnb-NF4 variant (~5 GB) of GLM-4-9B-0414, which uses
        # the Llama-style ``Glm4ForCausalLM`` architecture introduced in
        # transformers 4.52 — NOT the older fused-QKV ChatGLM modeling used
        # by GLM-4-9B-Chat. target_modules / response_template below reflect
        # the 0414 arch; if you swap back to a Chat-based GLM-4 you must
        # restore ``query_key_value`` etc.
        "hf_id": "unsloth/GLM-4-9B-0414-bnb-4bit",
        "response_template": "<|assistant|>\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 768,
        "trust_remote_code": True,
        "extra_eos_tokens": [],
    },
    "llama3_8b_instruct": {
        # Requires accepted Meta license and a Hugging Face token if the weights
        # are not already present locally. Uses Llama-3 turn tokens.
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": ["<|eot_id|>"],
    },
    "yi15_6b": {
        # Standard LlamaForCausalLM under the hood and a ChatML-style
        # chat_template (same "<|im_start|>assistant\n" marker as Qwen2.5),
        # so it slots into the existing LoRA / response-template plumbing
        # with no special casing. Replaces Baichuan2-7B-Chat, whose
        # tokenizer ships without a chat_template and breaks
        # apply_chat_template at generation time.
        # NOTE: Unsloth maintains a bnb-4bit pre-quant of the *base*
        # Yi-1.5-6B but NOT of the Chat variant, so we use the official
        # 01-ai fp16 repo (~12 GB cache) and let bitsandbytes 4-bit-quantize
        # at load time. Pair with DEEP_CLEAN=1 on tight 25 GB cloud disks.
        # Yi-1.5-Chat's tokenizer eos_token is NOT <|im_end|>, so add it as
        # an extra stop token for generate.py — otherwise the model runs to
        # max_new_tokens and fabricates a phantom next user/assistant turn.
        "hf_id": "01-ai/Yi-1.5-6B-Chat",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": ["<|im_end|>"],
    },
}


def apply_tokenizer_overrides(tok, cfg: dict) -> None:
    """Apply registry-provided tokenizer fixes before chat-template use."""
    chat_template = cfg.get("chat_template")
    if chat_template and not getattr(tok, "chat_template", None):
        tok.chat_template = chat_template


def list_model_ids() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def resolve(model_id_or_hf: str) -> Tuple[str, dict]:
    """Resolve either a short model_id or a raw HF id to (model_id, config).

    If ``model_id_or_hf`` is a known short id, return its config directly.
    If it matches a registered ``hf_id``, return the corresponding short id
    and config. Otherwise raise KeyError listing the valid ids.
    """
    if model_id_or_hf in MODEL_REGISTRY:
        return model_id_or_hf, MODEL_REGISTRY[model_id_or_hf]
    for mid, cfg in MODEL_REGISTRY.items():
        if cfg["hf_id"] == model_id_or_hf:
            return mid, cfg
    raise KeyError(
        f"Unknown model: {model_id_or_hf!r}. "
        f"Known short ids: {list(MODEL_REGISTRY)}"
    )
