"""Generate rehab assessment text from labels using a fine-tuned LoRA adapter.

Loads the registered base model + LoRA adapter, iterates over a subject
subset (filtered by split + fold + partition, or by ``--subjects``), and
writes a hyp/ref JSON file consumable by evaluate.py.

Example:
    python -m src.llm.generate \
        --model-id qwen25_3b \
        --adapter checkpoints/llm/qwen25_3b/fold1 \
        --suggestions patient_rehab_suggestions_100subjects.json \
        --split splits/3fold_patient_split_llm_100subjects.json \
        --fold 1 --partition test \
        --out outputs/llm/qwen25_3b/fold1_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_builder import (
    _get_fold,
    _load_suggestions,
    _split_val_test,
    normalize_rehab_text,
)
from .model_registry import apply_tokenizer_overrides, resolve
from .prompts import build_chat_messages


def _resolve_subjects(
    suggestions: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> List[str]:
    if args.subjects:
        wanted = [s.strip() for s in args.subjects.split(",") if s.strip()]
        return [sid for sid in wanted if sid in suggestions]

    if args.split is None:
        raise ValueError("Either --subjects or --split must be provided.")
    payload = json.loads(args.split.read_text(encoding="utf-8"))
    fold = _get_fold(payload, args.fold)
    train = list(map(str, fold["train_subjects"]))
    val, test = _split_val_test(list(map(str, fold["val_test_subjects"])))
    if args.partition == "train":
        return train
    if args.partition == "val":
        return val
    if args.partition == "test":
        return test
    if args.partition == "all":
        return sorted(set(train) | set(val) | set(test), key=lambda s: int(s))
    raise ValueError(f"Unknown partition: {args.partition}")


def _load_model(base: str, adapter: Optional[Path], load_4bit: bool, bf16: bool,
                trust_remote_code: bool, tokenizer_use_fast: Optional[bool] = None):
    """Load the base model, optionally attaching a LoRA adapter.

    Pass ``adapter=None`` to use the *un-fine-tuned base* model directly (the
    report path now defaults to the plain ``Yi-1.5-6B-Chat`` base — see
    ``backend/report.py``'s ``LLM_USE_ADAPTER`` switch). When ``adapter`` is a
    path, the LoRA weights are attached as before.
    """
    tok_kwargs = {"trust_remote_code": trust_remote_code}
    if tokenizer_use_fast is not None:
        tok_kwargs["use_fast"] = bool(tokenizer_use_fast)
    tok = AutoTokenizer.from_pretrained(base, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": trust_remote_code}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
        kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter is None:
        base_model.eval()
        return base_model, tok
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.eval()
    return model, tok


def _resolve_base(args: argparse.Namespace) -> tuple[str, dict]:
    """Pick the HF base id + per-model config from --model-id, --base, or
    the ``model_id.txt`` left in the adapter dir by train_lora."""
    if args.model_id:
        _, cfg = resolve(args.model_id)
        return cfg["hf_id"], cfg
    if args.base:
        _, cfg = resolve(args.base)
        return cfg["hf_id"], cfg
    pin = args.adapter / "model_id.txt"
    if pin.exists():
        _, cfg = resolve(pin.read_text(encoding="utf-8").strip())
        return cfg["hf_id"], cfg
    raise SystemExit(
        "Cannot resolve base model. Provide --model-id, --base, or train "
        "the adapter with the updated train_lora.py (writes model_id.txt)."
    )


_CHAT_TAG_SUFFIXES = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "</s>",
    "[INST]",
    "[/INST]",
)


def _resolve_eos_ids(tok, cfg: dict) -> List[int]:
    """Build the eos_token_id list passed to model.generate().

    Starts from the tokenizer's eos_token_id and appends any tokens declared
    in ``cfg["extra_eos_tokens"]`` whose ids resolve to something other than
    the unk token. transformers.generate accepts a list and stops on any
    member firing — this is what lets Yi-1.5 actually stop at <|im_end|>.
    """
    ids: List[int] = []
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    unk_id = tok.unk_token_id
    for name in cfg.get("extra_eos_tokens", []) or []:
        tid = tok.convert_tokens_to_ids(name)
        if tid is None or tid == unk_id:
            continue
        if tid not in ids:
            ids.append(int(tid))
    return ids


def _strip_trailing_chat_tags(text: str) -> str:
    """Defensively cut anything from the first chat-control marker onward.

    skip_special_tokens=True drops registered special tokens but Yi-1.5 et al.
    emit some chat markers as regular text fragments (e.g. when the model
    invents a phantom next turn after running past EOS), and those slip
    through. Trim them so BLEU/ROUGE see only the intended assistant turn.
    """
    cut = len(text)
    for tag in _CHAT_TAG_SUFFIXES:
        idx = text.find(tag)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].rstrip()


def _generate_one(
    model,
    tok,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    eos_ids: List[int],
) -> str:
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
    )
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False, num_beams=args.num_beams)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return _strip_trailing_chat_tags(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rehab text with LoRA adapter.")
    ap.add_argument("--model-id", default=None,
                    help="Short id from model_registry. If omitted, falls back "
                         "to --base, then to adapter/model_id.txt.")
    ap.add_argument("--base", default=None,
                    help="Raw HF id or local path. Optional if --model-id is set "
                         "or the adapter dir contains model_id.txt.")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--suggestions", type=Path,
                    default=Path("patient_rehab_suggestions_100subjects.json"))
    ap.add_argument("--split", type=Path, default=None,
                    help="Split JSON; required unless --subjects is given.")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--partition", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--subjects", default="",
                    help="Optional comma list of subject_ids; overrides --split/--partition.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--sample", action="store_true",
                    help="Use sampling (temperature/top_p) instead of beam search.")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load base model in bf16/fp16 instead of 4-bit.")
    args = ap.parse_args()

    bf16 = not args.fp16
    hf_id, cfg = _resolve_base(args)
    suggestions = _load_suggestions(args.suggestions)
    subjects = _resolve_subjects(suggestions, args)
    if not subjects:
        raise SystemExit("No subjects selected for generation.")
    print(f"[generate] hf_id={hf_id}  n_subjects={len(subjects)}")

    model, tok = _load_model(
        hf_id, args.adapter, load_4bit=not args.no_4bit, bf16=bf16,
        trust_remote_code=cfg["trust_remote_code"],
    )
    apply_tokenizer_overrides(tok, cfg)
    device = next(model.parameters()).device
    eos_ids = _resolve_eos_ids(tok, cfg)
    print(f"[generate] eos_token_ids={eos_ids}")

    rows: List[Dict[str, object]] = []
    for sid in subjects:
        item = suggestions[sid]
        messages = build_chat_messages(
            subject_id=sid,
            demographics=item["demographics"],
            labels=item["labels"],
            rehab_text=None,
        )
        hyp = _generate_one(model, tok, messages, args, device, eos_ids)
        rows.append({
            "subject_id": sid,
            "source": item.get("source", ""),
            "prompt_user": messages[1]["content"],
            "hyp": hyp,
            "ref": normalize_rehab_text(item),
            "labels": item["labels"],
        })
        print(f"  S{sid} ({item.get('source','')}): {hyp[:60]}…")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate] wrote {len(rows)} predictions → {args.out}")


if __name__ == "__main__":
    main()
