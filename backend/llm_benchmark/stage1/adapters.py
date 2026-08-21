"""Unified model routing for Stage 1.

The catalog is read from the existing ``llm_settings`` module. No model list is
duplicated here. The real local adapter reuses the existing low-level
``llm.model_registry``/``llm.generate`` functions; tests inject a mock provider
and never load weights or call a network.
"""

from __future__ import annotations

import gc
import json
import re
import time
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

from .prompt import STAGE1_PROMPT_VERSION

PARAMETER_NAMES = ("temperature", "top_p", "max_tokens")


@dataclass(frozen=True)
class ModelCapabilities:
    temperature: bool
    top_p: bool
    max_tokens: bool
    parameter_mapping: dict[str, str]

    def unsupported(self) -> tuple[str, ...]:
        return tuple(name for name in PARAMETER_NAMES if not getattr(self, name))


@dataclass(frozen=True)
class ExistingModelSpec:
    model_id: str
    provider: str
    display_name: str
    model_ref: str
    weight_path: str
    call_entrypoint: str
    capabilities: ModelCapabilities
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class ModelCallResult:
    model_id: str
    provider: str
    raw_text: str
    response_time_ms: float
    actual_parameters: dict[str, Any]
    unsupported_parameters: tuple[str, ...]
    ignored_parameters: tuple[str, ...] = ()


class ModelAdapterError(RuntimeError):
    def __init__(self, message: str, *, trace: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.trace = trace or {}


class _GenerationPrefillTokenizer:
    """Forward a tokenizer while seeding only the assistant JSON boundary."""

    def __init__(self, tokenizer: Any, prefill: str):
        self._tokenizer = tokenizer
        self._prefill = prefill

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return str(self._tokenizer.apply_chat_template(*args, **kwargs)) + self._prefill

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def _local_capabilities() -> ModelCapabilities:
    # ``llm.generate._generate_one`` accepts these three values. max_tokens is
    # mapped to the existing function's max_new_tokens argument.
    return ModelCapabilities(
        temperature=True,
        top_p=True,
        max_tokens=True,
        parameter_mapping={
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_new_tokens",
        },
    )


def discover_existing_models(settings: Optional[Mapping[str, Any]] = None) -> tuple[ExistingModelSpec, ...]:
    """Read the currently configured six model entries without adding any."""

    if settings is None:
        import llm_settings

        settings = llm_settings.read_settings()
    models = list(settings.get("models") or [])
    specs: list[ExistingModelSpec] = []
    for item in models:
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        provider = str(item.get("provider") or "").strip().lower()
        if provider == "local":
            entrypoint = (
                "Stage1ModelRouter → ExistingModelAdapter → "
                "llm.generate._load_model + llm.generate._generate_one"
            )
            capabilities = _local_capabilities()
        else:
            entrypoint = f"provider={provider}（当前Stage1未发现该provider的既有六模型配置）"
            capabilities = ModelCapabilities(False, False, False, {})
        specs.append(ExistingModelSpec(
            model_id=model_id,
            provider=provider,
            display_name=str(item.get("name") or model_id),
            model_ref=str(item.get("model_id") or model_id),
            weight_path=str(item.get("weight_path") or ""),
            call_entrypoint=entrypoint,
            capabilities=capabilities,
            raw_config=dict(item),
        ))
    if not specs:
        raise ModelAdapterError("llm_settings中未发现可用于Stage1的模型")
    return tuple(specs)


class ExistingModelCatalog:
    """Read-only lookup over the existing model settings."""

    def __init__(self, specs: tuple[ExistingModelSpec, ...] | None = None):
        self.specs = specs or discover_existing_models()
        self._by_id = {spec.model_id: spec for spec in self.specs}

    @classmethod
    def from_existing_settings(cls) -> "ExistingModelCatalog":
        return cls(discover_existing_models())

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(spec.model_id for spec in self.specs)

    def get(self, model_id: str) -> ExistingModelSpec:
        try:
            return self._by_id[str(model_id)]
        except KeyError as exc:
            raise ModelAdapterError(
                f"Stage1 model_id不在现有llm_settings中：{model_id}；已有={list(self.model_ids)}"
            ) from exc

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": spec.model_id,
                "provider": spec.provider,
                "display_name": spec.display_name,
                "model_ref": spec.model_ref,
                "weight_path": spec.weight_path,
                "call_entrypoint": spec.call_entrypoint,
                "parameter_support": {
                    name: bool(getattr(spec.capabilities, name)) for name in PARAMETER_NAMES
                },
                "parameter_mapping": dict(spec.capabilities.parameter_mapping),
            }
            for spec in self.specs
        ]


class ExistingModelAdapter:
    """Provider-neutral adapter exposing the requested Stage1 call signature."""

    def __init__(
        self,
        catalog: ExistingModelCatalog | None = None,
        *,
        mock_generate: Optional[Callable[..., str]] = None,
    ):
        self.catalog = catalog or ExistingModelCatalog.from_existing_settings()
        self.mock_generate = mock_generate
        self.last_result: Optional[ModelCallResult] = None
        self.last_error_trace: dict[str, Any] = {}
        self.last_raw_attempts: list[str] = []

    def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        return self.invoke(
            model_id,
            system_prompt,
            user_prompt,
            temperature,
            top_p,
            max_tokens,
        ).raw_text

    def invoke(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> ModelCallResult:
        spec = self.catalog.get(model_id)
        started = time.perf_counter()
        actual = {
            "temperature": temperature if spec.capabilities.temperature else None,
            "top_p": top_p if spec.capabilities.top_p else None,
            "max_tokens": max_tokens if spec.capabilities.max_tokens else None,
        }
        unsupported = spec.capabilities.unsupported()
        ignored = tuple(name for name in unsupported if actual[name] is None)
        self.last_raw_attempts = []
        try:
            if self.mock_generate is not None:
                raw = str(self.mock_generate(
                    spec,
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    **actual,
                ))
            elif spec.provider == "local":
                raw = self._generate_local(spec, system_prompt, user_prompt, actual)
            else:
                raise ModelAdapterError(f"当前Stage1没有provider适配器：{spec.provider}")
        except Exception as exc:  # noqa: BLE001 - provider boundary
            trace = {
                "model_id": spec.model_id,
                "provider": spec.provider,
                "actual_parameters": actual,
                "unsupported_parameters": list(unsupported),
                "ignored_parameters": list(ignored),
                "response_time_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.last_error_trace = trace
            self.last_raw_attempts = list(getattr(self, "last_raw_attempts", []))
            if isinstance(exc, ModelAdapterError):
                raise ModelAdapterError(str(exc), trace=trace) from exc
            raise ModelAdapterError(str(exc), trace=trace) from exc
        result = ModelCallResult(
            model_id=spec.model_id,
            provider=spec.provider,
            raw_text=raw,
            response_time_ms=(time.perf_counter() - started) * 1000,
            actual_parameters=actual,
            unsupported_parameters=unsupported,
            ignored_parameters=ignored,
        )
        self.last_result = result
        self.last_error_trace = {}
        if not self.last_raw_attempts:
            self.last_raw_attempts = [raw]
        return result

    def _generate_local(
        self,
        spec: ExistingModelSpec,
        system_prompt: str,
        user_prompt: str,
        actual: Mapping[str, Any],
    ) -> str:
        """Reuse the existing registry/loader/generator; no new model path."""

        try:
            import torch
            from llm.generate import _generate_one, _load_model, _resolve_eos_ids
            from llm.model_registry import apply_tokenizer_overrides, resolve
        except Exception as exc:  # pragma: no cover - absent on this dev Mac
            raise ModelAdapterError(
                "现有local provider运行时缺少llm.model_registry/llm.generate或GPU依赖；"
                "mock测试不需要这些依赖"
            ) from exc

        _, registry_cfg = resolve(spec.model_ref)
        base = spec.weight_path if Path(spec.weight_path).exists() else registry_cfg["hf_id"]
        active = spec.raw_config
        adapter_dir = active.get("adapter_dir") or os.environ.get("LLM_ADAPTER_DIR", "")
        use_adapter = bool(active.get("use_adapter")) if "use_adapter" in active else _env_flag("LLM_USE_ADAPTER", False)
        adapter = Path(str(adapter_dir)) if use_adapter and adapter_dir else None
        load_4bit = _env_flag("LLM_LOAD_4BIT", True)
        bf16 = not _env_flag("LLM_FP16", False)
        model = tok = None
        try:
            model, tok = _load_model(
                base,
                adapter,
                load_4bit=load_4bit,
                bf16=bf16,
                trust_remote_code=bool(registry_cfg.get("trust_remote_code", True)),
            )
            apply_tokenizer_overrides(tok, registry_cfg)
            device = next(model.parameters()).device
            eos_ids = _resolve_eos_ids(tok, registry_cfg)
            if (
                spec.model_ref == "baichuan2_7b_chat"
                and "program_fact_layer_llm_advice_v2" not in system_prompt
            ):
                # Keep the legacy compatibility path for the older seeded
                # two-section protocol.  Advice-only v2 has no seeded prefix;
                # clearing EOS there lets Baichuan run into unrelated corpus
                # continuation instead of ending its assistant turn.
                eos_ids = []
            def generate_segment(segment_user_prompt: str, prefill: str) -> str:
                args = SimpleNamespace(
                    max_new_tokens=int(actual["max_tokens"]),
                    repetition_penalty=1.05,
                    sample=float(actual["temperature"]) > 0.0,
                    temperature=float(actual["temperature"]),
                    top_p=float(actual["top_p"]),
                    num_beams=1,
                    generation_prefill=prefill,
                )
                generation_tok = _GenerationPrefillTokenizer(tok, prefill) if prefill else tok
                raw_segment = _generate_one(
                    model,
                    generation_tok,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": segment_user_prompt},
                    ],
                    args,
                    device,
                    eos_ids,
                )
                return prefill + raw_segment if prefill else raw_segment

            prefill = _stage1_generation_prefill(spec, registry_cfg)
            first_raw = generate_segment(user_prompt, prefill)
            self.last_raw_attempts = [first_raw]

            if spec.model_ref == "baichuan2_7b_chat":
                first_obj = _extract_stage1_json(first_raw)
                first_plan = first_obj.get("rehabilitation_plan") if first_obj else None
                if not (
                    isinstance(first_obj, dict)
                    and isinstance(first_obj.get("integrated_assessment"), str)
                    and isinstance(first_plan, list)
                    and len(first_plan) == 3
                ):
                    plan_prefill = '{\n  "rehabilitation_plan": ['
                    plan_raw = generate_segment(
                        user_prompt + _BAICHUAN_PLAN_FORMAT_SUFFIX,
                        plan_prefill,
                    )
                    self.last_raw_attempts.append(plan_raw)
                    plan_obj = _extract_stage1_json(plan_raw)
                    plan = plan_obj.get("rehabilitation_plan") if plan_obj else None
                    if (
                        isinstance(first_obj, dict)
                        and isinstance(first_obj.get("integrated_assessment"), str)
                        and isinstance(plan, list)
                        and len(plan) == 3
                    ):
                        # Only serialize unchanged model-produced fields; no
                        # action, goal, reason, precaution is invented here.
                        return json.dumps(
                            {
                                "integrated_assessment": first_obj["integrated_assessment"],
                                "rehabilitation_plan": plan,
                            },
                            ensure_ascii=False,
                        )
            return first_raw
        finally:
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _stage1_generation_prefill(spec: ExistingModelSpec, registry_cfg: Mapping[str, Any]) -> str:
    """Seed only serialization boundaries; never add clinical instructions."""

    if registry_cfg.get("generation_mode") == "segmented_clinical_json":
        return '</think>\n{"integrated_assessment": "'
    if spec.model_ref == "baichuan2_7b_chat" and registry_cfg.get("chat_template"):
        return '{\n  "integrated_assessment": "'
    return ""


_BAICHUAN_PLAN_FORMAT_SUFFIX = (
    "\n\n【仅序列化续接】请继续使用相同的九项临床输入和原始医学任务，"
    "现在只返回 rehabilitation_plan 字段的合法 JSON 对象；必须恰好包含3个对象，"
    "每个对象只能有 action、goal、reason、precaution 四个字段。"
    "不要输出综合评估、思维过程、Markdown、解释或其他问题。"
)


def _extract_stage1_json(text: str) -> Optional[dict[str, Any]]:
    """Extract a complete JSON object without repairing model content."""

    normalized = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    normalized = re.sub(r"```(?:json|JSON)?", "", normalized).replace("```", "")
    decoder = json.JSONDecoder()
    for index, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except (json.JSONDecodeError, TypeError):
            # Some chat checkpoints emit literal newlines/tabs inside a JSON
            # string. Escape only those JSON control characters so the parser
            # can inspect the model-produced fields; do not alter raw output
            # or invent/repair any clinical text.
            try:
                value, _ = decoder.raw_decode(_escape_json_string_controls(normalized[index:]))
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(value, dict):
            return value
    return None


def _escape_json_string_controls(text: str) -> str:
    """Make literal JSON control characters parseable without changing words."""

    escaped: list[str] = []
    in_string = False
    slash_escaped = False
    replacements = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
    for char in text:
        if in_string:
            if slash_escaped:
                escaped.append(char)
                slash_escaped = False
            elif char == "\\":
                escaped.append(char)
                slash_escaped = True
            elif char == '"':
                escaped.append(char)
                in_string = False
            elif char in replacements:
                escaped.append(replacements[char])
            elif ord(char) < 0x20:
                escaped.append(f"\\u{ord(char):04x}")
            else:
                escaped.append(char)
        else:
            escaped.append(char)
            if char == '"':
                in_string = True
    return "".join(escaped)


class Stage1ModelRouter:
    """Callable bridge from Stage1 runner messages to the unified adapter."""

    def __init__(self, adapter: ExistingModelAdapter):
        self.adapter = adapter
        self.last_trace: dict[str, Any] = {}
        self.last_raw_attempts: list[str] = []

    def __call__(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        model_id = str(kwargs.pop("model_id"))
        try:
            result = self.adapter.invoke(
                model_id,
                messages[0]["content"],
                messages[1]["content"],
                float(kwargs.pop("temperature")),
                float(kwargs.pop("top_p", 1.0)),
                int(kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 2048))),
            )
        except ModelAdapterError as exc:
            self.last_trace = dict(exc.trace or self.adapter.last_error_trace or {})
            self.last_raw_attempts = list(self.adapter.last_raw_attempts)
            raise
        self.last_raw_attempts = list(self.adapter.last_raw_attempts)
        self.last_trace = {
            "model_id": result.model_id,
            "provider": result.provider,
            "response_time_ms": result.response_time_ms,
            "actual_parameters": dict(result.actual_parameters),
            "unsupported_parameters": list(result.unsupported_parameters),
            "ignored_parameters": list(result.ignored_parameters),
        }
        return result.raw_text


def build_existing_stage1_config() -> Any:
    from .runner import Stage1RunConfig

    catalog = ExistingModelCatalog.from_existing_settings()
    return Stage1RunConfig(model_ids=list(catalog.model_ids))


__all__ = [
    "ExistingModelAdapter",
    "ExistingModelCatalog",
    "ExistingModelSpec",
    "ModelAdapterError",
    "ModelCallResult",
    "ModelCapabilities",
    "PARAMETER_NAMES",
    "Stage1ModelRouter",
    "build_existing_stage1_config",
    "discover_existing_models",
]
