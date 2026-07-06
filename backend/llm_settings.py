"""Persistent report-LLM selection settings.

The report pipeline can call several kinds of LLM backends:

* ``remote``   - local/remote HTTP service exposing ``/generate_messages``
* ``local``    - in-process transformers model loaded by ``backend.report``
* ``deepseek`` - OpenAI-compatible DeepSeek API

Only one model is active for report generation at a time. The settings are
stored outside source control so researchers can switch verified baseline
models from the Model Settings page without editing ``.env`` or restarting the
full stack.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_PATH = Path(
    os.environ.get(
        "LLM_SETTINGS_PATH",
        str(Path(__file__).resolve().parent / "config" / "llm_settings.json"),
    )
)


def _default_remote_url() -> str:
    return os.environ.get("LLM_REMOTE_URL", "http://127.0.0.1:6007").strip().rstrip("/")


def _original_model_path(filename: str) -> str:
    root = os.environ.get(
        "LLM_ORIGINAL_MODEL_ROOT",
        "/root/autodl-tmp/Qwen_data",
    ).rstrip("/")
    return f"{root}/{filename}"


def _first_existing_path(paths: List[str]) -> str:
    for item in paths:
        if item and Path(item).exists():
            return item
    return paths[0] if paths else ""


def _default_model_path(filename: str, extra_candidates: Optional[List[str]] = None) -> str:
    root = os.environ.get(
        "LLM_MODEL_ROOT",
        "/root/autodl-tmp/rehab_project/models",
    ).rstrip("/")
    candidates = [f"{root}/{filename}"]
    candidates.extend(extra_candidates or [])
    return _first_existing_path(candidates)


def _default_models() -> List[Dict[str, Any]]:
    remote_url = _default_remote_url()
    return [
        {
            "id": "qwen25_7b_gguf",
            "name": "Qwen2.5-7B-Instruct GGUF",
            "vendor": "Qwen",
            "origin": "国产",
            "provider": "remote",
            "model_id": "qwen25_7b",
            "remote_url": remote_url,
            "weight_path": _default_model_path(
                "qwen2.5-7b-instruct-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            ),
            "enabled": True,
            "description": "GGUF 回退/对照报告模型，通过本机 HTTP 服务调用。",
            "report_ready": True,
        },
        {
            "id": "qwen3_8b_hf",
            "name": "Qwen3-8B",
            "vendor": "Qwen",
            "origin": "国产",
            "provider": "local",
            "model_id": "qwen3_8b",
            "weight_path": _default_model_path(
                "Qwen3-8B",
                extra_candidates=[_original_model_path("Qwen3-8B")],
            ),
            "enabled": True,
            "description": "当前云端推荐报告模型，HF 原版格式，可作为中文基线和后续微调基座。",
            "report_ready": True,
        },
        {
            "id": "deepseek_r1_distill_qwen7b",
            "name": "DeepSeek-R1-Distill-Qwen-7B",
            "vendor": "DeepSeek",
            "origin": "国产",
            "provider": "local",
            "model_id": "deepseek_r1_distill_qwen7b",
            "weight_path": _default_model_path(
                "DeepSeek-R1-Distill-Qwen-7B",
                extra_candidates=[_original_model_path("DeepSeek-R1-Distill-Qwen-7B")],
            ),
            "enabled": True,
            "description": "DeepSeek 蒸馏模型候选；当前端到端报告 JSON 结构校验未通过，暂不用于线上报告。",
            "report_ready": False,
        },
        {
            "id": "baichuan2_7b_chat",
            "name": "Baichuan2-7B-Chat",
            "vendor": "Baichuan",
            "origin": "国产",
            "provider": "local",
            "model_id": "baichuan2_7b_chat",
            "weight_path": _default_model_path(
                "Baichuan2-7B-Chat",
                extra_candidates=[_original_model_path("Baichuan2-7B-Chat")],
            ),
            "enabled": True,
            "description": "百川中文对话模型候选，用于国产模型横向比较。",
            "report_ready": False,
        },
        {
            "id": "glm4_9b",
            "name": "GLM-4-9B",
            "vendor": "Zhipu",
            "origin": "国产",
            "provider": "local",
            "model_id": "glm4_9b",
            "weight_path": _default_model_path(
                "GLM-4-9B-0414",
                extra_candidates=[
                    _original_model_path("GLM-4-9B-0414"),
                    _original_model_path("GLM-4-9B-Chat"),
                ],
            ),
            "enabled": True,
            "description": "智谱 GLM 系列候选，用于国产模型横向比较。",
            "report_ready": False,
        },
        {
            "id": "mistral7b_v03",
            "name": "Mistral-7B-Instruct-v0.3",
            "vendor": "Mistral",
            "origin": "国外",
            "provider": "local",
            "model_id": "mistral7b_v03",
            "weight_path": _default_model_path(
                "Mistral-7B-Instruct-v0.3",
                extra_candidates=[_original_model_path("Mistral-7B-Instruct-v0.3")],
            ),
            "enabled": True,
            "description": "国外通用指令模型候选，可作为英文/国际基线。",
            "report_ready": False,
        },
        {
            "id": "llama3_8b_instruct",
            "name": "Llama-3-8B-Instruct",
            "vendor": "Meta",
            "origin": "国外",
            "provider": "local",
            "model_id": "llama3_8b_instruct",
            "weight_path": _default_model_path(
                "Meta-Llama-3-8B-Instruct",
                extra_candidates=[
                    _original_model_path("Meta-Llama-3-8B-Instruct"),
                    _original_model_path("Llama-3-8B-Instruct"),
                ],
            ),
            "enabled": True,
            "description": "国外通用指令模型候选，可作为国际基线。",
            "report_ready": False,
        },
    ]


def _default_settings() -> Dict[str, Any]:
    models = _default_models()
    preferred = os.environ.get("LLM_ACTIVE_MODEL_ID", "").strip()
    active = preferred if preferred in {m["id"] for m in models} else models[0]["id"]
    return {
        "schema_version": "rehab.llm_settings.v1",
        "active_model_id": active,
        "models": models,
    }


def _merge_with_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_settings()
    merged = deepcopy(defaults)
    by_id = {m["id"]: m for m in merged["models"]}
    for item in raw.get("models") or []:
        mid = item.get("id")
        if not mid:
            continue
        if mid in by_id:
            default_item = by_id[mid]
            saved_item = dict(item)
            # Old servers may have persisted a weight_path before a model was
            # downloaded. If that stale path does not exist but the current
            # default path does, heal the runtime config automatically so newly
            # added weights are detected without exposing paths in the UI.
            if (
                str(default_item.get("provider") or "").lower() == "local"
                and not _path_exists(saved_item.get("weight_path"))
                and _path_exists(default_item.get("weight_path"))
            ):
                saved_item.pop("weight_path", None)
            default_item.update(saved_item)
        else:
            merged["models"].append(item)
            by_id[mid] = item
    active = raw.get("active_model_id")
    if active in by_id:
        merged["active_model_id"] = active
    return merged


def read_settings() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return _default_settings()
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return _merge_with_defaults(raw if isinstance(raw, dict) else {})
    except Exception as exc:  # noqa: BLE001
        print(f"[llm-settings][warn] failed to read {CONFIG_PATH}: {exc}")
        return _default_settings()


def settings_configured() -> bool:
    return CONFIG_PATH.exists() or bool(os.environ.get("LLM_ACTIVE_MODEL_ID", "").strip())


def write_settings(settings: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="llm_settings_", suffix=".json", dir=str(CONFIG_PATH.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        tmp.replace(CONFIG_PATH)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def get_model(settings: Optional[Dict[str, Any]], model_id: str) -> Optional[Dict[str, Any]]:
    data = settings or read_settings()
    for model in data.get("models") or []:
        if model.get("id") == model_id:
            return model
    return None


def active_model() -> Dict[str, Any]:
    settings = read_settings()
    model = get_model(settings, settings.get("active_model_id", ""))
    if model is None:
        return _default_models()[0]
    return model


def update_active_model(model_id: str) -> Dict[str, Any]:
    settings = read_settings()
    model = get_model(settings, model_id)
    if model is None:
        valid = [m.get("id") for m in settings.get("models") or []]
        raise KeyError(f"Unknown LLM model id: {model_id}. Valid ids: {valid}")
    decorated = decorate_model(model, settings.get("active_model_id", ""), probe=False)
    if not decorated.get("available"):
        raise ValueError(
            f"LLM model is not ready: {model_id}. "
            "请先配置可用的服务/权重，并确认该模型已通过端到端报告结构校验。"
        )
    settings["active_model_id"] = model_id
    write_settings(settings)
    return settings


def update_model_settings(model_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    settings = read_settings()
    model = get_model(settings, model_id)
    if model is None:
        valid = [m.get("id") for m in settings.get("models") or []]
        raise KeyError(f"Unknown LLM model id: {model_id}. Valid ids: {valid}")

    allowed = {"weight_path", "remote_url", "enabled", "adapter_dir", "use_adapter"}
    for key, value in patch.items():
        if key not in allowed or value is None:
            continue
        if key in {"weight_path", "remote_url", "adapter_dir"}:
            model[key] = str(value).strip()
        elif key in {"enabled", "use_adapter"}:
            model[key] = bool(value)
    write_settings(settings)
    return settings


def _path_exists(path: Any) -> bool:
    text = str(path or "").strip()
    return bool(text) and Path(text).exists()


def _remote_health(remote_url: str) -> Dict[str, Any]:
    url = remote_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:  # noqa: S310 - local/admin URL
            body = resp.read(4096).decode("utf-8", errors="replace")
        data = json.loads(body) if body else {}
        return {
            "reachable": True,
            "loaded": bool(data.get("loaded", data.get("status") == "ok")),
            "status": data.get("status") or ("ok" if data else "unknown"),
            "detail": data,
        }
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"reachable": False, "loaded": False, "status": "unreachable", "error": str(exc)}


def decorate_model(model: Dict[str, Any], active_id: str, probe: bool = True) -> Dict[str, Any]:
    out = deepcopy(model)
    provider = str(out.get("provider") or "").lower()
    out["is_active"] = out.get("id") == active_id
    out["configured"] = bool(out.get("enabled", True))
    out["report_ready"] = bool(out.get("report_ready", True))
    out["health"] = None

    if provider == "remote":
        url = str(out.get("remote_url") or "").strip()
        out["configured"] = out["configured"] and bool(url)
        if probe and url:
            out["health"] = _remote_health(url)
            available = bool(out["health"].get("reachable") and out["health"].get("loaded"))
        else:
            available = out["configured"]
        available = bool(available and out["report_ready"])
    elif provider == "deepseek":
        out["configured"] = out["configured"] and bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
        available = bool(out["configured"] and out["report_ready"])
    elif provider == "local":
        out["weight_exists"] = _path_exists(out.get("weight_path"))
        available = bool(out["configured"] and out["weight_exists"] and out["report_ready"])
    else:
        available = False

    out["available"] = available
    if out["is_active"]:
        out["status"] = "active"
    elif available:
        out["status"] = "ready"
    elif out["configured"] and not out["report_ready"]:
        out["status"] = "candidate"
    else:
        out["status"] = "not_ready"
    return out


def settings_payload(probe: bool = True) -> Dict[str, Any]:
    settings = read_settings()
    active_id = settings.get("active_model_id", "")
    models = [decorate_model(model, active_id, probe=probe) for model in settings.get("models") or []]
    active = next((model for model in models if model.get("is_active")), None)
    return {
        "schema_version": settings.get("schema_version", "rehab.llm_settings.v1"),
        "config_path": str(CONFIG_PATH),
        "active_model_id": active_id,
        "active_model": active,
        "models": models,
    }


__all__ = [
    "active_model",
    "settings_configured",
    "settings_payload",
    "update_active_model",
    "update_model_settings",
]
