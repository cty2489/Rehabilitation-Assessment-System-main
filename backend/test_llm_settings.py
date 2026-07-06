import json

import pytest

import llm_settings
import report
from llm.model_registry import resolve


def test_default_settings_include_baseline_candidates():
    payload = llm_settings.settings_payload(probe=False)
    ids = {model["id"] for model in payload["models"]}

    assert payload["active_model_id"] == "qwen25_7b_gguf"
    assert {
        "qwen25_7b_gguf",
        "qwen3_8b_hf",
        "deepseek_r1_distill_qwen7b",
        "baichuan2_7b_chat",
        "glm4_9b",
        "mistral7b_v03",
        "llama3_8b_instruct",
    }.issubset(ids)


def test_update_active_model_persists_to_config(tmp_path, monkeypatch):
    config_path = tmp_path / "llm_settings.json"
    model_root = tmp_path / "models"
    (model_root / "Qwen3-8B").mkdir(parents=True)
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", config_path)
    monkeypatch.setenv("LLM_MODEL_ROOT", str(model_root))

    llm_settings.update_active_model("qwen3_8b_hf")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["active_model_id"] == "qwen3_8b_hf"
    assert llm_settings.active_model()["id"] == "qwen3_8b_hf"


def test_update_active_model_rejects_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", tmp_path / "llm_settings.json")

    with pytest.raises(KeyError):
        llm_settings.update_active_model("not-a-model")


def test_local_candidate_without_weight_path_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ROOT", str(tmp_path / "models"))

    payload = llm_settings.settings_payload(probe=False)
    qwen3 = next(model for model in payload["models"] if model["id"] == "qwen3_8b_hf")

    assert qwen3["weight_exists"] is False
    assert qwen3["available"] is False
    assert qwen3["status"] == "not_ready"


def test_update_active_model_rejects_not_ready_local_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", tmp_path / "llm_settings.json")
    monkeypatch.setenv("LLM_MODEL_ROOT", str(tmp_path / "models"))

    with pytest.raises(ValueError):
        llm_settings.update_active_model("qwen3_8b_hf")


def test_update_model_settings_persists_weight_path(tmp_path, monkeypatch):
    config_path = tmp_path / "llm_settings.json"
    model_path = tmp_path / "Qwen3-8B"
    model_path.mkdir()
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", config_path)

    llm_settings.update_model_settings("qwen3_8b_hf", {"weight_path": str(model_path)})

    payload = llm_settings.settings_payload(probe=False)
    qwen3 = next(model for model in payload["models"] if model["id"] == "qwen3_8b_hf")
    assert qwen3["weight_path"] == str(model_path)
    assert qwen3["weight_exists"] is True
    assert qwen3["available"] is True


def test_qwen_data_original_hf_paths_are_detected(tmp_path, monkeypatch):
    qwen_data = tmp_path / "Qwen_data"
    qwen3 = qwen_data / "Qwen3-8B"
    deepseek = qwen_data / "DeepSeek-R1-Distill-Qwen-7B"
    qwen3.mkdir(parents=True)
    deepseek.mkdir(parents=True)
    monkeypatch.setenv("LLM_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("LLM_ORIGINAL_MODEL_ROOT", str(qwen_data))

    payload = llm_settings.settings_payload(probe=False)
    by_id = {model["id"]: model for model in payload["models"]}

    assert by_id["qwen3_8b_hf"]["weight_exists"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["weight_exists"] is True
    assert by_id["qwen3_8b_hf"]["available"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["available"] is True


def test_settings_candidates_match_model_registry():
    payload = llm_settings.settings_payload(probe=False)
    for model in payload["models"]:
        if model["provider"] == "local":
            resolve(model["model_id"])


def test_report_provider_uses_env_until_ui_config_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", tmp_path / "llm_settings.json")
    monkeypatch.delenv("LLM_ACTIVE_MODEL_ID", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    model_root = tmp_path / "models"
    (model_root / "Qwen3-8B").mkdir(parents=True)
    monkeypatch.setenv("LLM_MODEL_ROOT", str(model_root))

    assert report.llm_provider() == "deepseek"

    llm_settings.update_active_model("qwen3_8b_hf")

    assert report.llm_provider() == "local"
