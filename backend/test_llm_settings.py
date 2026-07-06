import json

import pytest

import llm_settings
import report


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
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", config_path)

    llm_settings.update_active_model("qwen3_8b_hf")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["active_model_id"] == "qwen3_8b_hf"
    assert llm_settings.active_model()["id"] == "qwen3_8b_hf"


def test_update_active_model_rejects_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", tmp_path / "llm_settings.json")

    with pytest.raises(KeyError):
        llm_settings.update_active_model("not-a-model")


def test_report_provider_uses_env_until_ui_config_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", tmp_path / "llm_settings.json")
    monkeypatch.delenv("LLM_ACTIVE_MODEL_ID", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")

    assert report.llm_provider() == "deepseek"

    llm_settings.update_active_model("qwen3_8b_hf")

    assert report.llm_provider() == "local"
