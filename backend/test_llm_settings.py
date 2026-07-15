import json

try:
    import pytest
except ModuleNotFoundError as exc:  # unittest discovery without dev dependencies
    import unittest

    raise unittest.SkipTest("install backend/requirements-dev.txt to run pytest tests") from exc

import llm_settings
import report
from llm.model_registry import resolve


def test_default_settings_include_baseline_candidates():
    payload = llm_settings.settings_payload(probe=False)
    ids = {model["id"] for model in payload["models"]}

    assert payload["active_model_id"] == "qwen3_8b_hf"
    assert {
        "qwen3_8b_hf",
        "deepseek_r1_distill_qwen7b",
        "baichuan2_7b_chat",
        "glm4_9b",
        "mistral7b_v03",
        "internlm3_8b",
    }.issubset(ids)
    assert "qwen25_7b_gguf" not in ids
    assert "llama3_8b_instruct" not in ids


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


def test_inaccessible_model_path_is_treated_as_unavailable(tmp_path, monkeypatch):
    fallback = tmp_path / "available-model"
    fallback.mkdir()
    real_exists = llm_settings.Path.exists

    def guarded_exists(path):
        if str(path).startswith("/denied"):
            raise PermissionError("not allowed")
        return real_exists(path)

    monkeypatch.setattr(llm_settings.Path, "exists", guarded_exists)

    assert llm_settings._first_existing_path(["/denied/model", str(fallback)]) == str(fallback)
    assert llm_settings._path_exists("/denied/model") is False


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
    baichuan = qwen_data / "Baichuan2-7B-Chat"
    glm = qwen_data / "GLM-4-9B-Chat"
    mistral = qwen_data / "Mistral-7B-Instruct-v0.3"
    internlm = qwen_data / "InternLM3-8B-Instruct"
    qwen3.mkdir(parents=True)
    deepseek.mkdir(parents=True)
    baichuan.mkdir(parents=True)
    glm.mkdir(parents=True)
    mistral.mkdir(parents=True)
    internlm.mkdir(parents=True)
    monkeypatch.setenv("LLM_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("LLM_ORIGINAL_MODEL_ROOT", str(qwen_data))

    payload = llm_settings.settings_payload(probe=False)
    by_id = {model["id"]: model for model in payload["models"]}

    assert by_id["qwen3_8b_hf"]["weight_exists"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["weight_exists"] is True
    assert by_id["baichuan2_7b_chat"]["weight_exists"] is True
    assert by_id["glm4_9b"]["weight_exists"] is True
    assert by_id["mistral7b_v03"]["weight_exists"] is True
    assert by_id["qwen3_8b_hf"]["available"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["report_ready"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["available"] is True
    assert by_id["deepseek_r1_distill_qwen7b"]["status"] == "ready"
    assert by_id["baichuan2_7b_chat"]["report_ready"] is True
    assert by_id["baichuan2_7b_chat"]["available"] is True
    assert by_id["baichuan2_7b_chat"]["status"] == "ready"
    assert by_id["glm4_9b"]["report_ready"] is True
    assert by_id["glm4_9b"]["available"] is True
    assert by_id["glm4_9b"]["status"] == "ready"
    assert by_id["mistral7b_v03"]["report_ready"] is True
    assert by_id["mistral7b_v03"]["available"] is True
    assert by_id["mistral7b_v03"]["status"] == "ready"
    assert by_id["internlm3_8b"]["report_ready"] is True
    assert by_id["internlm3_8b"]["available"] is True
    assert by_id["internlm3_8b"]["status"] == "ready"


def test_saved_missing_weight_path_heals_to_existing_default(tmp_path, monkeypatch):
    config_path = tmp_path / "llm_settings.json"
    qwen_data = tmp_path / "Qwen_data"
    (qwen_data / "Baichuan2-7B-Chat").mkdir(parents=True)
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", config_path)
    monkeypatch.setenv("LLM_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("LLM_ORIGINAL_MODEL_ROOT", str(qwen_data))
    config_path.write_text(
        json.dumps({
            "schema_version": "rehab.llm_settings.v1",
            "active_model_id": "qwen25_7b_gguf",
            "models": [
                {
                    "id": "qwen25_7b_gguf",
                    "provider": "remote",
                    "remote_url": "http://127.0.0.1:6008",
                },
                {
                    "id": "baichuan2_7b_chat",
                    "weight_path": str(tmp_path / "models" / "Baichuan2-7B-Chat"),
                }
            ],
        }),
        encoding="utf-8",
    )

    payload = llm_settings.settings_payload(probe=False)
    baichuan = next(model for model in payload["models"] if model["id"] == "baichuan2_7b_chat")

    assert payload["active_model_id"] == "qwen3_8b_hf"
    assert "qwen25_7b_gguf" not in {model["id"] for model in payload["models"]}
    assert baichuan["weight_path"] == str(qwen_data / "Baichuan2-7B-Chat")
    assert baichuan["weight_exists"] is True


def test_removed_default_candidates_are_hidden_from_saved_config(tmp_path, monkeypatch):
    config_path = tmp_path / "llm_settings.json"
    monkeypatch.setattr(llm_settings, "CONFIG_PATH", config_path)
    config_path.write_text(
        json.dumps({
            "schema_version": "rehab.llm_settings.v1",
            "active_model_id": "llama3_8b_instruct",
            "models": [
                {
                    "id": "llama3_8b_instruct",
                    "provider": "local",
                    "model_id": "llama3_8b_instruct",
                    "enabled": True,
                    "report_ready": False,
                }
            ],
        }),
        encoding="utf-8",
    )

    payload = llm_settings.settings_payload(probe=False)

    assert payload["active_model_id"] == "qwen3_8b_hf"
    assert "llama3_8b_instruct" not in {model["id"] for model in payload["models"]}
    with pytest.raises(KeyError):
        llm_settings.update_active_model("llama3_8b_instruct")


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
