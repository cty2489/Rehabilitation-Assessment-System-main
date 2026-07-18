import json
from pathlib import Path

import pytest

import knowledge_admin


def _write_runtime(root: Path, *, source_id: str = "SRC-001") -> None:
    collection = root / "rehab_knowledge_trial_v0_2"
    collection.mkdir(parents=True)
    counts = {
        "total_entries": 1,
        "indexable_entries": 1,
        "demo_ready_entries": 1,
        "clinical_ready_entries": 0,
        "excluded_entries": 0,
        "chunks": 1,
        "evaluation_questions": 0,
        "sources": 1,
    }
    manifest = {
        "schema_version": "rehab.knowledge.manifest.v1",
        "collection_id": "rehab_knowledge_trial_v0_2",
        "created_at_utc": "2026-07-18T00:00:00+00:00",
        "source": {
            "document_id": "rehab_rag_expert_review_draft/0.1",
            "filename": "review.json",
            "sha256": "abc123",
        },
        "trial_release": {
            "release_id": "internal-rag-trial-v0.1",
            "expert_verified": False,
            "clinical_ready": False,
            "warning": "仅供内部测试。",
            "allowed_usage": ["内部检索测试"],
            "prohibited_usage": ["正式临床报告"],
        },
        "counts": counts,
    }
    quality = {
        "schema_version": "rehab.knowledge.quality.v1",
        "collection_id": "rehab_knowledge_trial_v0_2",
        "counts": counts,
    }
    entry = {
        "knowledge_id": "KB-EMG-001",
        "entry_version": "0.1",
        "title": "测试肌电指标",
        "category": "EMG",
        "system_key": "resting_emg_level",
        "knowledge_status": "blocked_current_implementation",
        "knowledge_status_label": "阻断：当前实现不可进入临床知识库",
        "content": "核心结论",
        "allowed_interpretation": "仅可同条件复测。",
        "prohibited_interpretation": "不得自动诊断。",
        "acquisition_and_algorithm_requirements": "固定采集协议。",
        "reference_range_policy": "不显示通用范围。",
        "implementation_action": "修正命名。",
        "applicable_population": ["成人脑卒中后上肢康复"],
        "review_notes": ["待专家复核"],
        "aliases": ["测试指标"],
        "source": {
            "document_id": "rehab_rag_expert_review_draft/0.1",
            "filename": "review.json",
            "sha256": "abc123",
            "original_entry_number": 1,
            "source_ids": [source_id],
        },
        "governance": {
            "source_status": "pending_expert_review",
            "expert_review_status": "pending",
            "expert_verified": False,
            "trial_release_id": "internal-rag-trial-v0.1",
        },
        "status": {
            "indexable": True,
            "demo_ready": True,
            "clinical_ready": False,
            "issues": ["internal_trial_only"],
        },
    }
    source = {
        "schema_version": "rehab.knowledge.source.v1",
        "source_id": "SRC-001",
        "title": "测试来源",
        "year": 2025,
        "source_type": "方法学共识",
        "evidence_tier": "A",
        "url": "https://example.test/source",
        "scope": "测试范围",
        "note": "",
        "knowledge_ids": ["KB-EMG-001"],
    }
    (collection / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (collection / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False), encoding="utf-8"
    )
    (collection / "entries.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (collection / "sources.jsonl").write_text(
        json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def test_status_separates_mapping_from_clinical_readiness(tmp_path, monkeypatch):
    _write_runtime(tmp_path)
    monkeypatch.setenv("KNOWLEDGE_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_COLLECTION", "rehab_knowledge_trial_v0_2")
    monkeypatch.setenv("RAG_MODE", "off")

    payload = knowledge_admin.status_payload(
        app_version="cloud-server-v1.2.0",
        build_commit="abcdef1",
        report_model="qwen3_8b_hf",
    )

    assert payload["available"] is True
    assert payload["counts"]["mapped_biomarkers"] == 1
    assert payload["counts"]["clinical_ready_biomarkers"] == 0
    assert payload["versions"]["content_release"] == "internal-rag-trial-v0.1"
    assert payload["versions"]["index_collection"] == "rehab_knowledge_trial_v0_2"
    assert payload["validation"]["valid"] is False
    assert payload["validation"]["issues"] == ["系统指标映射为 1/26"]


def test_entry_detail_uses_structured_sources(tmp_path, monkeypatch):
    _write_runtime(tmp_path)
    monkeypatch.setenv("KNOWLEDGE_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_COLLECTION", "rehab_knowledge_trial_v0_2")

    detail = knowledge_admin.entry_payload("KB-EMG-001")["entry"]
    filtered = knowledge_admin.entries_payload(query="resting_emg_level")

    assert detail["sources"][0]["source_id"] == "SRC-001"
    assert detail["prohibited_interpretation"] == "不得自动诊断。"
    assert filtered["total"] == 1


def test_missing_structured_source_reference_is_rejected(tmp_path, monkeypatch):
    _write_runtime(tmp_path, source_id="SRC-404")
    monkeypatch.setenv("KNOWLEDGE_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_COLLECTION", "rehab_knowledge_trial_v0_2")

    with pytest.raises(knowledge_admin.KnowledgeUnavailable, match="不存在的来源"):
        knowledge_admin.load_snapshot()


def test_source_reverse_link_must_match_entry(tmp_path, monkeypatch):
    _write_runtime(tmp_path)
    source_path = (
        tmp_path / "rehab_knowledge_trial_v0_2" / "sources.jsonl"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["knowledge_ids"] = []
    source_path.write_text(
        json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("KNOWLEDGE_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_COLLECTION", "rehab_knowledge_trial_v0_2")

    with pytest.raises(knowledge_admin.KnowledgeUnavailable, match="反向关联不一致"):
        knowledge_admin.load_snapshot()
