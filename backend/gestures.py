"""The 26-gesture rehabilitation hand-training library + selection space.

The report's “推荐手势组合（从26个中动态选取）” section recommends a combination
of hand gestures drawn from a fixed library of 26. The clinical team supplies
this library via ``backend/config/gestures_26.json`` (same schema as
``backend/config/gestures_26.example.json`` and ``_g`` below). **Until that
reviewed runtime file is provided, the library is considered "not ready"**
(``library_ready()`` is False): the report renders a placeholder for the gesture
section and the LLM is NOT asked to pick gestures — so an un-fine-tuned base
model can't invent gesture names. The ``_SEED_GESTURES_26`` list below is kept
only as a reference/example and is NOT used by the report while not ready, so an
inferred seed never masquerades as the clinical team's real library.

Consumer: the **LLM** (``backend/report.py::reason_clinical``) receives the
library as its *controlled selection space* and decides the final combination,
dosing (辅助力度 / 重复次数) and weekly plan. Code validates that every gesture
it picks exists in the library (anti-hallucination). If the LLM can't produce a
valid plan, the report keeps the rest of the clinical reasoning and skips only
the gesture section.

Each gesture:
    id            stable key
    name          中文名称 (must match what the LLM is told it may pick)
    purpose       训练目的
    default_force 默认辅助力度 (%, or a string like "电动抬腕辅助：开")
    stages        applicable Brunnstrom stages [1..6]
    tags          semantic tags used by the rule selector / LLM hints
                  (e.g. "extension", "flexion", "pinch", "adl", "wrist",
                   "isolation", "spasticity_risk")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "gestures_26.json"


def _g(gid: str, name: str, purpose: str, default_force: str,
       stages: Sequence[int], tags: Sequence[str]) -> Dict[str, Any]:
    return {
        "id": gid,
        "name": name,
        "purpose": purpose,
        "default_force": default_force,
        "stages": list(stages),
        "tags": list(tags),
    }


# Seed library (26). Names of the first 6 mirror the template example verbatim.
_SEED_GESTURES_26: List[Dict[str, Any]] = [
    _g("ext_index", "健侧伸食指+患侧被动伸指", "打破伸指协同模式（抗伸指-屈腕协同）",
       "患侧辅助力度：70%", [2, 3, 4], ["extension", "isolation", "mirror"]),
    _g("fist", "健侧握拳+患侧被动握拳", "维持屈曲功能，但注意不要过度强化屈肌",
       "患侧辅助力度：60%", [2, 3, 4, 5], ["flexion", "grip"]),
    _g("oppose", "健侧对指（拇指-食指捏）", "促进精细动作分离",
       "患侧辅助力度：80%", [3, 4, 5, 6], ["pinch", "isolation", "fine"]),
    _g("ext_pinky", "健侧伸小指+患侧被动伸小指", "训练单指分离运动",
       "患侧辅助力度：75%", [3, 4, 5], ["extension", "isolation", "fine"]),
    _g("cylinder", "健侧圆柱抓握（模拟拿水杯）", "ADL实用性训练",
       "患侧辅助力度：65%", [3, 4, 5, 6], ["adl", "grip"]),
    _g("wrist_ext", "健侧腕背伸+患侧被动腕背伸", "腕手联动训练",
       "电动抬腕辅助：开", [2, 3, 4, 5], ["wrist", "extension"]),
    # — extension / isolation family —
    _g("ext_all", "五指同步伸展", "整体伸指肌群激活，对抗屈肌优势",
       "患侧辅助力度：75%", [3, 4, 5], ["extension", "grip"]),
    _g("ext_thumb", "拇指外展伸展", "恢复拇指伸展与外展分离",
       "患侧辅助力度：75%", [3, 4, 5, 6], ["extension", "isolation", "thumb"]),
    _g("ext_middle", "伸中指+其余屈曲", "高难度单指分离训练",
       "患侧辅助力度：80%", [4, 5, 6], ["extension", "isolation", "fine"]),
    _g("wrist_radial", "腕桡侧偏移", "腕部多向控制，配合伸指",
       "患侧辅助力度：70%", [3, 4, 5], ["wrist", "isolation"]),
    # — grasp family —
    _g("spherical", "球形抓握（模拟握球）", "球形抓握ADL训练，增大目标物降低震颤影响",
       "患侧辅助力度：65%", [4, 5, 6], ["adl", "grip"]),
    _g("hook", "勾状抓握（提物）", "提物功能，维持手内肌长度",
       "患侧辅助力度：60%", [3, 4, 5], ["grip", "adl"]),
    _g("lateral_pinch", "侧捏（钥匙捏）", "侧方抓握ADL，拇指带动",
       "患侧辅助力度：70%", [4, 5, 6], ["pinch", "adl", "thumb"]),
    _g("tip_pinch", "指尖捏（捏小物）", "精细指尖控制",
       "患侧辅助力度：80%", [5, 6], ["pinch", "fine"]),
    _g("tripod", "三指捏（写字握姿）", "书写相关精细抓握",
       "患侧辅助力度：80%", [5, 6], ["pinch", "fine", "adl"]),
    _g("palmar_grasp", "掌心抓握", "基础掌侧抓握，维持抓握模式",
       "患侧辅助力度：60%", [2, 3, 4], ["grip"]),
    # — opposition / thumb family —
    _g("oppose_all", "拇指依次对指（拇-食-中-环-小）", "拇指对指序列，精细协调",
       "患侧辅助力度：80%", [5, 6], ["pinch", "isolation", "fine", "thumb"]),
    _g("thumb_circle", "拇指环转", "拇指多向活动度",
       "患侧辅助力度：75%", [4, 5, 6], ["thumb", "isolation"]),
    # — wrist / forearm family —
    _g("wrist_flex_ext", "腕屈伸交替", "腕关节主动控制",
       "患侧辅助力度：65%", [3, 4, 5], ["wrist"]),
    _g("forearm_supinate", "前臂旋后", "前臂旋转配合手部功能",
       "患侧辅助力度：70%", [3, 4, 5, 6], ["forearm", "adl"]),
    _g("forearm_pronate", "前臂旋前", "前臂旋转配合手部功能",
       "患侧辅助力度：70%", [3, 4, 5, 6], ["forearm", "adl"]),
    # — alternating anti-synergy —
    _g("ext_then_flex", "伸指后立即辅助屈指（交替）", "抗协同交替模式，避免单纯强化屈肌",
       "患侧辅助力度：70%", [3, 4, 5], ["extension", "flexion", "isolation"]),
    _g("release_after_grip", "抓握后主动松开", "训练随意松手（抓-放分离）",
       "患侧辅助力度：70%", [3, 4, 5], ["extension", "grip"]),
    # — gestures the template flags as low-value / risky for mid stages —
    _g("ok_sign", "OK手势（拇食指成环）", "复杂对指组合（与早中期功能改善相关性较低）",
       "患侧辅助力度：80%", [5, 6], ["pinch", "fine", "low_value_early"]),
    _g("rapid_grip", "连续快速抓放", "速度性抓握（易诱发痉挛，慎用）",
       "患侧辅助力度：55%", [5, 6], ["grip", "spasticity_risk"]),
    _g("strong_fist", "用力握拳保持", "握力强化（高张力期慎用，易强化屈肌）",
       "患侧辅助力度：50%", [5, 6], ["flexion", "grip", "spasticity_risk"]),
]


def _config_library() -> Optional[List[Dict[str, Any]]]:
    """Return the clinical team's library from the JSON override, or None.

    None means the override file is absent / empty / malformed — i.e. the real
    26-gesture library has not been supplied yet (placeholder mode).
    """
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception as exc:  # noqa: BLE001
            print(f"[gestures][warn] failed to read {_CONFIG_PATH}: {exc}; library not ready")
    return None


def library_ready() -> bool:
    """True once the clinical team supplied ``config/gestures_26.json``.

    While False, the report shows a placeholder gesture section and the LLM is
    not asked to select gestures (avoids an un-fine-tuned base model inventing
    gesture names). The example/seed library does NOT count as ready.
    """
    return _config_library() is not None


def load_library() -> List[Dict[str, Any]]:
    """Return the 26-gesture library, preferring the JSON override if present.

    Falls back to the inferred seed only for reference/dev; report code keys off
    ``library_ready()`` and won't use the seed as the selection space.
    """
    return _config_library() or list(_SEED_GESTURES_26)


GESTURES_26: List[Dict[str, Any]] = load_library()


def gesture_names() -> List[str]:
    return [g["name"] for g in load_library()]


def validate_selection(names: Sequence[str]) -> List[str]:
    """Keep only names that exist in the library (anti-hallucination)."""
    valid = set(gesture_names())
    return [n for n in names if n in valid]


__all__ = [
    "GESTURES_26",
    "load_library",
    "library_ready",
    "gesture_names",
    "validate_selection",
]
