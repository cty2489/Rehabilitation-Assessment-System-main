"""Torch-free clinical reading lookups for the predicted scores.

Extracted from ``inference.py`` so lightweight modules (``report_builder.py``)
can translate a Hand-MAS grade / Brunnstrom stage into a one-line clinical
reading without importing the heavy DL inference stack (torch/mne/…).
``inference.py`` re-exports these so its public behaviour is unchanged.
"""
from __future__ import annotations

from typing import Dict

# Hand MAS (Modified Ashworth) — 6 ordinal grades.
HAND_TONE_READING: Dict[str, str] = {
    "0": "被动活动时未见肌张力增加",
    "1": "肌张力轻度增高，被动活动末端可触及轻微卡顿",
    "1+": "肌张力轻中度增高，被动活动前半程出现卡顿、后半程仍可活动",
    "2": "肌张力中度增高，整个活动范围内阻力明显，但肢体仍易于被动活动",
    "3": "肌张力重度增高，被动活动困难",
    "4": "受累部位呈强直状态，难以被动屈伸",
}

# Brunnstrom hand recovery stage — clinically 2–6 for this model.
BRUNNSTROM_READING: Dict[int, str] = {
    1: "处于弛缓期，手部无主动运动",
    2: "出现联合反应，可见极少量肌肉活动但无随意抓握",
    3: "可引出共同运动，能完成钩状抓握但难以主动伸展",
    4: "出现部分分离运动，可完成侧方抓握与拇指带动的小范围松开",
    5: "分离运动较明显，可完成对掌、球形与圆柱抓握，手指能部分协同伸展",
    6: "手功能接近正常，可完成各类抓握与大部分精细动作，仅速度或协调性略逊于健侧",
}


def hand_tone_reading(value: str) -> str:
    return HAND_TONE_READING.get(str(value), "肌张力分级结果")


def brunnstrom_reading(stage: int) -> str:
    try:
        stage = int(stage)
    except (TypeError, ValueError):
        return "手功能分期结果"
    return BRUNNSTROM_READING.get(stage, "手功能分期结果")


__all__ = [
    "HAND_TONE_READING",
    "BRUNNSTROM_READING",
    "hand_tone_reading",
    "brunnstrom_reading",
]
