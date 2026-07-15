"""Pure device timestamp-unit and sampling-rate inference."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def infer_device_timebase(t: np.ndarray, default: float) -> Tuple[float, float, str]:
    """Return ``(rate_hz, seconds_per_unit, unit)`` for a device time column."""
    values = np.asarray(t, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(default), 1.0, "assumed_seconds"

    steps = np.diff(values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return float(default), 1.0, "assumed_seconds"
    dt = float(np.median(steps))

    candidates = (
        ("seconds", 1.0 / dt, 1.0),
        ("milliseconds", 1000.0 / dt, 1e-3),
        ("microseconds", 1_000_000.0 / dt, 1e-6),
    )
    unit, fs, seconds_per_unit = min(
        candidates,
        key=lambda item: abs(float(np.log(item[1] / float(default)))),
    )
    ratio = max(fs / float(default), float(default) / fs)
    if ratio > 1.5:
        raise ValueError(
            f"设备时间列推算采样率异常：{fs:.3f} Hz，预期约 {default:.1f} Hz；"
            "请确认时间单位和采样率配置"
        )
    return float(fs), float(seconds_per_unit), unit


__all__ = ["infer_device_timebase"]
