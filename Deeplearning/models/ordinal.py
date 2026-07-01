from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OrdinalConfig:
    thresholds: tuple[float, ...]
    score_min: float = 0.0
    score_max: float = 20.0
    blend_weight: float = 0.25
    dropout: float = 0.1
    label_smoothing: float = 0.02
    ordinal_weight: float = 0.2


def parse_thresholds(text: str | None, score_min: float = 0.0, score_max: float = 20.0) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        # Default bins tailored to the current MHH FMA range while still valid on [0, 20].
        return (15.5, 17.0, 18.5, 19.5)
    raw = [item.strip() for item in str(text).split(",") if item.strip()]
    if not raw:
        return ()
    values = sorted({float(item) for item in raw})
    if values[0] <= score_min or values[-1] >= score_max:
        raise ValueError(
            f"Ordinal thresholds must lie strictly inside ({score_min}, {score_max}), got {values}"
        )
    return tuple(values)


def cumulative_targets(y_true_fma: torch.Tensor, thresholds: Sequence[float], label_smoothing: float = 0.0) -> torch.Tensor:
    if y_true_fma.ndim != 1:
        raise ValueError(f"Expected 1D FMA targets for ordinal conversion, got {tuple(y_true_fma.shape)}")
    if len(thresholds) == 0:
        return torch.zeros((y_true_fma.shape[0], 0), device=y_true_fma.device, dtype=y_true_fma.dtype)
    threshold_tensor = torch.tensor(list(thresholds), device=y_true_fma.device, dtype=y_true_fma.dtype)
    target = (y_true_fma.unsqueeze(1) > threshold_tensor.unsqueeze(0)).to(y_true_fma.dtype)
    if label_smoothing > 0:
        smooth = min(max(float(label_smoothing), 0.0), 0.49)
        target = target * (1.0 - smooth) + 0.5 * smooth
    return target


def ordinal_probs_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return monotonic cumulative probs and class probs from ordinal logits."""
    if logits.ndim != 2:
        raise ValueError(f"Expected ordinal logits [batch, thresholds], got {tuple(logits.shape)}")
    cumulative = torch.sigmoid(logits)
    if cumulative.shape[1] > 1:
        # Enforce monotonic decreasing P(y > threshold) from low to high thresholds.
        flipped = torch.flip(cumulative, dims=[1])
        flipped = torch.cummax(flipped, dim=1).values
        cumulative = torch.flip(flipped, dims=[1])
    class_probs = []
    prev = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
    for index in range(cumulative.shape[1]):
        current = cumulative[:, index : index + 1]
        class_probs.append(torch.clamp(prev - current, min=0.0, max=1.0))
        prev = current
    class_probs.append(torch.clamp(prev, min=0.0, max=1.0))
    probs = torch.cat(class_probs, dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return cumulative, probs


def class_centers(thresholds: Sequence[float], score_min: float = 0.0, score_max: float = 20.0) -> np.ndarray:
    bounds = [float(score_min), *[float(value) for value in thresholds], float(score_max)]
    centers = [0.5 * (left + right) for left, right in zip(bounds[:-1], bounds[1:])]
    return np.asarray(centers, dtype=np.float32)


class OrdinalRegressionWrapper(nn.Module):
    """Adds a lightweight ordinal auxiliary head without breaking single-task FMA regression.

    The base model must support forward(..., return_features=True) and return
    (prediction, pooled_features). The wrapper keeps a regression output while also
    exposing ordinal logits that can be used for auxiliary supervision and optional
    inference-time blending.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        thresholds: Sequence[float],
        score_min: float = 0.0,
        score_max: float = 20.0,
        blend_weight: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.thresholds = tuple(float(value) for value in thresholds)
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.blend_weight = float(max(0.0, min(1.0, blend_weight)))
        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, max(1, len(self.thresholds))),
        )
        centers = class_centers(self.thresholds, self.score_min, self.score_max)
        self.register_buffer("class_centers", torch.tensor(centers, dtype=torch.float32), persistent=False)

    def forward(self, emg: torch.Tensor, kin: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_prediction, features = self.base_model(emg, kin, return_features=True)
        logits = self.ordinal_head(features)
        if len(self.thresholds) == 0:
            return {
                "raw_fma": raw_prediction,
                "fma": raw_prediction,
                "ordinal_logits": logits[:, :0],
            }
        _, class_probs = ordinal_probs_from_logits(logits)
        ordinal_expected = torch.sum(class_probs * self.class_centers.view(1, -1), dim=1, keepdim=True)
        blended = (1.0 - self.blend_weight) * raw_prediction + self.blend_weight * ordinal_expected
        return {
            "raw_fma": raw_prediction,
            "fma": blended,
            "ordinal_logits": logits,
            "ordinal_expected": ordinal_expected,
            "ordinal_class_probs": class_probs,
        }


class OrdinalAuxiliaryLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        thresholds: Sequence[float],
        ordinal_weight: float = 0.2,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.thresholds = tuple(float(value) for value in thresholds)
        self.ordinal_weight = float(max(0.0, ordinal_weight))
        self.label_smoothing = float(max(0.0, min(0.49, label_smoothing)))

    def forward(self, prediction: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor, raw_target_fma: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(prediction, dict):
            regression_pred = prediction.get("raw_fma", prediction.get("fma"))
            ordinal_logits = prediction.get("ordinal_logits")
        else:
            regression_pred = prediction
            ordinal_logits = None
        if regression_pred is None:
            raise ValueError("OrdinalAuxiliaryLoss requires a regression prediction tensor.")
        base_loss = self.base_loss(regression_pred.squeeze(-1), target)
        if ordinal_logits is None or ordinal_logits.numel() == 0 or self.ordinal_weight <= 0.0:
            return base_loss
        if raw_target_fma is None:
            raise ValueError("OrdinalAuxiliaryLoss requires raw_target_fma for ordinal supervision.")
        ordinal_target = cumulative_targets(raw_target_fma.view(-1), self.thresholds, self.label_smoothing)
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_target)
        return base_loss + self.ordinal_weight * ordinal_loss
