from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualTemporalBlock(nn.Module):
    """Small residual 1D conv block.

    Input shape: [batch, channels, time].
    Output shape: [batch, channels, time].
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            _group_norm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(x.shape)}")
        return self.activation(x + self.block(x))


class StableSignalEncoder(nn.Module):
    """Dual-branch encoder used before optional fusion modules.

    Input shape: [batch, in_channels, time].
    Output shape: [batch, feature_dim, time].
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32, feature_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            _group_norm(hidden_channels),
            nn.GELU(),
            ResidualTemporalBlock(hidden_channels, kernel_size=5, dilation=1, dropout=dropout),
            ResidualTemporalBlock(hidden_channels, kernel_size=5, dilation=2, dropout=dropout),
            nn.Conv1d(hidden_channels, feature_dim, kernel_size=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(x.shape)}")
        return self.encoder(x)


class TemporalAttentionPooling(nn.Module):
    """Lightweight temporal attention pooling.

    Input shape: [batch, feature_dim, time].
    Output shape: [batch, feature_dim].
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(16, feature_dim // 2)
        self.scorer = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, feature_dim, time], got {tuple(x.shape)}")
        logits = self.scorer(x).squeeze(1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(1), dim=-1)
        return pooled, weights


class DualBranchConvRegressor(nn.Module):
    """Improved small-sample dual-branch EMG/KIN regressor.

    Forward input:
    - emg: [batch, 12, time]
    - kin: [batch, 63, time]

    Forward output:
    - fma: [batch, 1]
    """

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        hidden_channels: int = 32,
        feature_dim: int = 64,
        dropout: float = 0.1,
        use_temporal_attention: bool = False,
    ):
        super().__init__()
        self.emg_channels = emg_channels
        self.kin_channels = kin_channels
        self.use_temporal_attention = use_temporal_attention
        self.emg_encoder = StableSignalEncoder(emg_channels, hidden_channels, feature_dim, dropout=dropout)
        self.kin_encoder = StableSignalEncoder(kin_channels, hidden_channels, feature_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1),
            _group_norm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualTemporalBlock(feature_dim, kernel_size=3, dilation=1, dropout=dropout),
        )
        self.temporal_attention = TemporalAttentionPooling(feature_dim, dropout=dropout) if use_temporal_attention else None
        regressor_hidden = max(16, feature_dim // 2)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, regressor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden, 1),
        )

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if emg.ndim != 3 or kin.ndim != 3:
            raise ValueError(
                "Expected EMG and KIN tensors shaped [batch, channels, time], "
                f"got emg={tuple(emg.shape)}, kin={tuple(kin.shape)}"
            )
        if emg.shape[1] != self.emg_channels:
            raise ValueError(f"Expected {self.emg_channels} EMG channels, got {emg.shape[1]}")
        if kin.shape[1] != self.kin_channels:
            raise ValueError(f"Expected {self.kin_channels} KIN channels, got {kin.shape[1]}")
        if emg.shape[0] != kin.shape[0] or emg.shape[-1] != kin.shape[-1]:
            raise ValueError(f"Batch/time mismatch: emg={tuple(emg.shape)}, kin={tuple(kin.shape)}")

        emg_features = self.emg_encoder(emg)
        kin_features = self.kin_encoder(kin)
        fused = self.fusion(torch.cat([emg_features, kin_features], dim=1))
        if self.temporal_attention is None:
            pooled = fused.mean(dim=-1)
        else:
            pooled, _ = self.temporal_attention(fused)
        prediction = self.regressor(pooled)
        if return_features:
            return prediction, pooled
        return prediction
