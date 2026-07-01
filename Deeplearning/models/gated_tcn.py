from __future__ import annotations
import torch
import torch.nn as nn

def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class SeparableTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * int(dilation)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels),
            _group_norm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            _group_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class AttentionTemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, feature_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            _group_norm(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            SeparableTemporalBlock(hidden_channels, dilation=1, dropout=dropout),
            SeparableTemporalBlock(hidden_channels, dilation=2, dropout=dropout),
            SeparableTemporalBlock(hidden_channels, dilation=4, dropout=dropout),
        )
        self.to_feature = nn.Sequential(
            nn.Conv1d(hidden_channels, feature_dim, kernel_size=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )
        self.att = nn.Sequential(
            nn.Conv1d(feature_dim, max(16, feature_dim // 2), kernel_size=1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(max(16, feature_dim // 2), 1, kernel_size=1),
        )
        self.stats = nn.Sequential(
            nn.Linear(in_channels * 5, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.to_feature(self.blocks(self.proj(x)))
        w = torch.softmax(self.att(feat).squeeze(1), dim=-1)
        pooled = torch.sum(feat * w.unsqueeze(1), dim=-1)
        stat = torch.cat([
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x.abs().mean(dim=-1),
            torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-6),
            x[..., -1] - x[..., 0],
        ], dim=1)
        return pooled + self.stats(stat)

class GatedTCNFusionRegressor(nn.Module):
    def __init__(self, emg_channels=12, kin_channels=63, hidden_channels=32, feature_dim=64, dropout=0.1):
        super().__init__()
        self.emg_channels = emg_channels
        self.kin_channels = kin_channels
        self.emg_encoder = AttentionTemporalEncoder(emg_channels, hidden_channels, feature_dim, dropout)
        self.kin_encoder = AttentionTemporalEncoder(kin_channels, hidden_channels, feature_dim, dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, max(16, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, feature_dim // 2), 1),
        )

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False):
        if emg.ndim != 3 or kin.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got emg={tuple(emg.shape)}, kin={tuple(kin.shape)}")
        if emg.shape[1] != self.emg_channels or kin.shape[1] != self.kin_channels:
            raise ValueError(f"Bad channel count: emg={emg.shape[1]}, kin={kin.shape[1]}")
        ef = self.emg_encoder(emg)
        kf = self.kin_encoder(kin)
        ctx = torch.cat([ef, kf, torch.abs(ef - kf), ef * kf], dim=1)
        gate = self.gate(ctx)
        fused = gate * ef + (1.0 - gate) * kf
        feat = torch.cat([fused, ef, kf, torch.abs(ef - kf)], dim=1)
        pred = self.regressor(feat)
        if return_features:
            return pred, fused
        return pred
