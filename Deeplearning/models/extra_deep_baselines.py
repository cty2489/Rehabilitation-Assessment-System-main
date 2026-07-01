from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFusionRegressor(nn.Module):
    """Flattened MLP fusion baseline.

    This is a non-temporal deep baseline. It flattens the aligned EMG/KIN input
    and predicts FMA through fully connected layers.
    """

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        seq_len: int = 64,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.emg_channels = emg_channels
        self.kin_channels = kin_channels
        self.seq_len = seq_len
        in_dim = (emg_channels + kin_channels) * seq_len
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(feature_dim, 1)

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False):
        x = torch.cat([emg, kin], dim=1)
        x = x.flatten(start_dim=1)
        feat = self.net(x)
        pred = self.regressor(feat)
        if return_features:
            return pred, feat
        return pred


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(16, dim // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(16, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class BiLSTMAttentionRegressor(nn.Module):
    """BiLSTM with temporal attention over aligned EMG/KIN tokens."""

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        hidden_dim: int = 64,
        feature_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        input_dim = emg_channels + kin_channels
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = TemporalAttentionPooling(feature_dim, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(32, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, feature_dim // 2), 1),
        )

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False):
        x = torch.cat([emg, kin], dim=1).transpose(1, 2)  # [B,T,C]
        x = self.input_norm(x)
        z, _ = self.lstm(x)
        z = self.proj(z)
        feat, _ = self.pool(z)
        pred = self.regressor(feat)
        if return_features:
            return pred, feat
        return pred


class TransformerFusionRegressor(nn.Module):
    """Transformer fusion baseline with modality-aware aligned tokens."""

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        hidden_dim: int = 96,
        feature_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_proj = nn.Linear(emg_channels + kin_channels, feature_dim)
        self.modality_gate = nn.Sequential(
            nn.Linear(emg_channels + kin_channels, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        if feature_dim % nhead != 0:
            nhead = 2 if feature_dim % 2 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=max(hidden_dim * 4, feature_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = TemporalAttentionPooling(feature_dim, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(32, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, feature_dim // 2), 1),
        )

    def _positional_encoding(self, length: int, dim: int, device, dtype):
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 2)))
        pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
        if dim > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False):
        x = torch.cat([emg, kin], dim=1).transpose(1, 2)  # [B,T,C]
        z = self.input_proj(x)
        z = z * self.modality_gate(x)
        z = z + self._positional_encoding(z.shape[1], z.shape[2], z.device, z.dtype)
        z = self.encoder(z)
        feat, _ = self.pool(z)
        pred = self.regressor(feat)
        if return_features:
            return pred, feat
        return pred


class MultiScaleTemporalCNNRegressor(nn.Module):
    """InceptionTime-style multi-scale temporal CNN baseline."""

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        hidden_dim: int = 48,
        feature_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        in_channels = emg_channels + kin_channels
        kernels = [3, 5, 9, 15]
        branch_dim = max(16, hidden_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_dim, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_dim),
                nn.GELU(),
                nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(branch_dim),
                nn.GELU(),
            )
            for k in kernels
        ])
        fused_dim = branch_dim * len(kernels)
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(32, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, feature_dim // 2), 1),
        )

    def forward(self, emg: torch.Tensor, kin: torch.Tensor, return_features: bool = False):
        x = torch.cat([emg, kin], dim=1)
        z = torch.cat([branch(x) for branch in self.branches], dim=1)
        z = self.fusion(z)
        feat = self.pool(z).squeeze(-1)
        pred = self.regressor(feat)
        if return_features:
            return pred, feat
        return pred
