from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(c):
    g = min(8, int(c))
    while g > 1 and c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class Block(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=(k // 2) * d, dilation=d, groups=c),
            _gn(c), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(c, c, 1), _gn(c),
        )
    def forward(self, x):
        return F.gelu(x + self.net(x))

class Branch(nn.Module):
    def __init__(self, cin, h, f, p):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, h, 1), _gn(h), nn.GELU())
        self.s = nn.Conv1d(h, h, 3, padding=1, groups=h)
        self.m = nn.Conv1d(h, h, 5, padding=2, groups=h)
        self.l = nn.Conv1d(h, h, 9, padding=4, groups=h)
        self.mix = nn.Sequential(
            nn.Conv1d(h * 3, h, 1), _gn(h), nn.GELU(),
            Block(h, 5, 1, p), Block(h, 5, 2, p), Block(h, 5, 4, p),
            nn.Conv1d(h, f, 1), _gn(f), nn.GELU(),
        )
        self.att = nn.Sequential(nn.Conv1d(f, max(16, f // 2), 1), nn.Tanh(), nn.Dropout(p), nn.Conv1d(max(16, f // 2), 1, 1))
        self.stats = nn.Sequential(nn.Linear(cin * 6, f), nn.LayerNorm(f), nn.GELU(), nn.Dropout(p))
        self.out = nn.Sequential(nn.LayerNorm(f * 3), nn.Linear(f * 3, f), nn.GELU(), nn.Dropout(p))
    def forward(self, x):
        z = self.stem(x)
        z = self.mix(torch.cat([self.s(z), self.m(z), self.l(z)], dim=1))
        w = torch.softmax(self.att(z).squeeze(1), dim=-1)
        att = torch.sum(z * w.unsqueeze(1), dim=-1)
        dx = x[..., 1:] - x[..., :-1]
        st = torch.cat([x.mean(-1), x.std(-1, unbiased=False), x.abs().mean(-1), torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-6), x[..., -1] - x[..., 0], dx.abs().mean(-1)], dim=1)
        return self.out(torch.cat([att, z.mean(-1) + z.amax(-1), self.stats(st)], dim=1))

class TaskAwarePrototypeTCNRegressor(nn.Module):
    uses_task_metadata = True
    def __init__(self, emg_channels=12, kin_channels=63, hidden_channels=32, feature_dim=64, task_embedding_dim=16, dropout=0.1, score_min=14.0, score_max=20.0, residual_scale=1.0, num_tasks=30, num_trials=3):
        super().__init__()
        centers = torch.arange(float(score_min), float(score_max) + 1e-6, 1.0, dtype=torch.float32)
        self.register_buffer("score_centers", centers, persistent=False)
        self.residual_scale = float(residual_scale)
        self.task_emb = nn.Embedding(int(num_tasks) + 1, int(task_embedding_dim))
        self.trial_emb = nn.Embedding(int(num_trials) + 1, max(4, int(task_embedding_dim) // 2))
        meta_dim = int(task_embedding_dim) + max(4, int(task_embedding_dim) // 2)
        self.emg = Branch(emg_channels, hidden_channels, feature_dim, dropout)
        self.kin = Branch(kin_channels, hidden_channels, feature_dim, dropout)
        self.meta = nn.Sequential(nn.LayerNorm(meta_dim), nn.Linear(meta_dim, feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.LayerNorm(feature_dim * 5), nn.Linear(feature_dim * 5, feature_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim * 6), nn.Linear(feature_dim * 6, feature_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(feature_dim * 2, feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(feature_dim, int(centers.numel()))
        self.res = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.res.weight)
        nn.init.zeros_(self.res.bias)
    def forward(self, emg, kin, task_id=None, trial_number=None, return_features=False):
        b = emg.shape[0]
        if task_id is None:
            task_id = torch.zeros(b, device=emg.device, dtype=torch.long)
        if trial_number is None:
            trial_number = torch.zeros(b, device=emg.device, dtype=torch.long)
        task_id = torch.clamp(task_id.to(emg.device, dtype=torch.long), 0, self.task_emb.num_embeddings - 1)
        trial_number = torch.clamp(trial_number.to(emg.device, dtype=torch.long), 0, self.trial_emb.num_embeddings - 1)
        meta = self.meta(torch.cat([self.task_emb(task_id), self.trial_emb(trial_number)], dim=1))
        ef, kf = self.emg(emg), self.kin(kin)
        diff, prod = torch.abs(ef - kf), ef * kf
        gate = self.gate(torch.cat([ef, kf, diff, prod, meta], dim=1))
        fused = gate * ef + (1.0 - gate) * kf
        feat = self.trunk(torch.cat([fused, ef, kf, diff, prod, meta], dim=1))
        pred = torch.sum(torch.softmax(self.cls(feat), dim=1) * self.score_centers.view(1, -1), dim=1, keepdim=True)
        pred = pred + self.residual_scale * torch.tanh(self.res(feat))
        return (pred, feat) if return_features else pred
