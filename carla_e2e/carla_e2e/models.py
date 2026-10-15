"""Small CNN policy for end-to-end driving (fits a 4GB GPU)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyDrivingNet(nn.Module):
    """Image (B,3,H,W) -> (B,3) = (steer, throttle, brake).

    Steering is unbounded in the head; the env clips it to [-1,1].
    Throttle/brake heads are passed through a sigmoid so they stay in [0,1].
    Small enough to train on a GTX 1650.
    """

    def __init__(self, img_h: int = 80, img_w: int = 160,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        # compute flattened feature size from a dummy forward pass
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 3, img_h, img_w)).flatten(1).shape[1]
        self.flat = int(flat)
        self.head = nn.Sequential(
            nn.Linear(self.flat, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        f = self.conv(x).flatten(1)
        return self.head(f)


def get_model(name: str, **kw):
    if name in ("tiny_cnn", "TinyDrivingNet"):
        return TinyDrivingNet(**kw)
    if name in ("temporal_cnn", "TemporalDrivingNet"):
        return TemporalDrivingNet(**kw)
    if name in ("gru", "GRUDrivingNet"):
        return GRUDrivingNet(**kw)
    raise ValueError(f"unknown model backbone: {name}")


class TemporalDrivingNet(nn.Module):
    """Frame-stacked CNN: (B, T*3, H, W) -> (B, 3).

    The simplest way to give a policy memory: feed the last T frames as extra
    input channels. Cheap (no recurrence) and it lets the network compare
    consecutive frames, which is what makes traffic-light state changes and
    lead-vehicle closing rate observable at all.
    """

    def __init__(self, img_h: int = 80, img_w: int = 160, n_frames: int = 4,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.n_frames = n_frames
        c_in = 3 * n_frames
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c_in, img_h, img_w)).flatten(1).shape[1]
        self.flat = int(flat)
        self.head = nn.Sequential(
            nn.Linear(self.flat, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class GRUDrivingNet(nn.Module):
    """Per-frame CNN encoder + GRU over time: (B, T, 3, H, W) -> (B, 3).

    Shares one encoder across frames (far fewer parameters than frame stacking)
    and lets the recurrent state carry information indefinitely, which suits
    stop-and-go car following better than a fixed window.
    """

    def __init__(self, img_h: int = 80, img_w: int = 160, n_frames: int = 8,
                 hidden_dim: int = 256, gru_hidden: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.n_frames = n_frames
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            feat = self.encoder(torch.zeros(1, 3, img_h, img_w)).flatten(1).shape[1]
        self.feat = int(feat)
        self.proj = nn.Linear(self.feat, gru_hidden)
        self.gru = nn.GRU(gru_hidden, gru_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        # x: (B, T, 3, H, W)
        b, t = x.shape[0], x.shape[1]
        f = self.encoder(x.reshape(b * t, *x.shape[2:])).flatten(1)
        f = torch.relu(self.proj(f)).reshape(b, t, -1)
        out, _ = self.gru(f)
        return self.head(out[:, -1])
