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
    raise ValueError(f"unknown model backbone: {name}")
