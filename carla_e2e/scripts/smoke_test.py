"""Smoke test: prove the PyBullet backend + dataset + training run end-to-end.

No GPU, no CARLA, minimal disk. Validates:
  - PyBulletEnv reset/step/render + expert oracle
  - a tiny train loop reduces imitation loss
  - DAgger query_expert_control returns a valid action

Run:
    source .venv/bin/activate
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.env import make_env
from carla_e2e.models import get_model
from carla_e2e.dataset import BCDataset


def main():
    env = make_env(backend="pybullet")
    obs, info = env.reset(autopilot=True)
    assert obs is not None and obs.ndim == 3, f"bad obs shape {None if obs is None else obs.shape}"
    assert obs.shape[:2] == (80, 160), f"expected (80,160,3) got {obs.shape}"
    expert = env.query_expert_control()
    assert expert.shape == (3,) and np.all(np.isfinite(expert)), f"bad expert {expert}"
    print(f"[smoke] reset ok; obs={obs.shape}; expert_action={expert.round(3)}")

    # collect a few expert frames in-memory
    X, Y = [], []
    for _ in range(64):
        obs, info = env.step(autopilot=True)
        if obs is None:
            continue
        arr = np.asarray(obs, np.float32) / 255.0
        arr = (arr.transpose(2, 0, 1) - np.array(BCDataset.MEAN).reshape(3, 1, 1)) \
              / np.array(BCDataset.STD).reshape(3, 1, 1)
        X.append(arr); Y.append(info["expert_action"])
    env.close()
    assert len(X) > 0, "no frames collected"

    X = torch.tensor(np.stack(X)).float(); Y = torch.tensor(np.stack(Y)).float()
    model = get_model("tiny_cnn", img_h=80, img_w=160, hidden_dim=64, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    loss0 = crit(model(X), Y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = crit(model(X), Y)
        loss.backward()
        opt.step()
    loss1 = crit(model(X), Y).item()
    print(f"[smoke] BC loss {loss0:.4f} -> {loss1:.4f}  (should decrease)")
    assert loss1 < loss0, "training did not reduce loss"
    print("[smoke] ALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
