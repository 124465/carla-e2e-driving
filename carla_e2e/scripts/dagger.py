"""DAgger: Dataset Aggregation for end-to-end driving.

Reduces covariate shift by rolling out the *current* policy, having the expert
label the states the policy actually visits, and re-training on the aggregated
dataset. Each round:
  1. run policy (with beta-mix: at each step, with prob beta let the policy act,
     else let the expert act) and record (image, *expert* action)
  2. append to the aggregated dataset
  3. re-train the policy on the full aggregated dataset from scratch

Usage (inside the py3.10 venv, CARLA running in another terminal):
    python scripts/dagger.py --config configs/base.yaml

Dependencies: same as train.py (pytorch + carla). Uses model/training sections
of the config plus the new `dagger` section.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.factory import build_env
from carla_e2e.models import get_model
from carla_e2e.dataset import BCDataset

try:
    import yaml
except ImportError:
    yaml = None


QUERY_EXPERT_MODE = "autopilot"  # or "behavior_agent" (needs CARLA agents on PYTHONPATH)


def to_tensor(obs, mean, std, size):
    arr = np.asarray(obs, dtype=np.float32) / 255.0
    arr = (arr.transpose(2, 0, 1) - mean) / std
    return torch.from_numpy(arr).float()


def expert_label(env: CarlaEnv):
    if QUERY_EXPERT_MODE == "autopilot":
        return env.query_expert_control()
    raise NotImplementedError(QUERY_EXPERT_MODE)


def run_round(round_i, beta, cfg, model, device, writer):
    backend = cfg.get("backend", "pybullet")
    env = build_env(cfg, backend=backend)
    m_cfg, d_cfg, dg = cfg["model"], cfg["data"], cfg["dagger"]
    mean = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    std = np.array(BCDataset.STD).reshape(3, 1, 1)
    out_dir = os.path.abspath(dg["out_dir"])
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, f"meta_dagger_{round_i:02d}.jsonl")

    model.eval()
    rng = np.random.default_rng(dg["seed"] + round_i)
    n_new = 0
    with open(meta_path, "w") as mf:
        for ep in range(dg["episodes_per_round"]):
            obs, _ = env.reset(autopilot=False)
            for step in range(dg["steps_per_episode"]):
                if obs is None:
                    obs = env._get_observation()
                    if obs is None:
                        continue
                # beta-mix: let policy act with prob beta, else expert acts
                if rng.random() < beta:
                    x = to_tensor(obs, mean, std, d_cfg["img_size"])[None].to(device)
                    with torch.no_grad():
                        action = model(x).cpu().numpy()[0]
                    action[1:] = np.clip(action[1:], 0.0, 1.0)
                    obs, _ = env.step(action, autopilot=False)
                else:
                    _, info = env.step(None, autopilot=True)
                    obs = env._get_observation()
                # expert labels the *current* state regardless of who acted
                expert = expert_label(env)
                if expert is None or obs is None:
                    continue
                fname = f"dagger{round_i:02d}_{ep:02d}_{step:05d}.png"
                Image.fromarray(obs).save(os.path.join(img_dir, fname))
                mf.write(json.dumps({"img": f"images/{fname}",
                                     "action": expert.tolist()}) + "\n")
                n_new += 1
    env.close()
    print(f"[dagger] round {round_i}: +{n_new} expert-labeled samples -> {meta_path}")
    return n_new


def retrain(round_i, cfg, device):
    """Re-train from scratch on ALL aggregated data (BC + dagger metas)."""
    tr, m, d = cfg["training"], cfg["model"], cfg["data"]
    # include both behavior-cloning meta_* and dagger meta_dagger_* files
    # (BCDataset already globs meta_run*.jsonl; add dagger files manually)
    base_ds = BCDataset(d["raw_dir"], tuple(d["img_size"]))
    dagger_dir = os.path.abspath(cfg["dagger"]["out_dir"])
    dagger_samples = []
    import glob
    for p in sorted(glob.glob(os.path.join(dagger_dir, "meta_dagger_*.jsonl"))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dagger_samples.append((
                    os.path.join(dagger_dir, rec["img"]),
                    np.asarray(rec["action"], dtype=np.float32),
                ))
    # combine into a single dataset object
    ds = BCDataset.__new__(BCDataset)
    ds.raw_dir = d["raw_dir"]
    ds.img_size = tuple(d["img_size"])
    ds.samples = base_ds.samples + dagger_samples
    print(f"[dagger] aggregated dataset size: {len(ds)}")

    n_val = max(1, int(len(ds) * tr["val_split"]))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    tl = torch.utils.data.DataLoader(train_ds, batch_size=tr["batch_size"],
                                    shuffle=True, num_workers=tr["num_workers"])
    vl = torch.utils.data.DataLoader(val_ds, batch_size=tr["batch_size"],
                                     shuffle=False, num_workers=tr["num_workers"])

    model = get_model(m["backbone"], img_h=d["img_size"][0], img_w=d["img_size"][1],
                      hidden_dim=m["hidden_dim"], dropout=m["dropout"]).to(device)
    w = torch.tensor(tr["loss_weights"], device=device).float()
    crit = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                           weight_decay=tr["weight_decay"])
    ckpt = os.path.join(tr["checkpoint_dir"], f"dagger_best_r{round_i:02d}.pt")
    os.makedirs(tr["checkpoint_dir"], exist_ok=True)
    best_val = float("inf")
    for epoch in range(tr["epochs"]):
        model.train()
        tl_ = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = (crit(model(x), y) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl_ += loss.item() * x.size(0)
        model.eval()
        vl_ = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                vl_ += (crit(model(x), y) * w).mean().item() * x.size(0)
        tl_, vl_ = tl_ / len(train_ds), vl_ / len(val_ds)
        print(f"[dagger] round {round_i} epoch {epoch:02d}  train={tl_:.4f} val={vl_:.4f}")
        if vl_ < best_val:
            best_val = vl_
            torch.save(model.state_dict(), ckpt)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = args.device
    dg = cfg["dagger"]

    # round 0: start from a behavior-cloned policy if available, else random init
    model = None
    start_ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "best.pt")
    if os.path.exists(start_ckpt):
        m = cfg["model"]; d = cfg["data"]
        model = get_model(m["backbone"], img_h=d["img_size"][0], img_w=d["img_size"][1],
                          hidden_dim=m["hidden_dim"], dropout=m["dropout"]).to(device)
        model.load_state_dict(torch.load(start_ckpt, map_location=device))
        print("[dagger] loaded initial BC policy from", start_ckpt)
    else:
        print("[dagger] no BC checkpoint found; DAgger round 0 starts from random init")

    beta = dg["mix_prob"]
    for r in range(dg["rounds"]):
        print(f"\n===== DAgger round {r} (beta={beta:.2f}) =====")
        if model is not None:
            run_round(r, beta, cfg, model, device, None)
        model = retrain(r, cfg, device)
        beta *= dg["beta_decay"]
    print("[dagger] done. final policy:",
          os.path.join(cfg["training"]["checkpoint_dir"],
                       f"dagger_best_r{dg['rounds']-1:02d}.pt"))


if __name__ == "__main__":
    main()
