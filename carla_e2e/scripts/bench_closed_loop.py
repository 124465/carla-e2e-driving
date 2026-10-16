"""Paired closed-loop benchmark across policies.

Offline metrics do not predict closed-loop driving (measured: a frame-stacked
policy with offline steer correlation +0.912 drove 23.4 m off the lane centre,
while a single-frame policy scoring +0.697 held 1.08 m). Model selection must
therefore be done in closed loop.

To make the comparison fair, every policy is evaluated on the SAME episodes:
the environment is re-created with a fixed seed for each policy, so start
position, traffic layout and lead-brake events are identical. Any difference is
attributable to the policy.

Usage:
    python scripts/bench_closed_loop.py --episodes 3 --steps 400
    python scripts/bench_closed_loop.py --ckpt BC=checkpoints/v7_tiny_cnn_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEFAULT_MODELS = [
    ("BC_single", "checkpoints/v7_tiny_cnn_best.pt", "tiny_cnn", 1),
    ("stack_T4", "checkpoints/v7_temporal_cnn_best.pt", "temporal_cnn", 4),
    ("dagger_r00", "checkpoints/dagger_best_r00.pt", "tiny_cnn", 1),
    ("dagger_r01", "checkpoints/dagger_best_r01.pt", "tiny_cnn", 1),
    ("gru_T8", "checkpoints/v7_gru_best.pt", "gru", 8),
]


def make_input(stack, backbone, mean, std, device):
    arrs = []
    for f in stack:
        a = np.asarray(f, dtype=np.float32) / 255.0
        arrs.append((a.transpose(2, 0, 1) - mean) / std)
    seq = np.stack(arrs)
    if backbone == "temporal_cnn":
        x = torch.from_numpy(seq)[None].float().to(device)
        return x.reshape(x.shape[0], x.shape[1] * 3, *x.shape[3:])
    if backbone == "gru":
        return torch.from_numpy(seq)[None].float().to(device)
    return torch.from_numpy(seq[-1])[None].float().to(device)


def run_policy(ckpt, backbone, n_frames, episodes, steps, seed, device,
               cfg, hidden, dropout):
    from carla_e2e.dynamic_env import DynamicDrivingEnv, DynamicSceneConfig
    from carla_e2e.models import get_model
    from carla_e2e.dataset import BCDataset

    kw = dict(img_h=cfg["data"]["img_size"][0], img_w=cfg["data"]["img_size"][1],
              hidden_dim=hidden, dropout=dropout)
    if backbone != "tiny_cnn":
        kw["n_frames"] = n_frames
    model = get_model(backbone, **kw).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    mean = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    std = np.array(BCDataset.STD).reshape(3, 1, 1)

    # identical episodes for every policy
    scfg = dict(cfg.get("dynamic", {}))
    scfg["seed"] = seed
    env = DynamicDrivingEnv(DynamicSceneConfig(**scfg))

    lat, off, viol, dist_all, surv = [], 0, 0, [], 0
    try:
        for ep in range(episodes):
            obs, info = env.reset(autopilot=False)
            stack = []
            prev = info["position"]
            d = 0.0
            for t in range(steps):
                stack.append(obs)
                if len(stack) > n_frames:
                    stack.pop(0)
                while len(stack) < n_frames:
                    stack.insert(0, stack[0])
                x = make_input(stack, backbone, mean, std, device)
                with torch.no_grad():
                    a = model(x).cpu().numpy()[0]
                a[1:] = np.clip(a[1:], 0.0, 1.0)
                obs, info = env.step(a, autopilot=False)
                lat.append(abs(float(info["lane_offset"])))
                off += int(bool(info.get("off_lane")))
                viol += int(bool(info.get("red_light_violation")))
                d += float(np.linalg.norm(info["position"] - prev))
                prev = info["position"]
            dist_all.append(d)
            surv += 1
    finally:
        env.close()
    lat = np.array(lat)
    return {
        "lane_mean": float(lat.mean()),
        "lane_p95": float(np.percentile(lat, 95)),
        "lane_max": float(lat.max()),
        "off_pct": 100.0 * off / max(1, len(lat)),
        "red_violations": int(viol),
        "distance": float(np.mean(dist_all)),
        "survival": f"{surv}/{episodes}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", action="append", default=None,
                    help="extra NAME=path[:backbone[:T]] entry")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    hidden = cfg["model"]["hidden_dim"]
    dropout = 0.0

    models = list(DEFAULT_MODELS)
    for spec in (args.ckpt or []):
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        bb = parts[1] if len(parts) > 1 else "tiny_cnn"
        T = int(parts[2]) if len(parts) > 2 else (1 if bb == "tiny_cnn" else 4)
        models.append((name, parts[0], bb, T))

    results = []
    for name, ckpt, bb, T in models:
        if not os.path.exists(ckpt):
            print(f"[bench] skip {name}: {ckpt} missing")
            continue
        r = run_policy(ckpt, bb, T, args.episodes, args.steps, args.seed,
                       args.device, cfg, hidden, dropout)
        r.update(name=name, backbone=bb, n_frames=T, checkpoint=ckpt)
        results.append(r)
        print(f"[bench] {name:12s} lane={r['lane_mean']:6.3f} "
              f"P95={r['lane_p95']:6.2f} off={r['off_pct']:5.1f}% "
              f"red={r['red_violations']:3d} dist={r['distance']:6.1f} "
              f"surv={r['survival']}", flush=True)

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_closed_loop.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 88)
    print(f"{'model':12s} {'backbone':13s} {'T':>2s} {'laneMean':>9s} "
          f"{'laneP95':>8s} {'off%':>6s} {'red':>4s} {'dist':>7s} {'surv':>5s}")
    print("-" * 88)
    for r in sorted(results, key=lambda x: x["lane_mean"]):
        print(f"{r['name']:12s} {r['backbone']:13s} {r['n_frames']:2d} "
              f"{r['lane_mean']:9.3f} {r['lane_p95']:8.2f} {r['off_pct']:6.1f} "
              f"{r['red_violations']:4d} {r['distance']:7.1f} {r['survival']:>5s}")
    print("\n(lane half-width = 1.5 m; lower laneMean/off% and 0 red is better)")
    print("wrote logs/bench_closed_loop.json")


if __name__ == "__main__":
    main()
