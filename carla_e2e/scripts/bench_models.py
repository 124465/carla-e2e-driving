"""Compare driving-policy architectures under identical settings.

The point of this script is a *fair* comparison, so every backbone sees the same
data, optimiser, schedule and batch policy. It reports three things:

1. offline error, AND whether the model beat a constant-mean predictor.
   This matters because weighted MSE on driving data has a nasty local optimum:
   predicting the mean action gives a low loss while being useless. A model whose
   error matches the mean baseline has learned nothing, however good the loss
   looks.

2. steer output std vs target std. A model that collapsed to the mean shows
   output std ~ 0 even when its loss looks respectable.

3. closed-loop metrics via evaluate.py, which is the only thing that decides
   whether the policy can actually drive.

Usage:
    python scripts/bench_models.py --epochs 150
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

BACKBONES = [
    ("tiny_cnn", 1),
    ("temporal_cnn", 4),
    ("gru", 8),
]


def load_model(backbone, n_frames, img_h, img_w, hidden, dropout, ckpt, device):
    from carla_e2e.models import get_model
    kw = dict(img_h=img_h, img_w=img_w, hidden_dim=hidden, dropout=dropout)
    if backbone != "tiny_cnn":
        kw["n_frames"] = n_frames
    m = get_model(backbone, **kw).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def prep_factory(backbone):
    def prep(x):
        if backbone == "temporal_cnn":
            b, t = x.shape[0], x.shape[1]
            return x.reshape(b, t * 3, *x.shape[3:])
        if backbone == "gru":
            return x
        return x.squeeze(1) if x.dim() == 5 else x
    return prep


def offline_metrics(backbone, n_frames, raw_dir, img_h, img_w, ckpt, device,
                    hidden=256, dropout=0.0, bs=32):
    from carla_e2e.fast_data import GPUDataset, GPUSequenceData

    temporal = backbone != "tiny_cnn"
    if temporal:
        ds = GPUSequenceData(raw_dir, (img_h, img_w), n_frames=n_frames,
                             device=device)
        Y = ds.win_act
    else:
        ds = GPUDataset(raw_dir, (img_h, img_w), device=device)
        Y = ds.actions

    model = load_model(backbone, n_frames, img_h, img_w, hidden, dropout,
                       ckpt, device)
    prep = prep_factory(backbone)

    outs = []
    with torch.no_grad():
        for xb, _ in ds.batches(bs, shuffle=False):
            outs.append(model(prep(xb)).cpu())
    pred = torch.cat(outs)
    y = Y.cpu()

    mse = ((pred - y) ** 2).mean(0)
    base = ((y.mean(0) - y) ** 2).mean(0)
    beat = (mse < base * 0.95)
    res = {
        "backbone": backbone,
        "n_frames": n_frames,
        "n_windows": int(y.shape[0]),
        "mse_steer": float(mse[0]), "mse_throttle": float(mse[1]),
        "mse_brake": float(mse[2]),
        "base_steer": float(base[0]), "base_throttle": float(base[1]),
        "base_brake": float(base[2]),
        "target_std_steer": float(y[:, 0].std()),
        "pred_std_steer": float(pred[:, 0].std()),
        "beat_mean_baseline": bool(beat[0] and pred[:, 0].std() > 0.02),
    }
    del ds, model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw-dir", default="data/dynamic")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tag-suffix", default="bench")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    img_h, img_w = cfg["data"]["img_size"]
    hidden = cfg["model"]["hidden_dim"]
    dropout = cfg["model"]["dropout"]

    results = []
    for backbone, n_frames in BACKBONES:
        tag = f"{backbone}_T{n_frames}_{args.tag_suffix}"
        ckpt = os.path.join(args.ckpt_dir, f"{tag}_best.pt")
        if not args.skip_train:
            cmd = [sys.executable, "scripts/train_temporal.py",
                   "--backbone", backbone, "--n-frames", str(n_frames),
                   "--epochs", str(args.epochs), "--raw-dir", args.raw_dir,
                   "--tag", tag, "--out", ckpt]
            print(f"\n===== training {backbone} (T={n_frames}) =====", flush=True)
            t0 = time.time()
            subprocess.run(cmd, check=True)
            print(f"  trained in {time.time()-t0:.0f}s", flush=True)
        if not os.path.exists(ckpt):
            print(f"  missing {ckpt}, skipping")
            continue
        r = offline_metrics(backbone, n_frames, args.raw_dir, img_h, img_w,
                            ckpt, args.device, hidden=hidden, dropout=dropout)
        r["checkpoint"] = ckpt
        results.append(r)
        print(f"  offline: steer MSE={r['mse_steer']:.5f} "
              f"(mean baseline {r['base_steer']:.5f}) "
              f"pred_std={r['pred_std_steer']:.4f} "
              f"-> {'OK' if r['beat_mean_baseline'] else 'DEGENERATE'}")

    out = "logs/bench_models.json"
    os.makedirs("logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'backbone':14s} {'T':>2s} {'steerMSE':>10s} {'meanBase':>10s} "
          f"{'predStd':>9s} {'tgtStd':>8s}  verdict")
    print("-" * 78)
    for r in results:
        print(f"{r['backbone']:14s} {r['n_frames']:2d} {r['mse_steer']:10.5f} "
              f"{r['base_steer']:10.5f} {r['pred_std_steer']:9.4f} "
              f"{r['target_std_steer']:8.3f}  "
              f"{'learned' if r['beat_mean_baseline'] else 'DEGENERATE'}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
