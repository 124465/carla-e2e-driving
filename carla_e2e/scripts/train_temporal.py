"""Train a driving policy with optional temporal (multi-frame) input.

Supports the three backbones so single-frame and temporal policies can be
compared under identical data / schedule / optimiser settings:

    tiny_cnn      : 1 frame  -> (B,3,H,W)
    temporal_cnn  : T stacked frames -> (B,3T,H,W)
    gru           : T frames -> (B,T,3,H,W)

Everything is GPU-resident (no DataLoader) so the comparison measures the model,
not the input pipeline.

Usage:
    python scripts/train_temporal.py --backbone temporal_cnn --n-frames 4
    python scripts/train_temporal.py --backbone tiny_cnn   --n-frames 1
    python scripts/train_temporal.py --backbone gru        --n-frames 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.fast_data import GPUDataset, GPUSequenceData
from carla_e2e.models import get_model

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--backbone", default="temporal_cnn",
                    choices=["tiny_cnn", "temporal_cnn", "gru"])
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--raw-dir", default="data/dynamic")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None, help="checkpoint path")
    ap.add_argument("--tag", default=None, help="tensorboard run tag")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["training"]
    m_cfg = cfg["model"]
    img_h, img_w = cfg["data"]["img_size"]
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else tr["lr"]

    torch.manual_seed(tr.get("seed", 42))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    temporal = args.backbone != "tiny_cnn"
    n_frames = args.n_frames if temporal else 1

    # ---------------- data (GPU resident) ----------------
    if temporal:
        ds = GPUSequenceData(args.raw_dir, (img_h, img_w), n_frames=n_frames,
                             device=device)
    else:
        ds = GPUDataset(args.raw_dir, (img_h, img_w), device=device)

    n_total = len(ds)
    n_val = max(1, int(n_total * tr["val_split"]))
    perm = torch.randperm(n_total, device=device)
    va, trn = perm[:n_val], perm[n_val:]

    # Keep uint8 images on the device and normalize per batch. Materialising the
    # whole split as float32 needs N*T*3*H*W*4 bytes (T=4 -> ~5.9 GB, T=8 ->
    # ~12 GB) and OOMs a 4 GB card, which is exactly how the first attempt at
    # this comparison failed. `gather`/`normalize` already do the conversion, so
    # store index tensors instead of image tensors.
    if temporal:
        tr_idx = trn
        va_idx = va
        Ytr = ds.win_act[trn]
        Yva = ds.win_act[va]

        def get_x(idx):
            return ds.gather(idx)[0]
    else:
        tr_idx = trn
        va_idx = va
        Ytr = ds.actions[trn]
        Yva = ds.actions[va]

        def get_x(idx):
            return ds.normalize(ds.images[idx])

    if args.batch_size == "auto":
        free, _ = torch.cuda.mem_get_info() if device.type == "cuda" else (2**31, 1)
        per = 1.6 * n_frames * ((img_h * img_w) / (80 * 160))
        bs = int(max(32, min(512, (free / 2**20 - 512) / max(per, 1))))
        bs = 2 ** (bs.bit_length() - 1)
        bs = min(bs, max(16, len(trn) // 4))
    else:
        bs = int(args.batch_size)

    # ---------------- model ----------------
    kw = dict(img_h=img_h, img_w=img_w, hidden_dim=m_cfg["hidden_dim"],
              dropout=m_cfg["dropout"])
    if args.backbone == "temporal_cnn":
        model = get_model("temporal_cnn", n_frames=n_frames, **kw).to(device)
        def prep(x):
            b, t = x.shape[0], x.shape[1]
            return x.reshape(b, t * 3, *x.shape[3:])
    elif args.backbone == "gru":
        model = get_model("gru", n_frames=n_frames, **kw).to(device)
        def prep(x):
            return x
    else:
        model = get_model("tiny_cnn", **kw).to(device)
        def prep(x):
            return x.squeeze(1) if x.dim() == 5 else x

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[temporal] backbone={args.backbone} T={n_frames} params={n_params/1e6:.3f}M "
          f"train={len(trn)} val={len(va)} batch={bs} amp={args.amp}")

    w = torch.tensor(tr["loss_weights"], device=device).float()
    crit = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    tag = args.tag or f"{args.backbone}_T{n_frames}"
    log_dir = os.path.join(tr["log_dir"], tag)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    out = args.out or os.path.join(tr["checkpoint_dir"], f"{tag}_best.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def run_epoch(train: bool):
        model.train(train)
        total, seen = 0.0, 0
        idx_all, Y = (tr_idx, Ytr) if train else (va_idx, Yva)
        n = len(idx_all)
        order = torch.randperm(n, device=device) if train \
            else torch.arange(n, device=device)
        with torch.set_grad_enabled(train):
            for i in range(0, n, bs):
                sel = order[i:i + bs]
                # normalise only this batch (uint8 stays resident on the GPU)
                xb = prep(get_x(idx_all[sel])); yb = Y[sel]
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=args.amp and device.type == "cuda"):
                    loss = (crit(model(xb), yb) * w).mean()
                if train:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                total += loss.item() * xb.size(0); seen += xb.size(0)
        return total / max(1, seen)

    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        te = time.time()
        trl = run_epoch(True)
        val = run_epoch(False)
        dt = time.time() - te
        writer.add_scalar("loss/train", trl, ep)
        writer.add_scalar("loss/val", val, ep)
        if val < best:
            best = val
            torch.save(model.state_dict(), out)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[temporal] ep {ep:02d} train={trl:.4f} val={val:.4f} "
                  f"{dt:.2f}s")
    writer.close()
    print(f"[temporal] best val={best:.4f} -> {out}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
