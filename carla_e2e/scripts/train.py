"""Train a behavior-cloning driving policy (GPU-accelerated).

Two data paths:
  * --data gpu  (default when CUDA is available): whole dataset pre-loaded into
    VRAM as uint8, normalized on-device. No DataLoader workers, no per-epoch
    disk decode -> GPU stays saturated.
  * --data disk : the original on-disk BCDataset with a DataLoader (CPU
    fallback, or when the dataset is too large to fit in VRAM).

Usage (inside the py3.10 venv):
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --data gpu --batch-size auto --amp
    python scripts/train.py --data disk          # original behaviour
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.dataset import BCDataset
from carla_e2e.fast_data import GPUDataset
from carla_e2e.models import get_model

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    if yaml is None:
        raise SystemExit("pyyaml not installed; install it in the venv")
    with open(path) as f:
        return yaml.safe_load(f)


def pick_auto_batch(n_samples, device, img_h, img_w, reserve_mib=512, max_frac=0.25):
    """Largest power-of-two batch that fits VRAM *and* keeps enough steps/epoch.

    A batch that swallows the whole dataset degenerates into full-batch gradient
    descent (unstable, few updates per epoch), so we also cap the batch at
    max_frac of the training set to preserve several optimizer steps per epoch.
    """
    if device.type != "cuda":
        return min(64, max(8, n_samples // 8))
    free, _total = torch.cuda.mem_get_info()
    free_mib = free / 2**20
    per_sample_mib = max(0.6, (img_h * img_w) / (80 * 160) * 1.6)
    usable = max(128.0, free_mib - reserve_mib)
    bs_vram = int(usable / per_sample_mib)
    bs_frac = max(32, int(n_samples * max_frac))
    bs = max(32, min(bs_vram, bs_frac, 2048))
    return 2 ** (bs.bit_length() - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", choices=["auto", "gpu", "disk"], default="auto")
    ap.add_argument("--batch-size", default="auto",
                    help="'auto' or an integer")
    ap.add_argument("--amp", action="store_true", help="use fp16 autocast")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dagger-dir", default=None,
                    help="also include DAgger data from this dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg, m_cfg, d_cfg = cfg["training"], cfg["model"], cfg["data"]
    device = torch.device(args.device)
    img_h, img_w = d_cfg["img_size"]
    epochs = args.epochs or tr_cfg["epochs"]

    torch.manual_seed(tr_cfg.get("seed", 42))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[train] device={device} ({torch.cuda.get_device_name(0)})")

    # ---------- data ----------
    mode = args.data
    if mode == "auto":
        mode = "gpu" if device.type == "cuda" else "disk"

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    if mode == "gpu":
        ds = GPUDataset(d_cfg["raw_dir"], (img_h, img_w), device=device,
                        dagger_dir=args.dagger_dir)
        n_val = max(1, int(len(ds) * tr_cfg["val_split"]))
        perm = torch.randperm(len(ds), device=device)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        Xtr, Ytr = ds.normalize(ds.images[train_idx]), ds.actions[train_idx]
        Xva, Yva = ds.normalize(ds.images[val_idx]), ds.actions[val_idx]
        bs = pick_auto_batch(len(train_idx), device, img_h, img_w) \
            if args.batch_size == "auto" else int(args.batch_size)
        n_train, n_val_actual = len(train_idx), len(val_idx)
        print(f"[train] data=gpu(VRAM) train={n_train} val={n_val_actual} "
              f"batch={bs} amp={use_amp}")
    else:
        dataset = BCDataset(d_cfg["raw_dir"], (img_h, img_w))
        n_val = max(1, int(len(dataset) * tr_cfg["val_split"]))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        bs = tr_cfg["batch_size"] if args.batch_size == "auto" else int(args.batch_size)
        pin = device.type == "cuda"
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=tr_cfg["num_workers"], pin_memory=pin)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=tr_cfg["num_workers"], pin_memory=pin)
        n_train, n_val_actual = len(train_ds), len(val_ds)
        print(f"[train] data=disk train={n_train} val={n_val_actual} batch={bs}")

    model = get_model(m_cfg["backbone"], img_h=img_h, img_w=img_w,
                      hidden_dim=m_cfg["hidden_dim"],
                      dropout=m_cfg["dropout"]).to(device)
    w = torch.tensor(tr_cfg["loss_weights"], device=device).float()
    crit = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"],
                           weight_decay=tr_cfg["weight_decay"])

    os.makedirs(tr_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(tr_cfg["log_dir"], exist_ok=True)
    writer = SummaryWriter(tr_cfg["log_dir"])

    def run_train_epoch():
        model.train()
        total, seen = 0.0, 0
        if mode == "gpu":
            perm = torch.randperm(n_train, device=device)
            for i in range(0, n_train, bs):
                sel = perm[i:i + bs]
                xb, yb = Xtr[sel], Ytr[sel]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = (crit(model(xb), yb) * w).mean()
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                total += loss.item() * xb.size(0); seen += xb.size(0)
        else:
            for xb, yb in train_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = (crit(model(xb), yb) * w).mean()
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                total += loss.item() * xb.size(0); seen += xb.size(0)
        return total / max(1, seen)

    @torch.no_grad()
    def run_val_epoch():
        model.eval()
        if mode == "gpu":
            model_dtype_ok = True
            total = 0.0
            for i in range(0, n_val_actual, max(bs, 256)):
                xb, yb = Xva[i:i + bs], Yva[i:i + bs]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    total += (crit(model(xb), yb) * w).mean().item() * xb.size(0)
            return total / max(1, n_val_actual)
        total, seen = 0.0, 0
        for xb, yb in val_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                total += (crit(model(xb), yb) * w).mean().item() * xb.size(0)
            seen += xb.size(0)
        return total / max(1, seen)

    best_val = float("inf")
    t_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = run_train_epoch()
        val_loss = run_val_epoch()
        dt = time.time() - t0
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("time/epoch_s", dt, epoch)
        ips = n_train / dt
        print(f"[train] epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  {dt:.2f}s ({ips:.0f} samples/s)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(tr_cfg["checkpoint_dir"], "best.pt"))
    torch.save(model.state_dict(),
               os.path.join(tr_cfg["checkpoint_dir"], "last.pt"))
    writer.close()
    print(f"[train] done -> checkpoints/best.pt  (total {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
