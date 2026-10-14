"""Train a behavior-cloning driving policy.

Usage (inside the py3.10 venv):
    python scripts/train.py --config configs/base.yaml

Saves checkpoints to checkpoints/ and TensorBoard logs to logs/.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.dataset import BCDataset
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg["training"]
    m_cfg = cfg["model"]
    d_cfg = cfg["data"]

    torch.manual_seed(tr_cfg.get("seed", 42))

    dataset = BCDataset(d_cfg["raw_dir"], tuple(d_cfg["img_size"]))
    n_val = max(1, int(len(dataset) * tr_cfg["val_split"]))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=tr_cfg["batch_size"],
                          shuffle=True, num_workers=tr_cfg["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=tr_cfg["batch_size"],
                        shuffle=False, num_workers=tr_cfg["num_workers"])

    model = get_model(m_cfg["backbone"], img_h=d_cfg["img_size"][0],
                      img_w=d_cfg["img_size"][1], hidden_dim=m_cfg["hidden_dim"],
                      dropout=m_cfg["dropout"]).to(args.device)

    w = torch.tensor(tr_cfg["loss_weights"], device=args.device).float()
    crit = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"],
                           weight_decay=tr_cfg["weight_decay"])

    os.makedirs(tr_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(tr_cfg["log_dir"], exist_ok=True)
    writer = SummaryWriter(tr_cfg["log_dir"])

    best_val = float("inf")
    for epoch in range(tr_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = (crit(pred, y) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                loss = (crit(model(x), y) * w).mean()
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        print(f"[train] epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(tr_cfg["checkpoint_dir"], "best.pt"))
    torch.save(model.state_dict(),
               os.path.join(tr_cfg["checkpoint_dir"], "last.pt"))
    writer.close()
    print("[train] done -> checkpoints/best.pt")


if __name__ == "__main__":
    main()
