"""Collect expert demonstrations (works for both CARLA and PyBullet backends).

Usage (inside the py3.10 venv):
    python scripts/collect.py --runs 3 --steps 800
    # backend is read from configs/base.yaml (default: pybullet, SSD-friendly)

Saves:  data/raw/images/runXX_YYYYYY.png
        data/raw/meta_runXX.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.factory import build_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--backend", type=str, default=None,
                    help="override backend in config (pybullet|carla)")
    ap.add_argument("--raw-dir", type=str, default="data/raw")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml"))
    backend = args.backend or cfg.get("backend", "pybullet")

    raw = os.path.abspath(args.raw_dir)
    img_dir = os.path.join(raw, "images")
    os.makedirs(img_dir, exist_ok=True)

    env = build_env(cfg, backend=backend)
    try:
        for run in range(1, args.runs + 1):
            run_id = f"{run:02d}"
            meta_path = os.path.join(raw, f"meta_run{run_id}.jsonl")
            env.reset(autopilot=True)
            with open(meta_path, "w") as mf:
                for step in range(args.steps):
                    obs, info = env.step(autopilot=True)
                    if obs is None:
                        continue
                    action = info["expert_action"]
                    fname = f"run{run_id}_{step:06d}.png"
                    Image.fromarray(obs).save(os.path.join(img_dir, fname))
                    mf.write(json.dumps({"img": f"images/{fname}",
                                          "action": action.tolist()}) + "\n")
            print(f"[collect] run {run_id} done -> {args.steps} frames")
    finally:
        env.close()
    print("[collect] finished; total samples:", glob_meta(raw))


def glob_meta(raw):
    from glob import glob
    n = 0
    for p in glob(os.path.join(raw, "meta_run*.jsonl")):
        n += sum(1 for _ in open(p))
    return n


if __name__ == "__main__":
    main()
