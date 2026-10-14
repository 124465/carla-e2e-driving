"""Generate synthetic expert data so the train/eval pipeline can be validated
BEFORE CARLA finishes downloading (or on a machine without a GPU).

The synthetic 'image' is just colored noise; the 'action' is a simple function
of the image so the model has something to learn. This ONLY validates the code
path (dataset -> training -> checkpoint). Do NOT use it for the real project.

Usage:
    source .venv/bin/activate
    python scripts/make_dummy_data.py --samples 2000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = os.path.abspath(args.raw_dir)
    img_dir = os.path.join(raw, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta = os.path.join(raw, "meta_run00.jsonl")

    with open(meta, "w") as mf:
        for i in range(args.samples):
            # fake observation: road-ish gradient + noise
            base = rng.uniform(0, 255, size=(80, 160, 3)).astype(np.uint8)
            img = Image.fromarray(base)
            fname = f"run00_{i:06d}.png"
            img.save(os.path.join(img_dir, fname))
            # fake expert: steer correlates with horizontal brightness center
            gray = base.mean(2)
            cx = gray.mean(1).argmax() / 160.0 - 0.5
            steer = float(np.clip(cx * 2.0, -1, 1))
            throttle = float(rng.uniform(0.3, 0.8))
            brake = float(rng.uniform(0.0, 0.1))
            mf.write(json.dumps({"img": f"images/{fname}",
                                 "action": [steer, throttle, brake]}) + "\n")
    print(f"[dummy] wrote {args.samples} samples to {meta}")


if __name__ == "__main__":
    main()
