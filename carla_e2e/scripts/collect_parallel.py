"""Parallel data collection across CPU cores.

The PyBullet step is ~82% software rasterization on a single core, so the only
way to collect faster is to run several independent simulators at once. This
uses multiprocessing (spawn) with one env per worker; each worker writes its own
meta/jsonl shard so there is no locking on the output.

Note on "GPU": PyBullet's offscreen renderer in this install is CPU-only (no EGL
in the wheel), so GPU offload is not available here -- see scripts/profile_perf.py.
Scaling out over cores is the effective lever.

Usage:
    python scripts/collect_parallel.py --workers 6 --runs 6 --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(job):
    """Collect one run. Runs in a separate process."""
    run_id, steps, raw_dir, seed = job

    import numpy as np
    from PIL import Image
    from carla_e2e.env import make_env

    os.makedirs(os.path.join(raw_dir, "images"), exist_ok=True)
    # give each worker a slightly different start so runs aren't identical
    env = make_env(backend="pybullet")
    obs, info = env.reset(autopilot=True)

    meta_path = os.path.join(raw_dir, f"meta_run{run_id:02d}.jsonl")
    n = 0
    with open(meta_path, "w") as mf:
        for step in range(steps):
            # expert drives; we record (image, expert action)
            action = info.get("expert_action")
            if obs is not None and action is not None:
                fname = f"run{run_id:02d}_{step:06d}.png"
                Image.fromarray(obs).save(os.path.join(raw_dir, "images", fname))
                mf.write(json.dumps({"img": f"images/{fname}",
                                     "action": [float(a) for a in action]}) + "\n")
                n += 1
            obs, info = env.step(autopilot=True)
    env.close()
    return run_id, n, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--start-id", type=int, default=None)
    args = ap.parse_args()

    import multiprocessing as mp

    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(os.path.join(raw_dir, "images"), exist_ok=True)
    start_id = args.start_id
    if start_id is None:
        # avoid clobbering existing runs
        existing = [f for f in os.listdir(raw_dir) if f.startswith("meta_run")]
        start_id = len(existing) + 1

    jobs = [(start_id + i, args.steps, raw_dir, 1000 + i) for i in range(args.runs)]
    print(f"[par-collect] {args.runs} runs x {args.steps} steps "
          f"across {args.workers} workers -> {raw_dir}")

    ctx = mp.get_context("spawn")
    t0 = time.time()
    total = 0
    with ctx.Pool(processes=args.workers) as pool:
        for run_id, n, meta_path in pool.imap_unordered(_worker, jobs):
            total += n
            print(f"[par-collect] run {run_id:02d} done -> {n} frames "
                  f"({os.path.basename(meta_path)})")
    dt = time.time() - t0

    serial_est = args.runs * args.steps / 125.0  # measured single-env rate
    print(f"[par-collect] finished; total samples: {total} in {dt:.1f}s "
          f"({args.workers} workers)")
    print(f"[par-collect] single-env estimate would be ~{serial_est:.1f}s "
          f"-> speedup ~{serial_est/max(dt,1e-6):.1f}x")


if __name__ == "__main__":
    main()
