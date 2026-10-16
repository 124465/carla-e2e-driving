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
    """Collect one run. Runs in a separate process.

    `start_frac` places this episode at a distinct arc position along the track,
    so that a batch of runs jointly covers the whole circuit instead of every
    episode seeing a similar slice (which was why an earlier dataset produced
    per-run steering distributions of std 0.077 vs 0.226 and no transferable
    image->steer mapping).
    """
    run_id, steps, raw_dir, seed, backend, start_frac = job

    import numpy as np
    from PIL import Image

    from carla_e2e.factory import build_env

    cfg = {"backend": backend, "dynamic": {}, "pybullet": {}}
    env = build_env(cfg, backend=backend)

    # deterministic, evenly spread start along the track
    if backend == "dynamic":
        import math
        ring_len = 2 * math.pi * 46.0
        env.cfg.randomize_start = False
        obs, _ = env.reset(autopilot=True)
        # move the ego to the requested arc position
        if hasattr(env, "ego"):
            s0 = start_frac * ring_len
            from carla_e2e.dynamic_env import _RING_R, _LANE_OFFSETS
            r0 = _RING_R + _LANE_OFFSETS[env.ego.lane]
            th = s0 / _RING_R
            env.ego.s = s0
            env.ego.x = r0 * math.cos(th)
            env.ego.y = r0 * math.sin(th)
            env.ego.heading = env._lane_tangent(s0)
            env._last_lat = env._ego_lat_offset()
            env._sync_visuals()
            obs = env._render()
            _ = env._get_info()
    else:
        obs, _ = env.reset(autopilot=True)

    meta_path = os.path.join(raw_dir, f"meta_run{run_id:02d}.jsonl")
    n = 0
    with open(meta_path, "w") as mf:
        for step in range(steps):
            info = env._get_info()
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
    ap.add_argument("--backend", default="dynamic",
                    choices=["dynamic", "pybullet"])
    ap.add_argument("--start-id", type=int, default=None)
    args = ap.parse_args()

    import multiprocessing as mp

    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(os.path.join(raw_dir, "images"), exist_ok=True)
    start_id = args.start_id
    if start_id is None:
        existing = [f for f in os.listdir(raw_dir) if f.startswith("meta_run")]
        start_id = len(existing) + 1

    # Spread episode start positions evenly around the track so the batch covers
    # the whole circuit. Previously every episode started wherever the RNG put
    # it, which gave per-run steering std of 0.077 vs 0.226 and no mapping a
    # policy could generalise across.
    jobs = []
    for i in range(args.runs):
        frac = (i / max(1, args.runs)) if args.runs > 1 else 0.0
        jobs.append((start_id + i, args.steps, raw_dir, 1000 + i,
                     args.backend, frac))

    print(f"[par-collect] {args.runs} runs x {args.steps} steps "
          f"across {args.workers} workers, backend={args.backend} -> {raw_dir}")
    print(f"[par-collect] start positions spread over the track: "
          f"{[round(j[5], 2) for j in jobs]}")

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
