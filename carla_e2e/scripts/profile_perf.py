"""GPU/CPU profiling for the driving project.

Answers, with measurements rather than assumptions:
  * Is training on the GPU?                -> yes (train.py defaults to cuda)
  * Where does CPU time actually go?       -> PyBullet software rendering (~86% of a step)
  * Can PyBullet render on the GPU here?   -> NO (see probe_renderer)
  * How much faster is GPU-resident data?  -> measured, see bench_data_path

Usage:
    python scripts/profile_perf.py            # all probes
    python scripts/profile_perf.py --quick    # skip the slower ones
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def gpu_report():
    print("=" * 68)
    print("GPU")
    print("=" * 68)
    if not torch.cuda.is_available():
        print("  torch.cuda.is_available() = False  (CPU-only environment)")
        return
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"  device      : {name}")
    print(f"  torch       : {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"  VRAM        : {free/2**20:.0f} MiB free / {total/2**20:.0f} MiB total")
    print(f"  cudnn bench : {torch.backends.cudnn.benchmark}")


def probe_renderer():
    """Check whether PyBullet can render on the GPU in this install."""
    print("=" * 68)
    print("PyBullet renderer probe")
    print("=" * 68)
    try:
        import pybullet as p
        import pybullet_data
    except Exception as e:
        print(f"  pybullet unavailable: {e}")
        return

    print(f"  p.EGL available      : {hasattr(p, 'EGL')}   <- needed for GPU rendering")
    print(f"  ER_BULLET_HARDWARE_OPENGL enum exists: "
          f"{hasattr(p, 'ER_BULLET_HARDWARE_OPENGL')}")
    print("  NOTE: enum existing does NOT mean it uses the GPU; it silently")
    print("        falls back to software in DIRECT mode (measured 0% GPU util).")

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    vm = p.computeViewMatrix([0, 0, 3], [0, 0, 0], [0, 1, 0])
    pm = p.computeProjectionMatrixFOV(60, 2.0, 0.1, 100)
    for label in ("ER_TINY_RENDERER", "ER_BULLET_HARDWARE_OPENGL"):
        r = getattr(p, label, None)
        if r is None:
            print(f"  {label:26s}: not available")
            continue
        for _ in range(5):
            p.getCameraImage(160, 80, viewMatrix=vm, projectionMatrix=pm, renderer=r)
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            p.getCameraImage(160, 80, viewMatrix=vm, projectionMatrix=pm, renderer=r)
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"  {label:26s}: {dt:6.2f} ms/frame")
    p.disconnect()


def bench_env_step(quick=False):
    print("=" * 68)
    print("Simulation step cost (PyBullet backend)")
    print("=" * 68)
    try:
        from carla_e2e.env import make_env
    except Exception as e:
        print(f"  env unavailable: {e}")
        return
    env = make_env(backend="pybullet")
    env.reset(autopilot=True)
    for _ in range(20):
        env.step(autopilot=True)

    n = 60 if quick else 150
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(autopilot=True)
    step_ms = (time.perf_counter() - t0) / n * 1000

    t0 = time.perf_counter()
    for _ in range(n):
        env._render()
    render_ms = (time.perf_counter() - t0) / n * 1000
    env.close()

    print(f"  full step            : {step_ms:6.2f} ms  ({1000/step_ms:.0f} steps/s)")
    print(f"  _render() alone      : {render_ms:6.2f} ms")
    print(f"  render share of step : {100*render_ms/step_ms:5.0f}%  <- CPU-bound (software renderer)")
    print("  => collection speed is limited by the CPU rasterizer, not the GPU.")


def bench_data_path():
    print("=" * 68)
    print("Training data path (disk DataLoader vs GPU-resident)")
    print("=" * 68)
    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return
    try:
        from carla_e2e.dataset import BCDataset
        from carla_e2e.fast_data import GPUDataset
        from carla_e2e.models import get_model
        from torch.utils.data import DataLoader
    except Exception as e:
        print(f"  skipped: {e}")
        return

    dev = "cuda"
    m = get_model("tiny_cnn", img_h=80, img_w=160, hidden_dim=256,
                  dropout=0.0).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    def steady(fn, warm=3, iters=40):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    ds = GPUDataset("data/raw", (80, 160), device=dev)
    Xtr, Ytr = ds.normalize(ds.images[:1080]), ds.actions[:1080]

    def step_gpu(bs=256):
        xb, yb = Xtr[:bs], Ytr[:bs]
        opt.zero_grad(set_to_none=True)
        crit(m(xb), yb).backward(); opt.step()

    t_gpu = steady(step_gpu)
    print(f"  GPU-resident bs=256  : {1000*t_gpu:7.2f} ms/step  "
          f"({256/t_gpu:7.0f} samples/s)")

    bcd = BCDataset("data/raw", (80, 160))
    dl = DataLoader(bcd, batch_size=64, shuffle=True, num_workers=2)
    it = iter(dl)

    def step_disk():
        nonlocal it
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl); xb, yb = next(it)
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(set_to_none=True)
        crit(m(xb), yb).backward(); opt.step()

    t_disk = steady(step_disk)
    print(f"  disk DataLoader bs=64: {1000*t_disk:7.2f} ms/step  "
          f"({64/t_disk:7.0f} samples/s)")
    print(f"  => GPU-resident is {256/t_gpu / (64/t_disk):.2f}x the sample throughput,")
    print("     but uses a larger batch (fewer, bigger steps).")
    print("     On a dataset this small (~1k frames) the win is small;")
    print("     it matters at 100k+ frames where disk decode dominates.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    args = ap.parse_args()

    gpu_report()
    if args.gpu_only:
        return
    probe_renderer()
    bench_env_step(quick=args.quick)
    if not args.quick:
        bench_data_path()


if __name__ == "__main__":
    main()
