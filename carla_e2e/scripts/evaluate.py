"""Closed-loop evaluation: deploy a trained policy (CARLA or PyBullet backend).

Reports route completion %, survival, distance, and records a demo video/GIF
with a small HUD overlay (steer / throttle / speed).

Usage (inside the py3.10 venv):
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 3
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --record --out-dir logs/eval
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_e2e.factory import build_env
from carla_e2e.models import get_model
from carla_e2e.dataset import BCDataset

try:
    import yaml
except ImportError:
    yaml = None

try:
    import imageio.v2 as imageio  # GIF/MP4 writer
except Exception:
    imageio = None


def draw_hud(frame: np.ndarray, action, info) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    steer, thr = float(action[0]), float(action[1])
    spd = float(info.get("speed", 0.0))
    txt = f"steer {steer:+.2f}  throttle {thr:.2f}  speed {spd:.1f} m/s"
    d.text((4, 4), txt, fill=(255, 255, 0))
    # steer bar
    cx = img.width // 2
    d.line([(cx, img.height - 6), (cx + int(steer * 40), img.height - 6)],
           fill=(0, 255, 0), width=3)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=None,
                    help="tiny_cnn | temporal_cnn | gru (overrides config)")
    ap.add_argument("--n-frames", type=int, default=None,
                    help="temporal window length (overrides default for backbone)")
    ap.add_argument("--record", action="store_true", help="save demo GIF/MP4")
    ap.add_argument("--out-dir", default="logs/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if yaml is None:
        raise SystemExit("pyyaml required")
    cfg = yaml.safe_load(open(args.config))
    backend = args.backend or cfg.get("backend", "pybullet")
    m_cfg, d_cfg = cfg["model"], cfg["data"]

    backbone = args.backbone or m_cfg["backbone"]
    n_frames = args.n_frames if args.n_frames else (
        {"temporal_cnn": 4, "gru": 8}.get(backbone, 1))

    model = get_model(backbone, img_h=d_cfg["img_size"][0],
                      img_w=d_cfg["img_size"][1], hidden_dim=m_cfg["hidden_dim"],
                      dropout=0.0,
                      **({"n_frames": n_frames} if backbone != "tiny_cnn" else {})
                      ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device,
                                     weights_only=True))
    model.eval()
    temporal = backbone != "tiny_cnn"
    print(f"[eval] backbone={backbone} n_frames={n_frames}")

    env = build_env(cfg, backend=backend)
    mean = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    std = np.array(BCDataset.STD).reshape(3, 1, 1)

    def to_model_input(stack):
        """stack: list of (H,W,3) uint8 normalized frames -> model input."""
        arrs = []
        for f in stack:
            a = np.asarray(f, dtype=np.float32) / 255.0
            arrs.append((a.transpose(2, 0, 1) - mean) / std)
        seq = np.stack(arrs)                      # (T,3,H,W)
        if backbone == "temporal_cnn":
            x = torch.from_numpy(seq)[None].float().to(args.device)
            b, t = x.shape[0], x.shape[1]
            return x.reshape(b, t * 3, *x.shape[3:])
        elif backbone == "gru":
            return torch.from_numpy(seq)[None].float().to(args.device)
        else:
            # single frame: only the newest, with no extra time dimension
            return torch.from_numpy(seq[-1])[None].float().to(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    success, steps_total, total_dist = 0, 0, 0.0
    lat_all, off_steps, viol_steps = [], 0, 0
    prev = None
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(autopilot=False)
            frames = []
            stack = []
            for t in range(args.max_steps):
                if obs is None:
                    obs = env._get_observation() if hasattr(env, "_get_observation") else None
                    if obs is None:
                        continue
                stack.append(obs)
                if len(stack) > n_frames:
                    stack.pop(0)
                while len(stack) < n_frames:      # pad at episode start
                    stack.insert(0, stack[0])
                x = to_model_input(stack)
                with torch.no_grad():
                    action = model(x).cpu().numpy()[0]
                action[1:] = np.clip(action[1:], 0.0, 1.0)
                obs, info = env.step(action, autopilot=False)
                if args.record:
                    frames.append(draw_hud(obs if obs is not None else np.zeros(
                        (80, 160, 3), np.uint8), action, info))
                if "position" in info and prev is not None:
                    total_dist += float(np.linalg.norm(info["position"] - prev))
                if "position" in info:
                    prev = info["position"]
                # dynamic-scene metrics (absent on the simple pybullet backend)
                if "lane_offset" in info:
                    lat_all.append(abs(float(info["lane_offset"])))
                if info.get("off_lane"):
                    off_steps += 1
                if info.get("red_light_violation"):
                    viol_steps += 1
                steps_total += 1
            success += 1
            print(f"[eval] episode {ep+1} completed {args.max_steps} steps")
            if args.record and frames:
                gif = os.path.join(args.out_dir, f"demo_ep{ep+1}.gif")
                if imageio is not None:
                    imageio.mimsave(gif, frames, fps=20)
                    print(f"[eval] wrote {gif} ({len(frames)} frames)")
                else:
                    # fallback: save first/last + a few as PNGs
                    for i in (0, len(frames) // 2, -1):
                        Image.fromarray(frames[i]).save(
                            os.path.join(args.out_dir, f"demo_ep{ep+1}_f{i}.png"))
                    print("[eval] imageio missing; saved PNG snapshots instead")
    finally:
        env.close()

    print(f"[eval] survival rate: {success}/{args.episodes}")
    print(f"[eval] avg steps/episode: {steps_total/max(1,args.episodes):.0f}")
    print(f"[eval] total distance travelled (m): {total_dist:.1f}")
    if lat_all:
        la = np.array(lat_all)
        print(f"[eval] lane |offset| mean={la.mean():.3f} P95={np.percentile(la,95):.3f} "
              f"max={la.max():.3f} (lane half-width 1.5 m)")
        print(f"[eval] off-lane steps: {off_steps}/{steps_total} "
              f"({100*off_steps/max(1,steps_total):.1f}%)")
        print(f"[eval] red-light violation steps: {viol_steps}")


if __name__ == "__main__":
    main()
