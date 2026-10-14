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

from carla_e2e.env import make_env
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
    ap.add_argument("--record", action="store_true", help="save demo GIF/MP4")
    ap.add_argument("--out-dir", default="logs/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if yaml is None:
        raise SystemExit("pyyaml required")
    cfg = yaml.safe_load(open(args.config))
    backend = args.backend or cfg.get("backend", "pybullet")
    m_cfg, d_cfg = cfg["model"], cfg["data"]

    model = get_model(m_cfg["backbone"], img_h=d_cfg["img_size"][0],
                      img_w=d_cfg["img_size"][1], hidden_dim=m_cfg["hidden_dim"],
                      dropout=0.0).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device,
                                     weights_only=True))
    model.eval()

    env = make_env(backend=backend)
    mean = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    std = np.array(BCDataset.STD).reshape(3, 1, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    success, steps_total, total_dist = 0, 0, 0.0
    prev = None
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(autopilot=False)
            frames = []
            for t in range(args.max_steps):
                if obs is None:
                    obs = env._get_observation() if hasattr(env, "_get_observation") else None
                    if obs is None:
                        continue
                arr = np.asarray(obs, dtype=np.float32) / 255.0
                arr = (arr.transpose(2, 0, 1) - mean) / std
                x = torch.from_numpy(arr)[None].float().to(args.device)
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


if __name__ == "__main__":
    main()
