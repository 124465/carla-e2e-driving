"""Record a demo GIF of a policy driving the dynamic scene.

Renders a HUD with steering, throttle/brake, speed, lane offset and the traffic
light state, so the recording shows what the policy is reacting to rather than
just a moving picture.

Closed-loop performance depends strongly on the starting position (measured lane
offset ranging 0.26 m to 3.79 m across seeds), so `--seed` selects the episode
and the script reports the metrics for the episode it recorded.

Usage:
    python scripts/record_demo.py --checkpoint checkpoints/dagger_best_r01.pt --seed 2
    python scripts/record_demo.py --seed 4 --out logs/demo/drift.gif
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def hud(frame, action, info, t):
    """Compose the camera frame with a compact telemetry overlay."""
    img = Image.fromarray(frame).convert("RGB").resize((480, 240), Image.NEAREST)
    d = ImageDraw.Draw(img)
    steer, thr, brk = float(action[0]), float(action[1]), float(action[2])
    speed = float(info.get("speed", 0.0))
    lat = float(info.get("lane_offset", 0.0))
    light = info.get("light_state", "-")
    ldist = float(info.get("light_dist", 0.0))

    # top bar: light state, speed, lane offset
    colour = {"green": (60, 230, 90), "yellow": (245, 210, 40),
              "red": (240, 60, 60)}.get(light, (200, 200, 200))
    d.rectangle([0, 0, 480, 20], fill=(15, 15, 20))
    d.text((6, 4), f"light {light.upper():6s} {ldist:5.1f}m", fill=colour)
    d.text((210, 4), f"speed {speed:4.1f} m/s", fill=(230, 230, 230))
    lat_col = (90, 230, 120) if abs(lat) < 1.5 else (240, 90, 90)
    d.text((360, 4), f"lat {lat:+.2f}m", fill=lat_col)

    # bottom bar: steer / throttle / brake bars
    d.rectangle([0, 220, 480, 240], fill=(15, 15, 20))
    cx = 60
    d.text((4, 225), "steer", fill=(200, 200, 200))
    d.line([(cx, 230), (cx + int(steer * 40), 230)], fill=(80, 200, 255), width=4)
    d.line([(cx, 222), (cx, 238)], fill=(90, 90, 90), width=1)

    d.text((140, 225), "thr", fill=(200, 200, 200))
    d.rectangle([170, 224, 170 + int(thr * 60), 236], fill=(90, 220, 120))
    d.text((250, 225), "brk", fill=(200, 200, 200))
    d.rectangle([280, 224, 280 + int(brk * 60), 236], fill=(240, 90, 90))

    d.text((380, 225), f"t={t:4d}", fill=(160, 160, 160))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dagger_best_r01.pt")
    ap.add_argument("--backbone", default="tiny_cnn")
    ap.add_argument("--n-frames", type=int, default=1)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="logs/demo/demo.gif")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import yaml
    from carla_e2e.dataset import BCDataset
    from carla_e2e.dynamic_env import DynamicDrivingEnv, DynamicSceneConfig
    from carla_e2e.models import get_model

    cfg = yaml.safe_load(open(args.config))
    mean = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    std = np.array(BCDataset.STD).reshape(3, 1, 1)

    kw = dict(img_h=cfg["data"]["img_size"][0], img_w=cfg["data"]["img_size"][1],
              hidden_dim=cfg["model"]["hidden_dim"], dropout=0.0)
    if args.backbone != "tiny_cnn":
        kw["n_frames"] = args.n_frames
    model = get_model(args.backbone, **kw).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device,
                                     weights_only=True))
    model.eval()

    scfg = dict(cfg.get("dynamic", {}))
    scfg["seed"] = args.seed
    env = DynamicDrivingEnv(DynamicSceneConfig(**scfg))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames = []
    lat, off, red, dist, spd = [], 0, 0, 0.0, []
    obs, info = env.reset(autopilot=False)
    prev = info["position"]
    stack = []
    try:
        for t in range(args.steps):
            stack.append(obs)
            if len(stack) > args.n_frames:
                stack.pop(0)
            while len(stack) < args.n_frames:
                stack.insert(0, stack[0])
            arrs = [(np.asarray(f, np.float32) / 255.0).transpose(2, 0, 1)
                    for f in stack]
            seq = (np.stack(arrs) - mean) / std
            if args.backbone == "temporal_cnn":
                x = torch.from_numpy(seq)[None].float().to(args.device)
                x = x.reshape(x.shape[0], x.shape[1] * 3, *x.shape[3:])
            elif args.backbone == "gru":
                x = torch.from_numpy(seq)[None].float().to(args.device)
            else:
                x = torch.from_numpy(seq[-1])[None].float().to(args.device)

            with torch.no_grad():
                a = model(x).cpu().numpy()[0]
            a[1:] = np.clip(a[1:], 0.0, 1.0)

            frames.append(hud(obs, a, info, t))
            obs, info = env.step(a, autopilot=False)
            lat.append(abs(float(info["lane_offset"])))
            off += int(bool(info.get("off_lane")))
            red += int(bool(info.get("red_light_violation")))
            dist += float(np.linalg.norm(info["position"] - prev))
            prev = info["position"]
            spd.append(float(info["speed"]))
    finally:
        env.close()

    lat = np.array(lat)
    print(f"[demo] seed={args.seed} steps={args.steps} checkpoint={args.checkpoint}")
    print(f"[demo] lane |offset| mean={lat.mean():.3f} P95={np.percentile(lat,95):.2f} "
          f"max={lat.max():.2f} (half-width 1.5 m)")
    print(f"[demo] off-lane {off}/{len(lat)} ({100*off/len(lat):.1f}%)  "
          f"red violations {red}")
    print(f"[demo] distance {dist:.1f} m, mean speed {np.mean(spd):.2f} m/s")

    if imageio is not None:
        imageio.mimsave(args.out, frames, fps=args.fps)
        print(f"[demo] wrote {args.out} ({len(frames)} frames, "
              f"{os.path.getsize(args.out)/1e6:.1f} MB)")
    else:
        for i in (0, len(frames) // 2, len(frames) - 1):
            Image.fromarray(frames[i]).save(args.out.replace(".gif", f"_f{i}.png"))
        print("[demo] imageio missing; wrote PNG snapshots instead")


if __name__ == "__main__":
    main()
