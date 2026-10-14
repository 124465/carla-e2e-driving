# ros2_ws (driving-project subset)

This repository is a selective export of a ROS2 workspace. It contains **only** the
end-to-end autonomous-driving imitation-learning project:

- `carla_e2e/` — training/sim project (PyTorch + PyBullet backend, optional CARLA)
- `src/carla_e2e_pkg/` — ROS2 `ament_python` deployment node

Other packages in the original workspace (`yolo_detector`, `follower_*`, etc.)
and all build artifacts (`build/`, `install/`, `log/`), python venvs, and the
abandoned partial CARLA download are **intentionally git-ignored** — see
`.gitignore`.

See [`carla_e2e/README.md`](carla_e2e/README.md) for the full project write-up,
[`carla_e2e/RESUME_BLURB.md`](carla_e2e/RESUME_BLURB.md) for a résumé-ready
description, and [`carla_e2e/INTERVIEW_TALK.md`](carla_e2e/INTERVIEW_TALK.md) for
interview talking points.

## Quick start
```bash
# training side (py3.10 venv with torch cu121 + pybullet)
cd carla_e2e && source .venv/bin/activate
python scripts/smoke_test.py
python scripts/collect.py --runs 3 --steps 400
python scripts/train.py
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 3

# ROS2 deployment side (py3.12 venv w/ --system-site-packages + CPU torch)
source /opt/ros/jazzy/setup.bash
source carla_e2e/.venv312/bin/activate
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$(pwd)/carla_e2e
ros2 run carla_e2e_pkg driving_policy_node --checkpoint checkpoints/best.pt
```
