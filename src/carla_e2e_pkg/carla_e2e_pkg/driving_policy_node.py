"""ROS2 entry point for the CARLA end-to-end driving policy.

Single source of truth: the actual model/dataset code lives in
ros2_ws/carla_e2e/carla_e2e/ (the py3.10 training project). We import it by
adding that directory to sys.path, so there is exactly one copy of the network.

Build/run:
    cd ~/ros2_ws && colcon build --packages-select carla_e2e_pkg
    source install/setup.bash
    ros2 run carla_e2e_pkg driving_policy_node \
        --checkpoint /path/to/ros2_ws/carla_e2e/checkpoints/best.pt \
        --image-topic /carla/ego_vehicle/rgb_front/image \
        --control-topic /carla/ego_vehicle/vehicle_control_cmd
"""
from __future__ import annotations

import os
import sys
import argparse

# Locate the sibling training project so we can reuse carla_e2e.models / dataset
_TRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "carla_e2e",          # ros2_ws/carla_e2e
)
if os.path.isdir(_TRAIN_DIR):
    sys.path.insert(0, _TRAIN_DIR)

from carla_e2e.ros2_node import main  # noqa: E402  (import-safe wrapper)


if __name__ == "__main__":
    main()
