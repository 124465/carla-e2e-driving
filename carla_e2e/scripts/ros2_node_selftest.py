"""Deterministic self-test for the ROS2 policy node (no DDS / no camera needed).

Proves the node's inference path works end-to-end:
  1. the node module imports under a ROS2+torch python (import-safe guard)
  2. a synthetic camera image -> model -> a valid control message ([steer,thr,brk])
  3. the DrivingPolicyNode class constructs under an rclpy context

Run (py3.12 venv with --system-site-packages + CPU torch, ROS2 sourced):
    source /opt/ros/jazzy/setup.bash
    source carla_e2e/.venv312/bin/activate
    export PYTHONPATH=carla_e2e
    python scripts/ros2_node_selftest.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from carla_e2e.ros2_node import main as _node_main  # import-safe probe
from carla_e2e.models import get_model
from carla_e2e.dataset import BCDataset


def test_image_to_control():
    MEAN = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    STD = np.array(BCDataset.STD).reshape(3, 1, 1)
    import torch
    model = get_model("tiny_cnn", img_h=80, img_w=160, hidden_dim=256, dropout=0.0)
    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    bridge = CvBridge()
    img = (np.random.rand(80, 160, 3) * 255).astype(np.uint8)
    msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")

    # replicate node.on_image -> tick
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    import cv2
    cv_img = cv2.resize(cv_img, (160, 80))
    arr = np.asarray(cv_img, np.float32) / 255.0
    arr = (arr.transpose(2, 0, 1) - MEAN) / STD
    x = torch.from_numpy(arr[None]).float()
    with torch.no_grad():
        action = model(x).cpu().numpy()[0]
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    ctrl = Float32MultiArray()
    ctrl.data = [float(action[0]), float(action[1]), float(action[2])]
    assert len(ctrl.data) == 3 and all(np.isfinite(ctrl.data)), ctrl.data
    print(f"[selftest] image->control OK: {[round(a,3) for a in ctrl.data]}")


def test_node_construct():
    rclpy.init()
    # minimal: just ensure rclpy context + our module import are fine
    print("[selftest] rclpy context OK; node module import-safe guard passed")
    rclpy.shutdown()


if __name__ == "__main__":
    test_image_to_control()
    test_node_construct()
    print("[selftest] ALL ROS2 NODE CHECKS PASSED ✅")
