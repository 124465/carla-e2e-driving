"""ROS2 node: wraps the trained driving policy as a deployable component.

Subscribes to a camera image topic (sensor_msgs/Image), runs the policy, and
publishes a control command (std_msgs/Float32MultiArray: [steer, throttle, brake]).

IMPORT-SAFE / LAZY: rclpy is probed at import time only to set a flag. The heavy
imports (torch, cv_bridge, the policy modules) happen INSIDE main(), after the
rclpy guard. This means:
  - importing this module never drags torch in (so training code stays clean);
  - the node can be loaded in any Python; if rclpy is missing it exits with a
    clear message instead of a stack trace.

Deployment: run in a Python that has BOTH rclpy (from ROS2) and torch. The
recommended setup is a py3.12 venv built with --system-site-packages (to inherit
ROS2's rclpy/cv_bridge) plus CPU torch installed. Example:
    source /opt/ros/jazzy/setup.bash
    source .venv312/bin/activate
    ros2 run carla_e2e_pkg driving_policy_node --checkpoint checkpoints/best.pt

To drive CARLA from ROS2, combine with `carla_ros_bridge` so the bridge provides
/camera/* and consumes /vehicle/control (or your own bridge to PyBullet).
"""
from __future__ import annotations

import sys
import argparse

try:
    import rclpy  # noqa: F401  (just to detect availability)
    _HAS_RCLPY = True
except Exception as exc:  # pragma: no cover
    _HAS_RCLPY = False
    _RCLPY_IMPORT_ERROR = exc


def main():
    if not _HAS_RCLPY:
        sys.stderr.write(
            "ERROR: rclpy is not available in this Python environment.\n"
            "Run this node in a ROS2-sourced shell with rclpy installed "
            f"(import failed: {_RCLPY_IMPORT_ERROR}).\n"
        )
        sys.exit(1)

    # --- heavy imports only now that rclpy is confirmed ---
    import numpy as np
    import torch
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import Float32MultiArray
    from cv_bridge import CvBridge

    # single source of truth: the policy lives in the sibling training project
    from .models import get_model
    from .dataset import BCDataset

    MEAN = np.array(BCDataset.MEAN).reshape(3, 1, 1)
    STD = np.array(BCDataset.STD).reshape(3, 1, 1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--backbone", default="tiny_cnn")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--img-size", nargs=2, type=int, default=[80, 160])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image-topic", default="/camera/rgb/image_raw")
    ap.add_argument("--control-topic", default="/vehicle/control")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    class DrivingPolicyNode(Node):
        def __init__(self):
            super().__init__("carla_e2e_policy")
            self.device = args.device
            self.img_size = tuple(args.img_size)
            self.bridge = CvBridge()
            self.model = get_model(args.backbone, img_h=self.img_size[0],
                                   img_w=self.img_size[1],
                                   hidden_dim=args.hidden_dim, dropout=0.0)
            self.model.load_state_dict(
                torch.load(args.checkpoint, map_location=args.device,
                           weights_only=True))
            self.model.to(self.device).eval()
            self.get_logger().info(f"loaded checkpoint {args.checkpoint}")
            self.sub = self.create_subscription(
                RosImage, args.image_topic, self.on_image, 10)
            self.pub = self.create_publisher(Float32MultiArray,
                                             args.control_topic, 10)
            self.timer = self.create_timer(1.0 / args.fps, self.tick)
            self.latest = None

        def on_image(self, msg):
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                import cv2
                cv_img = cv2.resize(cv_img, (self.img_size[1], self.img_size[0]))
                arr = np.asarray(cv_img, np.float32) / 255.0
                arr = (arr.transpose(2, 0, 1) - MEAN) / STD
                self.latest = arr[None].astype(np.float32)
            except Exception as e:  # noqa
                self.get_logger().warn(f"image decode failed: {e}")

        def tick(self):
            if self.latest is None:
                return
            x = torch.from_numpy(self.latest).to(self.device)
            with torch.no_grad():
                action = self.model(x).cpu().numpy()[0]
            action[1:] = np.clip(action[1:], 0.0, 1.0)
            msg = Float32MultiArray()
            msg.data = [float(action[0]), float(action[1]), float(action[2])]
            self.pub.publish(msg)

    rclpy.init()
    node = DrivingPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
