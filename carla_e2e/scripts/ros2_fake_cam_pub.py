"""Test helper: publish synthetic images on /camera/rgb/image_raw.

Used to validate the driving_policy_node end-to-end without a real camera or
CARLA. Run in the ROS2-sourced py3.12 venv alongside:
    ros2 run carla_e2e_pkg driving_policy_node --checkpoint <...>
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np


class FakeCam(Node):
    def __init__(self):
        super().__init__("fake_cam")
        self.pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.timer = self.create_timer(0.05, self.cb)
        self.i = 0

    def cb(self):
        img = (np.random.rand(80, 160, 3) * 255).astype(np.uint8)
        msg = Image()
        msg.height, msg.width = 80, 160
        msg.encoding = "rgb8"
        msg.step = 160 * 3
        msg.data = img.tobytes()
        self.pub.publish(msg)
        self.i += 1


def main():
    rclpy.init()
    n = FakeCam()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
