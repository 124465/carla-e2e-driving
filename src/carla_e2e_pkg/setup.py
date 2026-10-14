from setuptools import setup, find_packages

# This package only needs to be importable under a ROS2-sourced env.
# The actual torch/carla training code lives under ros2_ws/carla_e2e/
# and is run from the py3.10 venv; do NOT add torch/carla here.
setup(
    name="carla_e2e_pkg",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["rclpy", "numpy", "opencv-python", "cv-bridge"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "driving_policy_node = carla_e2e_pkg.driving_policy_node:main",
        ],
    },
)
