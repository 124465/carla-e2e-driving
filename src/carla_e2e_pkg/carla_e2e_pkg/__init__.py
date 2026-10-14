"""ROS2 package wrapper for the CARLA end-to-end driving policy.

NOTE: to run the node you must pass --checkpoint pointing at a model trained by
carla_e2e/scripts/train.py (which lives in ros2_ws/carla_e2e/). The shared
model/dataset code is imported from that sibling directory by inserting it onto
sys.path, so the policy implementation stays single-source-of-truth.
"""
