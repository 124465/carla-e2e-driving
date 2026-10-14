"""PyBullet backend for the end-to-end driving project (lightweight, SSD-friendly).

Uses a kinematic bicycle model integrated in PyBullet for both the EXPERT and the
DEPLOYED policy. The racecar URDF is rendered (camera) so observations reflect the
vehicle's real position/orientation, but motion is driven by a clean, robust
bicycle integrator instead of fragile joint-torque control. This is the standard
pattern for imitation-learning demos: a deterministic expert produces (steer,
throttle) labels, the camera renders the trajectory, and a learned policy is
later rolled out with the same integrator.

Same interface as CarlaEnv so scripts/collect.py, train.py, evaluate.py and
dagger.py work unchanged:
    reset(autopilot) -> (obs, info)
    step(action, autopilot) -> (obs, info)
    query_expert_control() -> np.ndarray([steer, throttle, brake])
    close()

Runs fully offscreen (DIRECT) and at low resolution (160x80) to fit a 4GB GPU.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

ACTION_DIM = 3  # steer, throttle, brake (kept consistent with CarlaEnv)

# Kinematic model constants
_WHEELBASE = 0.3
_STADIUM_R = 46.0          # racing ring radius inside the stadium plane
_MAX_SPEED = 14.0          # m/s


@dataclass
class PyBulletConfig:
    image_width: int = 160
    image_height: int = 80
    fov: float = 90.0
    target_speed: float = 10.0    # m/s, expert cruise speed
    waypoint_radius: float = 1.5
    max_steps: int = 1000


def _make_waypoints(n: int = 80, r: float = _STADIUM_R):
    """Precomputed racing line: a slightly egg-shaped ring so steering must correct."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1.0 + 0.05 * math.cos(2 * a))
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    return np.array(pts, dtype=np.float32)


class PyBulletEnv:
    def __init__(self, cfg: PyBulletConfig | None = None):
        import pybullet as p
        import pybullet_data

        self.cfg = cfg or PyBulletConfig()
        self.p = p
        self.waypoints = _make_waypoints()
        self.wp_idx = 0
        self.state = dict(x=0.0, y=-_STADIUM_R, yaw=math.pi / 2, speed=0.0)

        self.client = p.connect(p.DIRECT)  # offscreen, no GUI -> low SSD/CPU
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        try:
            p.loadSDF("plane_stadium.sdf")
        except Exception:
            p.loadURDF("plane.urdf")
        self.car = p.loadURDF("racecar/racecar.urdf",
                              [self.state["x"], self.state["y"], 0.05],
                              p.getQuaternionFromEuler([0, 0, self.state["yaw"]]))
        self._sync_pose()

    # ----------------------------------------------------------- kinematics
    def _expert_control(self):
        pos = np.array([self.state["x"], self.state["y"]])
        # nearest waypoint, then look ahead along the path
        best_d, best_i = 1e9, 0
        for k in range(len(self.waypoints)):
            d = np.linalg.norm(self.waypoints[k] - pos)
            if d < best_d:
                best_d, best_i = d, k
        self.wp_idx = best_i
        target = self.waypoints[(self.wp_idx + 4) % len(self.waypoints)]
        desired = math.atan2(target[1] - pos[1], target[0] - pos[0])
        err = math.atan2(math.sin(desired - self.state["yaw"]),
                         math.cos(desired - self.state["yaw"]))
        steer = float(np.clip(err / (math.pi / 4), -1.0, 1.0))
        # slow down on sharp heading error, otherwise cruise
        throttle = float(np.clip(0.55 + 0.35 * (1 - abs(err) / (math.pi / 2)),
                                 0.15, 1.0))
        brake = 0.0
        return np.array([steer, throttle, brake], dtype=np.float32)

    def _integrate(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        # bicycle model (rear-axle). steer maps to front-wheel angle.
        speed = self.state["speed"]
        dt = self._dt
        accel = (throttle * 8.0) - (brake * 10.0) - 0.4 * speed  # drag
        speed = float(np.clip(speed + accel * dt, 0.0, _MAX_SPEED))
        yaw = self.state["yaw"] + (speed / _WHEELBASE) * math.tan(steer * 0.6) * dt
        x = self.state["x"] + speed * math.cos(yaw) * dt
        y = self.state["y"] + speed * math.sin(yaw) * dt
        # keep inside stadium
        rad = math.hypot(x, y)
        if rad > _STADIUM_R + 2.0:
            x *= (_STADIUM_R + 2.0) / rad
            y *= (_STADIUM_R + 2.0) / rad
            speed *= 0.5
        self.state.update(x=x, y=y, yaw=yaw, speed=speed)

    # PyBulletConfig lacks fixed_dt; add a default
    @property
    def _dt(self):
        return 0.05

    # ----------------------------------------------------------- camera
    def _sync_pose(self):
        self.p.resetBasePositionAndOrientation(
            self.car,
            [self.state["x"], self.state["y"], 0.05],
            self.p.getQuaternionFromEuler([0, 0, self.state["yaw"]]))

    def _cam_view(self):
        pos = (self.state["x"], self.state["y"], 0.05)
        yaw = self.state["yaw"]
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        cam_pos = np.array(pos) + fwd * 0.25 + np.array([0, 0, 0.12])
        cam_target = cam_pos + fwd * 2.0
        view = self.p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])
        aspect = self.cfg.image_width / self.cfg.image_height
        proj = self.p.computeProjectionMatrixFOV(self.cfg.fov, aspect, 0.1, 50)
        return view, proj

    def _render(self):
        view, proj = self._cam_view()
        img = self.p.getCameraImage(
            width=self.cfg.image_width, height=self.cfg.image_height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=self.p.ER_TINY_RENDERER)
        rgba = np.reshape(img[2], (self.cfg.image_height,
                                   self.cfg.image_width, 4))
        return rgba[:, :, :3].astype(np.uint8)

    # ----------------------------------------------------------- api
    def reset(self, autopilot: bool = True):
        self.state = dict(x=0.0, y=-_STADIUM_R, yaw=math.pi / 2, speed=0.0)
        self.wp_idx = 0
        self._sync_pose()
        self._last_obs = self._render()
        return self._last_obs, self._get_info()

    def _get_observation(self):
        """Return the most recent rendered frame (compatibility with CARLA API)."""
        if getattr(self, "_last_obs", None) is None:
            self._last_obs = self._render()
        return self._last_obs

    def step(self, action=None, autopilot: bool = True):
        if action is None:
            action = self._expert_control()
        self._integrate(action)
        self._sync_pose()
        self.p.stepSimulation()
        self._last_obs = self._render()
        return self._last_obs, self._get_info()

    def _get_info(self):
        expert = self._expert_control()
        return {
            "speed": float(self.state["speed"]),
            "expert_action": expert,
            "position": np.array([self.state["x"], self.state["y"]], dtype=np.float32),
        }

    def query_expert_control(self):
        return self._expert_control()

    def close(self):
        self.p.disconnect()
