"""Thin CARLA wrapper for expert data collection and closed-loop evaluation.

Runs fully offscreen so it fits a 4GB GPU (GTX 1650). The camera is a small
RGB sensor (160x80) mounted front-center. Action space is the standard CARLA
VehicleControl: (steer, throttle, brake).

The CARLA client library is imported lazily so that importing this module does
not require the carla package to be installed (e.g. for tests / docs builds).
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    town: str = "Town01"
    image_width: int = 160
    image_height: int = 80
    fov: float = 90.0
    fixed_delta_seconds: float = 0.05
    target_speed: float = 30.0  # km/h (used by autopilot expert)


ACTION_DIM = 3  # steer, throttle, brake


class CarlaEnv:
    """Minimal synchronous CARLA environment.

    reset(autopilot=True)  -> collect expert demonstrations
    reset(autopilot=False) -> deploy a learned policy
    step(action)           -> apply control, return (obs, info)
    """

    def __init__(self, cfg: EnvConfig):
        import carla  # lazy import

        self.cfg = cfg
        self.carla = carla
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(cfg.town)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = cfg.fixed_delta_seconds
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.vehicle = None
        self.camera = None
        self._image_queue: "queue.Queue" = queue.Queue(maxsize=1)

    # ----------------------------------------------------------- setup
    def _spawn_vehicle(self, autopilot: bool):
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        spawn = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, spawn)
        self.vehicle.set_autopilot(autopilot)

    def _spawn_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.cfg.image_width))
        bp.set_attribute("image_size_y", str(self.cfg.image_height))
        bp.set_attribute("fov", str(self.cfg.fov))
        transform = self.carla.Transform(
            self.carla.Location(x=1.5, z=1.4),
            self.carla.Rotation(pitch=0.0),
        )
        self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._on_image(img))

    def _on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        try:
            self._image_queue.put_nowait(arr)
        except queue.Full:
            pass  # drop oldest frame if not consumed

    # ----------------------------------------------------------- api
    def reset(self, autopilot: bool = True):
        if self.vehicle is None:
            self._spawn_vehicle(autopilot)
            self._spawn_camera()
        self.world.tick()
        return self._get_observation()

    def _get_observation(self):
        self.world.tick()
        try:
            return self._image_queue.get(timeout=5.0)
        except queue.Empty:
            return None

    def step(self, action=None, autopilot: bool = True):
        """Apply `action` (np.ndarray len 3) or follow autopilot if None."""
        if action is not None:
            control = self.carla.VehicleControl()
            control.steer = float(np.clip(action[0], -1.0, 1.0))
            control.throttle = float(np.clip(action[1], 0.0, 1.0))
            control.brake = float(np.clip(action[2], 0.0, 1.0))
            self.vehicle.apply_control(control)
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_info(self):
        v = self.vehicle.get_velocity()
        speed = 3.6 * float(np.linalg.norm([v.x, v.y, v.z]))  # km/h
        ctrl = self.vehicle.get_control()
        expert_action = np.array(
            [ctrl.steer, ctrl.throttle, ctrl.brake], dtype=np.float32
        )
        return {"speed": speed, "expert_action": expert_action}

    def close(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    # ----------------------------------------------------------- DAgger oracle
    def query_expert_control(self):
        """Expert oracle for DAgger.

        CARLA's built-in autopilot has no public "predict" API, so we briefly
        enable autopilot, tick once so it computes+applies its control, read it
        back, then disable autopilot. The returned control is the expert's
        action at the *current* camera state.

        NOTE: as a side effect the world advances one tick under the expert.
        For a cleaner oracle (control WITHOUT advancing the world) use
        carla.agents.navigation.behavior_agent.BehaviorAgent.run_step(), which
        requires the CARLA PythonAPI `agents` module on PYTHONPATH. See
        scripts/dagger.py (QUERY_EXPERT_MODE = "behavior_agent").
        """
        if self.vehicle is None:
            return None
        self.vehicle.set_autopilot(True)
        self.world.tick()
        c = self.vehicle.get_control()
        expert = np.array([c.steer, c.throttle, c.brake], dtype=np.float32)
        self.vehicle.set_autopilot(False)
        return expert


def make_env(backend: str = "pybullet", carla_cfg: "EnvConfig | None" = None,
             pybullet_cfg: "object | None" = None,
             scene_cfg: "object | None" = None):
    """Factory: returns a CarlaEnv / PyBulletEnv / DynamicDrivingEnv.

    All expose:
        reset(autopilot=True/False) -> (obs, info)
        step(action=None|np.array, autopilot=bool) -> (obs, info)
        query_expert_control() -> np.ndarray([steer, throttle, brake])
        close()

    Backends:
      * "pybullet"  - simple empty oval (original, easiest task)
      * "dynamic"   - ring road with two lanes, traffic lights, moving traffic
                      and car-following (harder task, richer info dict)
      * "carla"     - real CARLA simulator (needs the 8GB download)

    PyBullet is the default: lightweight, offscreen, SSD-friendly.
    """
    if backend == "carla":
        return CarlaEnv(carla_cfg or EnvConfig())
    elif backend == "pybullet":
        from .pybullet_env import PyBulletEnv, PyBulletConfig
        return PyBulletEnv(pybullet_cfg or PyBulletConfig())
    elif backend == "dynamic":
        from .dynamic_env import DynamicDrivingEnv, DynamicSceneConfig
        return DynamicDrivingEnv(scene_cfg or DynamicSceneConfig())
    raise ValueError(f"unknown backend: {backend}")
