"""Build an environment from a parsed config dict.

Central place so collect/evaluate/dagger/dagger all construct the same env for a
given `backend`, including the newer `dynamic` scene backend.
"""
from __future__ import annotations


def build_env(cfg: dict, backend: str | None = None, verbose: bool = False):
    """Return an env for the given config.

    Reads cfg["backend"] unless `backend` is passed explicitly, and forwards the
    matching sub-config section (carla / pybullet / dynamic).
    """
    from .env import make_env

    backend = backend or cfg.get("backend", "pybullet")

    if backend == "dynamic":
        from .dynamic_env import DynamicSceneConfig
        section = dict(cfg.get("dynamic", {}))
        # ignore keys that belong to other backends
        valid = {f for f in DynamicSceneConfig.__dataclass_fields__}
        section = {k: v for k, v in section.items() if k in valid}
        return make_env(backend="dynamic", scene_cfg=DynamicSceneConfig(**section))

    if backend == "pybullet":
        from .pybullet_env import PyBulletConfig
        section = dict(cfg.get("pybullet", {}))
        valid = {f for f in PyBulletConfig.__dataclass_fields__}
        section = {k: v for k, v in section.items() if k in valid}
        return make_env(backend="pybullet", pybullet_cfg=PyBulletConfig(**section))

    if backend == "carla":
        from .env import EnvConfig
        section = dict(cfg.get("carla", {}))
        valid = {f for f in EnvConfig.__dataclass_fields__}
        section = {k: v for k, v in section.items() if k in valid}
        return make_env(backend="carla", carla_cfg=EnvConfig(**section))

    raise ValueError(f"unknown backend: {backend}")
