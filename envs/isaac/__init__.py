"""
SO-101 Isaac Lab environments.

RL task suite for the SO-101 follower arm (single-arm and dual-arm) built on
Isaac Lab's manager-based API (Isaac Sim 6.0.1 backend). Tasks are registered
as Gymnasium environments and are backend-agnostic on the RL side: skrl and
rsl_rl both wrap the same envs.

Registered tasks:
    SO101-PickLift-Single-v0     single arm: grasp cube, lift above threshold
    SO101-PickPlace-Single-v0    single arm: grasp cube, place on target
    SO101-CylReach-Single-v0     single arm: reach a point above a fixed
                                 cylinder's end, hold
    SO101-CylGrasp-Dual-v0       dual arm: grasp opposite ends of a thin
                                 cylinder, lift together
    SO101-CylReach-Dual-v0       dual arm: reach target points above a static
                                 cylinder's ends

Geometry note: in dual-arm tasks the follower bases sit 18 in (0.4572 m)
apart in Y with parallel Z and X axes, matching the real rig (see
robots/so101/README.md).
"""

from . import tasks  # noqa: F401  (registers all gym tasks)

__all__ = ["tasks"]
