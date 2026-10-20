"""Task registry — importing this package registers all SO-101 Isaac tasks.

Registered tasks (single-arm vs dual-arm is part of the task id):

    Single-arm (one follower):
        SO101-PickLift-Single-v0        grasp cube, lift above threshold
        SO101-PickPlace-Single-v0       grasp cube, place on target disc
        SO101-CylReach-Single-v0        reach a point above a fixed
                                        cylinder's end, hold
    Dual-arm (two followers, bases 18 in apart, parallel Z/X axes):
        SO101-CylGrasp-Dual-v0          grasp opposite ends of a thin
                                        cylinder, lift together
        SO101-CylReach-Dual-v0          reach points above a fixed
                                        cylinder's ends, hold

RL-library entry points (both wrap the same envs — the algorithm layer is
backend-independent):
    rsl_rl_cfg_entry_point: rl.agents.rsl_rl_cfg:SO101PPORunnerCfg
    skrl_cfg_entry_point:   rl.agents:skrl_ppo_cfg.yaml
"""

from envs.isaac.tasks import (  # noqa: F401
    cylinder_grasp,
    cylinder_reach,
    cylinder_reach_single,
    pick_lift,
    pick_place,
)
