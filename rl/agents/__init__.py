"""RL agent configs (skrl + rsl_rl) shared by all SO-101 tasks.

The algorithm layer is backend-independent: the same PPO configs train
policies on Isaac Lab, mjlab (MuJoCo Warp), or plain MuJoCo gymnasium envs.
"""

from .rsl_rl_cfg import SO101PPORunnerCfg

__all__ = ["SO101PPORunnerCfg"]
