"""Gymnasium registration for the SO-101 single-arm pick-lift task."""

import gymnasium as gym

gym.register(
    id="SO101-PickLift-Single-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PickLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "rl.agents.rsl_rl_cfg:SO101PPORunnerCfg",
        "skrl_cfg_entry_point": "rl.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO101-PickLift-Single-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PickLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl.agents.rsl_rl_cfg:SO101PPORunnerCfg",
        "skrl_cfg_entry_point": "rl.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
