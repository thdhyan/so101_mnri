"""Gymnasium registration for the dual-arm cylinder grasp task."""

import gymnasium as gym

gym.register(
    id="SO101-CylGrasp-Dual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CylGraspEnvCfg",
        "rsl_rl_cfg_entry_point": "rl.agents.rsl_rl_cfg:SO101PPORunnerCfg",
        "skrl_cfg_entry_point": "rl.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO101-CylGrasp-Dual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CylGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl.agents.rsl_rl_cfg:SO101PPORunnerCfg",
        "skrl_cfg_entry_point": "rl.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
