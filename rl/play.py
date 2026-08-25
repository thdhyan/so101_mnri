"""Checkpoint playback for the SO-101 tasks (skrl agents).

    python -m rl.play --backend isaaclab --task SO101-PickLift-Single-v0 \
        --checkpoint rl/runs/isaaclab/.../agent.pt --episodes 5
    python -m rl.play --backend mujoco --task pick_lift --checkpoint rl/runs/.../agent.pt
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SO-101 policy playback")
    p.add_argument("--backend", choices=["isaaclab", "mujoco"], default="mujoco")
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--save-dir", default="rl/runs/play")
    return p.parse_args(argv)


def play_isaaclab(args):
    from isaaclab.app import AppLauncher

    # cameras enabled: task scenes spawn camera sensors (required by Kit)
    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app
    try:
        import gymnasium as gym

        import envs.isaac  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.memories.torch import RandomMemory

        from rl.train import SkrlPolicy, SkrlValue, ISAACLAB_TASKS

        assert args.task in ISAACLAB_TASKS
        spec = gym.spec(args.task)
        module, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
        env_cfg = getattr(importlib.import_module(module), cls_name)()
        env_cfg.scene.num_envs = 1
        env = ManagerBasedRLEnv(cfg=env_cfg)
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        obs_shape = tuple(wrapped.observation_space.shape)
        act_shape = tuple(wrapped.action_space.shape)
        models = {
            "policy": SkrlPolicy(obs_shape[0], act_shape[0], device).to(device),
            "value": SkrlValue(obs_shape[0], act_shape[0], device).to(device),
        }
        agent = PPO(
            models=models,
            memory=RandomMemory(memory_size=1, num_envs=1, device=device),
            cfg=PPO_CFG(),
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
        )
        agent.init_from_trained_model = None  # noqa - use skrl load below
        agent.load(args.checkpoint)

        state, _ = wrapped.reset()
        for ep in range(args.episodes):
            total = torch.zeros(1, device=device)
            for t in range(args.max_steps):
                with torch.no_grad():
                    actions = agent.act(state, timestep=t, timesteps=args.max_steps)[0]
                state, reward, done, info = wrapped.step(actions)
                total += reward
                if done.any():
                    print(f"episode {ep}: steps={t + 1} return={float(total[0]):.2f}")
                    break
        env.close()
    finally:
        simulation_app.close()


def play_mujoco(args):
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory

    from rl.train import MUJOCO_TASKS, SkrlPolicy, SkrlValue

    module_path, func_name = MUJOCO_TASKS[args.task].split(":")
    env = getattr(importlib.import_module(module_path), func_name)(n_envs=1)
    wrapped = wrap_env(env, ml_framework="torch")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    obs_shape = tuple(wrapped.observation_space.shape)
    act_shape = tuple(wrapped.action_space.shape)
    models = {
        "policy": SkrlPolicy(obs_shape[0], act_shape[0], device).to(device),
        "value": SkrlValue(obs_shape[0], act_shape[0], device).to(device),
    }
    agent = PPO(
        models=models,
        memory=RandomMemory(memory_size=1, num_envs=1, device=device),
        cfg=PPO_CFG(),
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )
    agent.load(args.checkpoint)

    for ep in range(args.episodes):
        state, _ = wrapped.reset()
        total = 0.0
        for t in range(args.max_steps):
            with torch.no_grad():
                actions = agent.act(state, timestep=t, timesteps=args.max_steps)[0]
            state, reward, done, info = wrapped.step(actions)
            total += float(reward)
            if done.any():
                break
        print(f"episode {ep}: steps={t + 1} return={total:.2f}")
    env.close()


def main(argv=None):
    args = parse_args(argv)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.backend == "isaaclab":
        play_isaaclab(args)
    else:
        play_mujoco(args)


if __name__ == "__main__":
    main()
