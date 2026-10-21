#!/usr/bin/env python3
"""
Record teleop demonstrations for SO-101 Isaac Lab tasks.

Records episodes to HDF5 dataset while teleoperating. Supports both
joint-space (SO-101 Leader) and XR controller teleop.

Usage:
    # Joint-space teleop recording
    ./isaaclab.sh -p scripts/record_demos.py \
        --task SO101-CylReach-Single-v0 \
        --dataset_file ./datasets/so101_cylreach_demos.hdf5 \
        --num_demos 10 \
        --step_hz 30 \
        --viz kit

    # XR controller recording
    ./isaaclab.sh -p scripts/record_demos.py \
        --task SO101-CylReach-Single-v0 \
        --dataset_file ./datasets/so101_cylreach_demos.hdf5 \
        --num_demos 10 \
        --step_hz 30 \
        --xr \
        --viz kit

Based on: Isaac Lab's record_demos.py + IsaacTeleop SO-101 integration
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record SO-101 teleop demos")
parser.add_argument(
    "--task", type=str, required=True,
    help="Isaac Lab task name"
)
parser.add_argument(
    "--dataset_file", type=str, required=True,
    help="Output HDF5 dataset file path"
)
parser.add_argument(
    "--num_demos", type=int, default=10,
    help="Number of demonstrations to record"
)
parser.add_argument(
    "--step_hz", type=int, default=30,
    help="Control frequency (Hz)"
)
parser.add_argument(
    "--num_success_steps", type=int, default=10,
    help="Consecutive success steps to mark episode as successful"
)
parser.add_argument(
    "--xr", action="store_true",
    help="Enable XR/CloudXR for headset teleop"
)
parser.add_argument(
    "--teleop-device", type=str, default="so101_leader",
    choices=["so101_leader", "xr_controller"],
    help="Teleop device type"
)
parser.add_argument(
    "--viz", type=str, default="kit", choices=["kit", "none"],
    help="Visualization backend"
)
parser.add_argument(
    "--headless", action="store_true",
)

AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after AppLauncher
import h5py
import numpy as np
import torch

import gymnasium as gym

from envs.isaac import so101  # noqa: F401
from envs.isaac.tasks.teleop_cfg import get_teleop_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_teleop import TeleopInterface


def save_episode(h5_file, episode_idx, episode_data, success: bool):
    """Save one episode to HDF5."""
    grp = h5_file.create_group(f"episode_{episode_idx:06d}")
    for key, value in episode_data.items():
        grp.create_dataset(key, data=np.array(value))
    grp.attrs["success"] = success
    grp.attrs["length"] = len(episode_data["actions"])


def main():
    print(f"[record] Task: {args_cli.task}")
    print(f"[record] Dataset: {args_cli.dataset_file}")
    print(f"[record] Target demos: {args_cli.num_demos}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args_cli.dataset_file), exist_ok=True)

    # Create environment
    env_cfg = gym.make(args_cli.task).unwrapped.cfg
    env_cfg.scene.num_envs = 1
    env_cfg.decimation = int(1 / (args_cli.step_hz * env_cfg.sim.dt))

    # Get teleop config
    teleop_cfg = get_teleop_cfg(args_cli.task)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    teleop_interface = TeleopInterface(teleop_cfg, env)

    # HDF5 dataset
    h5_file = h5py.File(args_cli.dataset_file, "w")
    h5_file.attrs["task"] = args_cli.task
    h5_file.attrs["step_hz"] = args_cli.step_hz
    h5_file.attrs["num_success_steps"] = args_cli.num_success_steps

    demos_recorded = 0
    demo_idx = 0

    try:
        while simulation_app.is_running() and demos_recorded < args_cli.num_demos:
            print(f"\n[record] Demo {demos_recorded + 1}/{args_cli.num_demos}")
            print("[record] Press 'B' in Kit viewport to START teleop for this episode")

            # Reset environment
            obs, _ = env.reset()

            # Episode buffers
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "infos": [],
            }

            # State machine: WAIT_FOR_START -> RECORDING -> EPISODE_END
            recording = False
            success_steps = 0

            while simulation_app.is_running():
                # Step teleop
                teleop_actions = teleop_interface.step()

                # Check if teleop is active (B pressed)
                if teleop_actions is not None and not recording:
                    recording = True
                    print("[record] Recording started...")

                if recording:
                    # Step environment with teleop actions
                    if teleop_actions is not None:
                        obs, reward, terminated, truncated, info = env.step(teleop_actions)
                    else:
                        # No teleop input - hold
                        zero_actions = torch.zeros_like(env.action_space.sample())
                        obs, reward, terminated, truncated, info = env.step(zero_actions)

                    # Store data
                    episode_data["observations"].append(obs["policy"].cpu().numpy())
                    episode_data["actions"].append(
                        teleop_actions.cpu().numpy() if teleop_actions is not None
                        else np.zeros(env.action_space.shape)
                    )
                    episode_data["rewards"].append(reward.cpu().numpy())
                    episode_data["dones"].append(terminated.cpu().numpy())
                    episode_data["infos"].append(info)

                    # Check success
                    if info.get("success", False):
                        success_steps += 1
                    else:
                        success_steps = 0

                    # Episode end conditions
                    if terminated or truncated:
                        episode_success = success_steps >= args_cli.num_success_steps
                        print(f"[record] Episode ended. Success: {episode_success}")
                        save_episode(h5_file, demo_idx, episode_data, episode_success)
                        if episode_success:
                            demos_recorded += 1
                        demo_idx += 1
                        break

                    # Check for manual stop (Escape key in Kit)
                    # This would need keyboard event handling - for now rely on timeout
                    if len(episode_data["actions"]) > 10000:  # max steps safety
                        print("[record] Max steps reached, ending episode")
                        save_episode(h5_file, demo_idx, episode_data, False)
                        demo_idx += 1
                        break

                else:
                    # Waiting for B key - step with zero actions
                    zero_actions = torch.zeros_like(env.action_space.sample())
                    env.step(zero_actions)

    except KeyboardInterrupt:
        print("\n[record] Interrupted")
    finally:
        h5_file.close()
        env.close()
        simulation_app.close()
        print(f"\n[record] Done. {demos_recorded} successful demos saved to {args_cli.dataset_file}")


if __name__ == "__main__":
    main()