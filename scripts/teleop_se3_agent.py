#!/usr/bin/env python3
"""
Teleoperation SE(3) agent for SO-101 Isaac Lab tasks.

This script runs the Isaac Lab environment with teleop control enabled.
It connects to the IsaacTeleop runtime (CloudXR) and applies the retargeted
actions to the simulated robot.

Usage:
    # Joint-space teleop (SO-101 Leader, no headset)
    ./isaaclab.sh -p scripts/teleop_se3_agent.py \
        --task SO101-CylReach-Single-v0 \
        --viz kit

    # XR controller teleop (with headset)
    ./isaaclab.sh -p scripts/teleop_se3_agent.py \
        --task SO101-CylReach-Single-v0 \
        --xr \
        --viz kit

Based on: Isaac Lab's teleop_se3_agent.py + IsaacTeleop SO-101 integration
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Add task-specific arguments
parser = argparse.ArgumentParser(description="SE(3) teleop agent for SO-101")
parser.add_argument(
    "--task", type=str, required=True,
    help="Isaac Lab task name (e.g., SO101-CylReach-Single-v0)"
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments (teleop typically uses 1)"
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
    "--headless", action="store_true",
    help="Run headless (no viewport)"
)
parser.add_argument(
    "--viz", type=str, default="kit", choices=["kit", "none"],
    help="Visualization backend"
)

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules (after AppLauncher)
import gymnasium as gym
import torch

from envs.isaac import so101  # noqa: F401 (registers tasks)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_teleop import TeleopInterface

# Local teleop configs
from envs.isaac.tasks.teleop_cfg import get_teleop_cfg


def main():
    """Run teleop agent."""
    print(f"[teleop] Task: {args_cli.task}")
    print(f"[teleop] Device: {args_cli.teleop_device}")
    print(f"[teleop] XR enabled: {args_cli.xr}")

    # Create environment
    env_cfg = gym.make(args_cli.task).unwrapped.cfg
    env_cfg.scene.num_envs = args_cli.num_envs

    # Get teleop config for this task
    teleop_cfg = get_teleop_cfg(args_cli.task)

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Initialize teleop interface
    teleop_interface = TeleopInterface(teleop_cfg, env)

    # Reset environment
    obs, _ = env.reset()
    print("[teleop] Environment ready. Press 'B' in Kit viewport to start teleop.")

    # Teleop control state machine: IDLE -> TELEOP -> PAUSED
    teleop_active = False

    try:
        while simulation_app.is_running():
            # Check for teleop start/pause (B key in Kit viewport)
            # This is handled by the TeleopInterface internally
            # For manual control, we can check keyboard events here

            # Step teleop interface (poll sources, retarget, etc.)
            teleop_actions = teleop_interface.step()

            if teleop_actions is not None:
                teleop_active = True

            # Apply teleop actions to environment
            if teleop_active and teleop_actions is not None:
                # teleop_actions is a dict matching the action space
                env.step(teleop_actions)
            else:
                # No teleop input - step with zero actions (hold position)
                zero_actions = torch.zeros_like(env.action_space.sample())
                env.step(zero_actions)

    except KeyboardInterrupt:
        print("\n[teleop] Interrupted")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()