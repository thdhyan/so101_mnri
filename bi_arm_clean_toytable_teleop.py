"""
Bi-arm teleoperation for LeIsaac-SO101-CleanToyTable-BiArm-v0.

Task: Pick two letter-E objects into the box, reset arms to rest state.
Hardware: Two SO101 Leader arms (left + right).

Usage:
    python bi_arm_clean_toytable_teleop.py \
        --left-port /dev/ttyACM0 \
        --right-port /dev/ttyACM1 \
        [--fps 60]
"""

import argparse
import logging
import sys
import time
import gymnasium as gym
from dataclasses import dataclass, asdict
from pprint import pformat

# Inject before IsaacLab parses argv

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    bi_so_leader,
    so_leader,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.envs.factory import make_env

ENV_SCRIPT = "LightwheelAI/leisaac_env:envs/bi_so101_fold_cloth.py"


@dataclass
class EnvWrap:
    """Thin wrapper so action_process can access env.device / num_envs."""
    env: gym.Env


def load_env() -> gym.Env:
    envs_dict = make_env(ENV_SCRIPT, n_envs=1, trust_remote_code=True)
    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][0].envs[0].unwrapped
    return env


def teleop_loop(teleop: Teleoperator, env: gym.Env, fps: int):
    from leisaac.devices.action_process import preprocess_device_action
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

    env_wrap = EnvWrap(env=env)
    obs, info = env.reset()

    print("\n[READY] Teleop running — Ctrl+C to stop.\n")
    while True:
        loop_start = time.perf_counter()

        if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(env, "bi_so101leader")

        raw_action = teleop.get_action()

        # bi_so_leader returns keys like "left_arm.shoulder_pan.pos"
        # Split into per-arm dicts, stripping the arm prefix and ".pos" suffix.
        left_joints = {
            k.removeprefix("left_arm.").removesuffix(".pos"): v
            for k, v in raw_action.items()
            if k.startswith("left_arm.")
        }
        right_joints = {
            k.removeprefix("right_arm.").removesuffix(".pos"): v
            for k, v in raw_action.items()
            if k.startswith("right_arm.")
        }

        processed_action = preprocess_device_action(
            dict(
                bi_so101_leader=True,
                joint_state={
                    "left_arm": left_joints,
                    "right_arm": right_joints,
                },
                motor_limits={
                    "left_arm": SO101_FOLLOWER_MOTOR_LIMITS,
                    "right_arm": SO101_FOLLOWER_MOTOR_LIMITS,
                },
            ),
            env_wrap,
        )

        obs, reward, terminated, truncated, info = env.step(processed_action)
        if terminated or truncated:
            print("[RESET] Episode done — resetting.")
            obs, info = env.reset()

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"\rtime: {loop_s * 1e3:.1f}ms  ({1/loop_s:.0f} Hz)  "
              f"reward: {reward:.3f}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-port", default="/dev/ttyACM0",
                        help="Serial port for LEFT leader arm")
    parser.add_argument("--right-port", default="/dev/ttyACM1",
                        help="Serial port for RIGHT leader arm")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    init_logging()
    logging.info("Left port: %s  Right port: %s  FPS: %d",
                 args.left_port, args.right_port, args.fps)

    teleop_cfg = bi_so_leader.BiSOLeaderConfig(
        left_arm_config=so_leader.SO101LeaderConfig(
            port=args.left_port,
            use_degrees=False,
        ),
        right_arm_config=so_leader.SO101LeaderConfig(
            port=args.right_port,
            use_degrees=False,
        ),
    )

    teleop = make_teleoperator_from_config(teleop_cfg)
    env = load_env()

    teleop.connect()

    # BiArm tasks require initialize() before any stepping
    if hasattr(env, "initialize"):
        env.initialize()

    try:
        teleop_loop(teleop=teleop, env=env, fps=args.fps)
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted.")
    finally:
        teleop.disconnect()
        env.close()


if __name__ == "__main__":
    main()
