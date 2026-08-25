"""
Real SO-101 leader arm (via LeRobot) -> MuJoCo follower sim, with optional
simultaneous mirroring to a REAL follower arm ("shadow teleop": one leader
drives the simulation and the physical arm at the same time).

    # leader -> sim only
    python scripts/teleop_leader_lerobot.py --env single --leader-port /dev/ttyACM0
    # leader -> sim + real follower at the same time
    python scripts/teleop_leader_lerobot.py --env single --leader-port /dev/ttyACM0 \
        --mirror-real --follower-port /dev/ttyACM1
    # dual-arm sim (two leaders)
    python scripts/teleop_leader_lerobot.py --env dual \
        --leader-port /dev/ttyACM0 --leader-port-2 /dev/ttyACM1
    # hardware-free CI check
    MUJOCO_GL=egl python scripts/teleop_leader_lerobot.py --env single --dry-run

Uses LeRobot's SO101 leader/follower classes, so calibration is shared with
the rest of the LeRobot tooling (`lerobot-calibrate --teleop.type=so101_leader
--teleop.id=<id>`). Leader readings arrive in degrees; the sim wants radians
and the real follower wants degrees — conversion handled here.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_env(env_name: str):
    if env_name == "single":
        from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig

        return SO101SingleArmEnv(SingleArmEnvConfig(task="none")), None
    from envs.mujoco.so101_dual_arm.env import SO101DualArmEnv, DualArmEnvConfig

    return SO101DualArmEnv(DualArmEnvConfig(task="none")), "dual"


def connect_leader(port: str, robot_id: str | None):
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    cfg = SOLeaderTeleopConfig(port=port, id=robot_id)
    leader = SOLeader(cfg)
    leader.connect()
    return leader


def connect_follower(port: str, robot_id: str | None):
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    cfg = SO101FollowerConfig(port=port, id=robot_id)
    follower = SO101Follower(cfg)
    follower.connect()
    return follower


def leader_to_rad(action_deg: dict[str, float]) -> np.ndarray:
    """Leader action dict (degrees, '<joint>.pos' keys) -> (6,) radians."""
    out = np.zeros(6)
    for i, j in enumerate(JOINT_ORDER):
        out[i] = math.radians(action_deg[f"{j}.pos"])
    return out


def rad_to_follower_action(targets_rad: np.ndarray) -> dict[str, float]:
    """(6,) radians -> follower action dict (degrees, '<joint>.pos' keys)."""
    return {f"{j}.pos": math.degrees(targets_rad[i]) for i, j in enumerate(JOINT_ORDER)}


def run(env, leaders, mirror, steps: int, hz: float = 60.0):
    """leaders: list of connected leader arms (1 for single, 2 for dual)."""
    obs, _ = env.reset()
    dt = 1.0 / hz
    n_arms = len(leaders)
    try:
        for step in range(steps):
            loop_start = time.perf_counter()
            for arm_idx, leader in enumerate(leaders):
                targets = leader_to_rad(leader.get_action())
                if n_arms == 1:
                    env.data.ctrl[:] = np.clip(
                        targets, env.action_space.low, env.action_space.high
                    )
                else:
                    lo = env.action_space.low[arm_idx * 6:(arm_idx + 1) * 6]
                    hi = env.action_space.high[arm_idx * 6:(arm_idx + 1) * 6]
                    env.data.ctrl[arm_idx * 6:(arm_idx + 1) * 6] = np.clip(targets, lo, hi)
                if mirror is not None and n_arms == 1:
                    mirror.send(rad_to_follower_action(targets))
            # step physics with the written ctrl (pass the current ctrl to keep
            # the env's own clipping consistent)
            env.step(env.data.ctrl.copy())
            if step % 60 == 0:
                print(f"[teleop] step {step}: ctrl={np.round(env.data.ctrl[:6], 2)}")
            wait = dt - (time.perf_counter() - loop_start)
            if wait > 0:
                time.sleep(wait)
    finally:
        if mirror is not None:
            mirror.robot.disconnect()
        for leader in leaders:
            leader.disconnect()


def run_dry_run(env, steps: int = 100, hz: float = 60.0):
    """Sinusoidal fake leader — no hardware, CI-safe."""
    obs, _ = env.reset()
    dt = 1.0 / hz
    mid = (env.action_space.high + env.action_space.low) / 2
    amp = (env.action_space.high - env.action_space.low) / 2 * 0.3
    for step in range(steps):
        t = step * dt
        targets = mid + amp * np.sin(2 * math.pi * 0.5 * t + np.arange(env.action_space.shape[0]))
        env.step(targets.astype(np.float32))
    err = float(np.abs(env.data.ctrl - mid).mean())
    print(f"[dry-run] {steps} steps OK; final |ctrl - mid| mean = {err:.3f} rad")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=["single", "dual"], default="single")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-port-2", default="/dev/ttyACM1", help="second leader (dual env)")
    parser.add_argument("--leader-id", default=None, help="LeRobot calibration id for the leader")
    parser.add_argument("--mirror-real", action="store_true",
                        help="also send the same targets to a real follower arm")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--follower-id", default=None)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="fake leader, headless, no hardware")
    args = parser.parse_args()

    env, kind = build_env(args.env)

    if args.dry_run:
        run_dry_run(env, steps=min(args.steps, 100))
        return

    leaders = [connect_leader(args.leader_port, args.leader_id)]
    if args.env == "dual":
        leaders.append(connect_leader(args.leader_port_2, args.leader_id))
    mirror = connect_follower(args.follower_port, args.follower_id) if args.mirror_real else None
    if args.mirror_real and args.env == "dual":
        raise SystemExit("--mirror-real supports the single-arm env only (one real follower)")

    try:
        run(env, leaders, mirror, args.steps, args.hz)
    except KeyboardInterrupt:
        print("\n[teleop] interrupted")


if __name__ == "__main__":
    main()
