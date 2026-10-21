"""Record success/failure videos from policy rollouts.

Runs the trained policy N times and records each episode as a video,
useful for demos and presentations.

Usage:
    python -m vla.record_video \
        --backend smolvla \
        --checkpoint vla/runs/smolvla/last \
        --episodes 10 \
        --output videos/duck_push

    # Real robot recording
    python -m vla.record_video \
        --backend smolvla \
        --checkpoint vla/runs/smolvla/last \
        --follower-port /dev/ttyACM1 \
        --global-cam 0 \
        --episodes 10 \
        --output videos/duck_push_real
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from vla.rollout import load_policy, run_rollout, _save_video


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["smolvla", "groot", "act"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env", choices=["single", "dual", "push_t"], default="single")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--output", default="videos/duck_push", help="Output directory (videos saved inside)")
    p.add_argument("--real", action="store_true", help="Record on real robot instead of sim")
    p.add_argument("--follower-port", default="/dev/ttyACM1")
    p.add_argument("--wrist-cam", type=int, default=None)
    p.add_argument("--global-cam", type=int, default=0)
    return p.parse_args(argv)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    print(f"Loading {args.backend} from {args.checkpoint}...")
    policy_fn = load_policy(args.backend, args.checkpoint)

    if args.real:
        _record_real(policy_fn, args, output_dir)
    else:
        _record_sim(policy_fn, args, output_dir)


def _record_sim(policy_fn, args, output_dir: Path):
    """Record videos from sim rollouts."""
    if args.env == "single":
        from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig
        env = SO101SingleArmEnv(SingleArmEnvConfig(task="none"))
    elif args.env == "dual":
        from envs.mujoco.so101_dual_arm.env import SO101DualArmEnv, DualArmEnvConfig
        env = SO101DualArmEnv(DualArmEnvConfig(task="none"))
    elif args.env == "push_t":
        from envs.mujoco.so101_single_arm_push_t.env import SO101SingleArmPushTEnv, SingleArmPushTEnvConfig
        env = SO101SingleArmPushTEnv(SingleArmPushTEnvConfig())

    successes = 0
    for ep in range(args.episodes):
        video_path = str(output_dir / f"episode_{ep:03d}.mp4")
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")

        result = run_rollout(
            policy_fn, env,
            max_steps=args.max_steps,
            hz=args.hz,
            save_video=video_path,
            verbose=True,
        )

        if result["success"]:
            successes += 1
            # Rename to indicate success
            success_path = str(output_dir / f"SUCCESS_{ep:03d}.mp4")
            os.rename(video_path, success_path)
            print(f"  ✓ SUCCESS → {success_path}")
        else:
            print(f"  ✗ FAILED → {video_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Recorded {args.episodes} videos to {output_dir}")
    print(f"Success rate: {successes}/{args.episodes} ({100*successes/args.episodes:.0f}%)")
    print(f"{'='*50}")

    env.close()


def _record_real(policy_fn, args, output_dir: Path):
    """Record videos from real robot deployment."""
    from vla.deploy import FollowerArm, RealCamera, FakeFollower

    # Connect hardware
    follower = FollowerArm(args.follower_port)
    global_cam = None
    wrist_cam = None
    try:
        global_cam = RealCamera(args.global_cam)
    except Exception as e:
        print(f"WARNING: global camera not available ({e})")
    try:
        if args.wrist_cam is not None:
            wrist_cam = RealCamera(args.wrist_cam)
    except Exception as e:
        print(f"WARNING: wrist camera not available ({e})")

    dt = 1.0 / args.hz

    try:
        for ep in range(args.episodes):
            video_path = str(output_dir / f"real_{ep:03d}.mp4")
            print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
            input("Place duck and press Enter to start...")

            frames = []
            for step in range(args.max_steps):
                state = follower.read_positions_rad()
                wrist_img = wrist_cam.read() if wrist_cam else np.zeros((480, 640, 3), dtype=np.uint8)
                global_img = global_cam.read() if global_cam else np.zeros((480, 640, 3), dtype=np.uint8)

                action = policy_fn({"wrist": wrist_img, "global": global_img}, state.astype(np.float32))
                follower.send_action_rad(action)

                # Record frame (global camera view)
                if global_img is not None and global_img.any():
                    frames.append(global_img)
                elif wrist_img is not None and wrist_img.any():
                    frames.append(wrist_img)

                if step % 30 == 0:
                    print(f"  step {step:4d}")

                import time
                time.sleep(dt)

            # Save video
            if frames:
                _save_video(frames, video_path, fps=int(args.hz))
                print(f"  Saved: {video_path}")

    except KeyboardInterrupt:
        print("\n\nRecording stopped.")
    finally:
        follower.disconnect()
        if global_cam:
            global_cam.close()
        if wrist_cam:
            wrist_cam.close()


if __name__ == "__main__":
    main()
