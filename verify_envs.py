"""
Verification script for both SO-101 MuJoCo environments.

Usage:
    conda activate isaac6
    cd /home/thakk100/Projects/so101_mnri
    python verify_envs.py --env single     # single arm, interactive viewer
    python verify_envs.py --env dual       # dual arm, interactive viewer
    python verify_envs.py --env both       # headless checks only (CI-safe)
    python verify_envs.py --env single --steps 200 --save-images
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def check_obs_shapes(obs: dict, env_name: str):
    print(f"\n  [{env_name}] Observations:")
    for k, v in obs.items():
        if k == "images":
            for cam, img in v.items():
                print(f"    images.{cam}: {img.shape} dtype={img.dtype}")
        else:
            if hasattr(v, "shape"):
                print(f"    {k}: {v.shape} dtype={v.dtype}")
            else:
                print(f"    {k}: {v}")


def test_single_arm(interactive: bool = False, n_steps: int = 100,
                    save_images: bool = False):
    print("\n" + "="*60)
    print("  SINGLE ARM ENV")
    print("="*60)
    from so101_single_arm_env.env import (
        SO101SingleArmEnv, SingleArmEnvConfig, CameraConfig, GoalConfig
    )

    cfg = SingleArmEnvConfig(
        task="push",
        render_mode="human" if interactive else "rgb_array",
    )
    env = SO101SingleArmEnv(cfg)

    print(f"  action_space : {env.action_space}")
    print(f"  max_steps    : {env._max_steps}")

    obs, info = env.reset()
    check_obs_shapes(obs, "single")

    if save_images:
        _save_camera_frames(obs["images"], "single_arm", 0)

    total_reward = 0.0
    t0 = time.time()
    for step in range(n_steps):
        action = env.action_space.sample() * 0.05   # small random moves
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if interactive:
            env.render()
        if step == n_steps // 2 and save_images:
            _save_camera_frames(obs["images"], "single_arm", step)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.time() - t0
    fps = n_steps / elapsed
    print(f"  {n_steps} steps in {elapsed:.2f}s  ({fps:.0f} FPS)")
    print(f"  total_reward: {total_reward:.3f}")
    print(f"  last obs.object_pos: {obs.get('object_pos')}")

    # Test custom camera positions
    print("\n  Testing custom camera positions...")
    custom_cfg = SingleArmEnvConfig(
        task="none",
        outside_cameras=[
            CameraConfig("outside_left",  pos=(-0.3, -0.5, 1.0), lookat=(0.35, 0.0, 0.9), fov=70),
            CameraConfig("outside_right", pos=( 0.7, -0.5, 1.0), lookat=(0.35, 0.0, 0.9), fov=70),
        ]
    )
    env2 = SO101SingleArmEnv(custom_cfg)
    obs2, _ = env2.reset()
    print(f"  custom outside_left  shape: {obs2['images']['outside_left'].shape}")
    print(f"  custom outside_right shape: {obs2['images']['outside_right'].shape}")
    env2.close()

    env.close()
    print("  SINGLE ARM: PASS")


def test_dual_arm(interactive: bool = False, n_steps: int = 100,
                  save_images: bool = False):
    print("\n" + "="*60)
    print("  DUAL ARM ENV")
    print("="*60)
    from so101_dual_arm_env.env import (
        SO101DualArmEnv, DualArmEnvConfig, CameraConfig
    )

    cfg = DualArmEnvConfig(
        task="push",
        render_mode="human" if interactive else "rgb_array",
    )
    env = SO101DualArmEnv(cfg)

    print(f"  action_space : {env.action_space}")
    print(f"  max_steps    : {env._max_steps}")

    obs, info = env.reset()
    check_obs_shapes(obs, "dual")

    if save_images:
        _save_camera_frames(obs["images"], "dual_arm", 0)

    total_reward = 0.0
    t0 = time.time()
    for step in range(n_steps):
        action = env.action_space.sample() * 0.05
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if interactive:
            env.render()
        if step == n_steps // 2 and save_images:
            _save_camera_frames(obs["images"], "dual_arm", step)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.time() - t0
    fps = n_steps / elapsed
    print(f"  {n_steps} steps in {elapsed:.2f}s  ({fps:.0f} FPS)")
    print(f"  total_reward: {total_reward:.3f}")
    env.close()
    print("  DUAL ARM: PASS")


def _save_camera_frames(images: dict, env_tag: str, step: int):
    """Save camera frames to /tmp/ as PNG files for visual inspection."""
    try:
        from PIL import Image
        out_dir = Path("/tmp/so101_verify")
        out_dir.mkdir(exist_ok=True)
        for cam_name, img in images.items():
            path = out_dir / f"{env_tag}_{cam_name}_step{step:04d}.png"
            Image.fromarray(img).save(path)
            print(f"  Saved: {path}")
    except ImportError:
        # Pillow not available — use matplotlib instead
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            out_dir = Path("/tmp/so101_verify")
            out_dir.mkdir(exist_ok=True)
            fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5))
            if len(images) == 1:
                axes = [axes]
            for ax, (cam_name, img) in zip(axes, images.items()):
                ax.imshow(img)
                ax.set_title(cam_name)
                ax.axis("off")
            path = out_dir / f"{env_tag}_cameras_step{step:04d}.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  (image save skipped: {e})")


def test_make_env_api(env_name: str):
    """Test the EnvHub make_env API specifically."""
    print(f"\n  Testing make_env API for {env_name}...")
    if env_name == "single":
        from so101_single_arm_env.env import make_env
        vec_env = make_env(n_envs=1)
    else:
        from so101_dual_arm_env.env import make_env
        vec_env = make_env(n_envs=1)

    obs, info = vec_env.reset()
    print(f"  vec_env.observation_space: OK")
    action = vec_env.action_space.sample()
    obs, rew, term, trunc, info = vec_env.step(action)
    print(f"  step() reward: {rew}")
    vec_env.close()
    print(f"  make_env API for {env_name}: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["single", "dual", "both"], default="single")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interactive", action="store_true",
                        help="Open MuJoCo passive viewer window")
    parser.add_argument("--save-images", action="store_true",
                        help="Save camera frames to /tmp/so101_verify/")
    args = parser.parse_args()

    run_single = args.env in ("single", "both")
    run_dual   = args.env in ("dual",   "both")

    if run_single:
        test_single_arm(interactive=args.interactive, n_steps=args.steps,
                        save_images=args.save_images)
        test_make_env_api("single")

    if run_dual:
        test_dual_arm(interactive=args.interactive, n_steps=args.steps,
                      save_images=args.save_images)
        test_make_env_api("dual")

    print("\n\nAll tests passed.")


if __name__ == "__main__":
    main()
