"""Closed-loop evaluation of a finetuned VLA checkpoint in MuJoCo sim.

Runs the policy in a loop: reads observations from the sim env, queries the
model for actions, steps the env, and records success/failure + video.

Usage:
    python -m vla.rollout --backend smolvla --checkpoint vla/runs/smolvla/last
    python -m vla.rollout --backend smolvla --checkpoint <path> --save-video rollout.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def load_policy(backend: str, checkpoint: str):
    """Load a trained VLA policy for inference.

    Returns an callable: policy(images, state) -> action (6,) radians.
    """
    if backend == "smolvla":
        return _load_smolvla(checkpoint)
    elif backend == "groot":
        return _load_groot(checkpoint)
    elif backend == "act":
        return _load_act(checkpoint)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _load_smolvla(checkpoint: str):
    """Load SmolVLA via LeRobot's policy interface."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig.from_pretrained(checkpoint)
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.eval()
    policy.to("cuda" if _cuda_available() else "cpu")

    def _infer(images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """images: {cam_name: (H, W, 3)}, state: (6,) → action (6,) radians."""
        import torch

        # Format observations for SmolVLA
        obs = {
            "observation.images.wrist": _to_tensor(images.get("wrist")),
            "observation.images.global": _to_tensor(images.get("global")),
            "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        return action.squeeze(0).cpu().numpy()

    return _infer


def _load_groot(checkpoint: str):
    """Load GR00T N1.6 for inference."""
    import torch

    def _infer(images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        # GR00T inference placeholder — implement with Isaac-GR00T's inference API
        raise NotImplementedError("GR00T inference needs Isaac-GR00T env (py3.10)")

    return _infer


def _load_act(checkpoint: str):
    """Load ACT policy via LeRobot."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig.from_pretrained(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.eval()
    policy.to("cuda" if _cuda_available() else "cpu")

    def _infer(images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        import torch

        obs = {
            "observation.images.wrist": _to_tensor(images.get("wrist")),
            "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        return action.squeeze(0).cpu().numpy()

    return _infer


def _to_tensor(img: np.ndarray | None):
    """(H, W, 3) uint8 → (1, 3, H, W) float32."""
    import torch
    if img is None:
        return torch.zeros(1, 3, 480, 640)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------

def run_rollout(
    policy_fn,
    env,
    max_steps: int = 600,
    hz: float = 30.0,
    save_video: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run one episode of the policy in the env.

    Returns:
        dict with keys: success, steps, duration_s, frames
    """
    dt = 1.0 / hz
    obs, _ = env.reset()

    frames = []
    total_steps = 0
    t_start = time.perf_counter()

    for step in range(max_steps):
        t0 = time.perf_counter()

        # Extract observations
        images = {
            "wrist": obs.get("images", {}).get("wrist", None),
            "global": obs.get("images", {}).get("overhead_cam",
                       obs.get("images", {}).get("global", None)),
        }

        # Joint state — try common key names
        state = (
            obs.get("joint_pos", None)
            or obs.get("observation.state", None)
            or obs.get("left_joint_pos", None)
        )

        if state is None:
            raise ValueError(f"No joint state found in obs keys: {list(obs.keys())}")

        # Query policy
        action = policy_fn(images, state.astype(np.float32))

        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        # Record video frame
        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Print progress
        if verbose and step % 30 == 0:
            success = info.get("success", False)
            print(f"  step {step:4d} | reward {reward:.3f} | success {success}")

        # Early termination on success
        if terminated:
            if verbose:
                print(f"  SUCCESS at step {step}")
            break

        # Wait for real-time
        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    duration = time.perf_counter() - t_start
    success = info.get("success", False)

    result = {
        "success": success,
        "steps": total_steps,
        "duration_s": duration,
        "frames": frames,
    }

    # Save video
    if save_video and frames:
        _save_video(frames, save_video, fps=int(hz))
        print(f"  Video saved to {save_video}")

    return result


def _save_video(frames: list[np.ndarray], path: str, fps: int = 30):
    """Save a list of RGB frames as an MP4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["smolvla", "groot", "act"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    p.add_argument("--env", choices=["single", "dual", "push_t"], default="single")
    p.add_argument("--episodes", type=int, default=1, help="Number of eval episodes")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--save-video", default=None, help="Output video path (e.g., rollout.mp4)")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args(argv)


def main():
    args = parse_args()

    # Build env
    if args.env == "single":
        from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig
        env = SO101SingleArmEnv(SingleArmEnvConfig(task="none"))
    elif args.env == "dual":
        from envs.mujoco.so101_dual_arm.env import SO101DualArmEnv, DualArmEnvConfig
        env = SO101DualArmEnv(DualArmEnvConfig(task="none"))
    elif args.env == "push_t":
        from envs.mujoco.so101_single_arm_push_t.env import SO101SingleArmPushTEnv, SingleArmPushTEnvConfig
        env = SO101SingleArmPushTEnv(SingleArmPushTEnvConfig())

    # Load policy
    print(f"Loading {args.backend} from {args.checkpoint}...")
    policy_fn = load_policy(args.backend, args.checkpoint)

    # Run eval
    results = []
    for ep in range(args.episodes):
        print(f"\n--- Eval episode {ep + 1}/{args.episodes} ---")
        video_path = args.save_video.replace(".mp4", f"_{ep}.mp4") if args.save_video else None
        result = run_rollout(
            policy_fn, env,
            max_steps=args.max_steps,
            hz=args.hz,
            save_video=video_path,
            verbose=args.verbose,
        )
        results.append(result)

    # Summary
    successes = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}")
    print(f"Results: {successes}/{args.episodes} successes")
    print(f"Average steps: {np.mean([r['steps'] for r in results]):.1f}")
    print(f"Average duration: {np.mean([r['duration_s'] for r in results]):.1f}s")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()
