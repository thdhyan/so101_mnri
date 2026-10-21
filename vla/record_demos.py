"""Record real-world teleop demonstrations to a LeRobotDataset.

Captures wrist camera + global camera images, follower joint states, and
leader arm actions at each timestep. Episodes are auto-segmented by pressing
Enter between demonstrations.

Usage:
    python -m vla.record_demos \
        --leader-port /dev/ttyACM0 \
        --follower-port /dev/ttyACM1 \
        --global-cam 0 \
        --dataset thakk100/so101_duck_push \
        --task "push the duck into the square tape" \
        --episodes 400

    # Dry run (no hardware, sinusoidal fake actions)
    python -m vla.record_demos --dry-run --episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
N_JOINTS = 6
DEFAULT_HZ = 30.0


# ---------------------------------------------------------------------------
# LeRobotDataset writer
# ---------------------------------------------------------------------------

class LeRobotDatasetWriter:
    """Minimal LeRobot v3 dataset writer.

    Stores episodes as individual .npz files + metadata, ready for
    lerobot-train / SmolVLA / GR00T consumption.
    """

    def __init__(self, repo_id: str, task: str, fps: float, root: str = "data"):
        self.repo_id = repo_id
        self.task = task
        self.fps = fps
        self.root = Path(root) / repo_id.replace("/", "_")
        self.root.mkdir(parents=True, exist_ok=True)

        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)

        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(exist_ok=True)

        self._episode_count = 0
        self._info = {
            "repo_id": repo_id,
            "task": task,
            "fps": fps,
            "n_joints": N_JOINTS,
            "cameras": ["wrist", "global"],
        }
        self._save_info()

    def _save_info(self):
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(self._info, f, indent=2)

    def save_episode(
        self,
        images_wrist: list[np.ndarray],
        images_global: list[np.ndarray],
        joint_states: list[np.ndarray],
        actions: list[np.ndarray],
    ) -> int:
        """Save one episode. Returns the episode index."""
        ep_idx = self._episode_count
        n_steps = len(actions)

        ep_data = {
            "episode_index": ep_idx,
            "task": self.task,
            "fps": self.fps,
            "n_steps": n_steps,
            # Images: (T, H, W, 3) uint8
            "images.wrist": np.stack(images_wrist),
            "images.global": np.stack(images_global),
            # Joint states: (T, 6) float32
            "observation.state": np.stack(joint_states),
            # Actions: (T, 6) float32
            "action": np.stack(actions),
        }

        path = self.episodes_dir / f"episode_{ep_idx:06d}.npz"
        np.savez_compressed(path, **ep_data)

        self._episode_count += 1
        print(f"  [saved] episode {ep_idx}: {n_steps} steps → {path.name}")
        return ep_idx

    @property
    def total_episodes(self) -> int:
        return self._episode_count


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

class CameraCapture:
    """OpenCV camera capture wrapper."""

    def __init__(self, camera_id: int | str, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(int(camera_id) if isinstance(camera_id, int) else camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_id}")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Leader arm interface (reuses LeRobot SO-101 leader)
# ---------------------------------------------------------------------------

def connect_leader(port: str, robot_id: str | None = None):
    """Connect to a real SO-101 leader arm via LeRobot."""
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    cfg = SOLeaderTeleopConfig(port=port, id=robot_id)
    leader = SOLeader(cfg)
    leader.connect()
    return leader


def leader_to_action(action_deg: dict[str, float]) -> np.ndarray:
    """Leader action dict (degrees, '<joint>.pos' keys) -> (6,) radians."""
    out = np.zeros(N_JOINTS)
    for i, j in enumerate(JOINT_ORDER):
        out[i] = math.radians(action_deg[f"{j}.pos"])
    return out


# ---------------------------------------------------------------------------
# Dry-run fake leader
# ---------------------------------------------------------------------------

class FakeLeader:
    """Sinusoidal fake leader for CI / hardware-free testing."""

    def __init__(self):
        self._t = 0.0

    def get_action(self, dt: float = 1.0 / DEFAULT_HZ) -> dict[str, float]:
        self._t += dt
        angles = 20.0 * np.sin(2 * math.pi * 0.3 * self._t + np.arange(N_JOINTS) * 0.5)
        return {f"{j}.pos": float(angles[i]) for i, j in enumerate(JOINT_ORDER)}

    def disconnect(self):
        pass


# ---------------------------------------------------------------------------
# Recording loop
# ---------------------------------------------------------------------------

def record_single_episode(
    leader,
    wrist_cam: CameraCapture | None,
    global_cam: CameraCapture | None,
    hz: float,
    max_steps: int = 1000,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Record one teleop episode until user presses Enter."""
    dt = 1.0 / hz
    images_wrist = []
    images_global = []
    joint_states = []
    actions = []

    print("  Recording... (press Enter to stop)")

    for step in range(max_steps):
        t0 = time.perf_counter()

        # Read leader action
        raw_action = leader.get_action() if hasattr(leader, "get_action") else leader.get_action()
        action = leader_to_action(raw_action)

        # Capture images
        wrist_img = wrist_cam.read() if wrist_cam else np.zeros((480, 640, 3), dtype=np.uint8)
        global_img = global_cam.read() if global_cam else np.zeros((480, 640, 3), dtype=np.uint8)

        # Store
        images_wrist.append(wrist_img)
        images_global.append(global_img)
        joint_states.append(action.copy())  # leader state = follower target
        actions.append(action.copy())

        # Print progress
        if step % 30 == 0:
            print(f"    step {step:4d} | joints: {np.round(np.degrees(action), 1)}")

        # Wait for next timestep
        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        # Check if user pressed Enter (non-blocking)
        import select
        import sys
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            print("  Stopped.")
            break

    return images_wrist, images_global, joint_states, actions


def run_recording(args):
    """Main recording loop."""
    # Connect cameras
    wrist_cam = None
    global_cam = None
    try:
        wrist_cam = CameraCapture(args.wrist_cam) if args.wrist_cam is not None else None
    except Exception as e:
        print(f"WARNING: wrist camera not available ({e}), recording blank frames")
    try:
        global_cam = CameraCapture(args.global_cam) if args.global_cam is not None else None
    except Exception as e:
        print(f"WARNING: global camera not available ({e}), recording blank frames")

    # Connect leader arm
    if args.dry_run:
        leader = FakeLeader()
    else:
        leader = connect_leader(args.leader_port, args.leader_id)

    # Dataset writer
    writer = LeRobotDatasetWriter(
        repo_id=args.dataset,
        task=args.task,
        fps=args.hz,
        root=args.data_root,
    )

    print(f"\n{'='*60}")
    print(f"Recording to: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Target episodes: {args.episodes}")
    print(f"FPS: {args.hz}")
    print(f"{'='*60}\n")

    try:
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
            input("Place the duck and press Enter to start recording...")

            imgs_w, imgs_g, states, acts = record_single_episode(
                leader, wrist_cam, global_cam,
                hz=args.hz,
                max_steps=args.max_steps,
            )

            if len(acts) < 10:
                print("  Too few steps, skipping episode.")
                continue

            writer.save_episode(imgs_w, imgs_g, states, acts)
            print(f"  Total episodes recorded: {writer.total_episodes}")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. {writer.total_episodes} episodes saved.")
    finally:
        if not args.dry_run:
            leader.disconnect()
        if wrist_cam:
            wrist_cam.close()
        if global_cam:
            global_cam.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--leader-port", default="/dev/ttyACM0", help="Leader arm serial port")
    p.add_argument("--leader-id", default=None, help="LeRobot calibration id")
    p.add_argument("--wrist-cam", type=int, default=None, help="Wrist camera device ID (None=skip)")
    p.add_argument("--global-cam", type=int, default=0, help="Global camera device ID (None=skip)")
    p.add_argument("--dataset", required=True, help="LeRobot repo id or local name")
    p.add_argument("--task", default="push the duck into the square tape", help="Language instruction")
    p.add_argument("--episodes", type=int, default=400, help="Number of episodes to record")
    p.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--hz", type=float, default=DEFAULT_HZ, help="Recording frequency")
    p.add_argument("--data-root", default="data", help="Root directory for datasets")
    p.add_argument("--dry-run", action="store_true", help="Fake leader, no hardware")
    return p.parse_args(argv)


def main():
    args = parse_args()
    run_recording(args)


if __name__ == "__main__":
    main()
