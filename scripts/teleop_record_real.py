#!/usr/bin/env python3
"""
Integrated real-world teleop + recording for SO-101 (5-DOF, no wrist_roll).
Leader arm -> real follower + camera recording to LeRobotDataset + rerun visualization.

Hardware:
  Leader:  /dev/ttyACM1  (mnri_arms_1_leader)
  Follower: /dev/ttyACM0 (mnri_arms_1_follower)
  Cameras: wrist=0 (laptop), global=5 (USB overhead)

Usage:
    python scripts/teleop_record_real.py \
        --leader-port /dev/ttyACM1 --leader-id mnri_arms_1_leader \
        --follower-port /dev/ttyACM0 --follower-id mnri_arms_1_follower \
        --wrist-cam 0 --global-cam 5 \
        --dataset thakk100/so101_duck_push \
        --task "push the duck into the square tape" \
        --episodes 100 --hz 30 --data-root data \
        --rerun
"""

import argparse
import math
import queue
import select
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Optional rerun
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

# LeRobot
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

# ── 5-DOF joint order (no wrist_roll) ────────────────────────────────────────
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
N_JOINTS = 5
DEFAULT_HZ = 30.0


class CameraCapture:
    """OpenCV camera wrapper."""

    def __init__(self, camera_id: int, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
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


class LeRobotDatasetWriter:
    """Minimal LeRobot v3 dataset writer with async background saving."""

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
        self._save_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._save_worker, daemon=True)
        self._worker.start()
        self._save_info()

    def _save_info(self):
        import json
        info = {
            "repo_id": self.repo_id,
            "task": self.task,
            "fps": self.fps,
            "n_joints": N_JOINTS,
            "cameras": ["wrist", "global"],
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _save_worker(self):
        while True:
            item = self._save_queue.get()
            if item is None:
                break
            ep_idx, ep_data, path, outcome, n_steps = item
            np.savez_compressed(path, **ep_data)
            print(f"  [saved] episode {ep_idx} [{outcome}]: {n_steps} steps → {path.name}")
            self._save_queue.task_done()

    def save_episode(self, images_wrist, images_global, joint_states, actions, outcome: str = "success") -> int:
        ep_idx = self._episode_count
        n_steps = len(actions)
        ep_data = {
            "episode_index": ep_idx,
            "task": self.task,
            "fps": self.fps,
            "n_steps": n_steps,
            "outcome": outcome,
            "images.wrist": np.stack(images_wrist),
            "images.global": np.stack(images_global),
            "observation.state": np.stack(joint_states),
            "action": np.stack(actions),
        }
        path = self.episodes_dir / f"episode_{ep_idx:06d}.npz"
        self._save_queue.put((ep_idx, ep_data, path, outcome, n_steps))
        self._episode_count += 1
        pending = self._save_queue.qsize()
        print(f"  [queued] episode {ep_idx} [{outcome}]: {n_steps} steps (saving in background, {pending} pending)")
        return ep_idx

    def flush(self):
        """Block until all queued episodes are saved."""
        self._save_queue.join()
        self._save_queue.put(None)  # stop worker

    @property
    def total_episodes(self) -> int:
        return self._episode_count


def leader_to_action(action_deg: dict[str, float]) -> np.ndarray:
    """Leader action dict (degrees) -> (5,) radians for follower."""
    out = np.zeros(N_JOINTS)
    for i, j in enumerate(JOINT_ORDER):
        out[i] = math.radians(action_deg[f"{j}.pos"])
    return out


def action_to_follower(action_rad: np.ndarray) -> dict[str, float]:
    """(5,) radians -> follower action dict (degrees)."""
    return {f"{j}.pos": math.degrees(action_rad[i]) for i, j in enumerate(JOINT_ORDER)}


def init_rerun(dataset_name: str):
    """Initialize rerun for live visualization."""
    if not RERUN_AVAILABLE:
        return False
    import rerun.blueprint as rrb
    rr.init(f"so101_teleop_{dataset_name}", spawn=True)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="Wrist Cam", origin="cameras/wrist"),
                rrb.Spatial2DView(name="Global Cam", origin="cameras/global"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Leader Joints", origin="joints/leader"),
                rrb.TimeSeriesView(name="Follower Joints", origin="joints/follower"),
            ),
        )
    )
    rr.send_blueprint(blueprint)
    return True


def log_to_rerun(step, leader_action, follower_obs, wrist_img, global_img):
    """Log teleop data to rerun."""
    if not RERUN_AVAILABLE:
        return
    # Use the correct rerun API
    rr.set_time("frame", sequence=step)
    for i, name in enumerate(JOINT_ORDER):
        rr.log(f"joints/leader/{name}", rr.Scalars(leader_action[i]))
        rr.log(f"joints/follower/{name}", rr.Scalars(follower_obs[i]))
    if wrist_img is not None:
        rr.log("cameras/wrist", rr.Image(wrist_img).compress(jpeg_quality=75))
    if global_img is not None:
        rr.log("cameras/global", rr.Image(global_img).compress(jpeg_quality=75))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--follower-id", required=True)
    parser.add_argument("--wrist-cam", type=int, default=0)
    parser.add_argument("--global-cam", type=int, default=5)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", default="push the duck into the square tape")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rerun", action="store_true", help="Enable rerun visualization")
    args = parser.parse_args()

    # Connect hardware
    print("Connecting leader...")
    leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
    leader = SOLeader(leader_cfg)
    leader.connect()

    print("Connecting follower...")
    follower_cfg = SO101FollowerConfig(port=args.follower_port, id=args.follower_id)
    follower = SO101Follower(follower_cfg)
    follower.connect()

    # Cameras
    print("Opening cameras...")
    wrist_cam = None
    global_cam = None
    for cam_id in [args.wrist_cam, args.global_cam]:
        if cam_id is None:
            continue
        for attempt in range(3):
            try:
                cam = CameraCapture(cam_id)
                if cam_id == args.wrist_cam:
                    wrist_cam = cam
                else:
                    global_cam = cam
                break
            except RuntimeError as e:
                if attempt == 2:
                    print(f"WARNING: camera {cam_id} failed: {e}")
                else:
                    time.sleep(0.5)

    # Dataset writer
    writer = LeRobotDatasetWriter(args.dataset, args.task, args.hz, args.data_root)

    # Rerun
    if args.rerun:
        if not init_rerun(args.dataset):
            print("Rerun not available, continuing without visualization")
            args.rerun = False

    print(f"\n{'='*60}")
    print(f"Recording to: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Target episodes: {args.episodes}")
    print(f"FPS: {args.hz}")
    print(f"Leader: {args.leader_port} ({args.leader_id})")
    print(f"Follower: {args.follower_port} ({args.follower_id})")
    print(f"{'='*60}\n")

    dt = 1.0 / args.hz
    step_count = 0

    try:
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
            input("Place the duck and press Enter to start recording...")

            images_wrist = []
            images_global = []
            joint_states = []
            actions = []

            for step in range(args.max_steps):
                t0 = time.perf_counter()

                # Read leader
                raw_action = leader.get_action()
                action_rad = leader_to_action(raw_action)

                # Send to follower (in degrees)
                follower_action = action_to_follower(action_rad)
                sent_action = follower.send_action(follower_action)

                # Read follower state (observation)
                obs = follower.get_observation()
                follower_pos = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], dtype=np.float32)
                follower_pos_rad = np.radians(follower_pos)

                # Capture images
                wrist_img = wrist_cam.read() if wrist_cam else np.zeros((480, 640, 3), dtype=np.uint8)
                global_img = global_cam.read() if global_cam else np.zeros((480, 640, 3), dtype=np.uint8)

                # Store
                images_wrist.append(wrist_img)
                images_global.append(global_img)
                joint_states.append(follower_pos_rad)
                actions.append(action_rad)

                # Progress
                if step % 30 == 0:
                    print(f"    step {step:4d} | leader: {np.round(np.degrees(action_rad), 1)}")

                # Rerun logging
                if args.rerun:
                    log_to_rerun(step_count, action_rad, follower_pos_rad, wrist_img, global_img)

                step_count += 1

                # Wait for next timestep
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                # Check for Enter key (stop episode)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    print("  Stopped.")
                    break

            if len(actions) < 10:
                print("  Too few steps, skipping episode.")
                continue

            print("  Outcome? [s]uccess / [f]ail / [d]iscard: ", end="", flush=True)
            outcome_key = input().strip().lower()
            if outcome_key == "d":
                print("  Discarded.")
                continue
            outcome = "fail" if outcome_key == "f" else "success"
            writer.save_episode(images_wrist, images_global, joint_states, actions, outcome)
            print(f"  Total episodes: {writer.total_episodes}")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. {writer.total_episodes} episodes queued.")
    finally:
        print("Flushing save queue...")
        writer.flush()
        print(f"Done. {writer.total_episodes} episodes saved to disk.")
        leader.disconnect()
        try:
            follower.disconnect()
        except Exception as e:
            print(f"Follower disconnect warning (non-fatal): {e}")
        if wrist_cam:
            wrist_cam.close()
        if global_cam:
            global_cam.close()


if __name__ == "__main__":
    main()