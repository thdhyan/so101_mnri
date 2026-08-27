"""Deploy a trained VLA policy on the real SO-101 follower arm.

Reads observations from real cameras + follower joint states, queries the
policy, and sends actions to the follower.

Usage:
    python -m vla.deploy \
        --backend smolvla \
        --checkpoint vla/runs/smolvla/last \
        --follower-port /dev/ttyACM1 \
        --global-cam 0

    # Dry run (policy in loop, but no real arm — just prints actions)
    python -m vla.deploy \
        --backend smolvla \
        --checkpoint vla/runs/smolvla/last \
        --dry-run
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from vla.rollout import load_policy, _to_tensor

JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
N_JOINTS = 6
DEFAULT_HZ = 30.0


# ---------------------------------------------------------------------------
# Real hardware interfaces
# ---------------------------------------------------------------------------

class FollowerArm:
    """SO-101 follower arm via Feetech STS3215 servos."""

    ADDR_GOAL_POSITION = 60
    ADDR_PRESENT_POSITION = 56
    LEN_PRESENT_POSITION = 2
    BAUDRATE = 1_000_000
    PROTOCOL_END = 0
    SERVO_IDS = [1, 2, 3, 4, 5, 6]
    TICKS_PER_RAD = 4096 / (2 * math.pi)

    def __init__(self, port: str):
        import scservo_sdk as sc

        self.sc = sc
        self.port_handler = sc.PortHandler(port)
        self.packet_handler = sc.PacketHandler(self.PROTOCOL_END)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Cannot open port: {port}")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise RuntimeError(f"Cannot set baudrate on {port}")

        self.group_read = sc.GroupSyncRead(
            self.port_handler, self.packet_handler,
            self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION,
        )
        self.group_write = sc.GroupSyncWrite(
            self.port_handler, self.packet_handler,
            self.ADDR_GOAL_POSITION, self.LEN_PRESENT_POSITION,
        )

        for sid in self.SERVO_IDS:
            self.group_read.addParam(sid)
            self.group_write.addParam(sid, [0] * self.LEN_PRESENT_POSITION)

    def read_positions_rad(self) -> np.ndarray:
        """Read current joint positions in radians."""
        result = self.group_read.txRxPacket()
        if result != self.sc.COMM_SUCCESS:
            raise RuntimeError(f"Read failed: {self.packet_handler.getTxRxResult(result)}")

        rad = np.zeros(N_JOINTS)
        for i, sid in enumerate(self.SERVO_IDS):
            ticks = self.group_read.getData(sid, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            rad[i] = (ticks - 2048) / self.TICKS_PER_RAD
        return rad

    def send_action_rad(self, action_rad: np.ndarray):
        """Send target joint positions in radians."""
        action_rad = np.clip(action_rad, -np.pi, np.pi)

        for i, sid in enumerate(self.SERVO_IDS):
            ticks = int(action_rad[i] * self.TICKS_PER_RAD + 2048)
            ticks = max(0, min(4095, ticks))
            self.group_write.changeParam(sid, [ticks & 0xFF, (ticks >> 8) & 0xFF])

        result = self.group_write.txRxPacket()
        if result != self.sc.COMM_SUCCESS:
            raise RuntimeError(f"Write failed: {self.packet_handler.getTxRxResult(result)}")

    def disconnect(self):
        self.port_handler.closePort()


class RealCamera:
    """OpenCV camera for real-world capture."""

    def __init__(self, device_id: int | str, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(int(device_id) if isinstance(device_id, int) else device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device_id}")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class FakeFollower:
    """Fake follower for dry-run testing."""

    def __init__(self):
        self._pos = np.zeros(N_JOINTS)

    def read_positions_rad(self) -> np.ndarray:
        return self._pos.copy()

    def send_action_rad(self, action_rad: np.ndarray):
        self._pos = action_rad.copy()

    def disconnect(self):
        pass


# ---------------------------------------------------------------------------
# Deploy loop
# ---------------------------------------------------------------------------

def run_deploy(
    policy_fn,
    follower,
    global_cam: RealCamera | None = None,
    wrist_cam: RealCamera | None = None,
    hz: float = DEFAULT_HZ,
    max_steps: int = 600,
    verbose: bool = True,
):
    """Run the policy on the real robot."""
    dt = 1.0 / hz

    print(f"\n{'='*60}")
    print("DEPLOYMENT RUNNING")
    print(f"  Control frequency: {hz} Hz")
    print(f"  Max steps: {max_steps}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        for step in range(max_steps):
            t0 = time.perf_counter()

            # Read follower state
            state = follower.read_positions_rad()

            # Capture images
            wrist_img = wrist_cam.read() if wrist_cam else np.zeros((480, 640, 3), dtype=np.uint8)
            global_img = global_cam.read() if global_cam else np.zeros((480, 640, 3), dtype=np.uint8)

            # Query policy
            images = {"wrist": wrist_img, "global": global_img}
            action = policy_fn(images, state.astype(np.float32))

            # Send to follower
            follower.send_action_rad(action)

            # Print progress
            if verbose and step % 30 == 0:
                print(f"  step {step:4d} | action: {np.round(np.degrees(action), 1)}°")

            # Wait for next timestep
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nDeployment stopped by user.")
    finally:
        follower.disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["smolvla", "groot", "act"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--follower-port", default="/dev/ttyACM1", help="Follower arm serial port")
    p.add_argument("--wrist-cam", type=int, default=None, help="Wrist camera device ID")
    p.add_argument("--global-cam", type=int, default=0, help="Global camera device ID")
    p.add_argument("--hz", type=float, default=DEFAULT_HZ)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--dry-run", action="store_true", help="Fake follower, no arm hardware")
    return p.parse_args(argv)


def main():
    args = parse_args()

    # Load policy
    print(f"Loading {args.backend} from {args.checkpoint}...")
    policy_fn = load_policy(args.backend, args.checkpoint)

    # Connect hardware
    if args.dry_run:
        follower = FakeFollower()
    else:
        follower = FollowerArm(args.follower_port)

    global_cam = None
    wrist_cam = None
    if not args.dry_run:
        try:
            global_cam = RealCamera(args.global_cam)
        except Exception as e:
            print(f"WARNING: global camera not available ({e})")
        try:
            if args.wrist_cam is not None:
                wrist_cam = RealCamera(args.wrist_cam)
        except Exception as e:
            print(f"WARNING: wrist camera not available ({e})")

    # Run
    run_deploy(
        policy_fn, follower,
        global_cam=global_cam,
        wrist_cam=wrist_cam,
        hz=args.hz,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
