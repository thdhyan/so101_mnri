#!/usr/bin/env python3
"""Live raw joint counts for one arm, to measure its TRUE travel by hand.

Why
---
`repair_calibration.py` can tell you whether the file matches what is stored in
the motors. It cannot tell you whether the stored range is *physically true*.
That distinction matters when a joint is pinned at the end of its calibrated
range on every read.

On this rig the leader reads:

    joint             leader  follower
    shoulder_lift        1%!      45%
    elbow_flex          99%!      51%

...while shoulder_pan, wrist_flex, wrist_roll and gripper all agree with the
follower to within 1-2%. So the leader's calibration is broadly right and the
problem is isolated to two joints sitting at their extremes. Either the joint
is mechanically stuck, or the calibration records more travel than the joint
has -- in which case no recalibration is needed, the range is just wrong, and
the fix is to correct range_min / range_max.

This tool is READ-ONLY. It shows live raw counts (the 0..4095 values the servo
actually reports) so you can push each joint to each end and note the real
numbers.

    python scripts/check_joint_range.py --arm leader          # live table
    python scripts/check_joint_range.py --arm leader --joint shoulder_lift

Then compare what you see against the calibrated range:

    python scripts/check_joint_range.py --arm leader --compare
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"
SERIALS = {"leader": "5AAF263743", "follower": "5AAF263600"}
JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def resolve_port(arm: str) -> str | None:
    import subprocess
    out = subprocess.run(
        ["bash", str(Path(__file__).resolve().parent / "resolve_arm_ports.sh"), "--emit"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return None
    d = dict(l.split("=", 1) for l in out.stdout.strip().splitlines() if "=" in l)
    return d.get("LEADER_PORT" if arm == "leader" else "FOLLOWER_PORT")


def open_bus(port: str, arm: str):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import MotorCalibration

    cal = {k: MotorCalibration(**v)
           for k, v in json.load(open(CALIB_DIR / f"mnri_arms_1_{arm}.json")).items()}
    motors = {
        n: Motor(i, "sts3215",
                 MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100)
        for i, n in enumerate(JOINTS, 1)
    }
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=cal)
    bus.connect(handshake=False)
    return bus


def show_raw(port: str, arm: str, joint: str | None, interval: float) -> int:
    cal = json.load(open(CALIB_DIR / f"mnri_arms_1_{arm}.json"))
    bus = open_bus(port, arm)
    names = [joint] if joint else JOINTS
    try:
        print(f"{arm} on {port}  -- live RAW counts (0..4095). Ctrl-C to stop.\n")
        print(f"{'joint':14s} {'raw':>6s} {'min':>6s} {'max':>6s} {'%span':>7s}")
        while True:
            raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
            for n in names:
                v = float(raw[n])
                lo, hi = cal[n]["range_min"], cal[n]["range_max"]
                bad = "" if 0 <= v <= 4095 else "  <-- CORRUPT"
                pct = (v - lo) / (hi - lo) * 100 if hi > lo else 0
                edge = "  <-- AT END" if pct < 3 or pct > 97 else ""
                print(f"{n:14s} {v:6.0f} {lo:6d} {hi:6d} {pct:6.0f}%{edge}{bad}")
            print("  " + "-" * 46 + f"   (refreshing, {interval}s)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        bus.disconnect()


def compare(port: str, arm: str) -> int:
    """Print the calibrated range and where the arm currently sits, side by side."""
    cal = json.load(open(CALIB_DIR / f"mnri_arms_1_{arm}.json"))
    bus = open_bus(port, arm)
    try:
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    finally:
        bus.disconnect()
    print(f"{arm} on {port}\n")
    print(f"{'joint':14s} {'raw':>6s} {'cal_min':>9s} {'cal_max':>9s} {'span':>6s} {'here':>7s}")
    for n in JOINTS:
        lo, hi = cal[n]["range_min"], cal[n]["range_max"]
        v = float(raw[n])
        pct = (v - lo) / (hi - lo) * 100 if hi > lo else 0
        edge = "  <-- AT END of calibrated range" if pct < 3 or pct > 97 else ""
        print(f"{n:14s} {v:6.0f} {lo:9d} {hi:9d} {hi - lo:6d} {pct:6.0f}%{edge}")
    print("\nIf a joint sits at 0% or 100% of its range and you CANNOT move it")
    print("away, the calibrated range is wider than the real travel. Note what it")
    print("reads at each physical end, then correct range_min / range_max by hand.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["leader", "follower"], required=True)
    ap.add_argument("--joint", choices=JOINTS, help="only show this joint")
    ap.add_argument("--compare", action="store_true",
                    help="one-shot: calibrated range vs current position")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    port = resolve_port(args.arm)
    if not port:
        print(f"ERROR: could not resolve the {args.arm} port by USB serial.", file=sys.stderr)
        return 1
    if args.compare:
        return compare(port, args.arm)
    return show_raw(port, args.arm, args.joint, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
