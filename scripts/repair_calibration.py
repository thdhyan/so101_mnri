#!/usr/bin/env python3
"""Repair motor-side calibration, resolving arms by USB serial.

Incident this exists for (2026-09-25): moving the arms to a different USB port
reversed their `/dev/ttyACM*` indices. A path-based config therefore wrote the
FOLLOWER's calibration into the LEADER's servos and vice versa. LeRobot only
warns with

    Mismatch between calibration values in the motor and the calibration file
    or no calibration file found
    Press ENTER to use provided calibration file ... or type 'c' ...

and pressing ENTER writes the *file's* calibration to whatever servos that port
happens to address. With the ports swapped that silently swaps the calibration.

This tool does the same write, but it resolves each role from its USB serial
first, so it cannot write to the wrong arm, and it verifies afterwards.

    python scripts/repair_calibration.py            # repair both, verify
    python scripts/repair_calibration.py --dry-run  # report only
    python scripts/repair_calibration.py --arm leader

NOTE: the write happens inside `bus.torque_disabled()`, whose context manager
re-enables torque on exit. That leaves the follower energised and holding its
current position. Power-cycle the arm, or run with --release-torque, if you want
it limp.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"
sys.path.insert(0, str(Path(__file__).resolve().parent))

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
FIELDS = ("homing_offset", "range_min", "range_max", "drive_mode")

# Probe-confirmed on 2026-09-25: the arm whose joints moved when the operator
# moved the arm in their hand is the LEADER.
SERIALS = {"leader": "5AAF263743", "follower": "5AAF263600"}


def port_for_serial(serial: str) -> str | None:
    for dev in sorted(Path("/dev").glob("ttyACM*")):
        proc = subprocess.run(["udevadm", "info", "--query=property", "--name", str(dev)],
                              capture_output=True, text=True, check=False)
        got = next((ln.split("=", 1)[1] for ln in proc.stdout.splitlines()
                    if ln.startswith("ID_SERIAL_SHORT=")), None)
        if got == serial:
            return str(dev)
    return None


def serial_of(port: str) -> str:
    proc = subprocess.run(["udevadm", "info", "--query=property", "--name", port],
                          capture_output=True, text=True, check=False)
    return next((ln.split("=", 1)[1] for ln in proc.stdout.splitlines()
                 if ln.startswith("ID_SERIAL_SHORT=")), "?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["leader", "follower", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--release-torque", action="store_true",
                    help="leave torque disabled after the write")
    args = ap.parse_args()

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import MotorCalibration

    roles = ["leader", "follower"] if args.arm == "both" else [args.arm]
    targets = []
    for role in roles:
        port = port_for_serial(SERIALS[role])
        if port is None:
            print(f"ERROR: no /dev/ttyACM* with serial {SERIALS[role]} ({role}).", file=sys.stderr)
            for dev in sorted(Path("/dev").glob("ttyACM*")):
                print(f"       present: {dev}  serial={serial_of(str(dev))}", file=sys.stderr)
            return 1
        targets.append((role, port))

    # Guard against a config that maps both roles onto one port.
    if len({p for _, p in targets}) != len(targets):
        print("ERROR: both roles resolved to the same port; refusing.", file=sys.stderr)
        return 1

    motors = {
        n: Motor(i, "sts3215",
                 MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100)
        for i, n in enumerate(JOINTS, 1)
    }

    print(f"{'role':9s} {'port':16s} {'serial':12s} {'before':>8s}  action")
    all_ok = True
    for role, port in targets:
        fpath = CALIB_DIR / f"mnri_arms_1_{role}.json"
        if not fpath.is_file():
            print(f"ERROR: missing calibration file {fpath}", file=sys.stderr)
            return 1
        want = {k: MotorCalibration(**v) for k, v in json.load(open(fpath)).items()}
        ser = serial_of(port)

        bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
        bus.connect(handshake=False)
        try:
            live = bus.read_calibration()
            matches = all(
                all(int(getattr(live[n], k)) == int(getattr(want[n], k)) for k in FIELDS)
                for n in JOINTS
            )
            if matches:
                print(f"{role:9s} {port:16s} {ser:12s} {'correct':>8s}  no write needed")
                continue

            diffs = [f"{n}" for n in JOINTS
                     if any(int(getattr(live[n], k)) != int(getattr(want[n], k)) for k in FIELDS)]
            before = f"{len(diffs)} bad"
            if args.dry_run:
                print(f"{role:9s} {port:16s} {ser:12s} {before:>8s}  would write "
                      f"mnri_arms_1_{role}.json  (differs: {', '.join(diffs)})")
                continue

            with bus.torque_disabled():
                bus.write_calibration(want)
            check = bus.read_calibration()
            ok = all(
                all(int(getattr(check[n], k)) == int(getattr(want[n], k)) for k in FIELDS)
                for n in JOINTS
            )
            all_ok &= ok
            print(f"{role:9s} {port:16s} {ser:12s} {before:>8s}  "
                  f"wrote mnri_arms_1_{role}.json  verified={ok}"
                  + ("" if ok else "   <-- MISMATCH"))
        finally:
            try:
                if args.release_torque:
                    bus.disable_torque()
                    print(f"{'':9s} {'':16s} {'':12s} {'':>8s}  torque released")
            except Exception as e:  # noqa: BLE001
                print(f"  (could not release torque on {port}: {e})", file=sys.stderr)
            bus.disconnect()

    print()
    if args.dry_run:
        print("dry run - nothing written")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
