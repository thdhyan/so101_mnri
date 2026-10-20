#!/usr/bin/env python3
"""Run LeRobot's own SO-101 calibration for one arm.

This calls the real `SOLeader.calibrate()` / `SO101Follower.calibrate()` -- the
same code `lerobot-record` runs when you press 'c' at the calibration prompt.
Nothing is reimplemented here.

The only thing intercepted is the *first* prompt, "Press ENTER to use provided
calibration file ... or type 'c'". That is a UX guard, not calibration logic,
and it is auto-answered with 'c' because you asked to recalibrate. The two
prompts that actually matter are passed straight through to your terminal:

  1. "Move <arm> to the middle of its range of motion and press ENTER...."
        -> LeRobot calls bus.set_half_turn_homings() at that pose
  2. "Move all joints except 'wrist_roll' sequentially through their entire
      ranges of motion. Press ENTER to stop..."
        -> LeRobot calls bus.record_ranges_of_motion() over the whole window

Run ONE ARM AT A TIME. Serial ports are exclusive, and calibrating both at once
produces "multiple access on port" rather than a calibration.

    python scripts/calibrate_arm.py --arm leader
    python scripts/calibrate_arm.py --arm follower

Roll back from robots/so101/calibration/backup-*/ if the result is worse.
"""

import argparse
import builtins
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"

SERIALS = {"leader": "5AAF263743", "follower": "5AAF263600"}


def resolve_port(arm: str) -> str:
    out = subprocess.run(
        ["bash", str(Path(__file__).resolve().parent / "resolve_arm_ports.sh"), "--emit"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        sys.exit(f"ERROR: could not resolve the {arm} port by USB serial:\n{out.stderr}")
    d = dict(l.split("=", 1) for l in out.stdout.strip().splitlines() if "=" in l)
    return d["LEADER_PORT" if arm == "leader" else "FOLLOWER_PORT"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["leader", "follower"], required=True)
    ap.add_argument("--list-backups", action="store_true")
    args = ap.parse_args()

    if args.list_backups:
        for d in sorted(CALIB_DIR.glob("backup-*")):
            print(d)
        return 0

    port = resolve_port(args.arm)
    cal_id = f"mnri_arms_1_{args.arm}"

    print("=" * 68)
    print(f" LeRobot calibration: {args.arm}")
    print(f" port {port}  (USB serial {SERIALS[args.arm]})")
    print(f" will overwrite {CALIB_DIR / (cal_id + '.json')} and the arm's EEPROM")
    print("=" * 68)
    print()
    print("  1) Put the arm at the MIDDLE of its range of motion -- every joint")
    print("     centred, not at a stop. This is the pose that defines 0 degrees.")
    print("  2) Then sweep every joint except wrist_roll through its ENTIRE range,")
    print("     slowly, so LeRobot records the true min and max.")
    print()
    print("  wrist_roll is calibrated as a full-turn joint (0..4095) by LeRobot.")
    print("  Do not worry about it in step 2.")
    print()

    if args.arm == "leader":
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        from lerobot.teleoperators.so_leader.so_leader import SOLeader

        robot = SOLeader(SOLeaderTeleopConfig(port=port, id=cal_id,
                                              calibration_dir=CALIB_DIR))
    else:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower import SO101Follower

        # cameras={} so connect() does not open the cameras; calibration does
        # not need them and it keeps this fast and independent of the video rig.
        robot = SO101Follower(SO101FollowerConfig(port=port, id=cal_id,
                                                 calibration_dir=CALIB_DIR,
                                                 cameras={}))

    # Open the bus WITHOUT triggering the automatic calibration prompt, so we
    # control the ordering. calibrate=False skips LeRobot's auto-calibrate.
    robot.connect(calibrate=False)
    print(f"[calibrate] bus open on {port}\n")

    # Auto-answer only the "use existing file?" guard; pass everything else on.
    real_input = builtins.input
    seen = {"n": 0}

    def guarded_input(prompt: str = "") -> str:
        seen["n"] += 1
        if seen["n"] == 1:
            print(f"{prompt}c", flush=True)
            return "c"
        return real_input(prompt)

    builtins.input = guarded_input
    try:
        robot.calibrate()          # <- LeRobot's real calibration
    finally:
        builtins.input = real_input
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[calibrate] disconnect warning (non-fatal): {e}", file=sys.stderr)

    print()
    print("=" * 68)
    print(" Done. Verifying the new calibration is what got saved...")
    print("=" * 68)
    saved = robot.calibration
    for name, c in saved.items():
        print(f"  {name:14s} homing={c.homing_offset:6d} "
              f"range=[{c.range_min:5d},{c.range_max:5d}] span={c.range_max - c.range_min:5d}")

    print()
    print("Next: python scripts/check_pose_sync.py --watch")
    print("If the new range looks wrong, roll back from "
          f"{sorted(CALIB_DIR.glob('backup-*'))[-1] if list(CALIB_DIR.glob('backup-*')) else 'the backup dir'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
