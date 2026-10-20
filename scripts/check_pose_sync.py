#!/usr/bin/env python3
"""Check how far the leader pose is from the follower pose, before recording.

Why this matters
----------------
`record_loop` calls `robot.send_action()` on its very first tick, with whatever
the leader currently reads. If the two arms are not near each other, that first
command asks six servos to travel a long way against gravity. They stall, draw
a large current, the supply rail sags, and the Feetech bus starts dropping
packets. That surfaces as:

    WARNING  Record loop is running slower (5.3 Hz) than the target FPS (15 Hz)
    ConnectionError: Failed to sync read 'Present_Position' on ids=[1..6]
                     after 3 tries. [TxRxResult] There is no status packet!
    ConnectionError: Failed to write 'Torque_Enable' on id_=1 with '0'
                     after 6 tries. [TxRxResult] There is no status packet!

which looks like a flaky cable but is actually the arm being told to lunge.

READ-ONLY. It reads positions on both arms and prints how far each joint has to
travel, so you can physically bring the leader to the follower's pose before
starting a recording.

    python scripts/check_pose_sync.py
    python scripts/check_pose_sync.py --watch          # live, until Ctrl-C
    python scripts/check_pose_sync.py --max-delta 5   # stricter gate
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
# The two arms that carry the most gravity load and matter most to match.
WEIGHTED = {"shoulder_lift": 1.5, "elbow_flex": 1.5, "shoulder_pan": 1.0,
            "wrist_flex": 0.7, "wrist_roll": 0.5, "gripper": 0.3}


def resolve_ports() -> tuple[str, str]:
    out = subprocess.run(
        ["bash", str(Path(__file__).resolve().parent / "resolve_arm_ports.sh"), "--emit"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        sys.exit(1)
    d = dict(l.split("=", 1) for l in out.stdout.strip().splitlines() if "=" in l)
    return d["LEADER_PORT"], d["FOLLOWER_PORT"]


def read_arm(port: str, arm: str) -> dict[str, float] | None:
    """Positions in DEGREES, via LeRobot's own normalisation. Never writes.

    Reads RAW counts first and rejects the sample if any is outside
    0..4095. The STS3215 position register is a 12-bit value, so a raw count
    outside that range is provably a corrupt read. This matters: the Feetech
    bus on this rig occasionally returns garbage (an observed follower
    shoulder_lift of -763), and normalising garbage produces nonsense joint
    angles that look exactly like a 262 deg pose mismatch.
    """
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import MotorCalibration
    except ImportError as e:  # noqa: BLE001
        print(f"ERROR: cannot import lerobot ({e})", file=sys.stderr)
        sys.exit(1)

    fpath = CALIB_DIR / f"mnri_arms_1_{arm}.json"
    if not fpath.is_file():
        print(f"ERROR: no calibration for '{arm}' at {fpath}", file=sys.stderr)
        sys.exit(1)
    cal = {k: MotorCalibration(**v) for k, v in json.load(open(fpath)).items()}
    motors = {
        n: Motor(i, "sts3215",
                 MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100)
        for i, n in enumerate(JOINTS, 1)
    }

    bus = None
    try:
        bus = FeetechMotorsBus(port=port, motors=motors, calibration=cal)
        # handshake=False: we only want to read, and the handshake refuses to
        # connect when any motor is missing.
        bus.connect(handshake=False)
        for attempt in range(4):
            raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
            bad = [n for n, v in raw.items() if not (0 <= float(v) <= 4095)]
            if not bad:
                # Re-read in normalised units only once the raw sample is sane.
                return {k: float(v) for k, v in bus.sync_read("Present_Position", num_retry=2).items()}
            print(f"  [{arm}] corrupt read on {', '.join(bad)} "
                  f"(raw {[round(float(raw[b]), 1) for b in bad]}), retrying "
                  f"{attempt + 1}/4", file=sys.stderr)
        print(f"ERROR: {arm} on {port} returned corrupt counts 4x in a row. "
              f"That is a bus/hardware fault, not a pose problem.", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"ERROR reading {arm} on {port}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    finally:
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:  # noqa: BLE001
                pass


def travel_pct(arm: str, joint: str, deg: float) -> float | None:
    """How far through its calibrated range a joint is, 0-100.

    Inverse of LeRobot's DEGREES normalisation
    (MotorsBus._normalize), which is NOT span- or homing-based:

        mid      = (range_min + range_max) / 2
        max_res  = model_resolution_table[model] - 1      # sts3215 -> 4095
        deg      = (raw - mid) * 360 / max_res

    so  raw = deg * max_res / 360 + mid,  and frac = (raw - range_min)/span.
    """
    fpath = CALIB_DIR / f"mnri_arms_1_{arm}.json"
    if not fpath.is_file():
        return None
    c = json.load(open(fpath))[joint]
    span = c["range_max"] - c["range_min"]
    if span <= 0:
        return None
    mid = (c["range_min"] + c["range_max"]) / 2.0
    raw = deg * 4095.0 / 360.0 + mid
    return (raw - c["range_min"]) / span * 100.0


def report(leader: dict[str, float], follower: dict[str, float], max_delta: float) -> tuple[float, list[str]]:
    deltas = {j: leader[j] - follower[j] for j in JOINTS}
    worst = max(abs(d) for d in deltas.values())
    print(f"{'joint':14s} {'leader':>9s} {'follower':>10s} {'delta':>9s}   move leader")
    for j in JOINTS:
        d = deltas[j]
        bar = "#" * min(30, int(abs(d) / 3))
        flag = "  <-- too far" if abs(d) > max_delta else ""
        print(f"{j:14s} {leader[j]:9.1f} {follower[j]:10.1f} {d:+9.1f}   {bar}{flag}")
    print(f"\nworst joint delta: {worst:.1f} deg   (gate: {max_delta:.1f} deg)")

    at_limit = []
    print("\ntravel through each joint's calibrated range (aim for ~50%):")
    print(f"  {'joint':14s} {'leader':>9s} {'follower':>9s}")
    for j in JOINTS:
        pl = travel_pct("leader", j, leader[j])
        pf = travel_pct("follower", j, follower[j])
        if pl is None or pf is None:
            continue
        gl = f"{pl:6.0f}%" + ("!" if (pl < 10 or pl > 90) else " ")
        gf = f"{pf:6.0f}%" + ("!" if (pf < 10 or pf > 90) else " ")
        print(f"  {j:14s} {gl:>9s} {gf:>9s}")
        if pl < 10 or pl > 90:
            at_limit.append(f"leader.{j} at {pl:.0f}%")
        if pf < 10 or pf > 90:
            at_limit.append(f"follower.{j} at {pf:.0f}%")
    if at_limit:
        print("\n  '!' = pinned at the end of travel, where moves stall.")
        print("  Steer the arm in your HAND toward ~50%. Move the arm on the table")
        print("  by hand (torque off) if IT is the one pinned.")
        print("  Note the leader's range is deliberately narrower than the")
        print("  follower's (~59% on shoulder_lift, ~55% on elbow_flex), so keep")
        print("  the follower near mid-travel to keep the two alignable.")
    return worst, at_limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-delta", type=float, default=10.0,
                    help="per-joint gate in degrees (default 10)")
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl-C")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    lp, fp = resolve_ports()
    print(f"leader   {lp}   follower {fp}   (resolved by USB serial)\n")

    while True:
        leader = read_arm(lp, "leader")
        follower = read_arm(fp, "follower")
        if leader is None or follower is None:
            return 1

        print("=" * 64)
        worst, _ = report(leader, follower, args.max_delta)
        if worst > args.max_delta:
            print("\nUNSAFE TO START: bring the leader to the follower's pose.")
            print("  The first send_action() would command a large move, the six")
            print("  servos stall against gravity, the rail sags and the bus drops")
            print("  packets. Move the arm in your HAND, not the one on the table.")
            print(f"\n  Re-run until every delta is under {args.max_delta:.0f} deg.")
        else:
            print("\nOK: arms are aligned enough to start recording.")
        if not args.watch:
            return 0 if worst <= args.max_delta else 2
        print(f"\n(refreshing every {args.interval}s -- Ctrl-C to stop)")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
