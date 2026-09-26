#!/usr/bin/env python3
"""Health check for a Feetech (SO-101) servo bus: which motors answer, and at
what rail voltage.

Symptom this was written for: `ConnectionError: Failed to write
'Torque_Enable' on id_=N ... There is no status packet!` (or `Incorrect status
packet!`) during `SOFollower.connect()`.

A daisy-chained bus losing motors is almost always a power problem, not a
software one. This reads each motor's Present_Voltage / Present_Current /
Present_Position so you can see the rail sag under load.

READ-ONLY: it never enables torque and never writes a target or a calibration.
It connects with handshake=False, because the bus handshake itself refuses to
connect when motors are missing -- which is exactly the situation you want to
inspect.

    python scripts/check_bus_health.py --port /dev/ttyACM0 --arm leader
    python scripts/check_bus_health.py --port /dev/ttyACM1 --arm follower --rounds 5
"""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--arm", choices=["leader", "follower"], default="follower")
    ap.add_argument("--rounds", type=int, default=3, help="repeat the scan this many times")
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between rounds")
    ap.add_argument("--warn-voltage", type=float, default=None,
                    help="flag motors below this. Present_Voltage is in 0.1 V, so the "
                         "raw 49/119 mean 4.9 V / 11.9 V. Defaults to 4.5 for the "
                         "leader (5 V rail) and 10.5 for the follower (12 V rail).")
    args = ap.parse_args()

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import MotorCalibration

    # The two arms run different rails: the leader only reads, so it lives on 5 V;
    # the follower has to hold torque against gravity, so it lives on 12 V.
    # Present_Voltage is reported in 0.1 V.
    warn_v = args.warn_voltage
    if warn_v is None:
        warn_v = 4.5 if args.arm == "leader" else 10.5

    fpath = CALIB_DIR / f"mnri_arms_1_{args.arm}.json"
    if fpath.is_file():
        calib = {k: MotorCalibration(**v) for k, v in json.load(open(fpath)).items()}
    else:
        print(f"[warn] no calibration at {fpath}, reading raw counts only")
        calib = {}

    motors = {
        n: Motor(i, "sts3215",
                 MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100)
        for i, n in enumerate(JOINTS, 1)
    }

    print(f"port {args.port}   arm={args.arm}   rounds={args.rounds}")
    print("READ-ONLY: no torque, no target, no calibration writes\n")

    counts = []
    for rnd in range(args.rounds):
        bus = FeetechMotorsBus(port=args.port, motors=motors, calibration=calib)
        try:
            bus.connect(handshake=False)
        except Exception as e:  # noqa: BLE001
            print(f"round {rnd + 1}: could not open the port at all: {e}")
            counts.append(0)
            time.sleep(args.gap)
            continue

        print(f"--- round {rnd + 1} ---")
        print(f"{'id':>3} {'name':14s} {'ok':>4s} {'V':>7s} {'mA':>7s} {'pos':>8s}")
        found = 0
        for name, motor in motors.items():
            v = c = p = None
            ok = False
            try:
                v = bus.read("Present_Voltage", name, normalize=False, num_retry=1)
                c = bus.read("Present_Current", name, normalize=False, num_retry=1)
                p = bus.read("Present_Position", name, normalize=False, num_retry=1)
                ok = True
                found += 1
            except Exception:  # noqa: BLE001
                ok = False
            flag = ""
            if ok and isinstance(v, (int, float)) and v / 10.0 < warn_v:
                flag = f"  <-- LOW (rail ~{v / 10.0:.1f} V, want >= {warn_v} V)"
            print(f"{motor.id:>3} {name:14s} {'yes' if ok else 'NO':>4s} "
                  f"{v if ok else '-':>7} {c if ok else '-':>7} {p if ok else '-':>8}{flag}")
        try:
            bus.disconnect()
        except Exception:  # noqa: BLE001
            pass
        counts.append(found)
        print(f"  -> {found}/6 motors answered\n")
        if rnd < args.rounds - 1:
            time.sleep(args.gap)

    print("=" * 66)
    if all(c == 6 for c in counts):
        print(f"RESULT: all 6 motors answered every round {counts} - bus looks healthy.")
        print("If connect() still fails on Torque_Enable/Lock, the problem is the")
        print("current SURGE as 6 servos energise, not a dead motor.")
    else:
        print(f"RESULT: motors answering per round = {counts}")
        print("The bus is dropping motors - that is a POWER problem, not software.")
        print("Check, in this order:")
        print("  1. Power supply current rating. 6x STS3215 stalling needs real")
        print("     current; a 5V/1A brick browns out exactly like this.")
        print("  2. USB topology: an UNPOWERED hub steals bus current. Power the")
        print("     arm directly, or from a hub with its own supply. A USB port")
        print("     that shares a lane with the cameras is the usual culprit.")
        print("  3. Reseat the power connector on the controller board and any")
        print("     daisy-chain extension between servos, at BOTH ends.")
        print("  4. Voltage sag: this script prints Present_Voltage per motor.")
        print("     Anything below ~10.5 V is too low to hold 6 servos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
