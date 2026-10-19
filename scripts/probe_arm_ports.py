#!/usr/bin/env python3
"""Read-only probe: stream SO-101 joint positions from one or more serial ports
so you can tell which physical arm is plugged into which port.

Strictly READ-ONLY. It opens the Feetech bus and calls sync_read on
Present_Position. It never writes calibration, never enables torque, and never
sends a target. Safe to run while the arms are powered and rigged up.

Move ONE arm at a time while this is running; the port whose numbers move is
that arm's port.

Usage:
    python scripts/probe_arm_ports.py --port /dev/ttyACM0 --port /dev/ttyACM1
    python scripts/probe_arm_ports.py --port /dev/ttyACM0 --port /dev/ttyACM1 \
        --out data/port_probe.jsonl

Prints a per-port table to the terminal and appends every sample to --out as
JSONL so the history survives a scrollback, for later analysis.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"

# Same motor table lerobot's SO101Follower/SOLeader use.
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def load_calibration(arm_id: str) -> dict[str, MotorCalibration]:
    """Load the repo's LeRobot-format calibration for an arm id, if present."""
    fpath = CALIB_DIR / f"{arm_id}.json"
    if not fpath.is_file():
        return {}
    with open(fpath) as f:
        return json.load(f)


def open_bus(port: str, arm_id: str | None) -> FeetechMotorsBus:
    calib = load_calibration(arm_id) if arm_id else {}
    bus = FeetechMotorsBus(port=port, motors=MOTORS, calibration=calib)
    bus.connect()
    return bus


def read_positions(bus: FeetechMotorsBus) -> dict[str, float]:
    """Read all motor positions once (normalized when calibration is loaded)."""
    raw = bus.sync_read("Present_Position", normalize=False)
    return {name: float(v) for name, v in raw.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", action="append", required=True, help="repeatable, e.g. /dev/ttyACM0")
    p.add_argument("--arm-id", action="append", default=[], help="optional LeRobot id per --port")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=0.0, help="0 = run until Ctrl-C")
    p.add_argument("--out", default="", help="optional JSONL output path")
    args = p.parse_args()

    ids = list(args.arm_id) + [""] * (len(args.port) - len(args.arm_id))

    buses = {}
    for port, arm_id in zip(args.port, ids, strict=False):
        try:
            buses[port] = open_bus(port, arm_id or None)
            print(f"[probe] {port:>16s}  OPEN  id={arm_id or '<none>'}", flush=True)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[probe] {port:>16s}  FAILED: {e}", file=sys.stderr, flush=True)

    if not buses:
        print("[probe] no ports opened, aborting", file=sys.stderr)
        return 1

    out_f = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "a")
        out_f.write(json.dumps({"event": "start", "ports": list(buses)}) + "\n")
        out_f.flush()

    print(
        "\n  Move ONE arm at a time. The port whose numbers change is that arm's port.\n"
        "  Ctrl-C to stop.\n",
        flush=True,
    )

    dt = 1.0 / args.hz
    t_start = time.perf_counter()
    n = 0
    try:
        while args.duration <= 0 or (time.perf_counter() - t_start) < args.duration:
            loop_t = time.perf_counter()
            sample = {"t": round(time.perf_counter() - t_start, 3)}
            row = []
            for port, bus in buses.items():
                try:
                    pos = read_positions(bus)
                except Exception as e:  # noqa: BLE001
                    sample[port] = {"error": str(e)}
                    row.append(f"{port.split('/')[-1]:>10s}: ERR")
                    continue
                sample[port] = pos
                row.append(f"{port.split('/')[-1]:>10s}: " + " ".join(f"{pos[k]:6.1f}" for k in MOTORS))

            if n % max(1, int(args.hz)) == 0:  # ~1 Hz terminal refresh
                print(f"  t={sample['t']:7.2f}s  " + "   |   ".join(row), flush=True)

            if out_f:
                out_f.write(json.dumps(sample) + "\n")
                if n % max(1, int(args.hz * 2)) == 0:
                    out_f.flush()

            n += 1
            wait = dt - (time.perf_counter() - loop_t)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\n[probe] stopped", flush=True)
    finally:
        if out_f:
            out_f.write(json.dumps({"event": "stop", "samples": n}) + "\n")
            out_f.close()
        for port, bus in buses.items():
            try:
                bus.disconnect()
                print(f"[probe] {port} closed", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[probe] {port} close warning: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
