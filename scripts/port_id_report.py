#!/usr/bin/env python3
"""Decide which /dev/ttyACM* is the leader and which is the follower.

Reads the JSONL written by scripts/probe_arm_ports.py and reports, per port,
how far each joint travelled over the recording window. Move exactly one arm
while the probe runs; the port that moves is that arm's port.

    python scripts/port_id_report.py data/port_id/probe.jsonl

    # and to write the answer into scripts/so101_env.sh
    python scripts/port_id_report.py data/port_id/probe.jsonl --write-env
"""

import argparse
import json
import re
import sys
from pathlib import Path

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
ENV_FILE = Path(__file__).resolve().parent / "so101_env.sh"
# Anything at or above this many counts of travel is treated as "this arm moved".
MOVE_THRESHOLD = 15


def load(path: Path) -> tuple[dict[str, list[list[float]]], float]:
    per_port: dict[str, list[list[float]]] = {}
    t_first = t_last = 0.0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or '"event"' in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for port, pos in row.items():
            if port == "t" or not isinstance(pos, dict) or "error" in pos:
                continue
            per_port.setdefault(port, []).append([float(pos[j]) for j in JOINTS])
        t_first = t_first or float(row.get("t", 0.0))
        t_last = float(row.get("t", t_last))
    return per_port, t_last - t_first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("probe", nargs="?", default="data/port_id/probe.jsonl")
    ap.add_argument("--threshold", type=int, default=MOVE_THRESHOLD)
    ap.add_argument("--moved-is", choices=["leader", "follower"],
                    help="declare what the arm that moved actually is, and write the ports")
    ap.add_argument("--write-env", action="store_true",
                    help=f"update LEADER_PORT/FOLLOWER_PORT in {ENV_FILE} (needs --moved-is)")
    args = ap.parse_args()

    path = Path(args.probe)
    if not path.is_file():
        print(f"ERROR: no probe file at {path}", file=sys.stderr)
        print("       Start the probe first (see scripts/so101_env.sh header).", file=sys.stderr)
        return 1

    per_port, span = load(path)
    if not per_port:
        print(f"ERROR: {path} has no usable samples", file=sys.stderr)
        return 1

    print(f"probe file : {path}")
    print(f"window     : {span:.1f}s, {min(len(v) for v in per_port.values())} samples/port")
    print()

    # Compare every port over the SAME number of samples, so a port that
    # produced fewer rows is not silently measured over a different window.
    n = min(len(v) for v in per_port.values())

    travel: dict[str, list[float]] = {}
    for port, rows in sorted(per_port.items()):
        window = rows[:n]
        spread = [
            max(r[i] for r in window) - min(r[i] for r in window) for i in range(len(JOINTS))
        ]
        travel[port] = spread
        print(f"{port}")
        for j, s in zip(JOINTS, spread):
            bar = "#" * min(40, int(s / 10))
            mark = "  <-- moved" if s >= args.threshold else ""
            print(f"   {j:14s} {s:7.0f} counts {bar}{mark}")
        print(f"   {'TOTAL':14s} {sum(spread):7.0f}{mark if max(spread) >= args.threshold else ''}")
        print()

    moved = [p for p, s in travel.items() if max(s) >= args.threshold]
    still = [p for p, s in travel.items() if max(s) < args.threshold]

    if len(moved) != 1:
        print("=" * 62)
        if not moved:
            print("RESULT: no arm moved during the window.")
            print("        Move ONE arm (the leader, in your hand) while the probe runs,")
            print("        then re-run this report.")
        else:
            print(f"RESULT: {len(moved)} ports moved: {', '.join(moved)} - ambiguous.")
            print("        Move exactly ONE arm at a time, or just tell me which arm")
            print("        is which and I will set the ports directly.")
        return 1 if not moved else 2

    moved_port = moved[0]
    print("=" * 62)
    print(f"RESULT: {moved_port} is the arm you moved.")
    if still:
        print(f"        {still[0]} did not move.")
    other_port = still[0] if still else None

    if not (args.write_env and args.moved_is):
        print()
        print("Tell me which arm you moved and I will wire the ports up:")
        print("  - the arm in your HAND  = LEADER   (drives the follower)")
        print("  - the arm on the TABLE  = FOLLOWER (the one that gets recorded)")
        print()
        print("To record it yourself, pass --moved-is with the role:")
        print(f"  python scripts/port_id_report.py {path} --write-env --moved-is leader")
        return 0

    if other_port is None:
        print(f"ERROR: need exactly one moving port to write both roles, got {len(moved)}",
              file=sys.stderr)
        return 2

    ports = {args.moved_is: moved_port, "follower" if args.moved_is == "leader" else "leader": other_port}
    print()
    print(f"Writing {ENV_FILE}:")
    print(f"   LEADER_PORT={ports['leader']}")
    print(f"   FOLLOWER_PORT={ports['follower']}")
    return write_env(ENV_FILE, ports["leader"], ports["follower"])


def write_env(env_file: Path, leader: str, follower: str) -> int:
    """Rewrite the LEADER_PORT / FOLLOWER_PORT lines, leaving the rest alone."""
    if not env_file.is_file():
        print(f"ERROR: {env_file} not found", file=sys.stderr)
        return 1
    text = env_file.read_text()
    for key, val in (("LEADER_PORT", leader), ("FOLLOWER_PORT", follower)):
        pattern = rf"(?m)^{key}=.*$"
        if not re.search(pattern, text):
            print(f"ERROR: no '{key}=' line in {env_file}", file=sys.stderr)
            return 1
        text = re.sub(pattern, f"{key}={val}", text)
    env_file.write_text(text)
    print(f"   updated {env_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
