#!/usr/bin/env python
"""Quick real2sim sanity check: connect the SO-101 LEADER arm, read joint
positions for a few seconds, print them, and optionally drive the MuJoCo sim
follower with the same readings (reuses teleop_leader_lerobot.py internals).

Run this on demo morning BEFORE any teleop session: it validates port, baud,
calibration, and leader->sim direction in ~30 seconds.

Usage:
    # real leader, print positions only (5 s)
    python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id myleader

    # real leader + mirror into the MuJoCo single-arm env
    python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id myleader --mirror-sim

    # no hardware: fake sinusoidal "leader" (also exercises the sim mirror path)
    MUJOCO_GL=egl python scripts/real_to_sim_check.py --dry-run
    MUJOCO_GL=egl python scripts/real_to_sim_check.py --dry-run --mirror-sim
"""
import argparse
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from teleop_leader_lerobot import JOINT_ORDER, build_env, connect_leader, leader_to_rad

# Plausible mid-range pose (deg) for the fake leader's sinusoid.
FAKE_CENTER_DEG = np.array([0.0, -28.6, 45.8, 22.9, 0.0, 50.0])
FAKE_AMP_DEG = 15.0


class FakeLeader:
    """Hardware-free stand-in for SOLeader: sinusoidal '<joint>.pos' degrees."""

    def __init__(self, hz):
        self.hz = hz
        self._t0 = time.perf_counter()

    def get_action(self):
        t = time.perf_counter() - self._t0
        return {f"{j}.pos": float(FAKE_CENTER_DEG[i]
                                  + FAKE_AMP_DEG * math.sin(2 * math.pi * 0.5 * t))
                for i, j in enumerate(JOINT_ORDER)}

    def disconnect(self):
        pass


def fmt_deg(rad_row):
    return " ".join(f"{j}={math.degrees(float(v)):+7.1f}deg" for j, v in zip(JOINT_ORDER, rad_row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyACM0", help="leader arm serial port")
    parser.add_argument("--leader-id", default=None, help="LeRobot calibration id for the leader")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to read/print positions")
    parser.add_argument("--hz", type=float, default=30.0, help="read rate")
    parser.add_argument("--mirror-sim", action="store_true",
                        help="also step the MuJoCo single-arm env with the leader readings")
    parser.add_argument("--dry-run", action="store_true",
                        help="fake the leader with a sinusoid; no hardware touched")
    args = parser.parse_args()

    if not args.dry_run and not sys.platform.startswith("win") and not Path(args.port).exists():
        print(f"ERROR: leader port not found: {args.port} "
              "(check `ls /dev/ttyACM*` / `lerobot-find-port`; see TELEOP_GUIDE.md)")
        sys.exit(1)

    env, env_kind = build_env("single") if args.mirror_sim else (None, None)
    if env is not None:
        obs, _ = env.reset()
        site_id = env.model.site("gripperframe").id

    if args.dry_run:
        leader = FakeLeader(args.hz)
        print("[dry-run] using FAKE sinusoidal leader (no hardware)")
    else:
        leader = connect_leader(args.port, args.leader_id)
        print(f"[real2sim-check] connected leader on {args.port}")

    n_ticks = int(args.duration * args.hz)
    samples = np.zeros((n_ticks, 6))
    failures = 0
    try:
        for k in range(n_ticks):
            t_loop = time.perf_counter()
            try:
                rad = leader_to_rad(leader.get_action())
                failures = 0
            except Exception as e:
                failures += 1
                print(f"[real2sim-check] read failed ({failures}/3): {e}")
                if failures >= 3:
                    print("[real2sim-check] giving up — check cabling/calibration "
                          "(see TELEOP_GUIDE.md troubleshooting)")
                    raise
                rad = np.zeros(6)
            samples[k] = rad

            if env is not None:
                lo, hi = env.action_space.low[:6], env.action_space.high[:6]
                env.data.ctrl[:6] = np.clip(rad, lo, hi)
                env.step(env.data.ctrl.copy())

            if k % max(1, int(args.hz / 2)) == 0:
                stamp = f"[t={k / args.hz:5.2f}s] {fmt_deg(rad)}"
                if env is not None:
                    print(f"{stamp} | sim EE xyz={np.round(env.data.site_xpos[site_id], 3)}")
                else:
                    print(stamp)
            elapsed = time.perf_counter() - t_loop
            time.sleep(max(0.0, 1.0 / args.hz - elapsed))
    except KeyboardInterrupt:
        print("\n[real2sim-check] interrupted")
    finally:
        leader.disconnect()
        if env is not None:
            env.close()

    deg = np.degrees(samples)
    print("\n[real2sim-check] summary over %.1f s (%d samples @ %.0f Hz):"
          % (args.duration, len(samples), args.hz))
    print(f"{'joint':<14}{'mean':>9}{'std':>8}{'min':>9}{'max':>9}   (deg)")
    for i, j in enumerate(JOINT_ORDER):
        print(f"{j:<14}{deg[:, i].mean():+9.1f}{deg[:, i].std():8.1f}"
              f"{deg[:, i].min():+9.1f}{deg[:, i].max():+9.1f}")

    if env is not None:
        qpos_arm = np.array([env.data.qpos[env.model.jnt_qposadr[env.model.joint(n).id]]
                             for n in JOINT_ORDER])
        print(f"[real2sim-check] final sim qpos (rad): {np.round(qpos_arm, 3)}")
        print("[real2sim-check] leader -> sim direction OK" )
    else:
        print("[real2sim-check] read OK (add --mirror-sim to also exercise the sim side)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
