#!/usr/bin/env python3
"""Profile the real record_loop and say where the milliseconds go.

LeRobot's "running slower than target FPS" warning names three possible causes
(camera FPS, policy inference, CPU starvation) but is a fixed string -- there is
no policy in this loop, so it is not diagnostic. This measures the actual
components instead.

Wraps the real methods with timing accumulators, runs for a fixed number of
iterations, and prints a breakdown. Everything is the genuine code path; only
timing is added.

    python scripts/profile_record_loop.py --seconds 10
    python scripts/profile_record_loop.py --seconds 10 --no-display
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# name -> accumulated (count, total_seconds, max_seconds)
STATS: dict[str, list[float]] = {}


def instrument(obj, attr: str, label: str) -> None:
    """Wrap obj.attr so every call is timed into STATS[label]."""
    original = getattr(obj, attr)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return original(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            s = STATS.setdefault(label, [0.0, 0.0, 0.0])
            s[0] += 1
            s[1] += dt
            s[2] = max(s[2], dt)

    setattr(obj, attr, timed)


def instrument_func(module, name: str, label: str) -> None:
    original = getattr(module, name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return original(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            s = STATS.setdefault(label, [0.0, 0.0, 0.0])
            s[0] += 1
            s[1] += dt
            s[2] = max(s[2], dt)

    setattr(module, name, timed)


def report(loops: int, wall: float) -> None:
    print("\n" + "=" * 72)
    print(f" record_loop breakdown over {loops} iterations ({wall:.1f}s wall, "
          f"{loops / wall:.1f} Hz)")
    print("=" * 72)
    print(f"{'component':32s} {'n':>5s} {'mean ms':>9s} {'max ms':>8s} {'share':>7s}")
    total = 0.0
    for label, (n, tot, mx) in sorted(STATS.items(), key=lambda kv: -kv[1][1]):
        if not n:
            continue
        mean_ms = tot / n * 1000
        total += tot
        share = tot / wall * 100 if wall else 0
        print(f"{label:32s} {int(n):5d} {mean_ms:9.2f} {mx * 1000:8.2f} {share:6.1f}%")
    print("-" * 72)
    print(f"{'accounted total':32s} {'':5s} {total / loops * 1000:9.2f} "
          f"{'':8s} {total / wall * 100 if wall else 0:6.1f}%")
    unaccounted = wall - total
    print(f"{'unaccounted (sleep/loop overhead)':32s} {'':5s} "
          f"{unaccounted / loops * 1000:9.2f} "
          f"{'':8s} {unaccounted / wall * 100 if wall else 0:6.1f}%")

    print()
    per_iter = {k: v[1] / loops * 1000 for k, v in STATS.items() if v[0]}
    for fps in (30, 25, 20, 15):
        budget = 1000.0 / fps
        worst = sorted(per_iter.items(), key=lambda kv: -kv[1])[:3]
        print(f"  to hit {fps:2d} Hz ({budget:5.1f} ms/iter) you would need to remove: "
              + ", ".join(f"{k.split('.')[-1]} {v:.1f}ms" for k, v in worst))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--no-display", action="store_true",
                    help="skip the rerun debug view to isolate its cost")
    ap.add_argument("--skip-pose-gate", action="store_true",
                    help="record even if the arms are misaligned (NOT advised)")
    args = ap.parse_args()

    # This connects the follower, which enables torque and drives the arm. A
    # large leader/follower mismatch would command a lunge on the first tick,
    # stall the servos and drop the bus -- the same failure as a real run.
    if not args.skip_pose_gate:
        import subprocess
        gate = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "check_pose_sync.py"),
             "--max-delta", "10"],
            capture_output=True, text=True, check=False,
        )
        if gate.returncode != 0:
            print(gate.stdout[-1500:])
            print("\nERROR: arms are not aligned, so this profile would command a lunge.")
            print("       Align them first, or pass --skip-pose-gate if you know it is safe.")
            return 1
        print(gate.stdout[-300:])

    sys.argv = [
        "lerobot-record",
        "--robot.type=so101_follower",
        "--teleop.type=so101_leader",
        "--display_data=" + ("false" if args.no_display else "true"),
        "--display_mode=rerun",
        "--display_ip=127.0.0.1",
        "--display_port=9876",
        "--play_sounds=false",
        # placeholders, replaced from so101_env.sh below
    ]
    env = Path(__file__).resolve().parent / "so101_env.sh"
    vals = {}
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            vals[k.strip()] = v.split("#", 1)[0].strip().strip("\"'")
    import subprocess
    out = subprocess.run(
        ["bash", str(Path(__file__).resolve().parent / "resolve_arm_ports.sh"), "--emit"],
        capture_output=True, text=True, check=False).stdout
    d = dict(l.split("=", 1) for l in out.strip().splitlines() if "=" in l)

    cams = "{}"
    sys.argv += [
        f"--robot.port={d['FOLLOWER_PORT']}",
        f"--robot.id=mnri_arms_1_follower",
        f"--robot.calibration_dir={Path(__file__).resolve().parents[1] / 'robots/so101/calibration'}",
        f"--robot.cameras={cams}",
        f"--teleop.port={d['LEADER_PORT']}",
        f"--teleop.id=mnri_arms_1_leader",
        f"--teleop.calibration_dir={Path(__file__).resolve().parents[1] / 'robots/so101/calibration'}",
        "--dataset.repo_id=thakk100/profile_scratch",
        "--dataset.single_task=profile",
        f"--dataset.root=/tmp/opencode/profile_ds",
        f"--dataset.fps={vals.get('FPS', 15)}",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.episode_time_s=999",
        "--dataset.num_episodes=1",
    ]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.so_leader import SOLeader
    from lerobot.teleoperators.so_leader import so_leader

    instrument(SOFollower, "get_observation", "robot.get_observation")
    instrument(SOFollower, "send_action", "robot.send_action")
    instrument(SOLeader, "get_action", "teleop.get_action")
    instrument(LeRobotDataset, "add_frame", "dataset.add_frame")

    # split get_observation into bus vs cameras
    _obs = SOFollower.get_observation

    def split_obs(self):
        t0 = time.perf_counter()
        obs = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        dt = time.perf_counter() - t0
        s = STATS.setdefault("  obs: bus sync_read", [0.0, 0.0, 0.0])
        s[0] += 1; s[1] += dt; s[2] = max(s[2], dt)
        obs_dict = {f"{m}.pos": v for m, v in obs.items()}
        for key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                t1 = time.perf_counter()
                obs_dict[key] = cam.read_latest()
                dt1 = time.perf_counter() - t1
                s1 = STATS.setdefault(f"  obs: camera {key}", [0.0, 0.0, 0.0])
                s1[0] += 1; s1[1] += dt1; s1[2] = max(s1[2], dt1)
        return obs_dict

    instrument(SOFollower, "get_observation", split_obs)

    try:
        from lerobot.utils import visualization_utils
        instrument_func(visualization_utils, "log_visualization_data", "rerun log_visualization_data")
        import lerobot.scripts.lerobot_record as lr
        if hasattr(lr, "log_visualization_data"):
            instrument_func(lr, "log_visualization_data", "rerun log_visualization_data (lr)")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not instrument visualization: {e}")

    import threading
    stop = threading.Event()

    def stopper():
        time.sleep(args.seconds)
        stop.set()
    threading.Thread(target=stopper, daemon=True).start()

    from lerobot.scripts.lerobot_record import main as record_main
    t0 = time.perf_counter()
    try:
        record_main()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        wall = time.perf_counter() - t0
        loops = max(int(STATS.get("dataset.add_frame", [1])[0]), 1)
        report(loops, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
