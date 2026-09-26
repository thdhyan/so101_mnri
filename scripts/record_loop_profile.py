#!/usr/bin/env python3
"""Timing accumulators for the real record_loop.

LeRobot's "running slower than target FPS" warning names camera FPS, policy
inference and CPU starvation, but it is a fixed string with no diagnostics --
and there is no policy in this loop at all, so none of its three suggestions
are grounded.

This wraps the real methods with timing accumulators. The actual code path is
untouched; only measurement is added. Installed by `record_so101.py
--profile <seconds>`, so the argv under test is the same one a real recording
uses, including the cameras.

Splitting `get_observation` into the bus read vs each camera matters: a real
"camera FPS" problem shows up there, not in the aggregate.
"""

from __future__ import annotations

import statistics
import time

STATS: dict[str, list[float]] = {}


def _acc(label: str, dt: float) -> None:
    s = STATS.setdefault(label, [0.0, 0.0, 0.0])
    s[0] += 1
    s[1] += dt
    s[2] = max(s[2], dt)


def _wrap(owner, attr: str, label: str) -> None:
    original = getattr(owner, attr)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return original(*a, **kw)
        finally:
            _acc(label, time.perf_counter() - t0)

    setattr(owner, attr, timed)


def install() -> bool:
    """Instrument the record path. Idempotent."""
    if STATS:
        return True
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
    except ImportError as e:  # noqa: BLE001
        print(f"[profile] cannot import lerobot ({e})", flush=True)
        return False

    _wrap(SOLeader, "get_action", "teleop.get_action")
    _wrap(SOFollower, "send_action", "robot.send_action")
    _wrap(LeRobotDataset, "add_frame", "dataset.add_frame")

    # Replace get_observation with a split version: it is the only place a
    # camera cost can hide, and the aggregate hides it.
    def split_obs(self):
        t0 = time.perf_counter()
        obs = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        _acc("obs: bus sync_read", time.perf_counter() - t0)
        out = {f"{m}.pos": v for m, v in obs.items()}
        for key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                t1 = time.perf_counter()
                out[key] = cam.read_latest()
                _acc(f"obs: camera {key}", time.perf_counter() - t1)
        return out

    SOFollower.get_observation = split_obs

    # rerun logging, wherever it is called from
    for mod in ("lerobot.utils.visualization_utils", "lerobot.scripts.lerobot_record"):
        try:
            import importlib
            m = importlib.import_module(mod)
            if hasattr(m, "log_visualization_data"):
                _wrap(m, "log_visualization_data", "rerun log_visualization")
        except Exception:  # noqa: BLE001
            pass
    return True


def report(loops: int, wall: float) -> None:
    print("\n" + "=" * 74)
    print(f" record_loop breakdown: {loops} iterations, {wall:.1f}s wall, "
          f"{loops / wall:.1f} Hz")
    print("=" * 74)
    print(f"{'component':30s} {'n':>6s} {'mean ms':>9s} {'p95 ms':>8s} {'max ms':>8s} {'share':>7s}")
    accounted = 0.0
    rows = []
    for label, (n, tot, mx) in STATS.items():
        if not n:
            continue
        accounted += tot
        rows.append((label, n, tot / n * 1000, mx * 1000, tot / wall * 100 if wall else 0))
    for label, n, mean_ms, mx_ms, share in sorted(rows, key=lambda r: -r[2]):
        print(f"{str(label):30s} {int(n):6d} {mean_ms:9.2f} {'':8s} {mx_ms:8.2f} {share:6.1f}%")

    unaccounted = max(wall - accounted, 0.0)
    print("-" * 74)
    print(f"{'ACCOUNTED':30s} {'':6s} {accounted / loops * 1000:9.2f} {'':8s} {'':8s} "
          f"{accounted / wall * 100 if wall else 0:6.1f}%")
    print(f"{'unaccounted (sleep/overhead)':30s} {'':6s} {unaccounted / loops * 1000:9.2f}")
    print(f"{'TOTAL per iteration':30s} {'':6s} {wall / loops * 1000:9.2f}")

    per_iter = {k: v[1] / loops * 1000 for k, v in STATS.items() if v[0]}
    print()
    for fps in (30, 25, 20, 15, 10):
        budget = 1000.0 / fps
        worst = sorted(per_iter.items(), key=lambda kv: -kv[1])[:3]
        need = sum(v for _, v in worst)
        verdict = "OK" if budget >= wall / loops * 1000 else f"short by {wall / loops * 1000 - budget:.0f}ms"
        print(f"  {fps:2d} Hz needs {budget:5.1f} ms/iter -> {verdict:22s} "
              f"biggest: " + ", ".join(f"{k.split(':')[-1].strip()} {v:.1f}ms" for k, v in worst))


def reset() -> None:
    STATS.clear()
