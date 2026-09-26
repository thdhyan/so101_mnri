#!/usr/bin/env python3
"""Per-episode success/failure labels for a LeRobotDataset.

LeRobot's dataset schema (v3.0) has no outcome field -- an episode is just
frames plus a task string. This keeps the labels in a sidecar next to the data
rather than patching the schema, so:

  * the dataset stays loadable by stock LeRobot for training
  * labels survive a LeRobot upgrade
  * nothing is deleted; a failed or discarded episode is still on disk and can
    be filtered at train time instead

Layout, written to <dataset_root>/meta/outcomes.json:

    {"version": 1,
     "episodes": {"0": {"outcome": "success", "n_frames": 450,
                        "duration_s": 30.0, "recorded_at": "..."},
                  "1": {"outcome": "failure", ...}}}

Outcomes: success | failure | unlabeled (not in the file) | discarded
(kept on disk, but marked so it can be excluded).

    python scripts/episode_outcomes.py --root data/thakk100_so101_single_arm_6dof
    python scripts/episode_outcomes.py --root ... --export successes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

VALID = ("success", "failure", "discarded")
OUTCOMES_FILE = Path("meta") / "outcomes.json"
VERSION = 1


def outcomes_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / OUTCOMES_FILE


def load(dataset_root: str | Path) -> dict:
    p = outcomes_path(dataset_root)
    if not p.is_file():
        return {"version": VERSION, "episodes": {}}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        print(f"WARNING: {p} is corrupt, starting a fresh label file", file=sys.stderr)
        return {"version": VERSION, "episodes": {}}


def save(dataset_root: str | Path, data: dict) -> None:
    p = outcomes_path(dataset_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)          # atomic: never leave a half-written label file


def record(dataset_root: str | Path, episode_index: int, outcome: str, **meta) -> None:
    """Label one episode. Unknown episodes are tracked so gaps are visible."""
    if outcome not in VALID:
        raise ValueError(f"outcome must be one of {VALID}, got {outcome!r}")
    data = load(dataset_root)
    data.setdefault("version", VERSION)
    data.setdefault("episodes", {})
    data["episodes"][str(episode_index)] = {
        "outcome": outcome,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **{k: v for k, v in meta.items() if v is not None},
    }
    save(dataset_root, data)


def total_episodes(dataset_root: str | Path) -> int:
    """How many episodes the dataset actually holds, from meta/info.json."""
    info = Path(dataset_root) / "meta" / "info.json"
    if info.is_file():
        try:
            n = json.loads(info.read_text()).get("total_episodes")
            if isinstance(n, int):
                return n
        except json.JSONDecodeError:
            pass
    return 0


def report(dataset_root: str | Path) -> int:
    data = load(dataset_root)
    eps = data.get("episodes", {})
    n_total = total_episodes(dataset_root)

    counts = {o: 0 for o in VALID}
    for rec in eps.values():
        counts[rec.get("outcome", "success")] = counts.get(rec.get("outcome", "success"), 0) + 1
    labeled = sum(counts.values())
    unlabeled = max(n_total - labeled, 0)

    print(f"dataset: {dataset_root}")
    if n_total:
        print(f"episodes in dataset: {n_total}")
    print(f"labeled:            {labeled}")
    for o in VALID:
        bar = "#" * min(40, counts[o])
        print(f"  {o:10s} {counts[o]:5d}  {bar}")
    if unlabeled:
        print(f"  {'unlabeled':10s} {unlabeled:5d}  (no s/f pressed for these)")

    if n_total:
        usable = counts["success"]
        print(f"\n  {usable}/{n_total} usable for success-only training "
              f"({100 * usable / n_total:.0f}%)")

    missing = [i for i in range(n_total) if str(i) not in eps]
    if missing:
        shown = ", ".join(str(i) for i in missing[:20])
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        print(f"\n  unlabeled episode indices: {shown}{more}")
    return 0


def export(dataset_root: str | Path, out_csv: str, only: set[str]) -> int:
    data = load(dataset_root)
    rows = [
        {"episode_index": int(i), **rec}
        for i, rec in sorted(data.get("episodes", {}).items(), key=lambda kv: int(kv[0]))
    ]
    if only:
        rows = [r for r in rows if r.get("outcome") in only]
    if not rows:
        print("nothing to export", file=sys.stderr)
        return 1
    fields = ["episode_index", "outcome", "n_frames", "duration_s", "recorded_at"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="dataset root (the dir with meta/)")
    ap.add_argument("--export", metavar="CSV", help="write labels to a CSV")
    ap.add_argument("--only", default="", help="comma-separated outcomes to include in --export")
    args = ap.parse_args()

    if args.export:
        only = {o.strip() for o in args.only.split(",") if o.strip()}
        return export(args.root, args.export, only)
    return report(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
