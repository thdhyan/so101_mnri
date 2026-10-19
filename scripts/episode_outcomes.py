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
     "episodes": {"0": {"outcome": "success", "zone": 3, "n_frames": 450,
                        "recorded_at": "..."},
                  "1": {"outcome": "failure", ...}}}

``zone`` is the duck start zone (1-10) the operator marked while placing the
duck; it is absent on episodes recorded before zones existed, and on takes where
no zone key was pressed. It is the coverage record: a dataset where every
episode shares one zone is one demonstration recorded many times.

Outcomes: success | failure | discarded | unlabeled, where "unlabeled" means a
zone was recorded but no human verdict. An episode with no entry at all was
never reviewed. Both are unusable for success-only training and are counted
separately by ``report()``.

    python scripts/episode_outcomes.py --root data/thakk100_pink_duck_in_a_box
    python scripts/episode_outcomes.py --root ... --export successes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

VALID = ("success", "failure", "discarded", "unlabeled")
# The outcomes a human actually decided. "unlabeled" is storable but is not a
# decision -- it is what an episode looks like when a start zone was recorded
# but no s/f was ever pressed, which is different from an episode nobody
# touched at all.
DECIDED = ("success", "failure", "discarded")
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

    counts = {o: 0 for o in DECIDED}
    for rec in eps.values():
        o = rec.get("outcome", "success")
        counts[o] = counts.get(o, 0) + 1
    decided = sum(counts.values())
    # An episode with no entry at all was never looked at; one stored as
    # "unlabeled" has a zone but no human verdict. Both are unusable, and the
    # distinction tells you whether the gap is a missing keypress or a missing
    # review.
    never_touched = max(n_total - len(eps), 0)
    unlabeled = counts.get("unlabeled", 0) + never_touched

    print(f"dataset: {dataset_root}")
    if n_total:
        print(f"episodes in dataset: {n_total}")
    print(f"labeled:            {decided}")
    for o in DECIDED:
        bar = "#" * min(40, counts[o])
        print(f"  {o:10s} {counts[o]:5d}  {bar}")
    if unlabeled:
        detail = f" ({never_touched} never reviewed)" if never_touched else ""
        print(f"  {'unlabeled':10s} {unlabeled:5d}  (no s/f pressed){detail}")

    if n_total:
        usable = counts["success"]
        print(f"\n  {usable}/{n_total} usable for success-only training "
              f"({100 * usable / n_total:.0f}%)")

    missing = [i for i in range(n_total) if str(i) not in eps]
    if missing:
        shown = ", ".join(str(i) for i in missing[:20])
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        print(f"\n  unlabeled episode indices: {shown}{more}")

    # Duck start zones. Coverage is the thing that actually decides whether the
    # dataset is useful, and it is invisible unless it is counted: 100 episodes
    # from one duck placement is one episode, recorded 100 times.
    zones: dict[int, int] = {}
    for rec in eps.values():
        z = rec.get("zone")
        if isinstance(z, int):
            zones[z] = zones.get(z, 0) + 1
    if zones or n_total:
        print("\n  duck start zones:")
        unzoned = max(n_total - sum(zones.values()), 0)
        for z in sorted(zones):
            bar = "#" * min(40, zones[z])
            print(f"    zone {z:<2d} {zones[z]:5d}  {bar}")
        if unzoned:
            print(f"    {'none':6s} {unzoned:5d}  (episode not attributed to a zone)")
        # Report the spread, not a fixed denominator: how many zones exist is
        # the operator's choice (this rig is taped as a 2x2, zones 1-4), and
        # hardcoding 10 would make a healthy 4-zone rig look like it is short
        # six zones.
        if zones:
            lo, hi = min(zones), max(zones)
            spread = f"zones {lo}-{hi}" if hi > lo else f"zone {lo}"
            print(f"    -> {len(zones)} of {hi - lo + 1} in use ({spread})"
                  + (f"; {unzoned} episodes unzoned" if unzoned else ""))
        elif n_total:
            print(f"    -> none recorded yet; {unzoned} episodes unzoned")

    # Prompts. Multi-task coverage has its own failure mode: a dataset can look
    # healthy by count while one prompt holds 95% of it, which trains a policy
    # that ignores the instruction.
    prompts: dict[str, int] = {}
    for rec in eps.values():
        t = rec.get("task")
        if isinstance(t, str):
            prompts[t] = prompts.get(t, 0) + 1
    if prompts:
        print("\n  prompts:")
        for t, c in sorted(prompts.items(), key=lambda kv: -kv[1]):
            bar = "#" * min(40, c)
            print(f"    {c:5d}  {bar}  {t}")
        print(f"    -> {len(prompts)} distinct prompt(s) recorded")
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
    fields = ["episode_index", "outcome", "zone", "task_id", "task", "n_frames",
              "duration_s", "recorded_at"]
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
