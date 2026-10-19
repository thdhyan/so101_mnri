#!/usr/bin/env python3
"""Offline checks for scripts/episode_task.py. No hardware, no cameras.

The claim under test is that per-episode prompts produce a *stock* multi-task
LeRobotDataset rather than a custom format, i.e. the only thing that changes is
the `task` string on each frame:

- a selected prompt is written into frame["task"]
- no prompt selected -> frame["task"] is left exactly as LeRobot set it
- the prompt is FROZEN for the whole episode, so a keypress mid-take cannot
  split one episode across two task strings
- the freeze releases at save and at clear (a 'd' re-record), so the retake can
  choose again
- the sidecar and the dataset agree on the prompt
- meta/tasks.parquet really does accumulate the distinct prompts, which is the
  part that makes it loadable and trainable by stock LeRobot

    .venv/bin/python scripts/test_episode_task.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def main() -> int:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import episode_outcome_keys as E
    import episode_task as T

    if not E.install():
        print("could not install episode_outcome_keys", file=sys.stderr)
        return 1
    if not T.install():
        print("could not install episode_task", file=sys.stderr)
        return 1

    td = tempfile.mkdtemp()
    try:
        root = Path(td) / "ds"
        feats = {
            "observation.state": {"dtype": "float32", "shape": [6],
                                  "names": [f"j{i}" for i in range(6)]},
            "action": {"dtype": "float32", "shape": [6], "names": [f"j{i}" for i in range(6)]},
        }
        ds = LeRobotDataset.create("test/tasks", 15, root=root, robot_type="so_follower",
                                   features=feats, use_videos=False)

        def frame(task="push the duck into the square tape"):
            # add_frame POPS "task", so every frame needs a fresh dict.
            return {"observation.state": np.zeros(6, dtype=np.float32),
                    "action": np.zeros(6, dtype=np.float32), "task": task}

        # The buffer lives on the writer: LeRobotDataset.add_frame delegates to
        # DatasetWriter.add_frame, which appends frame["task"] there. Re-read it
        # every time -- clear_episode_buffer/save_episode REPLACE the dict, so
        # a cached reference silently points at the previous episode.
        def buf():
            return list(ds.writer.episode_buffer["task"])

        print("\n  no prompt selected:")
        E.set_task(None, quiet=True)
        ds.add_frame(frame("stock single_task"))
        check(buf() == ["stock single_task"],
              "frame task left exactly as LeRobot set it (stock fallback path)")

        print("\n  a selected prompt is written:")
        ds.clear_episode_buffer()
        E.set_task(1, quiet=True)
        ds.add_frame(frame("stock single_task"))
        ds.add_frame(frame("stock single_task"))
        check(buf() == [E.tasks()[0]] * 2,
              "every frame carries the selected prompt, not single_task")

        print("\n  frozen for the episode:")
        E.set_task(4, quiet=True)          # pressed mid-take
        ds.add_frame(frame("stock single_task"))
        check(set(buf()) == {E.tasks()[0]},
              "a keypress mid-episode cannot split it across two prompts")
        check(T.current() == E.tasks()[0], "current() reports the frozen prompt")

        ds.save_episode()
        check(T.current() is None or T.current() != E.tasks()[0],
              "freeze releases after the episode is saved")

        print("\n  every prompt reaches tasks.parquet:")
        for n in range(1, len(E.tasks()) + 1):
            E.set_task(n, quiet=True)
            ds.add_frame(frame("stock single_task"))
            ds.save_episode()
        meta = root / "meta" / "tasks.parquet"
        check(meta.is_file(), "meta/tasks.parquet written")
        if meta.is_file():
            try:
                import pyarrow.parquet as pq
                tbl = pq.read_table(meta)
                stored = set(tbl.column("task").to_pylist()) if "task" in tbl.column_names \
                    else set(map(tuple, tbl.to_pydict().values()))
                check(len(E.tasks()) == len(stored),
                      f"all {len(E.tasks())} prompts accumulated ({len(stored)} found)")
                for t in E.tasks():
                    check(any(t == s for s in stored), f"tasks.parquet holds {t!r}")
            except Exception as ex:  # noqa: BLE001
                check(False, f"reading tasks.parquet: {type(ex).__name__}: {ex}")

        print("\n  sidecar agrees with the dataset:")
        E.set_task(3, quiet=True)
        E.set_zone(2, quiet=True)
        E._PENDING["outcome"] = "success"
        ds.add_frame(frame("stock single_task"))
        ds.save_episode()
        idx = int(ds.num_episodes) - 1
        eps = json.loads((root / "meta" / "outcomes.json").read_text())["episodes"]
        rec = eps[str(idx)]
        check(rec.get("task") == E.tasks()[2], "sidecar records the prompt string")
        check(rec.get("task_id") == 3, "sidecar records the prompt index")
        check(rec.get("zone") == 2, "sidecar still records the zone")

        print("\n  re-record releases the freeze but keeps the prompt:")
        E.set_task(5, quiet=True)
        ds.add_frame(frame("stock single_task"))
        check(T.current() == E.tasks()[4], "prompt is frozen while the take is live")
        ds.clear_episode_buffer()
        check(T._frozen is None, "a 'd' re-record unfreezes the prompt")
        check(T.current() == E.tasks()[4],
              "the retake keeps the same prompt (same task being attempted again)")
        E.set_task(None, quiet=True)
        E.set_zone(None, quiet=True)

        print("\n  dataset schema untouched:")
        info = json.loads((root / "meta" / "info.json").read_text())
        check(not any("prompt" in k or "task_id" in k for k in info.get("features", {})),
              "no custom feature added; the prompt rides on LeRobot's own 'task'")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
