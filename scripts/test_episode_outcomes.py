#!/usr/bin/env python3
"""Offline test for the episode outcome keys (s / f / d) and the label file.

No hardware: it exercises the real patched keyboard dispatch and a real
LeRobotDataset with one dummy frame per episode, then checks that

  * 's' / 'f' set the pending outcome and 'd' requests a re-record
  * other keys still reach LeRobot's own handler
  * save_episode writes meta/outcomes.json with the right episode index
  * the report counts success / failure / unlabeled correctly

    python scripts/test_episode_outcomes.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_outcome_keys as E  # noqa: E402
import episode_outcomes as O  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def main() -> int:
    # --- grab the dispatch we handed to create_key_listener ----------------
    # NOTE: install() must run AFTER the spy is in place, because install
    # captures create_key_listener as its "_orig_create". Patching afterwards
    # would bypass the wrapper entirely and test nothing.
    from lerobot.utils import keyboard_input as K

    holder: dict = {}

    def capture(dispatch, **kw):
        holder["dispatch"] = dispatch
        return None

    real_create = K.create_key_listener
    K.create_key_listener = capture
    try:
        ok = E.install()
        check(ok, "install() succeeded (with spy in place)")
        if not ok:
            return 1
        # init_keyboard_listener is patched by install(); calling it now routes
        # through create_with_outcomes -> our capture, giving us the inner
        # wrapper. The spy must still be installed for this call.
        listener, events = K.init_keyboard_listener()
    finally:
        K.create_key_listener = real_create

    dispatch = holder.get("dispatch")
    check(callable(dispatch), "wrapped dispatch was registered with create_key_listener")
    if dispatch is None:
        FAILS.append("no dispatch captured")
        return 1

    # events dict is shared with ours (captured by the patched init)
    check(events is E._EVENTS or events is not None, "events dict captured")
    E._EVENTS.update(events)
    E._EVENTS["episode_outcome"] = None

    # --- key behaviour -----------------------------------------------------
    print("\n  key handling:")
    E._PENDING.clear()
    dispatch("s")
    check(E._PENDING.get("outcome") == "success", "'s' -> success")

    E._PENDING.clear()
    dispatch("f")
    check(E._PENDING.get("outcome") == "failure", "'f' -> failure")

    E._PENDING.clear()
    E._EVENTS["rerecord_episode"] = False
    E._EVENTS["exit_early"] = False
    dispatch("d")
    check(E._PENDING.get("outcome") == "discarded", "'d' -> discarded")
    check(E._EVENTS.get("rerecord_episode") is True, "'d' requests a re-record")
    check(E._EVENTS.get("exit_early") is True, "'d' ends the episode early")

    # passthrough: LeRobot's own keys must still work
    print("\n  passthrough:")
    for key in ("n", "q", "r", "right", "left", "esc"):
        E._PENDING.clear()
        E._EVENTS["rerecord_episode"] = False
        E._EVENTS["stop_recording"] = False
        dispatch(key)
        check(E._PENDING.get("outcome") is None, f"'{key}' is not intercepted")

    # --- end-to-end: real LeRobotDataset, real patched save_episode ---------
    print("\n  end-to-end label write:")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    td = tempfile.mkdtemp()
    try:
        root = Path(td) / "ds"
        feats = {
            "observation.state": {"dtype": "float32", "shape": [6],
                                  "names": [f"j{i}" for i in range(6)]},
            "action": {"dtype": "float32", "shape": [6], "names": [f"j{i}" for i in range(6)]},
        }
        ds = LeRobotDataset.create("test/outcomes", 15, root=root, robot_type="so_follower",
                                   features=feats, use_videos=False)
        import numpy as np
        # NOTE: add_frame does `frame.pop("task")` (dataset_writer.py:228), so
        # the dict is MUTATED. A fresh dict per episode is required -- real
        # recording builds one per frame, so it never hits this.
        def make_frame():
            return {"observation.state": np.zeros(6, dtype=np.float32),
                    "action": np.zeros(6, dtype=np.float32), "task": "t"}

        for idx, outcome in ((0, "success"), (1, "failure"), (2, None)):
            ds.add_frame(make_frame())
            E._PENDING.clear()
            if outcome:
                E._PENDING["outcome"] = outcome
            E._EVENTS["episode_outcome"] = outcome
            ds.save_episode()

        data = json.loads((root / "meta" / "outcomes.json").read_text())
        eps = data["episodes"]
        check(eps.get("0", {}).get("outcome") == "success", "episode 0 labelled success")
        check(eps.get("1", {}).get("outcome") == "failure", "episode 1 labelled failure")
        check("2" not in eps, "episode 2 left unlabeled when no key pressed")

        # The dataset schema must be untouched: outcomes live in a sidecar, so
        # meta/info.json gains no outcome field and stock LeRobot is unaffected.
        # (Re-instantiating LeRobotDataset here would hit the HF Hub for a
        # version of this fake repo_id, which fails offline, so read the
        # metadata directly instead.)
        info = json.loads((root / "meta" / "info.json").read_text())
        check(info.get("total_episodes") == 3, "dataset recorded 3 episodes")
        check(not any("outcome" in k for k in info.get("features", {})),
              "no outcome field leaked into the dataset schema")
        check((root / "meta" / "outcomes.json").is_file(),
              "outcomes live in a sidecar, not the schema")

        # rejected outcomes are refused rather than silently stored
        try:
            O.record(root, 99, "banana")
            check(False, "invalid outcome rejected")
        except ValueError:
            check(True, "invalid outcome rejected")

        print("\n  report:")
        check(O.report(root) == 0, "report() returns 0")

        csv_path = Path(td) / "out.csv"
        check(O.export(root, str(csv_path), {"success"}) == 0 and csv_path.is_file(),
              "CSV export written")
        if csv_path.is_file():
            rows = csv_path.read_text().strip().splitlines()
            check(len(rows) == 2, "CSV filtered to the 1 success (+header)")
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
