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

    # Must be the SAME object LeRobot's record_loop reads, not a copy: a copy
    # silently swallows 'd' and the --profile stop.
    check(events is E._EVENTS, "events dict is LeRobot's own, not a copy")

    # --- key behaviour -----------------------------------------------------
    print("\n  key handling:")
    E._PENDING.clear()
    events["exit_early"] = False
    dispatch("s")
    check(E._PENDING.get("outcome") == "success", "'s' -> success")
    check(events.get("exit_early") is True, "'s' ends the episode")

    E._PENDING.clear()
    events["exit_early"] = False
    dispatch("f")
    check(E._PENDING.get("outcome") == "failure", "'f' -> failure")
    check(events.get("exit_early") is True, "'f' ends the episode")

    E._PENDING.clear()
    E._EVENTS["rerecord_episode"] = False
    E._EVENTS["exit_early"] = False
    dispatch("d")
    check(E._PENDING.get("outcome") is None, "'d' leaves no label behind")
    check(events.get("rerecord_episode") is True, "'d' requests a re-record (LeRobot's dict)")
    check(events.get("exit_early") is True, "'d' ends the episode early (LeRobot's dict)")

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

        # 's' during a take that is then re-recorded must not label the retake.
        ds.add_frame(make_frame())
        dispatch("s")
        ds.clear_episode_buffer()
        check(E._PENDING.get("outcome") is None and E._EVENTS["episode_outcome"] is None,
              "re-record clears a label pressed during the discarded take")

        data = json.loads((root / "meta" / "outcomes.json").read_text())
        eps = data["episodes"]
        check(eps.get("0", {}).get("outcome") == "success", "episode 0 labelled success")
        check(eps.get("1", {}).get("outcome") == "failure", "episode 1 labelled failure")
        check(eps.get("1", {}).get("n_frames") == 1, "n_frames is the episode's own length")
        check("2" not in eps, "episode 2 left unlabeled when no key pressed")

        # --- duck start zones -------------------------------------------------
        # A zone is chosen while the duck is placed and applies to the episode
        # saved next. The important asymmetry: it SURVIVES a re-record (the
        # duck has not moved), and is consumed by exactly one save.
        print("\n  duck start zones:")
        E._PENDING.clear()
        E.set_zone(3, quiet=True)
        ds.add_frame(make_frame())
        dispatch("s")
        check(E._PENDING.get("outcome") == "success", "digit-zone take also labels outcome")

        # 'd' drops the take but the duck is still in zone 3.
        ds.clear_episode_buffer()
        check(E._PENDING_ZONE == 3, "zone SURVIVES a re-record (duck did not move)")

        # The retake: same zone, and this time it is actually saved.
        ds.add_frame(make_frame())
        dispatch("s")
        ds.save_episode()
        check(E._PENDING_ZONE is None, "zone consumed by the save it was attached to")

        retake = int(ds.num_episodes) - 1
        eps = json.loads((root / "meta" / "outcomes.json").read_text())["episodes"]
        check(eps[str(retake)].get("zone") == 3, f"episode {retake} carries zone 3")
        check(eps[str(retake)].get("outcome") == "success", f"episode {retake} labelled success")

        # digit mapping: digits now select PROMPTS, letters select zones
        check(len(E.PROMPT_KEYS) == len(E.tasks()),
              f"one prompt key per configured prompt (got {len(E.PROMPT_KEYS)})")
        check(set(E.PROMPT_KEYS) == {str(i) for i in range(1, len(E.tasks()) + 1)},
              "prompt keys are 1..N with no gaps")
        check(E.ZONE_KEYS.get("i") == 1 and E.ZONE_KEYS.get("l") == 4,
              f"letters map to zones 1-4 (got {E.ZONE_KEYS})")
        check(not (set(E.ZONE_KEYS) & set(E.PROMPT_KEYS)),
              "zone keys and prompt keys do not collide")

        # prompt selection -> what actually lands in the frame's `task`
        print("\n  prompts:")
        E.set_task(2, quiet=True)
        check(E.current_task_id() == 2, "digit 2 selects prompt 2")
        check(E.current_task() == E.tasks()[1], "prompt 2 resolves to TASK_PROMPTS[1]")
        E.set_task(None, quiet=True)
        check(E.current_task() is None, "prompt can be cleared -> falls back to RECORD_TASK")
        before = len(E.PROMPT_KEYS)
        E.set_task(before + 5, quiet=True)
        check(E.current_task() is None, "an out-of-range prompt number is rejected, not clamped")

        # a zone with no s/f at all is stored as 'unlabeled', not dropped
        E.set_zone(7, quiet=True)
        ds.add_frame(make_frame())
        ds.save_episode()
        eps = json.loads((root / "meta" / "outcomes.json").read_text())["episodes"]
        zoned = int(ds.num_episodes) - 1
        check(eps[str(zoned)].get("zone") == 7, "zoned take is kept even with no verdict")
        check(eps[str(zoned)].get("outcome") == "unlabeled", "zoned take with no s/f is 'unlabeled'")

        # 'z' clears a zone pressed by mistake
        E.set_zone(5, quiet=True)
        dispatch("z")
        check(E._PENDING_ZONE is None, "z clears the pending zone")

        # coverage nudge fires on the 3rd consecutive take from one zone
        E._ZONE_RUN, E._ZONE_RUN_LAST = 0, None
        for i in range(E.ZONE_RUN_WARN):
            E.set_zone(4, quiet=True)
        check(E._ZONE_RUN == E.ZONE_RUN_WARN, "same-zone run is counted")
        E.set_zone(6, quiet=True)
        check(E._ZONE_RUN == 1, "changing zone resets the run")
        check(E._PENDING_ZONE == 6, "new zone is pending after a reset")
        E.set_zone(None, quiet=True)

        # the prompt must survive a re-record, like the zone: the retake is the
        # same task being attempted again
        E.set_task(3, quiet=True)
        E.set_zone(1, quiet=True)
        ds.add_frame(make_frame())
        dispatch("d")
        check(E.current_task() == E.tasks()[2] and E._PENDING_ZONE == 1,
              "prompt AND zone survive a 'd' re-record")
        E.set_task(None, quiet=True)
        E.set_zone(None, quiet=True)

        # The dataset schema must be untouched: outcomes live in a sidecar, so
        # meta/info.json gains no outcome field and stock LeRobot is unaffected.
        # (Re-instantiating LeRobotDataset here would hit the HF Hub for a
        # version of this fake repo_id, which fails offline, so read the
        # metadata directly instead.)
        info = json.loads((root / "meta" / "info.json").read_text())
        # 3 from the loop above, +1 zoned retake, +1 zoned with no verdict.
        check(info.get("total_episodes") == 5, "dataset recorded 5 episodes")
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
            # episode 0 and the zoned retake are the two successes.
            check(len(rows) == 3, "CSV filtered to the 2 successes (+header)")
            check("zone" in rows[0], "CSV carries the zone column")
            check(any(r.startswith(f"{retake},") and ",3," in r for r in rows[1:]),
                  f"CSV row for episode {retake} carries zone 3")
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
