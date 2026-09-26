#!/usr/bin/env python3
"""Per-episode success/failure labels + retry, for SO-101 recording.

Wires into LeRobot's real keyboard pipeline rather than prompting, because the
key listener already owns stdin -- an ``input()`` between episodes would fight
it for the TTY.

LeRobot routes every key from both backends (pynput and the terminal reader)
through one ``dispatch`` callback passed to ``create_key_listener``. This wraps
that callback so:

    s   label the episode just saved as success
    f   label it as failure
    d   discard the in-progress episode and re-record it (alias of LeRobot's
        own left-arrow / 'r' re-record)
    n / right   end the episode early and keep it
    r / left    discard and re-record
    q / esc     stop

The label is attached in a ``save_episode`` wrapper, i.e. after LeRobot has
committed the episode, so the index is final and correct. Labels go to
``meta/outcomes.json`` (see scripts/episode_outcomes.py) -- the dataset schema
is untouched, so stock LeRobot can still train on it.

Failed and discarded episodes stay on disk. Nothing is deleted; filter at train
time instead.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_outcomes  # noqa: E402

logger = logging.getLogger("so101_outcomes")

# events dict, captured from init_keyboard_listener
_EVENTS: dict = {}
# outcome chosen since the last save
_PENDING: dict[str, str] = {}

HELP = (
    "\n  episode controls:  s=success  f=failure  d/n=discard+re-record  q=quit\n"
)


def install() -> bool:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils import keyboard_input
    except ImportError as e:  # noqa: BLE001
        print(f"[outcomes] cannot import lerobot ({e}); labels disabled", file=sys.stderr)
        return False

    # 1. Capture the events dict that LeRobot creates for the key listener.
    _orig_init = keyboard_input.init_keyboard_listener

    def init_capturing():
        listener, events = _orig_init()
        _EVENTS.clear()
        _EVENTS.update(events)
        # LeRobot's events dict has no outcome slot; add ours.
        _EVENTS.setdefault("episode_outcome", None)
        print(HELP, flush=True)
        return listener, events

    keyboard_input.init_keyboard_listener = init_capturing
    # lerobot_record imported the symbol directly, so patch it there too.
    try:
        import lerobot.scripts.lerobot_record as lr
        lr.init_keyboard_listener = init_capturing
    except ImportError:
        pass

    # 2. Intercept keys before LeRobot maps them, to catch s / f / d.
    _orig_create = keyboard_input.create_key_listener

    def create_with_outcomes(dispatch, *, controls_help: str = ""):
        def wrapped(name: str) -> None:
            key = (name or "").lower()
            if key == "s":
                _PENDING["outcome"] = "success"
                _EVENTS["episode_outcome"] = "success"
                print("\n  [outcome] success", flush=True)
                return
            if key == "f":
                _PENDING["outcome"] = "failure"
                _EVENTS["episode_outcome"] = "failure"
                print("\n  [outcome] failure", flush=True)
                return
            if key == "d":
                # Same as LeRobot's own left/'r': drop this take, try again.
                _PENDING["outcome"] = "discarded"
                _EVENTS["episode_outcome"] = "discarded"
                _EVENTS["rerecord_episode"] = True
                _EVENTS["exit_early"] = True
                print("\n  [outcome] discarded - re-recording this episode", flush=True)
                return
            dispatch(name)

        return _orig_create(wrapped, controls_help=controls_help)

    keyboard_input.create_key_listener = create_with_outcomes

    # 3. Attach the label after the episode is committed.
    _orig_save = LeRobotDataset.save_episode

    def save_and_label(self, *a, **kw):
        result = _orig_save(self, *a, **kw)
        outcome = _PENDING.pop("outcome", None) or _EVENTS.get("episode_outcome")
        _EVENTS["episode_outcome"] = None
        if outcome is None:
            print("\n  [outcome] unlabeled (press s or f before the next episode to label it)",
                  flush=True)
            return result
        idx = int(getattr(self, "num_episodes", 1)) - 1
        try:
            episode_outcomes.record(
                str(self.root), idx, outcome,
                n_frames=getattr(self, "num_frames", None),
            )
            print(f"  [outcome] episode {idx} -> {outcome}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[outcomes] could not write label for episode {idx}: {e}", file=sys.stderr)
        return result

    LeRobotDataset.save_episode = save_and_label
    logger.info("episode outcome controls installed (s/f/d)")
    return True
