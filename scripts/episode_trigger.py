#!/usr/bin/env python3
"""Turn-based recording: record only when the operator says the scene is ready.

## Why this exists

LeRobot's `record()` alternates two calls to the same `record_loop`:

    record_loop(..., control_time_s=episode_time_s, dataset=dataset)   # RECORDS
    record_loop(..., control_time_s=reset_time_s)                      # no dataset

The reset call is the same loop with no `dataset`, so `add_frame` is never
reached and **no frames are written while you reset** -- the frames of you
moving the duck are not in the dataset. That part is fine.

The problem is the *boundary*. The reset loop ends on a timer or on
``events["exit_early"]``, which LeRobot's own ``n`` key sets. So the usual
"press n to skip the rest of the reset" makes the next episode start recording
immediately -- while your hand is still in frame and the duck is still moving
into place. Those frames *are* recorded, and they teach the policy that a hand
in shot at t=0 is normal.

It also means the reset length is a guess rather than a decision, and that you
cannot mark where the duck started without a timer racing you.

## What this does

Puts an explicit gate between the two phases:

    WAITING    the robot still tracks the leader (so the arm returns to rest
               and the live view keeps running), but nothing is written.
               Press the zone digits here, while placing the duck.
    RECORDING  frames are written. s / f label and end, d discards.

Turn key: **SPACE** (also ``r`` or ``n`` while WAITING, for muscle memory).
``s``/``f``/``d`` are ignored while WAITING so a stray keypress mid-reset
cannot discard or label an episode.

The first episode waits for a turn key too, so the session never starts
recording before you are ready.

Implemented by wrapping LeRobot's `record_loop`:

- a call **with** a dataset is a recording call -> wait for the turn key first
- a call **without** a dataset is the reset call -> replace ``events`` with a
  gate whose ``exit_early`` is true only once the turn key has been pressed,
  and give it an effectively unbounded ``control_time_s``

The gate still delegates every other key to LeRobot's real events dict, so
``q`` keeps working and the reset loop keeps driving the arm.

Set ``TURN_KEY=0`` (env) to fall back to LeRobot's timer-driven reset.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("so101_turn")

# Turn keys accepted while WAITING. LeRobot binds n/r/q itself, so consuming
# them here during the reset is what makes the phase boundary deterministic.
TURN_KEYS = ("space", "r", "n", "enter")
KEY_ALIASES = {" ": "space", "spacebar": "space"}

# Live state, read by episode_outcome_keys to suppress s/f/d while WAITING.
_waiting = False
_turn = False
_ever_armed = False
_installed = False

HELP = (
    "\n  turn-based:  press SPACE when the duck is placed -> recording starts\n"
    "                s / f  label + end the episode      d  discard it\n"
    "                q      finalize and quit\n"
)


def is_waiting() -> bool:
    """True while the operator is setting up and nothing is being recorded."""
    return _waiting


def _key(name: str) -> str:
    k = (name or "").lower().strip()
    return KEY_ALIASES.get(k, k)


class _ResetGate:
    """Stand-in for LeRobot's events dict during the reset call.

    ``exit_early`` is the only key that decides when a `record_loop` ends, and
    LeRobot sets it from `n`. Here it reads True only after the operator has
    pressed the turn key, and writing False to it (what the loop does just
    before breaking) just consumes our turn flag. Everything else -- crucially
    `stop_recording` -- passes straight through to the real dict, so `q` still
    quits and the loop still drives the arm.
    """

    __slots__ = ("_real",)

    def __init__(self, real: dict):
        self._real = real

    def __getitem__(self, key):
        if key == "exit_early":
            return _turn
        return self._real[key]

    def __setitem__(self, key, value):
        if key == "exit_early":
            global _turn
            _turn = False
            return
        self._real[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key == "exit_early" or key in self._real


def arm(*, quiet: bool = False) -> None:
    """Called on a turn keypress: leave WAITING, let the episode record."""
    global _turn, _waiting
    _turn = True
    _waiting = False
    if not quiet:
        print("\n  [turn] recording", flush=True)


def _wait_for_turn(events: dict, what: str) -> None:
    """Block until the operator presses the turn key.

    Deliberately does not command the robot: the arm holds its pose under
    torque, which is what you want while you are reaching for the duck. The
    reset call reaches its own wait *inside* the original loop, so the arm is
    still being driven there.
    """
    global _turn, _waiting
    _waiting = True
    _turn = False
    if not events.get("stop_recording"):
        print(f"\n  [turn] {what}: place the duck, then press SPACE to record",
              flush=True)
    while not _turn:
        if events.get("stop_recording"):
            _waiting = False
            return
        time.sleep(0.05)


def install() -> bool:
    global _installed
    if os.environ.get("TURN_KEY", "1") in ("0", "false", "False"):
        print("[turn] disabled by TURN_KEY=0 -- using LeRobot's timer reset",
              file=sys.stderr)
        return False

    try:
        import lerobot.scripts.lerobot_record as lr
        from lerobot.utils import keyboard_input
    except ImportError as e:  # noqa: BLE001
        print(f"[turn] cannot import lerobot ({e}); turn-based control disabled",
              file=sys.stderr)
        return False

    _orig_loop = lr.record_loop

    def gated_record_loop(*, dataset=None, events=None, control_time_s=0.0, **kw):
        global _waiting, _turn, _ever_armed
        if dataset is not None:
            # A recording call. Never start writing frames before the operator
            # says the scene is ready -- this is what keeps the first second of
            # an episode from containing a hand and a moving duck.
            if not _ever_armed:
                _wait_for_turn(events, "first episode")
                _ever_armed = True
            _waiting = False
            return _orig_loop(dataset=dataset, events=events,
                              control_time_s=control_time_s, **kw)

        # The reset call: same loop, no dataset, so nothing is written. Make its
        # length operator-controlled instead of a timer, and stop `n` from
        # cutting it short.
        _waiting = True
        _turn = False
        print("\n  [turn] reset: place the duck, press SPACE to start the next "
              "episode", flush=True)
        try:
            return _orig_loop(dataset=None, events=_ResetGate(events),
                              control_time_s=3600.0, **kw)
        finally:
            _waiting = False
            _turn = False

    lr.record_loop = gated_record_loop

    # Intercept the turn key. Installed BEFORE episode_outcome_keys wraps the
    # same hook, so outcome keys are seen first and everything else falls
    # through to here.
    _orig_create = keyboard_input.create_key_listener

    def create_with_turn(dispatch, *, controls_help: str = ""):
        def wrapped(name: str) -> None:
            if _key(name) in TURN_KEYS and _waiting:
                arm()
                return
            dispatch(name)

        return _orig_create(wrapped, controls_help=controls_help)

    keyboard_input.create_key_listener = create_with_turn
    try:
        lr.create_key_listener = create_with_turn
    except AttributeError:
        pass

    # init_keyboard_listener is what hands the real events dict to record();
    # capture it so the gate can be built and the pre-arm wait can poll it.
    _orig_init = keyboard_input.init_keyboard_listener

    def init_capturing():
        listener, ev = _orig_init()
        globals()["_EVENTS"] = ev
        return listener, ev

    keyboard_input.init_keyboard_listener = init_capturing
    try:
        lr.init_keyboard_listener = init_capturing
    except AttributeError:
        pass

    print(HELP, flush=True)
    _installed = True
    logger.info("turn-based recording installed (SPACE arms each episode)")
    return True
