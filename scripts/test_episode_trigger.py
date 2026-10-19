#!/usr/bin/env python3
"""Offline checks for scripts/episode_trigger.py. No hardware, no cameras.

The trigger's whole job is a timing boundary, so most of these checks are about
*when* the patched record_loop returns:

- the reset call must NOT return on its own, and must NOT return because 'n'
  set events["exit_early"] -- only the turn key ends it
- a turn key must release it
- the first recording call must wait for a turn key too
- 'q' must still quit, i.e. the gate has to pass stop_recording through
- s/f/d must be inert while waiting, so a stray keypress mid-reset cannot label
  or discard an episode that was never recorded

LeRobot's real record_loop commands a robot, so it is replaced with a stub that
reproduces only the exit condition under test.

    .venv/bin/python scripts/test_episode_trigger.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def main() -> int:
    import lerobot.scripts.lerobot_record as lr
    from lerobot.utils import keyboard_input as K

    import episode_trigger as T
    import episode_outcome_keys as E

    # --- stub the real loop, keeping only its exit condition ----------------
    calls: list[tuple[bool, float]] = []

    def stub(*, dataset=None, events=None, control_time_s=0.0, **kw):
        calls.append((dataset is not None, control_time_s))
        # Mirrors the real loop: leave on exit_early, else on the timer. Capped
        # so a 3600s reset cannot hang the test.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < min(control_time_s or 0.0, 0.4):
            if events["exit_early"]:
                events["exit_early"] = False
                break
            time.sleep(0.01)
        return "returned"

    real_loop = lr.record_loop
    lr.record_loop = stub

    # --- install, with a spy on create_key_listener -------------------------
    holder: dict = {}

    def capture(dispatch, **kw):
        holder["dispatch"] = dispatch
        return None

    real_create = K.create_key_listener
    K.create_key_listener = capture
    try:
        ok = T.install()
        check(ok, "install() succeeded (with spy in place)")
        if not ok:
            lr.record_loop = real_loop
            return 1
        # Routes through create_with_turn -> capture, giving the inner wrapper.
        _listener, events = K.init_keyboard_listener()
    finally:
        K.create_key_listener = real_create
    dispatch = holder.get("dispatch")
    check(callable(dispatch), "wrapped dispatch registered with create_key_listener")

    # record() passes its own events dict in, so use a plain stand-in.
    ev = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    # --- the gate -----------------------------------------------------------
    print("\n  reset gate:")
    g = T._ResetGate(ev)
    check(g["stop_recording"] is False, "gate passes stop_recording through")
    ev["stop_recording"] = True
    check(g["stop_recording"] is True, "gate sees a later stop_recording (q still quits)")
    ev["stop_recording"] = False

    ev["exit_early"] = True          # what LeRobot's 'n' does
    check(g["exit_early"] is False,
          "gate ignores events['exit_early'] set by 'n' -- the reset cannot be cut short")
    ev["exit_early"] = False

    T.arm(quiet=True)
    check(g["exit_early"] is True, "gate reports exit_early after the turn key")
    g["exit_early"] = False           # the loop consumes it
    check(g["exit_early"] is False, "consuming exit_early clears the turn flag")

    ev["rerecord_episode"] = True
    check(g["rerecord_episode"] is True, "gate passes rerecord_episode through")
    ev["rerecord_episode"] = False

    # --- the reset call blocks until the turn key ---------------------------
    print("\n  reset call is operator-controlled:")
    done = threading.Event()

    def run_reset():
        lr.record_loop(dataset=None, events=ev, control_time_s=60.0)
        done.set()

    T._waiting, T._ever_armed = False, True
    t = threading.Thread(target=run_reset, daemon=True)
    t.start()
    blocked = not done.wait(0.25)
    check(blocked, "reset call does not return on its own")
    check(T.is_waiting(), "is_waiting() is True during the reset")

    ev["exit_early"] = True          # a stray 'n' mid-reset
    time.sleep(0.15)
    check(not done.is_set(), "a stray 'n' still does not end the reset")
    ev["exit_early"] = False

    T.arm(quiet=True)
    check(done.wait(2.0), "turn key ends the reset")
    check(not T.is_waiting(), "is_waiting() is False once armed")
    check(calls[-1][0] is False and calls[-1][1] > 60.0,
          f"reset got an unbounded control_time_s (got {calls[-1][1]})")

    # --- the first recording call waits too ---------------------------------
    print("\n  first recording call waits for a turn key:")
    T._ever_armed = False
    rec_done = threading.Event()

    def run_record():
        lr.record_loop(dataset=object(), events=ev, control_time_s=1.0)
        rec_done.set()

    t = threading.Thread(target=run_record, daemon=True)
    t.start()
    check(not rec_done.wait(0.25), "recording call blocks before the operator is ready")
    check(T.is_waiting(), "is_waiting() is True while pre-arming")
    T.arm(quiet=True)
    check(rec_done.wait(2.0), "turn key starts the recording")
    check(calls[-1][0] is True, "the recording call really got a dataset")

    # subsequent recordings must NOT block: the reset already gated them
    rec_done.clear()
    t = threading.Thread(target=run_record, daemon=True)
    t.start()
    check(rec_done.wait(1.5), "later episodes start immediately (the reset gated them)")

    # --- s/f/d are inert while waiting --------------------------------------
    print("\n  outcome keys are inert while resetting:")
    T._waiting = True
    E._PENDING.clear()
    for key, field in (("s", "outcome"), ("f", "outcome"), ("d", "rerecord_episode")):
        ev["exit_early"] = False
        ev["rerecord_episode"] = False
        dispatch(key)
        check(E._PENDING.get("outcome") is None and ev["rerecord_episode"] is False,
              f"'{key}' does nothing while waiting")
        check(ev["exit_early"] is False, f"'{key}' does not end anything while waiting")

    # zone digits MUST still work while waiting -- that is when you set them
    E.set_zone(6, quiet=True)
    check(E._PENDING_ZONE == 6, "zone digits still work while waiting")
    E.set_zone(None, quiet=True)

    # turn key reaches the trigger through the same dispatch
    T._waiting = True
    T._turn = False
    dispatch("space")
    check(T._turn is True, "SPACE arms the episode through the real key path")
    T._waiting = False
    T._turn = False

    # and outside the reset, SPACE must reach LeRobot untouched
    T._waiting = False
    T._turn = False
    dispatch("space")
    check(T._turn is False, "SPACE outside the reset is not swallowed by the trigger")

    lr.record_loop = real_loop
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
