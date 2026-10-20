#!/usr/bin/env python3
"""Per-episode success/failure labels + retry, for SO-101 recording.

Wires into LeRobot's real keyboard pipeline rather than prompting, because the
key listener already owns stdin -- an ``input()`` between episodes would fight
it for the TTY.

LeRobot routes every key from both backends (pynput and the terminal reader)
through one ``dispatch`` callback passed to ``create_key_listener``. This wraps
that callback so:

    s   label the current episode success and end it (or its reset) now
    f   label it failure and end it (or its reset) now
    d   discard the in-progress episode and re-record it (alias of LeRobot's
        own left-arrow / 'r' re-record)
    n / right   end the episode early and keep it
    r / left    discard and re-record
    q / esc     stop

Duck start zone, which is what actually fixes state coverage:

    1..9, 0   set the start zone for the NEXT episode to be saved; 0 = zone 10
    z         clear the zone

A zone is chosen *before* the take, while the duck is being placed, and is
attached to the episode that follows. Unlike the outcome it deliberately
SURVIVES a ``d`` re-record: the duck is still sitting in the same zone, so the
retake is the same condition and must be logged as such. Clearing it there
would silently mark every retake as unzoned.

The label and the zone are attached in a ``save_episode`` wrapper, i.e. after
LeRobot has committed the episode, so the index is final and correct. Labels go
to ``meta/outcomes.json`` (see scripts/episode_outcomes.py) -- the dataset
schema is untouched, so stock LeRobot can still train on it.

Failed episodes stay on disk; filter at train time. A discarded ('d') take is
never saved -- LeRobot drops its buffer and re-records.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_outcomes  # noqa: E402
import episode_trigger  # noqa: E402


def _waiting() -> bool:
    """True while episode_trigger is holding the session in the reset phase.

    Imported lazily-tolerant: episode_trigger may not be installed (e.g. the
    offline tests, or TURN_KEY=0), in which case there is never a wait phase
    and the outcome keys behave exactly as before.
    """
    return episode_trigger.is_waiting()

logger = logging.getLogger("so101_outcomes")

# events dict, captured from init_keyboard_listener
_EVENTS: dict = {}
# outcome chosen since the last save
_PENDING: dict[str, str] = {}
# Duck start zone for the next saved episode. Deliberately NOT cleared on a
# re-record, unlike the outcome -- see the module docstring.
_PENDING_ZONE: int | None = None
# Prompt (task) selected for the next saved episode, and its 1-based index in
# TASK_PROMPTS. Also survives a re-record: the same task is being retried.
_PENDING_TASK: str | None = None
_PENDING_TASK_ID: int | None = None
# consecutive episodes taken from the same zone, to nudge for coverage
_ZONE_RUN = 0
_ZONE_RUN_LAST: int | None = None

# How many takes from one zone in a row before nagging. The whole point of
# marking zones is that 6 episodes from one duck placement is not a dataset, so
# the nudge fires while it can still change what you do next.
ZONE_RUN_WARN = 3

# Prompt selector: digit -> index into TASK_PROMPTS. Digits, because the zone
# keys moved to letters. TASK_PROMPTS comes from scripts/so101_env.sh.
PROMPT_KEYS = {str(i): i for i in range(1, 10)}

# Zone selector: letter -> zone number. Populated from ZONE_KEYS in
# scripts/so101_env.sh (default i/o/k/l -> zones 1-4, the taped 2x2 grid).
ZONE_KEYS: dict[str, int] = {}
DEFAULT_ZONE_KEYS = "i o k l"
DEFAULT_TASKS = "pick the duck|push the duck into the square tape"

HELP = ""


def load_config(env_file: Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Read TASK_PROMPTS and ZONE_KEYS out of scripts/so101_env.sh.

    Parsed rather than shelled out to, so this cannot disagree with the
    recording run. Env vars win, which is what the offline tests use.
    """
    env = {}
    path = env_file or (Path(__file__).resolve().parent / "so101_env.sh")
    if path.is_file():
        for line in path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            env[k.strip()] = v.split("#", 1)[0].strip().strip("\"'")

    raw_tasks = os.environ.get("TASK_PROMPTS", env.get("TASK_PROMPTS", DEFAULT_TASKS))
    tasks = [t.strip() for t in raw_tasks.split("|") if t.strip()]

    raw_zones = os.environ.get("ZONE_KEYS", env.get("ZONE_KEYS", DEFAULT_ZONE_KEYS))
    keys = [k.strip().lower() for k in raw_zones.replace(",", " ").split() if k.strip()]
    zone_keys = {k: i for i, k in enumerate(keys, 1)}

    # Both maps are module globals because the key handler reads them by name.
    # Assigning them here (not just returning) is what makes the handler see the
    # configured layout.
    global PROMPT_KEYS, ZONE_KEYS
    PROMPT_KEYS = {str(i): i for i in range(1, len(tasks) + 1)}
    ZONE_KEYS = zone_keys
    return tasks, zone_keys


def tasks() -> list[str]:
    """The prompt list, index 0 = key 1."""
    return load_config()[0]


def current_task() -> str | None:
    """Prompt for the next episode, or None to fall back to RECORD_TASK."""
    return _PENDING_TASK


def current_task_id() -> int | None:
    return _PENDING_TASK_ID


def current_zone() -> int | None:
    return _PENDING_ZONE


def set_task(task_id: int | None, *, quiet: bool = False) -> None:
    """Choose the prompt for the next episode. None clears it."""
    global _PENDING_TASK, _PENDING_TASK_ID
    if task_id is None:
        _PENDING_TASK, _PENDING_TASK_ID = None, None
        if not quiet:
            print("\n  [prompt] cleared -> falls back to the default task", flush=True)
        return
    tl = tasks()
    if not 1 <= task_id <= len(tl):
        print(f"\n  [prompt] no prompt {task_id} (have {len(tl)})", file=sys.stderr)
        return
    _PENDING_TASK, _PENDING_TASK_ID = tl[task_id - 1], task_id
    if not quiet:
        print(f"\n  [prompt] {task_id}: {_PENDING_TASK}", flush=True)


def set_zone(zone: int | None, *, quiet: bool = False) -> None:
    """Set (or clear) the start zone for the next episode to be saved."""
    global _PENDING_ZONE, _ZONE_RUN, _ZONE_RUN_LAST

    _PENDING_ZONE = None if zone is None else int(zone)
    if _PENDING_ZONE is None:
        _ZONE_RUN, _ZONE_RUN_LAST = 0, None
        if not quiet:
            print("\n  [zone] cleared", flush=True)
        return

    if _ZONE_RUN_LAST == _PENDING_ZONE:
        _ZONE_RUN += 1
    else:
        _ZONE_RUN, _ZONE_RUN_LAST = 1, _PENDING_ZONE

    if not quiet:
        print(f"\n  [zone] {_PENDING_ZONE} -> next episode", flush=True)
    if _ZONE_RUN == ZONE_RUN_WARN:
        keys = load_config()[1]
        others = [k for k, z in keys.items() if z != _PENDING_ZONE]
        if others:
            print(f"  [zone] {_ZONE_RUN} takes in a row from zone {_PENDING_ZONE}. "
                  f"Cycled elsewhere yet? e.g. {', '.join(others[:5])}", flush=True)


def _print_help() -> None:
    tl, zk = load_config()
    zlist = ", ".join(f"{k}={v}" for k, v in zk.items())
    print("\n  --- prompts (press before the take; applies to the next episode) ---",
          flush=True)
    for i, t in enumerate(tl, 1):
        print(f"    {i}  {t}", flush=True)
    print(f"\n  --- duck start zone: {zlist}  (z clears) ---", flush=True)
    print("  s=success+end  f=failure+end  d=discard  q=quit"
          "   SPACE=start recording\n", flush=True)


def install() -> bool:
    # Load the key layout before anything can dispatch a key, so ZONE_KEYS and
    # PROMPT_KEYS are populated even if the help banner never prints.
    load_config()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils import keyboard_input
    except ImportError as e:  # noqa: BLE001
        print(f"[outcomes] cannot import lerobot ({e}); labels disabled", file=sys.stderr)
        return False

    # 1. Capture the events dict that LeRobot creates for the key listener.
    _orig_init = keyboard_input.init_keyboard_listener

    def init_capturing():
        global _EVENTS
        listener, events = _orig_init()
        # Alias LeRobot's dict, never copy it: record_loop reads THIS object, so
        # a copy would swallow 'd' (rerecord/exit_early) and the profile stop.
        _EVENTS = events
        # LeRobot's events dict has no outcome slot; add ours.
        _EVENTS.setdefault("episode_outcome", None)
        _print_help()
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
            # While the operator is resetting (scripts/episode_trigger.py), only
            # the turn key and the zone digits mean anything. Letting s/f/d
            # through here would label or discard an episode that has not been
            # recorded yet, and 'd' would also set rerecord_episode, which makes
            # record() throw the next episode away.
            if key in ("s", "f", "d", "r") and _waiting():
                print(f"\n  [outcome] '{key}' ignored while resetting -- "
                      f"press SPACE to start recording", flush=True)
                return
            # Digits pick the prompt, letters pick the start zone. LeRobot's own
            # listener binds only n/r/q/arrows/esc and ignores everything else,
            # so both sets are free.
            if key in PROMPT_KEYS:
                set_task(PROMPT_KEYS[key])
                return
            if key in ZONE_KEYS:
                set_zone(ZONE_KEYS[key])
                return
            if key == "z":
                set_zone(None)
                return
            if key == "s":
                _PENDING["outcome"] = "success"
                _EVENTS["episode_outcome"] = "success"
                _EVENTS["exit_early"] = True
                print("\n  [outcome] success", flush=True)
                return
            if key == "f":
                _PENDING["outcome"] = "failure"
                _EVENTS["episode_outcome"] = "failure"
                _EVENTS["exit_early"] = True
                print("\n  [outcome] failure", flush=True)
                return
            if key == "d":
                # Same as LeRobot's own left/'r': drop this take, try again.
                # The take is never saved, so there is nothing to label; any
                # s/f already pressed is cleared with the buffer (see step 4).
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
        frames_before = int(getattr(self, "num_frames", 0))
        result = _orig_save(self, *a, **kw)
        outcome = _PENDING.pop("outcome", None) or _EVENTS.get("episode_outcome")
        _EVENTS["episode_outcome"] = None
        # The zone and prompt belong to the episode just committed. Read them
        # BEFORE clearing. A 'd' re-record never reaches here (the take is
        # dropped, not saved) and does not clear them: the retake is the same
        # duck placement attempting the same task.
        global _PENDING_ZONE, _PENDING_TASK, _PENDING_TASK_ID
        zone, task, task_id = _PENDING_ZONE, _PENDING_TASK, _PENDING_TASK_ID
        if outcome is None and zone is None and task is None:
            print("\n  [outcome] unlabeled (label with s/f DURING an episode or its reset; "
                  "a key pressed now applies to the next episode)", flush=True)
            return result
        idx = int(getattr(self, "num_episodes", 1)) - 1
        # record() requires a valid outcome, so an episode marked with only a
        # zone/prompt still needs one: treat it as unlabeled.
        try:
            episode_outcomes.record(
                str(self.root), idx, outcome or "unlabeled",
                # num_frames is the dataset total; the delta is this episode.
                n_frames=int(getattr(self, "num_frames", 0)) - frames_before,
                zone=zone,
                task=task,
                task_id=task_id,
            )
        except ValueError as e:
            print(f"[outcomes] not writing episode {idx}: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[outcomes] could not write label for episode {idx}: {e}", file=sys.stderr)
        else:
            bits = []
            if outcome is not None:
                bits.append(outcome)
            if task_id is not None:
                bits.append(f"prompt {task_id}")
            if zone is not None:
                bits.append(f"zone {zone}")
            print(f"  [outcome] episode {idx} -> {', '.join(bits) or 'unlabeled'}", flush=True)
        _PENDING_ZONE = None
        _PENDING_TASK = None
        _PENDING_TASK_ID = None
        return result

    LeRobotDataset.save_episode = save_and_label

    # 4. A re-recorded take is thrown away, so is any label pressed during it;
    #    otherwise it would land on the retake.
    _orig_clear = LeRobotDataset.clear_episode_buffer

    def clear_and_unlabel(self, *a, **kw):
        _PENDING.pop("outcome", None)
        _EVENTS["episode_outcome"] = None
        return _orig_clear(self, *a, **kw)

    LeRobotDataset.clear_episode_buffer = clear_and_unlabel
    logger.info("episode outcome controls installed (s/f/d)")
    return True
