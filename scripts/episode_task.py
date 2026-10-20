#!/usr/bin/env python3
"""Put the operator-selected prompt into the dataset as the per-frame `task`.

LeRobot's `record_loop` hardcodes the prompt for every frame:

    frame = {**observation_frame, **action_frame, "task": single_task}

where `single_task` is `--dataset.single_task`, one string for the whole run.
That is fine for a single-task dataset and wrong for this one, which has five.

LeRobot v3.0 supports per-frame task strings natively, so no schema patch is
needed. `DatasetWriter.add_frame` appends `frame["task"]` to the episode
buffer; `save_episode` then does `episode_tasks = list(set(tasks))`,
`save_episode_tasks(...)` and a per-frame `task_index`. `meta/tasks.parquet`
accumulates the distinct prompts. The result is a stock multi-task
LeRobotDataset that stock LeRobot trains on unchanged, and where
`meta/tasks.parquet` plus the `task_index` column say which prompt each episode
demonstrated.

So this wraps `add_frame` and overwrites `frame["task"]` with the prompt chosen
by the number keys (scripts/episode_outcome_keys.py). It writes the *same*
string for every frame of an episode, because `save_episode` takes the set of
tasks in the episode and a mixed episode would be ambiguous about what it
demonstrated.

When no prompt key was pressed, `task` is left exactly as LeRobot set it --
`--dataset.single_task` from scripts/so101_env.sh -- so the fallback path is
byte-identical to stock behaviour and the first 12 episodes already on disk
stay consistent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_outcome_keys as keys  # noqa: E402

logger = logging.getLogger("so101_task")

_ORIGINAL = None
# Guard so a mid-episode keypress cannot change the prompt halfway through and
# produce an episode with two task strings.
_frozen: str | None = None


def _reset() -> None:
    global _frozen
    _frozen = None


def current() -> str | None:
    """The prompt that would be written into the next frame."""
    return _frozen if _frozen is not None else keys.current_task()


def install() -> bool:
    global _ORIGINAL

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:  # noqa: BLE001
        print(f"[task] cannot import lerobot ({e}); per-episode prompts disabled",
              file=sys.stderr)
        return False

    if _ORIGINAL is not None:
        return True

    original = LeRobotDataset.add_frame

    def add_frame_with_task(self, frame: dict):
        global _frozen
        chosen = keys.current_task()
        if chosen is not None:
            # Freeze on the first frame of the episode so a keypress during the
            # take cannot split it across two prompts.
            if _frozen is None:
                _frozen = chosen
            frame["task"] = _frozen
        return original(self, frame)

    add_frame_with_task.__wrapped__ = original
    LeRobotDataset.add_frame = add_frame_with_task
    _ORIGINAL = original

    # Unfreeze when the episode buffer is cleared or saved, i.e. at the episode
    # boundary. clear_episode_buffer is also what a 'd' re-record calls, so the
    # retake gets to choose its prompt again -- same as the zone.
    _orig_save = LeRobotDataset.save_episode
    _orig_clear = LeRobotDataset.clear_episode_buffer

    def save_episode(self, *a, **kw):
        try:
            return _orig_save(self, *a, **kw)
        finally:
            _reset()

    def clear_episode_buffer(self, *a, **kw):
        try:
            return _orig_clear(self, *a, **kw)
        finally:
            _reset()

    LeRobotDataset.save_episode = save_episode
    LeRobotDataset.clear_episode_buffer = clear_episode_buffer

    n = len(keys.tasks())
    logger.info("per-episode prompts enabled (%d configured)", n)
    return True
