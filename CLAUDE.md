# CLAUDE.md

SO-101 arm: MuJoCo + Isaac envs, RL, IL, and real-robot data collection. Start
with [HANDOFF.md](HANDOFF.md) and [AGENT_TASKS.md](AGENT_TASKS.md). Python is
the uv venv at `.venv` (`.venv/bin/python`).

## Real-robot recording (`scripts/record_so101.sh`)

Path: `record_so101.sh` (config from `scripts/so101_env.sh`, arms resolved by
USB serial, read-only pose gate) → `record_so101.py` (installs patches) →
stock `lerobot.scripts.lerobot_record.main()`. Output is a stock LeRobot
**v3.0** dataset; do not add a custom writer or format.

Patches, all monkeypatches installed before LeRobot runs:
- `so101_torque_guard.py`: syncs `Goal_Position := Present_Position` before
  torque-on and enables the motors one at a time with retries. Only the
  follower enables torque.
- `episode_outcome_keys.py`: `s`/`f` label the episode and end it, `d` discards
  and re-records. Labels go to the `meta/outcomes.json` sidecar
  (`episode_outcomes.py`).
- `record_loop_profile.py`: only used with `--profile`.

Invariants (each one is a bug that has already happened):
- `episode_outcome_keys._EVENTS` must BE LeRobot's `events` dict, never a
  copy. With a copy, `d` and the `--profile` stop silently do nothing.
- A re-record (`clear_episode_buffer`) must clear any pending label, or it
  lands on the retake.
- LeRobot's `num_frames` is the dataset total. An episode's length is the
  change across `save_episode`.
- `--profile` stops mid-episode and LeRobot saves that take, so profiling
  writes to `<REPO_ID>_profile`, never to the real dataset.
- LeRobot writes `meta/episodes/` only every 10 episodes or in `finalize()`.
  A killed run leaves an unloadable dataset: `info.json` says N episodes,
  `meta/episodes/` is missing, and the parquet has no footer. Resume then
  fails with a misleading Hub 404. Quit with `q`; never Ctrl+C twice.
- Keys come from LeRobot's terminal reader (pynput is not installed), so the
  recording terminal must have focus.
- The dataset fps and camera width are baked into `meta/info.json`. The
  launcher refuses to resume with different settings.

Checks, no hardware needed:
```bash
.venv/bin/python scripts/test_episode_outcomes.py   # outcome keys + label file
bash scripts/record_so101.sh --dry-run              # argv accepted by LeRobot
```

Full guide: [docs/SO101_RECORD_SETUP.md](docs/SO101_RECORD_SETUP.md).

## Conventions

- Hardware and dataset settings go in `scripts/so101_env.sh`, not the scripts.
- Never hardcode `/dev/ttyACM*`. Enumeration order flips, and a wrong path has
  already written one arm's calibration into the other.
- Meshes and robot models live only under `robots/so101/`.
