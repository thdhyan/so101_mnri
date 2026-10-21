# Contributing

Thanks for helping with SO-101 MNRI. This repo has MuJoCo and Isaac envs, RL,
imitation learning, teleop and real-robot data collection for the SO-101 arm.

## Setup

```bash
uv sync                      # creates .venv
.venv/bin/python -c "import mujoco"
```

## Workflow

1. Branch off `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
2. Keep PRs small: **one feature per PR**. Do not mix sim envs, RL, teleop,
   IL and data-collection changes in one PR.
3. Use [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(sim): ...`, `fix(real): ...`, `docs: ...`.
4. In the PR description, say what changed, why, and how you tested it.
   If the change needs hardware, say which checks you ran on the real arm.
5. Never commit datasets, checkpoints, `data/`, `.venv/`, or secrets
   (`docker/.env`).

## Code conventions

- Match the style of the surrounding code. Python 3, type hints where the file
  already uses them.
- Meshes and robot models live only under `robots/so101/`. Do not copy them.
- Hardware and dataset settings go in `scripts/so101_env.sh`, not in scripts.
- Never hardcode `/dev/ttyACM*`. Resolve arms by USB serial
  (`scripts/resolve_arm_ports.sh`); enumeration order flips.
- Real-robot recording writes a stock LeRobot v3.0 dataset. Do not add a custom
  writer or format.

## Real-robot safety rules

- **Never recalibrate the arms.** `robots/so101/calibration/mnri_arms_1_*.json`
  are the verified calibration. If a joint reads wrong, inspect it with
  `scripts/check_joint_range.py --arm <arm> --compare` and fix
  `range_min`/`range_max` by hand.
- Quit recording with `q`, never with Ctrl+C twice, or the dataset is left
  unloadable.

## Checks before opening a PR

No hardware needed:

```bash
.venv/bin/python scripts/test_episode_outcomes.py
.venv/bin/python scripts/test_episode_task.py
.venv/bin/python scripts/test_episode_trigger.py
.venv/bin/python scripts/test_camera_flip.py
bash scripts/record_so101.sh --dry-run
```

Run the ones that touch your change, and any env smoke test for sim changes.

## CI

GitHub Actions runs on every PR:

| job | what it does |
|---|---|
| `lint` | `ruff check .` (errors only) and a compile check on every script |
| `unit` | `pytest tests/` — the calibration guard, the env name contract, the teleop regression |
| `sim-smoke` | `verify_manual.py --headless-check` for the single- and dual-arm envs |
| `recording` | the four recording-patch scripts above — only when they change, because LeRobot is a 1.7 GB install |

Install what CI installs with `pip install -r requirements-dev.txt`.

`tests/test_calibration_guard.py` is the tripwire for the safety rule below: if
anyone runs `calibrate_arm.py`, CI goes red.

## License

By contributing you agree that your contributions are licensed under the
[CC BY-NC 4.0](LICENSE) license of this repository.
