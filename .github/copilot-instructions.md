# SO-101 MNRI — instructions for Copilot

@.github/review-instructions.md

This repository is MuJoCo + Isaac Sim environments, RL/IL training and
real-robot data collection for the SO-101 arm. Before reviewing a change,
read `AGENT_TASKS.md` and `CONTRIBUTING.md`.

The rules that matter most, in case the import above is not resolved:

- **Never recalibrate the arms.** `robots/so101/calibration/*.json` are
  verified; `calibrate_arm.py` overwrites the file *and the arm's EEPROM*, and
  a new zero point makes the pose gate fail and the servos lunge against
  gravity.
- **Never hardcode `/dev/ttyACM*`** — resolve arms by USB serial.
- **Never duplicate meshes** — they are canonical under `robots/so101/`, and
  both envs `<include>` them with a relative `meshdir`.
- **Never hand-edit the Isaac USD** — it is generated from the URDF.
- Joint and site names are a stable contract across `scripts/`, `envs/` and
  the real-robot code.
- The recording path writes a **stock LeRobot v3.0** dataset; no custom
  writer, no custom format.
- Never commit `data/`, checkpoints, `.venv/` or `docker/.env`.
