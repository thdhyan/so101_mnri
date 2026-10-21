# Review instructions for this repo

These are the rules a reviewer has to enforce here. They are not style
preferences -- most of them are bugs that have already happened on this rig.

Read `AGENT_TASKS.md`, `CONTRIBUTING.md` and `robots/so101/README.md` for the
full picture. This file is the short list that a code-review bot should check
every time.

## Block these outright

**Never recalibrate the arms.** `robots/so101/calibration/mnri_arms_1_*.json`
are the verified calibration. `scripts/calibrate_arm.py` overwrites the file
**and the arm's EEPROM**. The arms are compared in degrees derived from those
files, so a new zero point makes the pose gate block every session or let a
99-degree mismatch through -- and `record_loop` calls `send_action()` on its
first tick, so six servos lunge against gravity, the rail sags and the
Feetech bus drops packets. It presents as a flaky cable, not as a bug.
If a joint really is at a limit it should not be at, **read** it with
`scripts/check_joint_range.py --arm <arm> --compare` and correct
`range_min`/`range_max` by hand.

A diff that touches `robots/so101/calibration/*.json` needs an explicit
"this is not a recalibration" explanation, or it is a bug.

**Never hardcode `/dev/ttyACM*`.** Enumeration order flips when the arms are
replugged or moved to another USB port, and on 2026-09-25 that wrote the
follower's calibration into the leader's servos. Resolve arms by USB serial
(`scripts/resolve_arm_ports.sh`).

**Never duplicate meshes or robot models.** They are canonical under
`robots/so101/`. Both envs `<include>` the shared files and set `meshdir`
relative to their own `scene.xml`, so a second copy under
`envs/mujoco/*/assets/` breaks the include path.

**Never hand-edit the Isaac USD.** `envs/isaac/assets/` is generated from
`robots/so101/so101.urdf` with `python -m envs.isaac.convert_urdf`.

**Never commit datasets, checkpoints, `data/`, `.venv/` or secrets**
(`docker/.env`).

## The names are a contract

Joint, actuator and site names are consumed by `scripts/`, `envs/isaac/` and
the real-robot code. Renaming one without updating every consumer breaks
things that have nothing to do with the rename.

- joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper` (optionally `left_`/`right_` prefixed)
- sites: `gripperframe`, `left_gripperframe`, `right_gripperframe`

`tests/test_env_contract.py` checks this. Prefer discovering cameras and
joints dynamically (`model.ncam`, `model.joint(name)`) over hardcoding lists.

## The recording path has rules too

- The dataset is a **stock LeRobot v3.0** dataset written by LeRobot's own
  `lerobot_record`. No custom writer, no custom format.
- `episode_outcome_keys._EVENTS` must **be** LeRobot's `events` dict, never a
  copy. With a copy, `d` and the `--profile` stop silently do nothing.
- A re-record (`clear_episode_buffer`) must clear any pending label, or it
  lands on the retake.
- `--profile` writes to `<REPO_ID>_profile`, never to the real dataset.
- LeRobot writes `meta/episodes/` only every 10 episodes or in `finalize()`.
  A killed run leaves an unloadable dataset -- quit with `q`, never Ctrl+C
  twice.
- Keys come from LeRobot's terminal reader, so the **recording terminal must
  have focus**.

## House style

- Hardware and dataset settings go in `scripts/so101_env.sh`, not in scripts.
- Python 3.12, the repo `.venv`; `MUJOCO_GL=egl` for headless rendering;
  Isaac needs `OMNI_KIT_ACCEPT_EULA=YES`.
- Type hints where the file around them uses them; match the surrounding
  style rather than imposing one.
- One feature per PR. Conventional Commits (`feat(sim):`, `fix(real):`,
  `docs:`). Branch off `main` as `feat/<topic>`, `fix/<topic>`,
  `docs/<topic>`.
- A PR says what changed, why, and how it was tested -- and which checks were
  run on the real arm, if it needs hardware.

## What CI checks

`lint` (ruff, errors only, plus a compile check), `unit` (pytest over
`tests/`), `sim-smoke` (the MuJoCo envs build, render and step), and
`recording` (the four recording-patch scripts, when that code changes).

If a PR changes a sim env, `sim-smoke` has to pass. If it changes the
recording path, the `recording` scripts have to pass. If it touches the
calibration, `tests/test_calibration_guard.py` will say so.
