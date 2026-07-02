# Agent tasks & plans

Open work for agents picking this repo up. Each task lists what to build,
where it goes, and how to validate it before calling it done. Read
`README.md` and `ENVS.md` first for the current state.

## Ground rules

- Robot meshes/kinematics/URDFs are canonical in `robots/so101/`. Never add
  a second copy of an STL or MJCF robot body under `envs/mujoco/*/assets/`
  — both envs `<include>` the shared files and set `meshdir` relative to
  their own `scene.xml` location (see "MuJoCo path resolution note" in
  `robots/so101/README.md` — this is a real footgun, read it before editing
  any XML with `<include>`).
- Joint/actuator/site names are a stable contract other scripts depend on:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
  (optionally `left_`/`right_` prefixed), sites `gripperframe` /
  `left_gripperframe` / `right_gripperframe`. Don't rename without updating
  every consumer (`env.py`, `scripts/*.py`).
- Discover cameras/joints dynamically (`model.ncam`, `model.joint(name)`,
  etc.) in new scripts rather than hardcoding lists — scenes gain cameras
  over time.
- Validate with `/home/thakk100/miniconda3/envs/so101/bin/python` (conda env
  `so101`), `MUJOCO_GL=egl` for headless rendering.
- Before finishing any task: run `scripts/verify_manual.py --env single
  --headless-check` and `--env dual --headless-check`, both must exit 0.
  Same for the 4 task envs: `--env pick_lift`, `--env pick_place`,
  `--env cyl_grasp`, `--env cyl_reach`.

## Open tasks

### 1. Isaac Lab port — `envs/isaac/`

Currently an empty placeholder (`.gitkeep`). Port the single-arm and
dual-arm envs to Isaac Lab, mirroring `envs/mujoco/*/env.py`'s observation
space, action space, task logic (`push`/`pull`/`none`), and camera setup.
Reuse `robots/so101/so101.urdf` / `so101_dual.urdf` as the USD-conversion
source (Isaac Lab's URDF importer) rather than hand-authoring a new model.

**Validate:** obs/action space shapes match the MuJoCo envs exactly (so
downstream training code doesn't need per-backend branches); render at
least one camera and confirm it's non-blank; reset/step run without error
for both `push` and `none` tasks.

### 2. mujoco-ar-viewer integration

Wire [Improbable-AI/mujoco-ar-viewer](https://github.com/Improbable-AI/mujoco-ar-viewer)
as a new driver for `scripts/teleop_gamepad_ik.py`'s `IKTeleop` class. The
class already exposes `set_ee_target(pos, quat=None)` / `set_gripper(openness)`
decoupled from the gamepad-reading loop — build an AR-viewer-based reader
that calls those methods per frame instead of polling `pygame` joystick
axes, likely as `scripts/teleop_ar_viewer.py` following the same structure
as `teleop_gamepad_ik.py` (`build_ik()` helper, `--dry-run` headless mode).

**Validate:** `--dry-run` mode runs headless (no AR device) with a scripted
target trajectory, same pattern as the existing `--dry-run` flags; final
EE-tracking error should be small (existing gamepad script achieves <0.1 cm
on a circular trajectory — use that as a rough bar).

### 3. Data collection / episode recording

No script currently saves rollouts to disk (images + joint states + actions)
in a training-ready format (e.g. LeRobot dataset format, given the `env.py`
docstrings mention EnvHub compatibility). Add a recording wrapper around
`teleop_gamepad_ik.py` / `teleop_leader_arm.py` that logs each control-loop
tick's `obs`, `action`, `reward` to disk, keyed by episode.

**Validate:** run `--dry-run` teleop with recording enabled, confirm output
files exist, load them back and check shapes/dtypes match `obs_space`.

### 4. Isaac / conda environment for `envs/isaac`

Once task 1 needs it: check existing conda envs (`isaac`, `env_isaaclab`
already exist per `conda env list`) for compatibility before creating a new
one — ask the user which to use, same as was done for the MuJoCo `so101`
env, don't assume.

## Known rough edges (not blocking, worth fixing opportunistically)

- `envs/mujoco/so101_single_arm/README.md` and `so101_dual_arm/README.md`
  still show the old `so101_single_arm_env.env` / `so101_dual_arm_env.env`
  import path from before the `envs/mujoco/` reorg — update to
  `envs.mujoco.so101_single_arm.env` when next touching those files.
- Root-level `view_single_arm.py` / `view_dual_arm.py` (matplotlib-based,
  hardcoded camera lists) are now superseded by `scripts/verify_manual.py`
  and `scripts/view_cameras.py` (dynamic camera discovery, OpenCV). Not
  removed since they may still be in someone's muscle memory — fine to
  delete once confirmed unused.
- `bi_arm_clean_toytable_teleop.py`, `capture_images.py`, `tune_cameras.py`,
  `test` at repo root predate this reorg and haven't been audited against
  the new `robots/so101/` layout — check they still import correctly before
  relying on them. `bi_arm_clean_toytable_teleop.py` also imports from
  `lerobot`, an external dependency not in this repo's requirements.
- `mujoco.mju_mat2Quat` requires float64 in/out buffers on the installed
  mujoco version — `envs/mujoco/so101_single_arm_pick_lift/env.py` and
  `so101_single_arm_pick_place/env.py`'s `_get_obs()` compute a float64
  scratch array then cast to float32 for `tcp_quat`. Follow that pattern if
  you add similar mat→quat conversions elsewhere; a bare float32 buffer
  raises `TypeError` at runtime.
