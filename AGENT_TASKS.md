# Agent tasks & plans

Open work for agents picking this repo up. **Start at HANDOFF.md** (current
state, verified commands, invariants) and `test.md` (failure archive) before
working in the venv/Isaac/camera areas. `README.md` + `ENVS.md` + `MDP.md`
cover the envs.

## Ground rules

- Robot meshes/kinematics/URDFs are canonical in `robots/so101/`. Never add
  a second copy of an STL or MJCF robot body under `envs/mujoco/*/assets/`
  — both envs `<include>` the shared files and set `meshdir` relative to
  their own `scene.xml` location (see "MuJoCo path resolution note" in
  `robots/so101/README.md` — this is a real footgun, read it before editing
  any XML with `<include>`).
- The Isaac USD (`envs/isaac/assets/`) is GENERATED from
  `robots/so101/so101.urdf` — regenerate with
  `python -m envs.isaac.convert_urdf`, never hand-edit.
- Joint/actuator/site names are a stable contract other scripts depend on:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
  (optionally `left_`/`right_` prefixed), sites `gripperframe` /
  `left_gripperframe` / `right_gripperframe`. Don't rename without updating
  every consumer (`env.py`, `scripts/*.py`, Isaac cfgs).
- Discover cameras/joints dynamically (`model.ncam`, `model.joint(name)`,
  etc.) in new scripts rather than hardcoding lists — scenes gain cameras
  over time.
- Validate with the repo `.venv` (python 3.12): `source .venv/bin/activate`.
  `MUJOCO_GL=egl` for headless MuJoCo rendering. Isaac needs
  `OMNI_KIT_ACCEPT_EULA=YES` (in `.envrc`) and free RAM/GPU (see test.md —
  silent deaths under resource pressure).
- Before finishing MuJoCo-side tasks: `scripts/verify_manual.py
  --headless-check` for every touched env, plus
  `scripts/validate_actions.py`. Isaac-side: `python -m
  envs.isaac.scripts.validate_actions`.

## Open tasks

### 1. Full RL training runs (both backends)

Smoke tests pass (3 iterations + checkpoint on isaaclab-skrl and
mujoco-skrl); no real training has run. Suggested first target:
`--backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs
4096` (default 1500 iterations), then rsl_rl, then the dual-arm cylinder
tasks. Tune actuator gains (stiffness 100 / damping 2.5 first guess) if the
arm sags or oscillates.

### 2. Image-based RL (user requirement)

Wire CNN encoders (ResNet-style, CLIP/FiLM, or YOLO-detector features) into
skrl/rsl_rl policies consuming camera obs. Isaac: `CameraCfg` sensors already
render (`env.scene["wrist_cam"].data.output["rgb"]`); MuJoCo: `obs["images"]`.
Keep the state-based path as the default/fallback.

### 3. Isaac teleop via isaacteleop + lerobot example

`isaacteleop` 1.3.131 installed. The official XR→SO-101 example lives in
lerobot SOURCE (`examples/isaac_teleop_to_so101/`): clone lerobot,
`pip install -e ".[feetech,kinematics,dataset]"` (replaces pip lerobot),
install `isaacteleop[cloudxr]`, accept the CloudXR EULA once
(`python -m isaacteleop.cloudxr --accept-eula`). VR headset required for the
XR path; the SO-101-leader path needs the C++ plugin built from IsaacTeleop
source.

### 4. mjlab backend smoke test

`--backend mjlab --task so101_pick_lift` (older plan, MJLAB_INTEGRATION.md)
is wired but untested on this venv. Run a short job on a free GPU; fix the
Runner cfg if the fork's API drifted (see test.md skrl section).

### 5. Sim-to-real: lerobot bridges

Done: `scripts/teleop_leader_lerobot.py` (real leader → sim, optional
--mirror-real to also drive the physical follower) and
`scripts/teleop_gamepad_ik.py --real-follower-port` (gamepad IK → sim +
real simultaneously). Untested on hardware — verify calibration flow
(`lerobot-calibrate`) and deg/rad conventions with a real arm.

### 6. GR00T-WholeBodyControl / GEAR-SONIC (research references)

Humanoid whole-body-control models (NVlabs/GR00T-WholeBodyControl,
nvidia/GEAR-SONIC sonic_v1_1) — NOT directly applicable to 6-DoF SO-101
arms. Keep as references for future embodiments; don't attempt integration
without a matching robot.

### 7. Docs polish

Root README/ENVS.md updated for the rework but still MuJoCo-centric in
places; per-env READMEs in envs/mujoco/*/ still show pre-reorg import paths
in spots. Refresh opportunistically.

## Known rough edges (not blocking, worth fixing opportunistically)

- Root-level `view_single_arm.py` / `view_dual_arm.py` (matplotlib,
  hardcoded cameras) superseded by `scripts/view_cameras.py` — delete once
  confirmed unused.
- `bi_arm_clean_toytable_teleop.py`, `capture_images.py`, `tune_cameras.py`,
  `test` at repo root predate the reorg — audit before relying on them.
- `mujoco.mju_mat2Quat` requires float64 buffers on the installed mujoco —
  follow the existing pattern in pick_lift/pick_place env.py if adding
  mat→quat conversions.
