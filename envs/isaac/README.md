# envs/isaac/ — Isaac Lab port (Isaac Sim 6.0.1)

Isaac Lab 3.0.0b2 tasks for the SO-101 arm, mirroring the MuJoCo envs in
`envs/mujoco/`. Registered task IDs (single- vs dual-arm is part of the id):

| Task ID | Arms | MuJoCo twin |
|---|---|---|
| `SO101-PickLift-Single-v0` | 1 | `so101_single_arm_pick_lift` |
| `SO101-PickPlace-Single-v0` | 1 | `so101_single_arm_pick_place` |
| `SO101-CylReach-Single-v0` | 1 | — (single-arm variant of cyl_reach) |
| `SO101-CylGrasp-Dual-v0` | 2 | `so101_dual_arm_cylinder_grasp` |
| `SO101-CylReach-Dual-v0` | 2 | `so101_dual_arm_cylinder_reach` |

Per-task MDP specs + screenshots: `tasks/<task>/MDP.md` (index: root
`MDP.md`).

## Layout

```
assets/so101/       USD converted from robots/so101/so101.urdf (generated — do not edit)
so101.py            ArticulationCfg (implicit PD servos), joint names, rig geometry
convert_urdf.py     one-shot URDF→USD regeneration
tasks/              env_cfg.py + gym registration per task; mdp/ shared terms
scripts/            render_cameras.py (screenshots), validate_actions.py (zero/random checks)
```

## Usage

```bash
source .venv/bin/activate   # + OMNI_KIT_ACCEPT_EULA=YES (in .envrc)

# regenerate the USD after any robots/so101/so101.urdf change
python -m envs.isaac.convert_urdf

# validate all tasks: reset + zero/random action rollouts, finite-reward checks
python -m envs.isaac.scripts.validate_actions

# regenerate per-task camera screenshots (writes tasks/<task>/images/)
python -m envs.isaac.scripts.render_cameras

# train (see rl/README.md)
python -m rl.train --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
```

## Geometry / conventions

- Dual-arm tasks spawn TWO instances of the single-arm USD: `LeftArm` at
  (0.18, **+0.2286**, 0.82), `RightArm` at (0.18, **−0.2286**, 0.82) —
  bases 18 in apart in Y, Z/X axes parallel (the real rig).
- Cameras: `wrist_cam(_left/_right)` on `gripper_frame_link` (aimed at the
  task object via calibrated quats in `tasks/common.py`) + global
  `overhead_cam` (top-down) and `front_cam` — all poses are cfg fields.
- Actions: absolute joint-position targets in radians (parity with the
  MuJoCo envs). Control 50 Hz, episodes 1000 steps.
- RL training strips camera sensors (state obs); set `enable_cameras=True`
  whenever a scene with cameras is created.

See `test.md` for the API landmines (offset-quat order, enable_cameras,
resource-starvation silent deaths).
