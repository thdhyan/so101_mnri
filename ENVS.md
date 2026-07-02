# SO-101 Environments

Two MuJoCo `gymnasium.Env` implementations built on the SO-101 follower arm
(6-DoF + gripper, STS3215 servos). Both share robot definitions from
`robots/so101/` (see that dir's README for provenance) and differ only in
world layout and task.

## Single arm — `envs/mujoco/so101_single_arm/`

One SO-101 arm at a table with a cube (task object) and a goal marker.

![Single arm setup](images/single_arm_overview.png)

- **Joints/actuators (6):** `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- **EE site:** `gripperframe`
- **Cameras (5):** `wrist` (gripper-mounted), `outside_left`, `outside_right`, `overhead_cam` (top-down), `front_cam`
- **Tasks:** `none` (data collection only), `push`, `pull`
- **Action space:** `(6,)` target joint positions, radians
- **Entry point:** `envs.mujoco.so101_single_arm.env.make_env(n_envs, use_async_envs, cfg)`

Camera streams tiled:

![Single arm camera streams](images/single_arm_cameras.png)

## Dual arm — `envs/mujoco/so101_dual_arm/`

Two SO-101 arms (`left_`/`right_` prefixed) facing each other across a wider
table, same cube/goal task.

![Dual arm setup](images/dual_arm_overview.png)

- **Joints/actuators (12):** left/right × `{shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper}`
- **EE sites:** `left_gripperframe`, `right_gripperframe`
- **Cameras (6):** `left_wrist`, `right_wrist`, `overhead_left`, `overhead_right`, `overhead_cam`, `front_cam`
- **Tasks:** `none`, `push`, `pull`
- **Action space:** `(12,)` = left 6 + right 6 target joint positions, radians
- **Entry point:** `envs.mujoco.so101_dual_arm.env.make_env(n_envs, use_async_envs, cfg)`

Camera streams tiled:

![Dual arm camera streams](images/dual_arm_cameras.png)

## Manually checking an environment

`scripts/verify_manual.py` is the go-to script for eyeballing an env before
trusting it — opens the interactive 3D viewer alongside a live tiled window
of every camera, so you can confirm robot geometry, camera framing, and
physics stability by hand.

```bash
conda activate so101
python scripts/verify_manual.py --env single     # interactive: 3D viewer + camera grid
python scripts/verify_manual.py --env dual
```

Controls: drag to orbit the 3D viewer, scroll to zoom. Press **ESC** in the
camera-grid window (or close either window) to exit. The robot starts at its
`home` keyframe pose and physics runs live (gravity, no control input), so
you'll see it sag/settle — that's expected with no actuator holding it,
useful for confirming joint limits and collision geometry look right.

For CI / non-interactive checks (no GUI, exits nonzero on failure):

```bash
python scripts/verify_manual.py --env single --headless-check
python scripts/verify_manual.py --env dual --headless-check
```

This compiles the scene, loads the keyframe, renders every camera and checks
none are blank, then resets/steps the env once. See also `verify_envs.py`
(repo root) for a more detailed per-env observation/action-space dump.

## Robot definitions

Shared meshes, MJCF kinematics, and URDFs live in `robots/so101/` — see
`robots/so101/README.md`.

## Teleop / data collection scripts

See `scripts/README.md` for gamepad IK teleop, real leader-arm teleop, and
the camera-viewer script.
