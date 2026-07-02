# SO-101 Dual Arm Cylinder Reach MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for two SO-101 follower
arms. Task: reach target points above the two ends of a **static** (fixed,
non-physical) cylinder — no grasping, no lifting. Success requires holding
both gripper sites within `pos_threshold` of their targets for `hold_steps`
consecutive steps.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0
```

## Usage

```python
from envs.mujoco.so101_dual_arm_cylinder_reach.env import make_env, DualArmCylinderReachConfig

env = make_env(n_envs=1)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # (12,) = left 6 + right 6
```

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `left_wrist` | left gripper body | Mounted on left SO-101 gripper |
| `right_wrist` | right gripper body | Mounted on right SO-101 gripper |
| `overhead_left` | world | Overhead camera, left workspace |
| `overhead_right` | world | Overhead camera, right workspace |

## Observations

```
left_joint_pos    (6,)    right_joint_pos   (6,)
left_joint_vel    (6,)    right_joint_vel   (6,)
left_ee_pos       (3,)    right_ee_pos      (3,)
cyl_end_a_pos     (3,)    world position of end A site (fixed)
cyl_end_b_pos     (3,)    world position of end B site (fixed)
left_target_pos   (3,)    end A + z_offset
right_target_pos  (3,)    end B + z_offset
images.{left_wrist,right_wrist,overhead_left,overhead_right}  (480,640,3) uint8
```

## Actions

12D joint position targets, radians: first 6 = left arm, last 6 = right arm.

## Reward

Dense: negative sum of both arms' distance to target + bonus while both
within threshold, `+10.0` on success (both held within `pos_threshold` for
`hold_steps` consecutive steps).

## Robot definitions

Shared meshes/MJCF live in `robots/so101/` — see `robots/so101/README.md`.
