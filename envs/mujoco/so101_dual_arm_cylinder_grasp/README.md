# SO-101 Dual Arm Cylinder Grasp MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for two SO-101 follower
arms. Task: left arm grasps one end of a thin freely-moving cylinder, right
arm grasps the opposite end, both lift it together above `lift_threshold`.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0
```

## Usage

```python
from envs.mujoco.so101_dual_arm_cylinder_grasp.env import make_env, DualArmCylinderGraspConfig

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
cyl_pos           (3,)    cylinder centre [m]
cyl_quat          (4,)    cylinder orientation [wxyz]
cyl_end_a_pos     (3,)    world position of end A site
cyl_end_b_pos     (3,)    world position of end B site
left_is_grasped   (1,)    0/1 — left gripper grasping end A
right_is_grasped  (1,)    0/1 — right gripper grasping end B
images.{left_wrist,right_wrist,overhead_left,overhead_right}  (480,640,3) uint8
```

## Actions

12D joint position targets, radians: first 6 = left arm, last 6 = right arm.

## Reward

Dense: sum of both arms' negative reach distance to their assigned end +
per-arm grasp bonus + (while both grasped) scaled lift-progress bonus,
`+10.0` on success (both grasped and cylinder lifted above `lift_threshold`).

## Robot definitions

Shared meshes/MJCF live in `robots/so101/` — see `robots/so101/README.md`.
