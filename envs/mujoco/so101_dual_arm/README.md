# SO-101 Dual Arm MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for two SO-101 follower arms.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0 Pillow
```

## Usage

```python
from so101_dual_arm_env.env import make_env, DualArmEnvConfig, CameraConfig

# Default (push task, tuned cameras)
env = make_env(n_envs=1)
obs, _ = env.reset()

# Custom camera positions
cfg = DualArmEnvConfig(
    task="push",
    left_wrist_camera=CameraConfig("left_wrist",   pos=(0.0, -0.043, -0.042), euler=(29.0, -6.0, 0.5), fov=75.0),
    right_wrist_camera=CameraConfig("right_wrist", pos=(0.0, -0.043, -0.042), euler=(29.0, -6.0, 6.0), fov=75.0),
    overhead_cameras=[
        CameraConfig("overhead_left",  pos=(0.637,  0.027, 1.248), euler=(-26.0,  0.5, -90.5), fov=80.0),
        CameraConfig("overhead_right", pos=(0.530, -0.491, 1.061), euler=(-67.0, -5.0,-159.5), fov=80.0),
    ],
)
env = make_env(cfg=cfg)
```

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `left_wrist` | left gripper body | Mounted on left SO-101 gripper |
| `right_wrist` | right gripper body | Mounted on right SO-101 gripper |
| `overhead_left` | world | Overhead camera covering left arm workspace |
| `overhead_right` | world | Overhead camera covering right arm workspace |

Camera POVs (tuned positions):

| left_wrist | right_wrist |
|-----------|------------|
| ![left_wrist](images/dual_left_wrist.png) | ![right_wrist](images/dual_right_wrist.png) |

| overhead_left | overhead_right |
|--------------|---------------|
| ![overhead_left](images/dual_overhead_left.png) | ![overhead_right](images/dual_overhead_right.png) |

## Observations

```
left_joint_pos    (6,)        left joint positions [rad]
left_joint_vel    (6,)        left joint velocities [rad/s]
left_ee_pos       (3,)        left end-effector position [m]
right_joint_pos   (6,)        right joint positions [rad]
right_joint_vel   (6,)        right joint velocities [rad/s]
right_ee_pos      (3,)        right end-effector position [m]
object_pos        (3,)        cube position [m]
object_quat       (4,)        cube orientation [wxyz]
object_vel        (3,)        cube linear velocity [m/s]
goal_pos          (3,)        goal position [m]
goal_quat         (4,)        goal orientation [wxyz]
images.left_wrist       (480,640,3)  uint8
images.right_wrist      (480,640,3)  uint8
images.overhead_left    (480,640,3)  uint8
images.overhead_right   (480,640,3)  uint8
```

## Actions

12D joint position targets normalized to `[-1, 1]`: first 6 = left arm, last 6 = right arm.

## Reward

Dense: `-dist - 0.05 * rot_err` + `10.0` success bonus when within 4 cm / 0.3 rad.

## Tasks

| task | description |
|------|-------------|
| `"push"` | Push cube to goal position |
| `"pull"` | Pull cube toward centre |
| `"none"` | No object — pure data collection |
