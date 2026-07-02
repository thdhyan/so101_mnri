# SO-101 Single Arm MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for the SO-101 follower arm.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0 Pillow
```

## Usage

```python
from so101_single_arm_env.env import make_env, SingleArmEnvConfig, CameraConfig

# Default (push task, tuned cameras)
env = make_env(n_envs=1)
obs, _ = env.reset()

# Custom camera positions
cfg = SingleArmEnvConfig(
    task="push",
    wrist_camera=CameraConfig("wrist", pos=(0.0, -0.043, -0.042), euler=(29.0, -6.0, 0.5), fov=75.0),
    outside_cameras=[
        CameraConfig("outside_left",  pos=(0.462, -0.110, 1.273), euler=(0.5,  3.5,  91.0), fov=65.0),
        CameraConfig("outside_right", pos=(0.382, -0.304, 1.066), euler=(65.0, 1.0,  -3.0), fov=65.0),
    ],
)
env = make_env(cfg=cfg)
```

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `wrist` | gripper body | Mounted on SO-101 gripper, looks toward fingertips |
| `outside_left` | world | Left-side fixed camera |
| `outside_right` | world | Right-side fixed camera (angled down) |

Camera POVs (tuned positions):

| wrist | outside_left | outside_right |
|-------|-------------|--------------|
| ![wrist](images/single_wrist.png) | ![outside_left](images/single_outside_left.png) | ![outside_right](images/single_outside_right.png) |

## Observations

```
joint_pos       (6,)        joint positions [rad]
joint_vel       (6,)        joint velocities [rad/s]
ee_pos          (3,)        end-effector position [m]
object_pos      (3,)        cube position [m]
object_quat     (4,)        cube orientation [wxyz]
object_vel      (3,)        cube linear velocity [m/s]
goal_pos        (3,)        goal position [m]
goal_quat       (4,)        goal orientation [wxyz]
images.wrist          (480,640,3)  uint8
images.outside_left   (480,640,3)  uint8
images.outside_right  (480,640,3)  uint8
```

## Actions

6D joint position targets normalized to `[-1, 1]`, mapped to joint limits.

## Reward

Dense: `-dist - 0.05 * rot_err` + `10.0` success bonus when within 4 cm / 0.3 rad.

## Tasks

| task | description |
|------|-------------|
| `"push"` | Push cube to goal position |
| `"pull"` | Pull cube toward robot |
| `"none"` | No object — pure data collection |
