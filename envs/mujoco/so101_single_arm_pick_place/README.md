# SO-101 Single Arm Pick-and-Place MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for the SO-101 follower arm.
Task: pick up the cube and place it on the target disc.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0
```

## Usage

```python
from envs.mujoco.so101_single_arm_pick_place.env import make_env, SingleArmPickAndPlaceEnvConfig

env = make_env(n_envs=1)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `wrist` | gripper body | Mounted on SO-101 gripper |
| `outside_left` | world | Left-side fixed camera |
| `outside_right` | world | Right-side fixed camera |

## Observations

```
joint_pos      (6,)        joint positions [rad]
joint_vel      (6,)        joint velocities [rad/s]
tcp_pos        (3,)        gripper site world position [m]
tcp_quat       (4,)        gripper site orientation [wxyz]
is_grasped     (1,)        0/1, gripper closed near cube
obj_pos        (3,)        cube position [m]
obj_quat       (4,)        cube orientation [wxyz]
target_pos     (3,)        target disc location [m]
tcp_to_obj     (3,)        tcp_pos - obj_pos
obj_to_target  (3,)        obj_pos - target_pos
images.wrist          (480,640,3)  uint8
images.outside_left   (480,640,3)  uint8
images.outside_right  (480,640,3)  uint8
```

## Actions

6D joint position targets, radians, clipped to joint limits.

## Reward

Dense: negative reach distance + grasp bonus + (while grasped) negative xy
distance to target, `+10.0` on success (on target, released, robot static).

## Robot definitions

Shared meshes/MJCF live in `robots/so101/` — see `robots/so101/README.md`.
