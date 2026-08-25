# SO-101 Single Arm Cube-Push-Bridge MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for the SO-101 follower arm.
Task: push the red cube **across a narrow bridge** (7 cm wide, over a 13 cm
gap) onto the far table section and into the green goal patch. Non-prehensile
pushing — careless pushes drop the cube into the gap (episode failure).

Adapted from a Franka belief-based cube-push/bridge project; see `MDP.md` for
reward provenance and the belief-observation rationale.

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0
```

## Usage

```python
from envs.mujoco.so101_single_arm_cube_push_bridge.env import (
    make_env, CubePushBridgeEnvConfig,
)

env = make_env(n_envs=1, cfg=CubePushBridgeEnvConfig(obs_mode="full"))
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Set `obs_mode="belief"` for the POMDP variant (cube pose hidden, finite
proprioceptive history instead).

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `wrist` | gripper body | Mounted on SO-101 gripper |
| `overhead_cam` | world | Top-down view of both tables + bridge |
| `front_cam` | world | Front view of the workspace |

All pos/euler/fov customizable via `CameraConfig` fields in `env.py`.

## Observations

`obs_mode="full"`:

```
joint_pos       (6,)        joint positions [rad]
joint_vel       (6,)        joint velocities [rad/s]
last_action     (6,)        previous applied action
cube_pos_root   (3,)        cube position in robot root frame [m]
cube_yaw_root   (1,)        cube yaw in root frame [rad]
goal_pos_root   (2,)        goal XY in root frame [m]
images.wrist          (480, 640, 3) uint8
images.overhead_cam   (480, 640, 3) uint8
images.front_cam      (480, 640, 3) uint8
```

`obs_mode="belief"`: same minus `cube_*_root`, plus `cube_init_pos_root (3,)`,
`cube_init_yaw_root (1,)`, and `history (90,)` = last 5 steps of
(`joint_pos`, `joint_vel`, `last_action`).

## Actions

6D absolute joint position targets, radians, clipped to joint limits.

## Reward

Dense cube-to-goal shaping (L2 + tanh kernel + signed progress +
velocity-toward-goal), EE-cube proximity, alive −0.05 and action-rate −0.01
penalties, +10 success bonus. Success: cube centre within 5 cm (XY) of the
goal patch on the far table. Failure: cube falls below table-top − 5 cm.
Full tables in `MDP.md`.

## Robot definitions

Shared meshes/MJCF live in `robots/so101/` — see `robots/so101/README.md`.

## Renders

![overview](../../images/cube_push_bridge_overview.png)

![cameras](images/cameras.png)
