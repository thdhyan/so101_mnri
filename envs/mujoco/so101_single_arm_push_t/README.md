# SO-101 Single Arm Push-T MuJoCo Environment

LeRobot EnvHub-compatible simulation environment for the SO-101 follower arm.
Task: push a T-shaped block onto a target T outline drawn on the table. The
gripper is held closed and used as a pusher (non-prehensile — no grasping).

## Setup

```bash
pip install mujoco>=3.1.0 gymnasium>=0.29.0 numpy>=1.24.0
```

## Usage

```python
from envs.mujoco.so101_single_arm_push_t.env import make_env, SingleArmPushTEnvConfig

env = make_env(n_envs=1)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Cameras

| Camera | Frame | Description |
|--------|-------|-------------|
| `wrist` | gripper body | Mounted on SO-101 gripper |
| `overhead` | world | Top-down over the workspace |
| `front` | world | Front view at eye height |

All camera pos/euler/fov are customizable via the `CameraConfig` fields in
`env.py` (applied at init, overriding XML).

## Observations

Flat policy vector: 6+6+3+3+3+6 = **(27,)**.

```
joint_pos    (6,)        joint positions [rad]
joint_vel    (6,)        joint velocities [rad/s]
tcp_pos      (3,)        gripper site world position [m]
t_pose       (3,)        T-block pose in robot root frame (x, y, yaw)
target_pose  (3,)        target pose in robot root frame (x, y, yaw)
last_action  (6,)        previous executed joint targets [rad]
images.wrist     (480,640,3)  uint8
images.overhead  (480,640,3)  uint8
images.front     (480,640,3)  uint8
```

## Actions

6D absolute joint position targets, radians, clipped to joint limits.
The gripper channel is pinned closed (`GRIPPER_CLOSED_POS`) — the arm pushes
with closed jaws.

## Reward

exp-kernel on T-centre XY distance to target centre + exp-kernel on wrapped
yaw error − action-rate penalty − per-step alive penalty, `+10.0` on success
(centre within 2.5 cm AND yaw within 15 deg). See `MDP.md`.

## Robot definitions

Shared meshes/MJCF live in `robots/so101/` — see `robots/so101/README.md`.

## Renders

![overview](../../images/push_t_overview.png)

![cameras](images/cameras.png)
