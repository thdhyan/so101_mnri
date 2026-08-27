# SO-101 Single Arm Cylinder Reach MuJoCo Environment

Single-arm variant of the dual-arm cylinder reach task. One SO-101 follower
arm reaches a target point above a fixed cylinder's end A.

## Usage

```python
from envs.mujoco.so101_single_arm_cylinder_reach.env import make_env

env = make_env(n_envs=1)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # (6,)
```

## RL Training

```bash
python -m rl.train --backend mujoco --task cyl_reach_single --algo skrl --max-iterations 5000
```
