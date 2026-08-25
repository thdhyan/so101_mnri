# SO-101 RL — MDP specs index

Per-task MDP documentation (observation space, action space, reward formulas,
termination, domain randomization, training commands and hyperparameters),
with live-rendered camera views embedded.

## MuJoCo gymnasium envs (`envs/mujoco/`)

| Task | Arms | MDP spec |
|---|---|---|
| Pick-lift | 1 | [`so101_single_arm_pick_lift/MDP.md`](mujoco/so101_single_arm_pick_lift/MDP.md) |
| Pick-and-place | 1 | [`so101_single_arm_pick_place/MDP.md`](mujoco/so101_single_arm_pick_place/MDP.md) |
| Push-T | 1 | [`so101_single_arm_push_t/MDP.md`](mujoco/so101_single_arm_push_t/MDP.md) |
| Cube push (ramp) | 1 | [`so101_single_arm_cube_push_ramp/MDP.md`](mujoco/so101_single_arm_cube_push_ramp/MDP.md) |
| Cube push (bridge) | 1 | [`so101_single_arm_cube_push_bridge/MDP.md`](mujoco/so101_single_arm_cube_push_bridge/MDP.md) |
| Cylinder grasp | 2 | [`so101_dual_arm_cylinder_grasp/MDP.md`](mujoco/so101_dual_arm_cylinder_grasp/MDP.md) |
| Cylinder reach | 2 | [`so101_dual_arm_cylinder_reach/MDP.md`](mujoco/so101_dual_arm_cylinder_reach/MDP.md) |

## Isaac Lab tasks (`envs/isaac/tasks/`, Isaac Sim 6.0.1)

| Task ID | Arms | MDP spec |
|---|---|---|
| `SO101-PickLift-Single-v0` | 1 | [`tasks/pick_lift/MDP.md`](isaac/tasks/pick_lift/MDP.md) |
| `SO101-PickPlace-Single-v0` | 1 | [`tasks/pick_place/MDP.md`](isaac/tasks/pick_place/MDP.md) |
| `SO101-CylReach-Single-v0` | 1 | [`tasks/cylinder_reach_single/MDP.md`](isaac/tasks/cylinder_reach_single/MDP.md) |
| `SO101-CylGrasp-Dual-v0` | 2 | [`tasks/cylinder_grasp/MDP.md`](isaac/tasks/cylinder_grasp/MDP.md) |
| `SO101-CylReach-Dual-v0` | 2 | [`tasks/cylinder_reach/MDP.md`](isaac/tasks/cylinder_reach/MDP.md) |

## Shared RL conventions

- **Reward design**: staged reach → grasp → lift/place dense terms; a
  per-step `alive` penalty **dominant** over `action_rate` (prevents
  reward-hacking by stalling — see `MJLAB_INTEGRATION.md`); sparse +10
  success bonus.
- **Termination**: two paths — success (within tolerance: 1 cm position /
  10° rotation) → `terminated`; timeout → `truncated`.
- **Training**: skrl and rsl_rl share identical PPO hyperparameters
  (`rl/agents/`); both wrap the same envs, so a policy trained with either
  library is evaluated the same way (`rl/play.py`).
- **Geometry**: dual-arm tasks place the follower bases 18 in
  (0.4572 m) apart in Y with parallel Z/X axes — matching the real rig
  (`robots/so101/README.md`).
