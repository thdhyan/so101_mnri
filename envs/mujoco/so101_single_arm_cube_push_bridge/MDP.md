# MDP spec — Cube push over bridge (single arm)

One SO-101 follower arm pushes a **4.5 cm red cube across a narrow bridge**
from a near table section onto a far table section, into a fixed **green goal
patch**. Non-prehensile: the closed gripper is used as a blunt pusher.
Pushing carelessly knocks the cube into the 13 cm gap — irrecoverable failure.

Task adapted from the Franka belief-based pushing project
(`~/Projects/ebasa/Belief-based-pushing/source/franka_cube_push_project/`,
RSS'20-style bridge benchmark of Wirnshofer et al.) — see *Provenance* below.

## Scene (all primitives; `assets/scene.xml`)

```
 y
 ^      near table          bridge       far table
 |   +-------------+    +--------+   +---------+
 |   |             │    │  7 cm  │   |         |
 |   |   cube ●→   │════│ bridge │═══│  ◎ goal |
 +--x---+───────────┘    +────────┘   +─────────+
     -0.05   0.42        0.42  0.55   0.55   0.75   (x, m)
        arm base x=0.18, table top z=0.82
```

| Element | Geometry | Notes |
|---|---|---|
| near table top | box half `(0.235, 0.35, 0.02)` @ `(0.185, 0, 0.80)` | x ∈ [−0.05, 0.42], top z = 0.82 |
| gap | — | 13 cm, x ∈ [0.42, 0.55] |
| bridge | box half `(0.065, 0.035, 0.02)` @ `(0.485, 0, 0.80)` | 7 cm wide, top flush with tables, collision on, 2 thin legs to floor |
| far table top | box half `(0.10, 0.35, 0.02)` @ `(0.65, 0, 0.80)` | x ∈ [0.55, 0.75], top z = 0.82 |
| cube | box half `0.0225`, mass 0.08 kg, freejoint | red, condim 4, friction `1.0 0.005 0.0001` |
| goal patch | cylinder r = 0.07, h = 5 mm @ `(0.64, 0, 0.8225)` | visual only (`contype=0 conaffinity=0`) |

Robot: shared `robots/so101/so101_follower.xml` include (base at
`(0.18, 0, 0.82)`, identity orientation). Meshes are never duplicated;
`meshdir` is set only in this top-level scene.xml.

Cameras (pos/euler/fov overridable at runtime via `CameraConfig` in `env.py`):

| Camera | Frame | Default pose |
|---|---|---|
| `wrist` | gripper body | from follower XML / `wrist_camera` cfg |
| `overhead_cam` | world | `(0.30, 0.0, 1.50)`, euler `(0, 0, 90)`, fovy 65° |
| `front_cam` | world | `(0.35, −1.05, 1.05)`, euler `(75, 0, 0)`, fovy 65° |

## Action space

`(6,)` float32 — **absolute** target joint positions in radians, clipped to
actuator `ctrlrange`. Position servos (STS3215), 30 Hz control. Joint order:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
For pushing, command the gripper toward its ctrlrange max (+1.745 rad) so the
jaws close into a blunt pusher.

## Observation space — two modes via `cfg.obs_mode`

Both modes are dicts including `images: {cam_name: (480, 640, 3) uint8}`.
Positions/yaw are expressed in the **robot root frame** (base body frame).
State-vector sizes below exclude images.

### `obs_mode="full"` (fully observed MDP) — 24 floats

| Field | Shape | Description |
|---|---|---|
| `joint_pos` | (6,) | absolute joint positions, rad |
| `joint_vel` | (6,) | joint velocities, rad/s |
| `last_action` | (6,) | previous applied action (home at reset) |
| `cube_pos_root` | (3,) | cube centre in robot root frame, m |
| `cube_yaw_root` | (1,) | cube yaw in root frame, rad (wrapped) |
| `goal_pos_root` | (2,) | goal patch XY in root frame, m |
| `images` | dict | RGB per camera |

### `obs_mode="belief"` (POMDP) — 114 floats

Cube pose is **removed**. Replaced by the cube's initial pose plus a finite
proprioceptive history — an approximation of the source project's particle-
filter belief conditioning (see Provenance).

| Field | Shape | Description |
|---|---|---|
| `joint_pos` | (6,) | absolute joint positions, rad |
| `joint_vel` | (6,) | joint velocities, rad/s |
| `last_action` | (6,) | previous applied action |
| `goal_pos_root` | (2,) | goal XY in root frame, m |
| `cube_init_pos_root` | (3,) | cube position AT RESET, root frame |
| `cube_init_yaw_root` | (1,) | cube yaw AT RESET, root frame |
| `history` | (90,) | last K=5 steps × (`joint_pos`(6) + `joint_vel`(6) + `last_action`(6)), concatenated; excludes the current instant; initialised to the home state |
| `images` | dict | RGB per camera |

## Reward

Total:

```
r = -0.05                                   # alive penalty
  - 0.01 · ‖a_t − a_{t−1}‖²                 # action rate
  - 0.60 · d_xy                             # cube-goal L2 distance
  + 2.00 · (1 − tanh(d_xy / 0.10))          # cube-goal tanh kernel
  + 80.0 · (d_prev − d_xy)                  # signed per-step progress
  + 3.00 · clip(v_cube·û_goal / 0.20, ±1)   # cube velocity toward goal
  + 0.15 · (1 − tanh(d_ee-cube_xy / 0.16))  # EE-cube proximity
  + 10.0 · 1[success]
```

with `d_xy = ‖cube_xy − goal_xy‖` and û_goal the unit vector cube → goal.

| Term | Weight | Source (Franka project, `cube_push/mdp/rewards.py` + `RewardsCfg`) |
|---|---|---|
| alive | −0.05/step | repo convention (pick-lift Isaac twin), dominant over action rate |
| action_rate_l2 | −0.01 | repo spec weight (source: −0.002) |
| `cube_goal_distance_l2` | −0.6 | source base PPO cfg, same weight |
| `cube_goal_distance_tanh` | +2.0, std 0.10 | source base PPO cfg, same weight & std |
| `cube_goal_progress` (signed) | ×80 | source base PPO cfg, same weight; signed-only avoids the oscillation farming the source found with paired positive-only progress |
| `cube_velocity_to_goal` | ×3.0, speed_scale 0.20 | source base PPO cfg, identical |
| `ee_cube_distance_tanh` | +0.15, std 0.16 | source base PPO cfg, identical |
| success bonus | +10 | repo convention (spec mandate); source used 25 @ 1 cm threshold |

## Termination

| Path | Condition | Effect |
|---|---|---|
| success | `d_xy < 0.05` m (cube centre inside 5 cm of goal centre, far table) | `terminated=True`, +10 bonus, `info["success"]=True` |
| fell off | cube z < 0.77 m (= table top − 5 cm): dropped into the gap or off any table edge | `terminated=True`, **no bonus**, `info["fell_off_bridge"]=True` |
| timeout | 25 s episode (750 steps @ 30 Hz) | `truncated=True` |

Success tolerance: default 5 cm (`TaskConfig.success_threshold`), matching the
7 cm visual patch; the source project's own threshold was much tighter
(1 cm precision-grade).

## Reset

- Arm at home `qpos = [0, −0.5, 0.8, 0.4, 0, 0]` (same as sibling envs);
  actuators hold home, `last_action` initialised to home.
- Cube at `(0.36, 0.0, 0.844)` + uniform XY jitter **±2 cm** (aligned roughly
  with the bridge), yaw 0.
- Goal fixed at `(0.64, 0.0)` on the far table.
- Belief-mode history buffer reset to home-state entries.

## Randomization

| What | Range | Mode |
|---|---|---|
| cube XY spawn | ±2 cm around `(0.36, 0)` | every reset |

(The Franka source additionally randomised cube scale 0.85–1.15× at startup,
mass ×0.75–1.35 per reset, joint-noise σ=0.02 rad, and ran a spawn-distance
curriculum; out of scope for this MuJoCo env.)

## Provenance & adaptations (Franka cube-push project)

Source: `franka_cube_push_project/tasks/manager_based/cube_push/` (flat-table
push to green patch) and `.../cube_bridge/` (bridge variant). Extracted:

- **Dense reward structure**: cube-goal L2 + tanh kernel + signed progress +
  velocity-to-goal + EE-proximity terms with their base-PPO weights/stds
  (`cube_push_env_cfg.RewardsCfg`, `mdp/rewards.py`). Copied nearly verbatim.
- **Success criterion**: cube-centre-to-goal-patch XY distance below threshold
  → terminate (`terminations.cube_reached_goal`).
- **Bridge failure semantics**: crossing failure = irrecoverable episode end
  (`cube_bridge/mdp.cube_fell_off_bridge`, citing Wirnshofer et al. RSS 2020).
- **Belief-conditioned POMDP variant** idea: actor never sees the true cube
  pose (`BeliefObservationsCfg`).

Adapted for SO-101 / plain MuJoCo gymnasium:

- Success tolerance loosened 1 cm → 5 cm (spec mandate; matches goal-patch
  radius); bonus 25 → 10; alive/action-rate weights follow this repo's
  conventions instead of Isaac-Lab magnitudes.
- Belief: the source maintains a 256-particle planar particle filter updated by
  contact/measurement models (`compressed_planar_belief`, ~40 hyperparameters).
  Here it is proxied by the cube's initial pose + K=5 finite proprioceptive
  history — cheap, Markov-embedable approximation; documented as such.
- Actions: absolute joint targets (SO-101 servos) instead of the source's
  relative-IK / high-level strategy actions; hence the source's EE pre-push /
  corridor shaping terms (`ee_pre_push_pose_tanh`,
  `ee_behind_cube_corridor_tanh`), tuned to that IK interface, were dropped,
  keeping only the simple EE-proximity term.
- Terminations not carried over: cube tilt >10°, cube lift >6 mm, workspace
  margin, goal overshoot penalties/terminations — replaced here by the single
  fall-off-the-bridge failure condition (the bridge makes falls the dominant
  irrecoverable mode); belief-shaping rewards (mode accuracy/concentration)
  belong to the particle filter and have no analogue here.
- Scene rebuilt from primitives (no USD/mesh assets); SO-101 geometry means
  the goal sits ~28 cm from the cube spawn vs the source's flat 20–30 cm push,
  and the reachable low-forward band starts near x≈0.36 world — verified
  empirically when placing the bridge/goal.

## Training

```bash
# plain MuJoCo (single env, CPU-friendly smoke)
python -m rl.train --backend mujoco --task cube_push_bridge --algo skrl --max-iterations 200
```

Not yet registered in the Isaac backend or `ENVS.md`; MuJoCo-side only.
