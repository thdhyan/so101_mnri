# MDP spec — Cube-push-ramp (single arm)

One SO-101 follower arm pushes a 4.5 cm red cube **up a 15° ramp** into a fixed
green goal patch near the ramp top, using its closed gripper as a blunt,
non-prehensile pusher. The core difficulty: ramp friction (μ = 0.24) is below
tan 15° ≈ 0.268, so **gravity pulls the cube back down whenever pushing
stops**. MuJoCo twin of the Franka cube-push task from
`Belief-based-pushing/source/franka_cube_push_project` (Isaac Lab), adapted to
the SO-101 and re-scoped onto an inclined plane.

## Scene

| Element | Geometry | Placement |
|---|---|---|
| SO-101 follower | shared `robots/so101/so101_follower.xml` include | base at (0.18, 0, 0.82), identity orientation |
| Table | same as pick-lift | top surface z = 0.82 |
| Ramp | flat box, 0.30 m (slope) × 0.15 m × 0.02 m thick, pitched −15° about +y | low edge flush with table at (0.28, 0, 0.82), rising toward +x; rises 7.8 cm over its length |
| Cube | 4.5 cm box, 0.08 kg, red, freejoint | rests flush on the ramp at s = 0.0587 m along-slope from the low edge, lateral jitter ±1.5 cm |
| Goal patch | visual-only green cylinder r = 7 cm, no collision (`contype=0 conaffinity=0`) | fixed on the ramp surface at s = 0.24 m along-slope |

Ramp/cube/goal geoms are primitives only — no new mesh files; `meshdir` is set
only in this env's top-level `scene.xml`. Cube–ramp sliding friction is 0.24
on both geoms so the cube creeps back down when unsupported (net slide
acceleration ≈ 0.10 m/s²).

## Provenance: Franka cube-push project → this env

Extracted from `franka_cube_push_project/tasks/manager_based/cube_push/`
(env_cfg, `mdp/rewards.py`, `mdp/terminations.py`, `mdp/belief.py`):

| Source element | Value there | What we kept |
|---|---|---|
| `cube_goal_distance_tanh` reward | w=2.0, `1 − tanh(d/std)`, std=0.10, planar XY cube→goal distance | kept verbatim as the dense shaping term |
| Success criterion `cube_reached_goal` / `GOAL_SUCCESS_THRESHOLD` | planar centre-to-centre distance < 0.010 m | kept verbatim (`success_threshold`, cfg-tunable) |
| `action_rate_l2` penalty | −0.002…−0.001 | kept form (Σ squared per-dim deltas), weight −0.01 per spec |
| Alive penalty (from sibling Isaac pick-lift style) | −0.05 dominant over action rate | kept |
| Success bonus | +25 flat (source) | +10 per spec for this env |
| Signed per-step progress `cube_goal_progress` | w=80 delta-of-distance | **rejected** — see rationale below |
| `ee_pre_push_pose_tanh`, `cube_velocity_to_goal`, tilt/lift/OOB terminations, OmniReset curriculum, domain rand | — | out of scope (single blunt pusher; minimal spec) |
| Planar particle-filter belief (`belief.py`, 256 particles, two-mode compression, RSS20-style hierarchy) | full learned-belief stack | replaced by finite-history proxy (below); particle filter documented as future work |

**Why the tanh kernel instead of per-step delta:** a signed delta term pays for
any distance decrease and punishes any increase, so on a slippery ramp it can
be farmed by push/release oscillation cycles — the source project itself
removed signed progress in `OmniResetHighLevelRewardsCfg` for exactly that
reason ("pays out net-positive reward for push/pull oscillation"). The state
based kernel `2·(1 − tanh(d/0.10))` is Markovian, identical in both obs modes,
and bounded in [0, 2].

## Observation space

Dict obs, two modes selected by `CubePushRampEnvConfig.obs_mode`.
All positions/yaw are expressed in the **robot root frame** (base body pose).
Cameras are separate: `obs["images"] = {cam_name: (H, W, 3) uint8}`.

### `"full"` mode (MDP) — concatenated policy vector: 24

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `joint_pos` | (6,) | float32 | absolute joint positions, rad |
| `joint_vel` | (6,) | float32 | joint velocities, rad/s |
| `cube_pos` | (3,) | float32 | cube position, root frame, m |
| `cube_yaw` | (1,) | float32 | cube yaw, root frame, rad |
| `goal_pos` | (2,) | float32 | goal patch xy, root frame, m |
| `last_action` | (6,) | float32 | previous applied action |

### `"belief"` mode (POMDP proxy) — concatenated policy vector: 114

Cube current pose is **removed**. Instead the policy gets the cube's initial
pose (known at reset) plus a finite history of proprioception/actions:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `joint_pos` | (6,) | float32 | absolute joint positions, rad |
| `joint_vel` | (6,) | float32 | joint velocities, rad/s |
| `goal_pos` | (2,) | float32 | goal patch xy, root frame, m |
| `last_action` | (6,) | float32 | previous applied action |
| `cube_init_pos` | (3,) | float32 | cube position AT RESET, root frame, m |
| `cube_init_yaw` | (1,) | float32 | cube yaw AT RESET, root frame, rad |
| `history` | (90,) | float32 | K=5 most recent steps × [joint_pos(6), joint_vel(6), last_action(6)], flattened oldest→newest |

**Belief note.** This is a practical finite-history approximation of belief
conditioning: contact forces implied by the joint history are the only signal
for localising the cube after it leaves its known spawn pose. The source
project's estimator — a 256-particle planar filter with contact-driven
resampling compressed to a two-mode belief, plus the RSS20 controller
hierarchy — is future work for this env; swap `history` for a compressed
particle encoding without changing the rest of the MDP.

## Action space

`(6,)` float32 — **absolute** target joint positions in radians, clipped to
actuator `ctrlrange`. Position servos (STS3215 model), 30 Hz control, 16
physics steps per action (dt = 2 ms). The gripper channel commands the jaw
angle; closed (~1.75 rad) turns the gripper into a blunt pusher. Home pose:
`[0, −0.85, 1.68, 0.68, 0, 1.745]` — the closed jaws parked low just behind
the cube's resting spot (a backstop at the ramp base).

## Reward

Per step:

```
r = 2.0 · (1 − tanh(d/0.10)) − 0.01 · ‖aₜ − aₜ₋₁‖₂² − 0.05 + 10 · 1[success]
```

where `d = ‖cube_xy − goal_xy‖` (planar, world == root frame here).

| Term | Weight | Formula | Notes |
|---|---|---|---|
| goal proximity | ×2.0 | `1 − tanh(d / 0.10)` | source `cube_goal_distance_tanh`; bounded [0, 2] |
| action rate | −0.01 | Σ_dims (aₜ − aₜ₋₁)² | source `action_rate_l2` form |
| alive | −0.05 | constant per step | dominant over action-rate scale |
| success bonus | +10.0 | once, on termination step | episode ends on success |

## Termination & success

| Path | Condition |
|---|---|
| `terminated` (success) | planar cube-centre→goal-centre distance < `success_threshold` = **0.010 m** (source tolerance) |
| `truncated` (timeout) | 25 s episode (750 steps @ 30 Hz); longer than flat-push tasks because ramp pushing is slow |

No tilt / out-of-bounds / lift terminations (source has them; out of scope
here — the ramp lip plus timeout bound failure naturally).

## Reset & randomization

| What | Range |
|---|---|
| cube lateral (y) spawn | ±1.5 cm around the ramp centreline, at s = 0.0587 m along-slope (never past the base region toward the goal) |
| cube orientation | fixed, flush with the ramp (pitch −15°, yaw 0) |
| arm | home keyframe, jaws closed |
| goal | static, never randomised |

The cube spawns 1 mm above the surface and settles for 15 physics steps before
the (known) initial pose is recorded for belief mode.

## Known dynamics (verified empirically)

* Unpushed, the cube creeps down-slope (≈0.10 m/s²) and comes to rest against
  the parked pusher / ramp lip — the home pose acts as a backstop, so episodes
  start from a stable, repeatable configuration.
* A slow face-on shove or drag up-slope moves the cube several cm; pausing
  lets it slide back onto the pusher — the intended core difficulty.
* Reaching d < 0.010 m requires sustained pushing through ~15 cm of rise;
  holding still farms only the small proximity residual at the backstop
  (≈ +0.09/step) and can never succeed.

## Usage

```python
from envs.mujoco.so101_single_arm_cube_push_ramp.env import make_env, CubePushRampEnvConfig

env = make_env(n_envs=1)                                        # full obs
env = make_env(n_envs=1, cfg=CubePushRampEnvConfig(obs_mode="belief"))  # POMDP proxy
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```
