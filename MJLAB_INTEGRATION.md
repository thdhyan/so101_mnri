# SO-101 → mjlab RL integration

Plan for training RL policies on the 4 SO-101 task envs using
[mjlab](https://github.com/mujocolab/mjlab) (GPU-batched MuJoCo Warp,
IsaacLab-style manager-based RL API) and skrl. Written for any agent
picking up this work — read this file in full before starting; it
supersedes ad-hoc exploration since the design decisions below (fork
choice, submodule layout, reward/DR design, termination policy) were
already made and confirmed with the user.

## Context

We have 6 working MuJoCo `gymnasium.Env` implementations for the SO-101 arm
(single/dual data-collection envs + 4 fixed-task envs: pick-lift,
pick-and-place, cylinder-grasp, cylinder-reach — see `ENVS.md`). All
currently run single-instance, CPU, no domain randomization, fixed
object/target spawn positions, no RL training wired up.

Goal: train policies for the 4 task envs (single/dual data-collection envs
have no fixed reward and are out of scope — confirmed with user) using
**mjlab** — with **skrl** as the RL library, plus domain randomization on
object/target spawn poses and object mass (IsaacLab-style `EventManager` +
`reset_root_state_uniform`).

Machine used for this work has an NVIDIA RTX 4060 Laptop GPU (CUDA 13.0
driver) — GPU training is viable; mjlab has no CPU training path, so a GPU
is a hard requirement for anyone continuing this work.

## Confirmed facts (from research, not assumptions)

- mjlab pins: `python>=3.10,<3.14`, `mujoco~=3.10.0`, `torch>=2.7.0` (cu128
  extra), `mujoco-warp~=3.10.0`, `warp-lang>=1.14.0`, `numpy<2.5`. The
  existing `so101` conda env (python 3.11.15, mujoco 3.10.0) satisfies these
  exactly — mjlab and skrl are installed into `so101`, no new conda env.
- mjlab: **user's fork** `https://github.com/thdhyan/mjlab`, added as a
  **git submodule** at `third_party/mjlab`, `pip install -e` into `so101`
  — fork instead of upstream so our own changes to mjlab stay in sync via
  the user's remote; submodule (not sibling checkout, not vendored/stripped
  copy) so the exact commit is pinned and tracked in this repo's git
  history without duplicating mjlab's own commit history into ours.
- skrl: **user's fork** `https://github.com/thdhyan/skrl`, **also a git
  submodule** at `third_party/skrl`, same treatment as mjlab, `pip install
  -e` into `so101`.
- mjlab does not depend on skrl or gymnasium directly — only ships an
  `RslRlVecEnvWrapper`. We write our own `SkrlVecEnvWrapper` (~80 lines,
  same shape as the rsl_rl one: expose `num_envs`/`device`/
  `observation_space`/`action_space`, `step`/`reset`/`state`/`render`/`close`,
  subclass `skrl.envs.wrappers.torch.Wrapper`).
- mjlab's `ManagerBasedRlEnv` requires re-expressing each task as MJCF +
  manager configs (`SceneCfg`, `RewardTermCfg`, `EventTermCfg`, etc.) — it
  does **not** run our existing `gymnasium.Env` `env.py` files directly.
  `env.py` reward/success logic is the reference spec to port, not reusable
  code.
- Best template: `src/mjlab/tasks/manipulation/lift_cube_env_cfg.py` +
  its `mdp/` folder (observations.py, rewards.py, terminations.py) — same
  shape as our pick-lift task (cube + target height, staged reach→lift
  reward, `action_rate_l2`/`joint_pos_limits` penalty terms).
- Object pose randomization on reset: `mdp.reset_root_state_uniform`
  (`EventTermCfg(mode="reset", params={"pose_range": ..., "velocity_range": ...})`).
  Other DR (friction, mass, etc.) via `mjlab.envs.mdp.dr.*` terms,
  `mode="startup"` or `"reset"`.

## Scope

In scope: 4 task envs (pick-lift, pick-place, cylinder-grasp,
cylinder-reach) get mjlab task configs, reward functions, domain
randomization, and an skrl training entry point.

Out of scope (confirmed with user): single/dual data-collection envs
(`push`/`pull`/`none`) — no fixed success criteria to port as reward terms,
skipped.

## Directory layout

```
third_party/
  mjlab/                           # git submodule -> thdhyan/mjlab, pip install -e
  skrl/                            # git submodule -> thdhyan/skrl, pip install -e
rl/
  tasks/
    so101_pick_lift/
      __init__.py                 # task registration (mjlab registry pattern)
      env_cfg.py                  # ManagerBasedRlEnvCfg: scene, actions, observations, rewards, events, terminations
      mdp/
        rewards.py                # reach/grasp/lift reward terms
        observations.py           # obs terms (mirrors env.py's obs dict fields)
        terminations.py           # success/timeout terms
        events.py                 # domain randomization: object pose, target pose, mass, friction
    so101_pick_place/             # same shape
    so101_cylinder_grasp/         # same shape, dual-arm scene entities
    so101_cylinder_reach/         # same shape, cylinder pose DR added (new — was static)
  skrl_wrapper.py                 # SkrlVecEnvWrapper(skrl.envs.wrappers.torch.Wrapper)
  train.py                        # skrl PPO training entry point, tyro CLI (--task so101_pick_lift, --num_envs 4096, etc.)
  play.py                         # load checkpoint, run policy in viewer for eval
  README.md                       # setup: submodule init + pip install -e commands, how to train/play
MDP.md                            # root index linking the 4 per-env MDP.md files
envs/mujoco/so101_single_arm_pick_lift/MDP.md      # per-env: obs/action/reward/termination/DR spec
envs/mujoco/so101_single_arm_pick_place/MDP.md
envs/mujoco/so101_dual_arm_cylinder_grasp/MDP.md
envs/mujoco/so101_dual_arm_cylinder_reach/MDP.md
```

Each task's MJCF scene is a **new file** under `rl/tasks/<name>/assets/`
(mjlab's `SceneCfg`/entity config wraps standard MJCF, but its scene
composition model may differ enough from a monolithic `<include>`d file
that a straight copy of `envs/mujoco/<name>/assets/scene.xml` might not
work unmodified — **validate this on the first task, pick-lift, before
repeating the pattern for the other 3**, see Risk section below). Whatever
the final shape, it must still `<include>` `robots/so101/so101_follower.xml`
/ `so101_dual_follower.xml` and follow the same `meshdir`-in-top-level-file
rule as `robots/so101/README.md`'s "MuJoCo path resolution note" — **do not
modify `robots/so101/`**, same constraint as `INTEGRATIONPLAN.md`'s rule.

## Per-env reward/DR design (ported from existing `env.py` logic)

| Env | Reward terms (weight — mirrors existing dense reward in `env.py`, plus new terms below) | Termination | Domain randomization |
|---|---|---|---|
| **pick-lift** | `reach_ee_object` (neg dist, continuous) + `grasp_bonus` (binary) + `lift_progress` (scaled 0-3, gated on grasp) + `action_rate_l2` (penalty) + `alive_penalty` (new, dominant weight, see below) + `success_bonus` (+10, sparse) | **success** (grasped AND height_gain > lift_threshold, within tolerance) **OR timeout** | cube xy spawn pos (`reset_root_state_uniform`, pose_range from existing `init_pos_noise` ±0.06/±0.06), **cube mass randomized heavy** (see below), gripper/table friction |
| **pick-place** | `reach_ee_object` + `grasp_bonus` + `place_progress` (neg xy dist to target, gated on grasp) + `action_rate_l2` + `alive_penalty` + `success_bonus` (+10) | **success** (within 1cm xy + 10° orientation tolerance, released, static) **OR timeout** | cube xy spawn (±0.05/±0.05, existing noise), target xy spawn (±0.05/±0.05, existing `target_pos_noise`), **cube mass randomized heavy**, friction |
| **cylinder-grasp** | per-arm `reach_ee_end` ×2 + per-arm `grasp_bonus` ×2 + `lift_progress` (gated on both grasped) + `action_rate_l2` ×2 + `alive_penalty` + `success_bonus` (+10) | **success** (both grasped AND height_gain > threshold, within tolerance) **OR timeout** | cylinder x spawn (±0.04, existing `init_pos_noise` — y stays centered between arms), **cylinder mass randomized heavy**, friction |
| **cylinder-reach** | per-arm `reach_target` ×2 (neg dist) + `hold_bonus` (both within threshold) + `action_rate_l2` ×2 + `alive_penalty` + `success_bonus` (+10, gated on hold_steps) | **success** (both within 1cm of target, held `hold_steps`) **OR timeout** | **new DR** (cylinder is static/fixed in current env): randomize cylinder x position ±0.05m at reset; both targets derived from cylinder end sites at `reset()` so they move with it automatically; no mass DR here — cylinder has no freejoint (fixed body, mass irrelevant to reach-only task) |

**`alive_penalty` (user-requested):** small negative reward per timestep,
weighted **larger in magnitude than `action_rate_l2`** (e.g. `alive_penalty`
weight ~`-0.05`/step vs `action_rate_l2` ~`-0.01`), scales with episode
length. Purpose: without it, a policy that oscillates near the goal (reach
in → back out → back in) to keep collecting per-step "reach" reward without
ever committing to the terminal state is a classic reward-hacking failure
mode in dense, distance-based rewards. Being dominant over `action_rate_l2`
matters — if the smoothness penalty were larger, the policy could minimize
cost by freezing in place (which `action_rate_l2` rewards) rather than by
finishing quickly; making `alive_penalty` the bigger term keeps "hurry up
and finish" the dominant incentive over "don't move."

**Termination policy — two ways an episode ends (user-requested):**
(1) **success termination**: `terminated=True` when the task's success
condition holds *within a tolerance band*, not exact alignment (see below);
(2) **timeout termination**: `truncated=True` at `max_steps` if success
never triggers. Both paths are real episode-enders. `alive_penalty` being
dominant over `action_rate_l2` is what keeps this combination
reward-hack-resistant: a policy can't profitably stall near success to
farm dense reward, because every extra step costs more via `alive_penalty`
than it could gain by not committing, so the optimal policy is to reach the
tolerance band and terminate as early as possible.

**Success tolerance (user-requested — no exact-alignment requirement):**
success conditions that involve reaching a target pose use a **position
tolerance of 1cm** and, where orientation matters, a **rotation tolerance
of 10°**, instead of the tighter/exact thresholds implied by a naive port
of `env.py`'s existing distance checks. Concretely:
- **pick-lift / cylinder-grasp** (grasp + lift): success = grasped AND
  height gain > `lift_threshold` — no orientation component, only the 1cm
  position-style tolerance applies to any position-based sub-check (e.g.
  grasp distance threshold), height-gain threshold itself is unchanged
  (it's a "lift at least this much" floor, not a target-alignment check).
- **pick-place**: success = cube within **1cm** xy of target position AND
  (if cube orientation is tracked as part of "placed correctly") within
  **10°** of target orientation, released, robot near-static.
- **cylinder-reach**: success = both gripper sites within **1cm** of their
  respective targets, held for `hold_steps` — tighten/replace the existing
  `pos_threshold` (currently 0.035m in `env.py`) with the new 1cm bound.
  Note this is *tighter* than the current 3.5cm threshold — flag as a
  deliberate change and confirm during validation that it's actually
  achievable, not just theoretically defined.

Where an env's existing `env.py` used a threshold looser than 1cm (e.g.
cylinder-reach's 3.5cm), the new tolerance is deliberately tighter — call
this out per-env in each `MDP.md` rather than silently changing behavior.

**`action_rate_l2`** (penalizes `‖action_t − action_{t-1}‖²`) — standard
mjlab/IsaacLab convention for smoother policies.

**Domain randomization — object mass (user-requested "make it heavy"):**
add a `randomize_body_mass` event term (`mode="startup"` or `"reset"`,
`mjlab.envs.mdp.dr`-style, or hand-written if no exact upstream term
matches) on the cube (pick-lift, pick-place) / cylinder (cylinder-grasp)
body. Range design: current masses are `cube: 0.08kg`, `cylinder: 0.05kg`
(from existing `scene.xml`) — randomize toward heavier, e.g. **cube
0.08–0.35kg, cylinder 0.05–0.25kg** (roughly 1x–4x nominal). Exact range is
a first-pass guess — validate empirically that the arm can still lift the
heaviest sampled mass at all before locking it in; tune down if the policy
can never succeed even with random exploration at the top of the range.

## WandB logging (user-requested)

`rl/train.py` logs to Weights & Biases via skrl's built-in W&B tracking
hook (skrl agents accept an `experiment` config dict with
`write_interval`/`wandb`/`wandb_kwargs` — confirm exact key names against
the `thdhyan/skrl` fork's `PPO`/agent config once the submodule is
initialized, since a fork may have renamed/extended these). Log per-task:
episode reward (total + each term broken out — `reach`, `grasp_bonus`,
`lift_progress`/`place_progress`, `action_rate_l2`, `alive_penalty`,
`success_bonus`), success rate, episode length, and sampled DR ranges
(mass, spawn position) for auditability. CLI flag `--wandb-project
so101-mjlab` (or similar) on `rl/train.py`, defaulting to on for real
training runs but skippable for short smoke-test runs (`--no-wandb` or
equivalent) so pipeline-health checks don't spam a W&B project with
throwaway runs.

## MDP.md format (per env)

One `MDP.md` per env under `envs/mujoco/<name>/`, consistent template:

```markdown
# MDP spec — <env name>

## Observation space
<table: field, shape, dtype, description — copied from existing env.py obs dict>

## Action space
<shape, units, range>

## Reward terms
<table: term name, weight, formula, gating condition>

## Termination
<two paths: success (with position/orientation tolerance — state the exact
values, e.g. 1cm / 10°) triggers `terminated`; `max_steps` timeout triggers
`truncated` if success never happens>

## Domain randomization (mjlab EventManager terms)
<table: term, mode (startup/reset), what's randomized, range — including object mass DR>
```

Root-level `MDP.md` is a short index linking to the 4 per-env files
(mirrors `ENVS.md`'s role relative to per-env `README.md`s).

## skrl training setup

- `rl/skrl_wrapper.py`: `SkrlVecEnvWrapper` wrapping `ManagerBasedRlEnv`,
  modeled directly on mjlab's `src/mjlab/rl/vecenv_wrapper.py`
  `RslRlVecEnvWrapper` (confirmed source: exposes `num_envs`, `device`,
  `observation_space`, `action_space`; `step()`/`reset()` translate
  `TensorDict` env output to what skrl's `Wrapper` base class expects).
- `rl/train.py`: PPO via skrl (`skrl.agents.torch.ppo.PPO`), one config per
  task, CLI via `tyro` (mjlab's own CLI convention) — `--task so101_pick_lift
  --num_envs 4096 --headless --wandb-project so101-mjlab`. WandB logging
  wired through skrl's tracker.
- `rl/play.py`: load a checkpoint, roll out in mjlab's viewer for visual
  sanity-check (mirrors `scripts/verify_manual.py`'s role for the
  hand-authored envs).

## Risk to validate early

mjlab's scene-authoring may expect entity-per-body decomposition rather
than pointing at a monolithic `<include>`d MJCF the way our gymnasium envs
do. Build and fully validate **pick-lift only first** (compiles, resets,
steps, renders, trains a short smoke-test) before copy-adapting the same
pattern to the other 3 tasks — avoids repeating a wrong structure 4x.

## Implementation tasks

Status as of this file's creation: `main` pushed to `origin`,
`third_party/mjlab` and `third_party/skrl` submodules added and confirmed
compatible with the `so101` conda env's existing python/mujoco versions.

- [x] Push current `main` state to `origin`.
- [x] Create `feature/mjlab` branch.
- [x] Add `https://github.com/thdhyan/mjlab` as git submodule at
      `third_party/mjlab`.
- [x] Add `https://github.com/thdhyan/skrl` as git submodule at
      `third_party/skrl`.
- [ ] `pip install -e third_party/mjlab` and `pip install -e
      third_party/skrl` into `so101`. Verify `import mjlab, skrl, torch,
      mujoco_warp` and `torch.cuda.is_available()` → `True`.
- [ ] Port **pick-lift only**: `rl/tasks/so101_pick_lift/` (scene, mdp
      terms including `alive_penalty` dominant over `action_rate_l2` and
      mass DR, `env_cfg.py` with success-OR-timeout termination and
      1cm/10° tolerance, task registration) + `rl/skrl_wrapper.py` +
      minimal `rl/train.py` with WandB wiring.
  - [ ] Instantiate via mjlab's registry, `reset()`/`step()` run for
        `num_envs=4` without error.
  - [ ] Reward sanity: finite/non-NaN, `alive_penalty` accumulates
        negatively and dominates `action_rate_l2` for a stalling policy.
  - [ ] Both termination paths reachable: `terminated` fires within
        1cm/10° tolerance under a scripted near-target trajectory;
        `truncated` fires at `max_steps` under a do-nothing policy.
  - [ ] DR sanity: reset 20x, log spawn position + sampled mass, confirm
        both vary within range, positions stay on-table, mass stays ≥
        nominal.
  - [ ] Liftability check: heaviest sampled mass is still liftable under a
        scripted near-optimal grasp+lift trajectory — narrow the mass DR
        range if not.
  - [ ] Short training smoke test (`--num_envs 256 --max_iterations 50`),
        confirm no crash, reward curves move, WandB run appears with
        per-term breakdown logged.
- [ ] Copy-adapt the validated pick-lift structure to pick-place,
      cylinder-grasp, cylinder-reach (each with its own reward/DR/tolerance
      per the table above). Repeat the same validation checklist per env.
- [ ] Write `rl/play.py` (checkpoint playback in mjlab viewer).
- [ ] Write `rl/README.md` (fork URLs, submodule init commands, train/play
      usage).
- [ ] Write the 4 per-env `MDP.md` files + root `MDP.md` index.
- [ ] Update `AGENT_TASKS.md` to reflect the new `rl/` dir, the two
      `third_party/` submodules, and any follow-up items (full training
      runs, hyperparameter tuning — not part of this integration itself).
- [ ] Commit (including `.gitmodules` and pinned submodule commits) and
      push `feature/mjlab` to `origin`.

## Verification checklist

1. `.gitmodules` present with both `third_party/mjlab` and
   `third_party/skrl` pinned; `git submodule status` shows both
   initialized, no `-` prefix.
2. `so101` env: `python -c "import mjlab, skrl, torch, mujoco_warp;
   print(skrl.__version__, torch.cuda.is_available())"` → prints version,
   `True`.
3. Each task's `env_cfg.py` instantiates via mjlab's registry;
   `reset()`/`step()` run for `num_envs=4` without error before scaling to
   full batch size.
4. Reward sanity per task: finite/non-NaN reward over a few hundred
   random-action steps; both `terminated` and `truncated` paths reachable
   (see task checklist above).
5. DR sanity: object/target position and mass vary within configured
   ranges across 20 resets; positions stay on-table; mass never drops
   below nominal.
6. Liftability check (pick-lift/cylinder-grasp): heaviest sampled mass is
   liftable under a scripted trajectory.
7. Short training smoke test per task (`--num_envs 256 --max_iterations
   50`): no crash, reward moves, WandB run logs per-term breakdown,
   success rate, episode length.
8. `AGENT_TASKS.md` updated to reflect this work landing.
9. `feature/mjlab` pushed to `origin` with all work, including submodule
   pins.
