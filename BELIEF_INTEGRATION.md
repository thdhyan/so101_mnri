# BELIEF INTEGRATION — RSS-20-style belief RL into SO-101 MNRI

How to run the belief-driven POMDP stack from the `thdhyan/skrl`
`feature/belief-pomdp-manipulation` fork inside THIS repo's tasks.

Source of the method: `~/Projects/ebasa/Belief-based-pushing` (Franka cube
push) — particle filter over object pose → compressed 24-float GMM-style
vector → policy acts on belief only. Full docs live in that repo's
`HANDOFF.md` and in the fork's `BELIEF_TRANSFER.md` +
`HANDOFF_ISAACLAB_BELIEF.md`.

---

## 0. Prerequisite — point the skrl submodule at the belief branch

`third_party/skrl` is currently pinned at plain `3cdc7f3b Release 2.1.0`.
The belief branch sits directly on top of that commit (additive only), and the
APIs this repo uses (`PPO_CFG`, `ExperimentCfg`, `compute(inputs dict)`) are
unchanged — verified.

```bash
cd third_party/skrl
git fetch origin
git checkout feature/belief-pomdp-manipulation   # 1388269b or newer
cd ../.. && git add third_party/skrl && git commit -m "Bump skrl submodule to belief branch"
# then re-install editable if your venv tracks the checkout:
source .venv/bin/activate && pip install -e third_party/skrl --no-deps
python -c "from skrl.belief import ParticleFilter, OmniResetCfg; print('belief ok')"
```

What you gain: `skrl/belief/torch/*` (particle filter, GMM compression,
entropy), the **OmniReset sampler + Isaac Lab event adapter**
(`skrl/belief/torch/isaaclab_omnireset.py`), `PomdpWrapper`, `BeliefDQN`,
and tests (`pytest third_party/skrl/tests/belief -q`).

---

## 1. Pick the pattern per backend

| Tasks | Pattern | Why |
|---|---|---|
| Isaac Lab: `SO101-*` | **B — filter inside the env** (obs term emits compressed belief) | Vectorized envs auto-reset internally; wrappers can't intercept per-episode resets; scales to 1024+ envs; keeps `rl/train.py`'s stock-PPO flow |
| MuJoCo push twins (`push_t`, `cube_push_ramp`, `cube_push_bridge`) | **A — belief observation wrapper** owning a `ParticleFilter` | Single-instance gymnasium envs; Python sees every reset |

Best first Isaac target: **`SO101-CylReach-Single-v0`** or
**`SO101-PickLift-Single-v0`** — object Z is known/fixed on the table, so the
planar XY(+yaw) filter from the source repo applies as-is; you hide the
currently-GT `object_position` / `cylinder_ends` obs terms.

---

## 2. Isaac Lab tasks (Pattern B) — step by step

### 2.1 Add the planar filter: `envs/isaac/tasks/mdp/belief.py`

Copy from the source repo
(`source/franka_cube_push_project/franka_cube_push_project/tasks/manager_based/cube_push/mdp/belief.py`)
and apply these renames/adaptations:

| Source (Franka push) | SO-101 version |
|---|---|
| `SceneEntityCfg("red_cube")` | `SceneEntityCfg("object")` |
| `ee_frame.data.target_pos_w[:, 0, :]` | same — your `FrameTransformerCfg` already has one target named `"ee"` |
| workspace clamp x∈[0.25,0.75], y∈[−0.28,0.28] | your reachable region (e.g. x∈[0.10,0.35], y∈[−0.20,0.20] around base — check against `CUBE_INIT_POS` ± reset ranges) |
| z constant `0.0203` (cube half-height) | `CUBE_SIZE[2]/2` (or cylinder half-height) |
| goal = fixed green patch | reach/grasp: target marker pos or nominal grasp point; pick_lift: lift column above init pos |
| `BELIEF_NUM_PARTICLES = 256` | keep 256 (cost is negligible vs sim) |

Everything else ports verbatim: contact-gated prediction, asymmetric
contact/free-space likelihoods, ESS resampling, `_two_component_features`
(24-dim output), diagnostics, posterior-sampling viz helpers.

### 2.2 Observation groups (the critical part)

skrl's `IsaacLabWrapper` reads groups BY NAME: `observations["policy"]` →
actor obs, `observations["critic"]` → privileged states. So:

```python
@configclass
class BeliefObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions         = ObsTerm(func=base_mdp.last_action)
        noisy_joint_pos = ObsTerm(func=belief_mdp.noisy_joint_pos, params={"std": 0.0174533})  # 1 deg, paper sensor model
        joint_vel       = ObsTerm(func=base_mdp.joint_vel_rel)
        ee_position     = ObsTerm(func=mdp.observations.ee_position_in_robot_root_frame)  # add if missing
        compressed_belief = ObsTerm(func=belief_mdp.compressed_planar_belief, params={...})  # 24-dim b̃
        ee_to_belief    = ObsTerm(func=belief_mdp.ee_to_belief_mean, params={"num_particles": 256})
        # object_position REMOVED — that's the whole point
        def __post_init__(self):
            self.enable_corruption = False      # belief/noise already modeled; don't corrupt log-vars!
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):                  # privileged, train-only
        joint_pos     = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel     = ObsTerm(func=base_mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.rewards.object_position_in_robot_root_frame)  # GT allowed here ONLY
        actions       = ObsTerm(func=base_mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    critic: CriticCfg = CriticCfg()
```

Rules:
- Group attribute names MUST be exactly `policy` / `critic`.
- Ground truth lives ONLY in `critic`. Actor sees proprioception + belief.
- `enable_corruption = False` on these groups (your current `PolicyCfg` sets
  `True`; corrupting the belief's log-variance dims breaks its scale).
- Symmetric shortcut for a first run: omit `critic` entirely — then
  `env.state()` returns None and everything trains on OBSERVATIONS alone.

### 2.3 Resets: OmniReset + matching belief prior

Replace the plain `reset_root_state_uniform` event with the shared sampler:

```python
from skrl.belief.torch.isaaclab_omnireset import reset_object_pose_omnireset
from skrl.belief.torch.omnireset import OmniResetCfg

reset_object_omnireset = EventTerm(
    func=reset_object_pose_omnireset,
    mode="reset",
    params={
        "object_cfg": SceneEntityCfg("object"),
        "cfg": OmniResetCfg(goal_xy=(<target_x>, <target_y>),
                            start_x_range=(...), start_y_range=(...),
                            workspace_bounds_xy=((...), (...))),
        "z": float(CUBE_SIZE[2] / 2),
    },
)
```

This gives reaching/corridor/near-goal start diversity (0.35/0.30/0.35) and
stores mode ids on `env._skrl_omnireset_mode_ids` (log with
`omnireset_mode_fractions(env)`). Pair with the filter-prior reset term from
the ported `belief.py` (`reset_planar_particle_belief_around_cube` — local +
secondary tangential mode + uniform recovery mass).

### 2.4 Register `-Belief-v0` variants

In each task's `__init__.py`, register e.g.
`SO101-PickLift-Single-Belief-v0` pointing at a
`PickLiftBeliefEnvCfg(ManagerBasedRLEnvCfg)` that swaps in
`BeliefObservationsCfg` + the omni-reset events. Keep the original tasks
untouched (they remain the fully-observable baselines).

### 2.5 Wire into `rl/train.py`

Three small changes:

1. Add the new IDs to `ISAACLAB_TASKS`.
2. Value network input: when the env has a `critic` group,
   `SkrlVecEnvWrapper.state()` returns the flattened critic vector, but
   `SkrlValue` currently sizes its input from `num_observations`. Add:

```python
class SkrlValuePrivileged(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space,
                       state_space=state_space, action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        in_dim = self.num_states or self.num_observations   # critic group size
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
    def compute(self, states, taken_actions=None, role=None):
        x = states["states"] if isinstance(states, dict) and states.get("states") is not None else states
        return self.net(x), {}
```

   (In `compute()`, PPO hands models `states["states"]` when present — the
   wrapper already flattens the critic group there.)
3. Env cfg selection: `train_isaaclab()` reads `spec.kwargs["env_cfg_entry_point"]`,
   which will now resolve to the Belief cfg class automatically for the new IDs.
   Nothing else changes — `build_skrl_ppo` stays stock PPO.

GPU budget note (8 GB laptop): belief tensors are `[num_envs, 256, ...]` —
trivial memory; the sim dominates. Start at `--num-envs 512–1024` (not 4096)
given the PhysX buffer tuning in `__post_init__`.

---

## 3. MuJoCo push twins (Pattern A, ~30 min)

For `push_t` / `cube_push_ramp` / `cube_push_bridge` single-instance envs,
wrap before `DropImagesObs` in `train_mujoco()`:

```python
from examples_style.belief_wrapper import BeliefObservationWrapper  # adapt from third_party/skrl/examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py

env = BeliefObservationWrapper(single_env,
                               hidden=[obj_x_idx, obj_y_idx],   # mask GT in measurement
                               nparticles=200, ng=2)
wrapped = wrap_env(env, wrapper="gymnasium")
```

- `motion_model_fn`: contact-gated push (copy the 15-line heuristic from the
  Franka `belief.py` §prediction) or constant-position + process noise.
- `measurement_model_fn`: Gaussian likelihood on visible dims (gripper/site
  distances).
- Baseline for comparison: wrap with `PomdpWrapper(hidden_obs_indices=[...])`
  instead — masked raw obs, no belief. This is exactly the A/B the CartPole
  proof ran (`--baseline`).
- Reset-time priors: use the wrapper's `OmniResetWrapper`
  (`third_party/skrl/skrl/envs/wrappers/torch/omnireset_wrapper.py`) with a
  `place_state_fn` calling `env.set_state(...)` equivalents.

---

## 4. Visualization

**Ready now:** `envs/isaac/scripts/belief_markers.py` — viewport overlay with
the requested styling: low-alpha particle spheres (opacity 0.06→0.75 ramped by
normalized weight²), yellow→red color ramp on probability mass, two
semi-transparent orange mode outlines, and a cyan "go-here" puck + drop-line
at the belief-mean target. Test it today without the filter (fake bimodal
belief animated in a bare stage):

```bash
python -m envs.isaac.scripts.belief_markers --collapse   # converging cloud
```

Once §2 lands, wire it into a play loop:

```python
overlay = BeliefViewportOverlay(num_particles=256)
overlay.bind_env(env.unwrapped)      # reads env._so101_belief (xy_w/weights)
while simulation_app.is_running():
    obs, _ = env.step(...)           # or your play loop
    overlay.update(); simulation_app.update()
```

- **MuJoCo**: draw `filter.get_particles()[..., :2]` as small spheres
  (`mujoco.user_scn.geoms`, yellow) + GMM means (orange). Posterior-sample
  instead of raw clouds — low-weight stragglers make raw clouds look frozen.
- Scalars worth mirroring to WandB alongside your existing mirror hook:
  `Belief/entropy`, ESS ratio, contact signal,
  `||belief_mean_xy − true_object_xy||`.

---

## 5. Suggested first milestones

1. Submodule bump + import check (§0).
2. `SO101-CylReach-Single-Belief-v0`: symmetric variant first (no critic),
   512 envs, 300 iterations — verify success-rate ≥ fully-observable baseline
   × 0.7 and belief error decaying after first contact.
3. Add privileged critic group + `SkrlValuePrivileged` — expect recovery of
   most of the remaining gap (this matches the source repo's rsl_rl result).
4. Port to `PickLift` (grasp+lift under hidden pose), then dual-arm CylGrasp
   (one filter per arm's object end, or shared filter over cylinder center).
5. MuJoCo push twins via Pattern A for cheap iteration + paper comparisons.

## 6. Repo-specific pitfalls

- Your `PolicyCfg.enable_corruption = True` default: turn OFF for belief terms
  (§2.2) — double-noising log-variance dims silently degrades the filter.
- `wrist_cam` etc.: training path strips cameras already; keep
  `enable_cameras=True` at AppLauncher regardless (existing invariant).
- Silent Isaac deaths = RAM/GPU starvation (~90%, per HANDOFF.md) — close
  other sim jobs before long belief runs; the filter adds host RAM only for
  checkpoints, not per-step overhead.
- Keep the invariant "RL layer never imports sim packages": `belief.py` lives
  in `envs/isaac/tasks/mdp/` (imports isaaclab — fine); anything imported by
  `rl/` must stay pure torch/skrl.
- Dual-arm: two ee frames (`left_/right_gripperframe`); run ONE filter over
  the shared cylinder center with contact evidence OR'd from both arms.
