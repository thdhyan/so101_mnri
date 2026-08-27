# SO-101 MNRI

MuJoCo **and** Isaac Sim environments for the SO-101 robot arm (single-arm
and dual-arm), with a backend-agnostic RL training layer (skrl / rsl_rl),
plus teleop and data-collection tooling.

Python 3.12 uv venv at repo root (`.venv`) — `source .venv/bin/activate`
(direnv users: `.envrc` also sets the Isaac EULA + uv cache vars). Full
onboarding: **[HANDOFF.md](HANDOFF.md)**. Failure archive: **[test.md](test.md)**.

## Environments

| | Single arm | Dual arm |
|---|---|---|
| | ![single](images/single_arm_overview.png) | ![dual](images/dual_arm_overview.png) |
| Path | `envs/mujoco/so101_single_arm/` | `envs/mujoco/so101_dual_arm/` |
| Joints | 6 | 12 (2×6) |
| Cameras | 5 (wrist + 2 side + overhead + front) | 6 (2 wrist + 2 side + overhead + front) |

Four fixed-task envs built on the same robot/world pattern:

| | Path | Arms | Task |
|---|---|---|---|
| Pick-lift | `envs/mujoco/so101_single_arm_pick_lift/` | 1 | grasp cube, lift above threshold |
| Pick-and-place | `envs/mujoco/so101_single_arm_pick_place/` | 1 | grasp cube, place on target disc |
| Push-T | `envs/mujoco/so101_single_arm_push_t/` | 1 | push a T-shaped block onto a target outline |
| Cube push (ramp) | `envs/mujoco/so101_single_arm_cube_push_ramp/` | 1 | push a cube along a ramp to a goal patch |
| Cube push (bridge) | `envs/mujoco/so101_single_arm_cube_push_bridge/` | 1 | push a cube across a narrow bridge to a goal patch |
| Cylinder grasp | `envs/mujoco/so101_dual_arm_cylinder_grasp/` | 2 | grasp opposite ends of a free cylinder, lift together |
| Cylinder reach | `envs/mujoco/so101_dual_arm_cylinder_reach/` | 2 | reach target points above a static cylinder's ends |

Full details, camera lists, and observation/action spaces: **[ENVS.md](ENVS.md)**.
Per-task MDP specs (reward formulas, obs/action spaces, termination, DR,
training params, camera screenshots): **[MDP.md](MDP.md)** index.

## Isaac Lab tasks (Isaac Sim 6.0.1)

Five registered gym tasks (single- and dual-arm), defined in `envs/isaac/`:

| Task ID | Arms | Task |
|---|---|---|
| `SO101-PickLift-Single-v0` | 1 | grasp cube, lift above threshold |
| `SO101-PickPlace-Single-v0` | 1 | grasp cube, place on target |
| `SO101-CylReach-Single-v0` | 1 | reach a point above a fixed cylinder's end, hold |
| `SO101-CylGrasp-Dual-v0` | 2 | grasp opposite ends of a thin cylinder, lift together |
| `SO101-CylReach-Dual-v0` | 2 | reach points above a fixed cylinder's ends |

Dual-arm tasks mirror the real rig: follower bases 18 in (0.4572 m) apart in
Y, Z and X axes parallel. Cameras (wrist + global overhead/front) are
configurable; screenshots in each task's `MDP.md`.

```bash
python -m envs.isaac.validate 2>/dev/null || python -m envs.isaac.scripts.validate_actions  # zero+random action check
python -m envs.isaac.scripts.render_cameras    # regenerate task screenshots
```

## RL training (skrl / rsl_rl — backend-agnostic)

The algorithm layer is independent of the sim backend:

```bash
python -m rl.train --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl   --num-envs 4096
python -m rl.train --backend isaaclab --task SO101-CylGrasp-Dual-v0  --algo rsl_rl --num-envs 2048
python -m rl.train --backend mujoco   --task pick_lift --algo skrl --device cpu
python -m rl.play   --backend mujoco  --task pick_lift --checkpoint <agent.pt>
```

All backends log to TensorBoard (`rl/runs/...`) **and** Weights & Biases by
default (`--no-wandb` to disable; `--wandb-entity` / `$WANDB_ENTITY` to pick
a team). Credentials are read from `~/.netrc` (`machine api.wandb.ai`) —
run `wandb login` once if not already authenticated.

Details: **[rl/](rl/)** (`rl/README.md`), shared PPO configs in `rl/agents/`.

## Quick run

Look at an environment (interactive 3D view + live camera streams):

```bash
python scripts/verify_manual.py --env single
python scripts/verify_manual.py --env dual
```

Headless sanity check (CI-safe, exits nonzero on failure):

```bash
python scripts/verify_manual.py --env single --headless-check
python scripts/verify_manual.py --env pick_lift --headless-check   # and pick_place, cyl_grasp, cyl_reach
python verify_envs.py --env both
```

Tile every camera into one grid, save a PNG without opening a window:

```bash
python scripts/view_cameras.py --env single --save single_cams.png
```

Use in Python:

```python
from envs.mujoco.so101_single_arm.env import make_env, SingleArmEnvConfig

env = make_env(n_envs=1, cfg=SingleArmEnvConfig(task="push"))
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Set `MUJOCO_GL=egl` (or `osmesa`) for headless rendering on machines without
a display.

## Imitation Learning (Duck-Push-to-Square)

Train a policy from teleoperation demonstrations instead of RL. Uses
leader arm teleoperation + wrist/global cameras → LeRobotDataset →
SmolVLA / GR00T / ACT finetuning.

```bash
# Record 400 demos with leader arm
python -m vla.record_demos --leader-port /dev/ttyACM0 --dataset thakk100/so101_duck_push \
    --task "push the duck into the square tape" --episodes 400

# Train (SmolVLA on laptop, ~15 min)
python -m vla.finetune --backend smolvla --dataset thakk100/so101_duck_push --steps 10000

# Deploy on real robot
python -m vla.deploy --backend smolvla --checkpoint vla/runs/smolvla/last
```

Full plan: **[IL_PLAN.md](IL_PLAN.md)** | Pipeline details: **[vla/](vla/)**

## Teleoperation

```bash
python scripts/teleop_gamepad_ik.py --env single         # gamepad -> Cartesian IK
python scripts/teleop_leader_arm.py --env single          # real SO-101 leader arm -> sim
```

Details, control mappings, and the `--dry-run` flags for hardware-free
testing: **[scripts/README.md](scripts/README.md)**.

`teleop_gamepad_ik.py`'s `IKTeleop` class is the integration point for
upcoming [mujoco-ar-viewer](https://github.com/Improbable-AI/mujoco-ar-viewer)
AR-driven teleop/data collection — the IK/physics loop doesn't change, only
the source of `set_ee_target()` calls.

## Robot definitions

Meshes, MJCF kinematics, and URDFs are canonical under `robots/so101/`
(shared by both envs, not duplicated) — see `robots/so101/README.md`.

## Repo layout

```
envs/
  mujoco/               6 gymnasium envs (see ENVS.md)
  isaac/                Isaac Lab port: 5 registered RL tasks, USD asset, scripts
robots/so101/            shared meshes, MJCF, URDF — single source of truth
rl/                      backend-agnostic training (skrl/rsl_rl): train.py, play.py, agents/
vla/                     VLA finetuning + IL pipeline: record, augment, finetune, rollout, deploy
scripts/                 teleop, camera viewer, manual verification, validation
third_party/             git submodules: mjlab, skrl (user forks)
IL_PLAN.md               imitation learning plan (duck-push-to-square task)
HANDOFF.md               start here when picking up the repo
test.md                  failure archive from the Aug 2026 rework
```

## Working with other agents

If you're an agent picking up follow-on work in this repo, start at
**[AGENT_TASKS.md](AGENT_TASKS.md)** — open tasks, validation steps, and
constraints (e.g. don't duplicate meshes outside `robots/so101/`).
