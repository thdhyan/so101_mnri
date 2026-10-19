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

## Real-robot data collection (single arm)

Leader arm → follower arm + global/wrist cameras → `LeRobotDataset` (**v3.0**,
written by LeRobot's own `lerobot-record`), with a live view in Rerun (native
desktop viewer, or the browser UI over SSH).

```bash
bash scripts/record_so101.sh             # pose gate, then record
bash scripts/record_so101.sh --dry-run   # check config, no hardware
bash scripts/record_so101.sh --profile=12  # time the loop (separate *_profile dataset)
```

Current dataset: `thakk100/pink_duck_in_a_box` → `data/thakk100_pink_duck_in_a_box`
(100 × 20 s at 15 fps, 640×480). Hardware, dataset, task prompts and camera
roles live in `scripts/so101_env.sh`.

### Do not recalibrate the arms

> **Do not recalibrate the arms — the current calibration is verified, and
> recalibrating one arm silently breaks the recording path.**
>
> `robots/so101/calibration/mnri_arms_1_{leader,follower}.json` are the
> **verified** calibration for this rig, and they are what makes the leader and
> the follower agree. Recalibrating one arm silently breaks the recording path:
>
> - The two arms are compared **in degrees**, and those degrees come from these
>   two files. Re-calibrating one arm moves its zero point, so the arms stop
>   agreeing, and the pose gate (`MAX_POSE_DELTA`, default 10°) either blocks
>   every session or lets a 99° mismatch through.
> - `record_loop` calls `send_action()` on its **very first tick**. A mismatch
>   asks six servos to lunge against gravity: they stall, the rail sags, and the
>   Feetech bus starts dropping packets. That presents as a flaky cable, not as
>   a software bug.
> - A calibration is only as good as the sweep. Miss a joint, or sweep it
>   lazily, and the recorded range ends up **wider than the real travel**, so
>   the joint then reads pinned at ~1 % or ~99 % on every read. That is exactly
>   what the follower's `shoulder_lift` had: recorded over the full 0..4095 raw
>   span, when the joint really travels 756..3097.
> - `scripts/calibrate_arm.py` overwrites the file **and the arm's EEPROM**.
>   There is no undo except a backup directory.
>
> The current files were calibrated on 2026-09-25 and checked against the live
> arms. The previous ones are kept in
> `robots/so101/calibration/backup-20260925-215302/`.
>
> If a joint really is at a limit it should not be at, **read** it first and
> correct the range by hand — do not recalibrate:
>
> ```bash
> .venv/bin/python scripts/check_joint_range.py --arm leader --compare
> # raw counts at each physical end, then edit range_min / range_max
> ```
>
> `scripts/repair_calibration.py` writes the motor-side table back, serial-resolved.

### Using the arms

Both arms are 6-motor Feetech STS3215 buses and are identified by **USB
serial, never by `/dev/ttyACM*` order** — enumeration order reverses when you
replug them or change USB ports, and on 2026-09-25 that wrote the follower's
calibration into the leader's servos.

| | serial | rail |
|---|---|---|
| leader | `5AAF263743` | 5 V |
| follower | `5AAF263600` | 12 V |

The rails differ, so do not flag the leader as low-voltage against a 12 V
threshold.

**Before every session:**

```bash
bash scripts/record_so101.sh --dry-run                    # the argv LeRobot will get
.venv/bin/python scripts/check_record_config.py --deep     # really opens both cameras
.venv/bin/python scripts/check_pose_sync.py --watch        # live leader-vs-follower delta
```

Then:

1. Plug both arms in and power them. **One arm check at a time** — serial
   ports are exclusive, and a second open gives
   `SerialException: multiple access on port`, which looks like a dead bus.
2. `bash scripts/resolve_arm_ports.sh` — which port each serial is on today.
3. Bring the **leader** to the follower's pose **by hand**. The launcher runs
   the pose gate itself and refuses to start (exit 2) while any joint is more
   than `MAX_POSE_DELTA` (default 10°) off. Move the arm in your hand; never
   let the follower drive itself into position.
4. Start the live view in a second terminal (optional, but it is how you see
   what the camera actually sees):

   ```bash
   .venv/bin/python scripts/so101_debug_server.py
   # open http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy
   # over SSH: ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <this-host>
   ```

5. Record:

   ```bash
   bash scripts/record_so101.sh
   ```

Recording is **turn-based**: nothing is written until you say the scene is
ready. `SPACE` is the turn key — press it once the duck is placed, and again
after each reset. That is what stops a take from starting while your hand is
still in frame.

Keys (the **recording terminal** must have focus, not the Rerun window):

| key | during the episode | during the reset |
|---|---|---|
| `1`–`5` | pick the task prompt for the next episode | same |
| `i` `o` `k` `l` | duck start zone 1–4 for the next episode | same |
| `z` | clear the pending zone | same |
| `SPACE` | turn key — the scene is ready, start recording | end the wait, start recording |
| `s` / `f` | label success / failure **and end** the episode | (re)label, skip the rest of the reset |
| `d` | discard this take and re-record | discard the episode just recorded |
| `n` | end now, keep it unlabeled | skip the rest of the reset |
| `q` | stop — **wait for `Exiting`**; never Ctrl+C twice (corrupts the dataset) | same |

The prompt and the zone are chosen *before* the take and survive a `d`
re-record, because the duck is still in the same place. Labels go to
`meta/outcomes.json` (sidecar; the v3 schema is untouched):
`python scripts/episode_outcomes.py --root data/thakk100_pink_duck_in_a_box`.

Full guide: **[docs/SO101_COLLECT_HANDOFF.md](docs/SO101_COLLECT_HANDOFF.md)** ·
capture reference: **[docs/SO101_RECORD_SETUP.md](docs/SO101_RECORD_SETUP.md)** ·
capture→train→deploy preprocessing: **[docs/PREPROCESSING.md](docs/PREPROCESSING.md)**

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

The arm calibration files (`robots/so101/calibration/`) are the real hardware's
verified calibration. **Do not recalibrate** — see
[Do not recalibrate the arms](#do-not-recalibrate-the-arms) above for what
that breaks and what to do instead.

## Repo layout

```
envs/
  mujoco/               6 gymnasium envs (see ENVS.md)
  isaac/                Isaac Lab port: 5 registered RL tasks, USD asset, scripts
robots/so101/            shared meshes, MJCF, URDF, arm calibration — single source of truth
rl/                      backend-agnostic training (skrl/rsl_rl): train.py, play.py, agents/
vla/                     VLA finetuning + IL pipeline: record, augment, finetune, rollout, deploy
scripts/                 real-arm data collection, teleop, checks, camera viewer, validation
docs/                    capture reference, collection handoff, preprocessing
third_party/             git submodules: mjlab, skrl (user forks)
IL_PLAN.md               imitation learning plan (duck-push-to-square task)
HANDOFF.md               start here when picking up the repo
test.md                  failure archive from the Aug 2026 rework
```

## Working with other agents

If you're an agent picking up follow-on work in this repo, start at
**[AGENT_TASKS.md](AGENT_TASKS.md)** — open tasks, validation steps, and
constraints (e.g. don't duplicate meshes outside `robots/so101/`).
