# HANDOFF — SO-101 MNRI (read this first)

For a fresh agent picking up this repo. Read `README.md` and `ENVS.md` for
background, then this file for current state. `test.md` records everything
that went wrong during the Aug 2026 rework — read it before debugging
anything in the venv/Isaac/camera areas.

## What this repo is now

MuJoCo **and** Isaac Sim simulation environments for the SO-101 arm
(single-arm + dual-arm), with a backend-agnostic RL training layer
(skrl / rsl_rl) on top, plus LeRobot-based real-robot teleop.

```
envs/mujoco/          6 gymnasium envs (2 data-collection + 4 RL tasks)
envs/isaac/           Isaac Lab port: 4 registered RL tasks + USD asset
  assets/so101/       USD converted from robots/so101/so101.urdf (DO NOT hand-edit)
  tasks/<task>/       env_cfg.py (scene+MDP), MDP.md (spec + screenshots), images/
  scripts/            render_cameras.py, validate_actions.py
rl/                   backend-agnostic trainers
  train.py            --backend {isaaclab,mujoco,mjlab} --algo {skrl,rsl_rl}
  play.py             checkpoint playback (skrl)
  agents/             shared PPO configs (rsl_rl runner cfg + skrl cfg)
  skrl_wrapper.py     mjlab-specific wrapper (legacy path)
  tasks/so101_pick_lift/   mjlab task (older plan, still functional path)
robots/so101/         canonical robot: meshes, MJCF, URDF (single source of truth)
scripts/              teleop (gamepad IK, leader-arm), camera viewers, validation
third_party/          git submodules: mjlab (user fork), skrl (user fork)
```

## Environment (critical — read test.md too)

- **Python 3.12.13** uv-managed `.venv` at repo root (replaced the old py3.11
  one in Aug 2026). Activate: `source .venv/bin/activate` (or use direnv —
  `.envrc` also sets `OMNI_KIT_ACCEPT_EULA=YES` and `UV_CACHE_DIR=/Storage/uvcache`).
- Installed: `isaacsim==6.0.1.0` (pip, needs `--extra-index-url
  https://pypi.nvidia.com`), `isaaclab==3.0.0b2.post1` (same extra index),
  torch 2.11.0+cu128, mujoco 3.10.0 (+mujoco-warp), mjlab 1.5.0 (editable,
  submodule), skrl 2.1.0 (editable, user fork — NOTE: new dataclass API,
  see test.md), rsl-rl-lib 5.4.0, lerobot 0.6.1 `[feetech]` (real SO-101
  control), isaacteleop 1.3.131 `[retargeters-lite]`, pygame/opencv/etc.
- GPU: RTX 4060 Laptop 8 GB. **Isaac needs ~4 GB RAM + GPU headroom** — if
  other sim jobs run simultaneously, env creation dies SILENTLY (no
  traceback, exit 0). Check `nvidia-smi` and `free -g` first.

## Verified-working commands (as of handoff)

```bash
source .venv/bin/activate

# MuJoCo envs: full validation (all 6 must pass)
MUJOCO_GL=egl python scripts/verify_manual.py --env single --headless-check  # also dual, pick_lift, pick_place, cyl_grasp, cyl_reach
MUJOCO_GL=egl python scripts/validate_actions.py          # zero+random actions, all 6 envs

# Isaac Lab tasks: validation + renders
python -m envs.isaac.scripts.validate_actions             # zero+random, all 4 tasks
python -m envs.isaac.scripts.render_cameras               # screenshots -> tasks/*/images/

# Training (GPU required for isaaclab/mjlab backends)
python -m rl.train --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
python -m rl.train --backend isaaclab --task SO101-CylGrasp-Dual-v0  --algo rsl_rl --num-envs 2048
python -m rl.train --backend mujoco --task pick_lift --algo skrl --max-iterations 200 --device cpu
python -m rl.play --backend mujoco --task pick_lift --checkpoint <path/to/agent.pt>
```

## Registered Isaac tasks

| ID | Arms | MuJoCo twin |
|---|---|---|
| `SO101-PickLift-Single-v0` | 1 | so101_single_arm_pick_lift |
| `SO101-PickPlace-Single-v0` | 1 | so101_single_arm_pick_place |
| `SO101-CylGrasp-Dual-v0` | 2 | so101_dual_arm_cylinder_grasp |
| `SO101-CylReach-Dual-v0` | 2 | so101_dual_arm_cylinder_reach |

Per-task specs (obs/action spaces, reward formulas, termination, DR,
training params, screenshots): `MDP.md` in each task folder + root `MDP.md`
index.

## Key invariants (don't break these)

1. **Robot truth lives in `robots/so101/`** — meshes, MJCF, URDF. The Isaac
   USD is generated: `python -m envs.isaac.convert_urdf`. Never edit the USD.
2. **Joint names** are a stable contract: `shoulder_pan, shoulder_lift,
   elbow_flex, wrist_flex, wrist_roll, gripper` (left_/right_ prefixed in
   dual). Sites: `gripperframe` / `left_gripperframe` / `right_gripperframe`.
3. **Dual-arm geometry**: bases 18 in = 0.4572 m apart in Y
   (±0.2286 m), Z/X axes parallel (identity orientation) — matches the real
   rig. Set in `so101_dual_follower.xml`, `so101_dual.urdf`, and
   `envs/isaac/so101.py` (LEFT_BASE_POS/RIGHT_BASE_POS).
4. **MuJoCo meshdir rule**: `<compiler meshdir=...>` resolves relative to the
   TOP-LEVEL loaded file — only scene.xml sets it, never the included
   follower XMLs (see robots/so101/README.md).
5. **Camera poses are user-customizable** everywhere: MuJoCo via
   `CameraConfig` fields in each env.py (applied at init, overriding XML);
   Isaac via cfg fields in `tasks/common.py` + env_cfg `__post_init__`.
6. **RL layer is backend-agnostic**: skrl/rsl_rl never import sim packages;
   Isaac Lab / MuJoCo / mjlab only provide envs. Keep it that way.

## Known sharp edges

- **Isaac Lab camera offset quats**: `CameraCfg.OffsetCfg.rot` is documented
  (x,y,z,w) but behaves as (w,x,y,z) through the spawner — use the
  `rot_wxyz=` params in `tasks/common.py` (calibrated values, see comments).
- **enable_cameras must be True** whenever an Isaac scene with camera sensors
  is created (training strips cameras from the cfg but still needs the flag).
- The skrl fork (third_party/skrl) uses the NEW dataclass API (`PPO_CFG`,
  `ExperimentCfg`, models take spaces in `Model.__init__`, `compute()` gets
  the inputs dict and must return 2 values). Don't copy old-skrl dict-config
  examples.
- Silent Isaac deaths (exit 0, log just stops after "SimulationContext
  cleared") = resource starvation (RAM/GPU) ~90% of the time. Retry after
  freeing resources before debugging code.

## Open work (priority order)

1. **Full training runs** — smoke tests pass (3 iterations, checkpoint
   saved) on both isaaclab and mujoco backends; nobody has run a real
   training yet. Suggested first target: pick-lift, 4096 envs, 1500 iters.
2. **Image-based RL** (user requirement): needs CNN encoders (ResNet-style,
  CLIP/FiLM, or YOLO-based detectors) in skrl/rsl_rl — neither is wired for
   image obs yet. Isaac TiledCamera/Camera sensors already render; the
   mujoco envs expose `obs["images"]`.
3. **Isaac teleop via isaacteleop**: package installed; the official
   XR→SO-101 example lives in lerobot SOURCE
   (`examples/isaac_teleop_to_so101/`) — requires cloning lerobot and
   `pip install -e ".[feetech,kinematics,dataset]"` (replaces the pip
   lerobot). CloudXR extra + headset needed for VR.
4. **GR00T-WholeBodyControl / GEAR-SONIC** (user-linked): humanoid
   whole-body models — NOT directly applicable to SO-101; treat as research
   references unless a new embodiment appears.
5. **mjlab path**: `--backend mjlab --task so101_pick_lift` exists (older
   plan, MJLAB_INTEGRATION.md); smoke-run it once on a free GPU.
6. Stale docs: root `README.md`/`ENVS.md` still describe only the MuJoCo
   side and reference the removed conda env — refresh when touching them.
