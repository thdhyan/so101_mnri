# test.md — what went wrong (Aug 2026 rework)

Chronological record of every failure hit while rebuilding the venv,
porting to Isaac Lab, fixing cameras/geometry, and wiring the RL trainers —
with the fix for each. Read this before debugging the same areas.

## 1. Environment / venv

| Problem | Cause | Fix |
|---|---|---|
| `uv pip install isaacsim[all,extscache]==6.0.1.0` failed: `mujoco-usd-converter` not found | uv's index-strategy only checks the first index containing a package | `--index-strategy unsafe-best-match --prerelease=allow` |
| Install died: `No space left on device` on `/` | uv cache (~16 GB) + isaacsim wheels live on the small home partition | `rm -rf ~/.cache/uv`, then `UV_CACHE_DIR=/Storage/uvcache` (exported in `.envrc`) |
| `mjlab` was missing from the old venv entirely | editable install was lost in the conda→uv migration | `uv pip install -e third_party/mjlab` into the new venv |
| mjlab install upgraded `warp-lang` 1.13→1.16 and `rsl-rl-lib` 5.0.1→5.4.0 | mjlab's deps | Harmless: isaacsim bundles its own warp; rsl-rl 5.4 works with isaaclab 3.0.0b2 |
| lerobot downgraded numpy 2.3.1→2.2.6 | lerobot pin | Fine: mjlab needs numpy<2.5 |
| mjlab env reset died: CUDA `vectorized_gather_kernel` device-side assert in `reset_joints_by_offset` (num_envs>1); `EntityData.soft_joint_pos_limits` etc. came back `(1, 6, 2)` instead of `(num_envs, 6, 2)` | mujoco-warp **3.10.0.2+** changed tensor indexing, breaking mjlab (upstream issue mjocolab/mjlab#1093); fork is at upstream main, so no sync helps | `uv pip install "mujoco-warp==3.10.0.1"` (satisfies mjlab's own `~=3.10.0` pin). Re-check the pin after future mjlab/warp upgrades |

## 2. Robot / URDF

| Problem | Cause | Fix |
|---|---|---|
| URDF→USD conversion failed: `ExpatError: not well-formed (invalid token) line 13` | `so101.urdf` header comment contained `--` (illegal inside XML comments). MuJoCo never parsed the URDF (it uses the MJCFs) so it went unnoticed for months | Rewrote the comment line in `robots/so101/so101.urdf` |
| USD written to `assets/so101/so101.usda`, not `assets/so101.usd` | UrdfConverter creates a subfolder named after the asset | Updated `SO101_USD` in `envs/isaac/so101.py` |
| `convert_mimic_connections_to_normal_joints` kwarg rejected | 3.0.0b2 converter API dropped it | Removed the kwarg |
| `gripper_frame_link` initially at risk of being merged away | UrdfConverterCfg `merge_fixed_joints=True` default | `merge_fixed_joints=False` — the TCP link must survive (EE frames + wrist cams attach to it) |
| Body prim paths differ from MJCF names | URDF importer nests under `Geometry/`: `Robot/Geometry/base_link/.../gripper_link/gripper_frame_link` | `gripper_frame_prim()` helper in `so101.py`; FrameTransformer source uses `<spawn>/Geometry/base_link` |

## 3. Isaac Lab API (3.0.0b2 — beta, lots of changes vs 2.x)

| Problem | Cause | Fix |
|---|---|---|
| `No module named 'omni.physics'` creating envs raw | SimulationApp alone doesn't load PhysX | Use `isaaclab.app.AppLauncher` FIRST, before other isaaclab imports |
| `randomize_object_mass` param error | 3.0 renamed `min_mass/max_mass` → `mass_distribution_params=(min,max)` | Updated all env cfgs |
| `CollisionPropertiesCfg(friction_combine=...)` TypeError | kwarg removed in 3.0 | Plain `CollisionPropertiesCfg()` |
| PhysX `PxgCudaDeviceMemoryAllocator failed` (scene creation) | Default 256 MB GPU heap chunks; 8 GB GPU shared with other jobs | `PhysxCfg(gpu_heap_capacity=32MB, gpu_temp_buffer_capacity=16MB, ...)` in every env cfg |
| **Cameras aimed at sky/floor** (global cams) | `CameraCfg.OffsetCfg.rot` is documented (x,y,z,w) but the spawner interprets it as **(w,x,y,z)** — verified empirically (identity authored as xyzw became a 180° Z-flip) | `euler_deg_to_quat()` returns wxyz; see comment in `tasks/common.py` |
| **Wrist cams aimed wrong / stale pose** | (a) same quat-order issue; (b) `update_latest_camera_pose=False` freezes body-attached cams at the pre-reset (zero-pose) pose | Explicit calibrated `rot_wxyz` constants + `update_latest_camera_pose=True` for body-attached cams in `tasks/common.py` |
| Left wrist cam misaimed while right worked | Left quat from offline look-at calibration didn't survive whatever transform the offset path applies | Y-mirror conjugate of the verified right quat: `(w,x,y,z) → (w,-x,y,-z)` — scene is mirror-symmetric |
| Silent death (exit 0, log stops after "SimulationContext cleared") during env creation/reset — RECURRING | ~90% resource starvation (RAM/GPU shared with other sim jobs; 8 GB GPU, 15 GB RAM). ~10%: scene has camera sensors but AppLauncher ran with `enable_cameras=False` | Check `nvidia-smi` + `free -g`, free resources, retry; ALWAYS `enable_cameras=True` when scenes have cameras (train.py/play.py/validate do) |
| `HydraEngine::render failure` + silent death mid-training | RTX renders cameras every `render_interval` steps headless; flaky under memory pressure | Training strips camera sensors from the cfg (state-based RL); renders only via `scripts/render_cameras.py` |
| mjlab training crashed at first PPO update: `output with shape [64, 1] doesn't match the broadcast shape [64, 64]` | `rl/skrl_wrapper.py` returned 1-D reward/terminated/truncated; skrl memory slots are `(N, 1)`, so PPO's time-limit bootstrapping (`next_values (N,1) * truncated (N,)`) broadcast to `(N, N)` | Wrapper now `.view(-1, 1)`s the three scalars, matching skrl's own isaaclab wrapper convention |
| `import lerobot` fails in the main venv: `huggingface-hub>=0.34.0,<1.0 is required ... found 1.28.0` | something (isaacsim dep chain) upgraded huggingface-hub past lerobot 0.6.1's `<1.0` pin | VLA training uses dedicated venvs (`.venv-vla-lerobot`, `.venv-vla-groot`) — see vla/README.md; don't "fix" by downgrading hf-hub in the main venv (isaacsim may need it) |
| `A camera was spawned without --enable_cameras` | Scene contains CameraCfg sensors | Launch with `enable_cameras=True` |

## 4. skrl fork (third_party/skrl) — NEW API, old examples don't work

| Problem | Cause | Fix |
|---|---|---|
| `PPO_DEFAULT_CONFIG` ImportError | Fork moved to dataclass configs | `from skrl.agents.torch.ppo import PPO, PPO_CFG` |
| `wrap_env(env, ml_framework=...)` TypeError | New signature: `wrap_env(env, wrapper="auto")` | `wrapper="gymnasium"` |
| `'SkrlPolicy' object has no attribute 'action_space'` | Mixin `__init__` reads spaces → `Model.__init__` must run FIRST and Model must be in the MRO | `class Policy(GaussianMixin, Model)` + call `Model.__init__` before the mixin |
| `linear(): input must be Tensor, not dict` | Fork's `GaussianMixin.act` passes the WHOLE inputs dict to `compute()` | Unpack inside compute: `states = inputs["states"] or inputs["observations"]` |
| `too many values to unpack (expected 2)` in gaussian act | Fork's compute contract: return `(mean_actions, {"log_std": ...})` — 2 values | Fixed return |
| `clip_predicted_values` kwarg rejected | Removed from PPO_CFG in the fork | Dropped (value_clip remains) |
| Device mismatch (cuda vs cpu) with `--device cpu` | Wrapper/agents parse skrl's GLOBAL `config.torch.device` (defaults to cuda) at construction | Set `skrl_config.torch.device = device` in `main()` BEFORE building anything + `wrapped._device = ...` |
| MuJoCo env obs space is a Dict incl. `images` → `.shape is None`; dict reached the net | RL wants flat state obs | `DropImagesObs` wrapper: strips `images`, flattens Dict→Box (fixed key order) |
| `make_env(n_envs=1)` returns SyncVectorEnv; ObservationWrapper rejected it | EnvHub-style envs are vectorized | `vec.envs[0]` before wrapping |

## 5. MuJoCo cameras / geometry

| Problem | Cause | Fix |
|---|---|---|
| `front_cam` showed sky in every env | euler `(0,55,0)` pitches about Y → view dir (−sin55, 0, −cos55) = toward −X (away from workspace) | euler `(75,0,0)` (roll about X looks down +Y); fixed in XML + env cfgs of all envs |
| Task envs had no global cameras | Only the 2 data-collection envs had overhead_cam/front_cam | Added both to all 4 task envs (XML + CameraConfig lists), poses runtime-customizable |
| Wrist cam XML defaults showed mostly the arm itself | XML defaults ≠ tuned env CameraConfig values | Synced shared follower XMLs to the tuned cfg values |

## 6. Process / environment hazards (non-code)

- **Silent Isaac death checklist** (in order): `free -g` (need >4 GB
  available), `nvidia-smi` (other jobs?), `enable_cameras=True`?, then retry —
  the crash is intermittent even when healthy.
- The user runs other GPU sims concurrently (k1_velocity, g1_warehouse_sim).
  Coordinate before long Isaac jobs; never kill their processes without asking.
- `pip` does not exist in the uv venv — use `uv pip install --python
  .venv/bin/python ...` or activate first.
- Kit logs land in `.venv/lib/python3.12/site-packages/isaacsim/kit/logs/`
  — check the kit_*.log when debugging silent deaths.

## What was validated at handoff time

- MuJoCo: all 6 envs `--headless-check` PASS; zero+random action rollouts
  finite on all 6 (`scripts/validate_actions.py`).
- Isaac: all 4 tasks zero+random action rollouts finite
  (`envs/isaac/scripts/validate_actions.py`); camera renders for all tasks
  (`scripts/render_cameras.py`); skrl training smoke test (3 iterations,
  64 envs) completes with checkpoint.
- MuJoCo backend training smoke test completes with checkpoint (CPU).
- rsl_rl path: configured but only smoke-run via the same wrapper stack —
  run a short job before trusting it.
