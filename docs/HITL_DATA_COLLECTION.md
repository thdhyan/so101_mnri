# HITL Training & Data Collection for SO-101 Duck-Push

Status: **filled progressively** (resilience: file written with full skeleton first, sections rewritten one at a time).
Scope: read-only research + this file. No code or config was modified.
Task: one SO-101 6-DoF arm, imitation learning, single task — *push a pink rubber duck into a square taped on the table*. Leader-arm teleop drives follower. 2 cameras (global `/dev/video2`, wrist `/dev/video4`). Output: stock LeRobot **v3.0** dataset. Target model: SmolVLA (or ACT / GR00T).

---

## 0. What exists and is reusable in this repo

**Recording path (canonical).** `scripts/record_so101.sh` → `scripts/record_so101.py` → stock `lerobot.scripts.lerobot_record.main()`. Patches install as monkeypatches *before* LeRobot runs (established pattern): `so101_torque_guard.py`, `episode_outcome_keys.py`, `so101_camera_flip.py`, `record_loop_profile.py` (`--profile=N`). Dataset `data/thakk100_pink_duck_in_a_box`, stock-loadable v3.0, 640×480, 15 Hz, 6-DoF. Hardware/dataset config in `scripts/so101_env.sh`.

**Outcome labels (existing HITL hook, but training-side dead end today).** `scripts/episode_outcome_keys.py` + `scripts/episode_outcomes.py`: `s`/`f` label + end, `d` discard/retake, `n` end unlabeled, `r` discard+retake. Labels land in `meta/outcomes.json` **sidecar** (`episode_index`, `outcome`, `n_frames`, `recorded_at`) because the LeRobot v3 schema has no outcome field. `episode_outcomes.py` already exports CSV, reports usable fraction. **Nothing reads `outcomes.json` at training time.** Tested by `scripts/test_episode_outcomes.py`.

**Preflight / health (already solves half of section B).** `scripts/check_pose_sync.py` (refuses >10° leader/follower mismatch, `--watch`), `scripts/check_record_config.py --deep` (really opens cameras), `scripts/check_bus_health.py`, `scripts/probe_arm_ports.py`, `scripts/port_id_report.py`, `scripts/resolve_arm_ports.sh`, `scripts/tune_collection_cameras.py`, `scripts/validate_actions.py`, `scripts/record_loop_profile.py`.

**Training / deploy.** `vla/finetune.py` (unified wrapper: ACT, diffusion, VQ-BeT, SmolVLA, Pi0/Pi0Fast, **GR00T N1.7 3B**, XVLA, EVO1, TDMPC; per-backend default steps), `vla/deploy.py`, `vla/rollout.py`, `vla/record_video.py`, `vla/augment.py`, `vla/gr00t_prepare_dataset.py` (v3→v2 for GR00T), `scripts/remote_train.sh`, `scripts/slurm_train.sh`. `docs/PREPROCESSING.md` is the authoritative train/deploy parity contract.

**Second, non-canonical recorder.** `vla/record_demos.py` — hand-rolled, no torque guard, no outcome keys, no flip. Do not use for new data (per `docs/SO101_COLLECT_HANDOFF.md`).

**Sim envs that exist:** `envs/mujoco/so101_single_arm_push_t`, `so101_single_arm_cube_push_bridge`, `so101_single_arm_cube_push_ramp` (+ the other single-arm envs). **No duck-push env, no verified scripted/RL solution policy.**

**Constraints any proposal must respect.** Dataset stays stock-loadable (extra metadata → sidecar). Hardware config in `scripts/so101_env.sh`. Monkeypatches install before LeRobot runs. Never hardcode `/dev/ttyACM*`. Quit with `q`, never Ctrl+C twice (`meta/episodes/` flushes only every 10 episodes or in `finalize()`).

---

## 0b. Hard numbers (measured — do not re-derive)

| | |
|---|---|
| episodes so far | **6**, all labelled `success`, essentially **one duck placement** → state coverage ≈ 0 |
| configured / planned | 100 configured, **300 planned** |
| size | **5.85 MiB/episode, 20.7 KiB/frame** |
| scaling | 100 eps ≈ **0.6 GiB**, 400 eps ≈ **2.3 GiB**; **310 GiB free** |
| bottleneck | **operator hours**, ~**1 min/episode** — *not* storage |
| record loop | camera 30 fps, dataset/control target 15 Hz, achieved **~12.6 Hz** (unprofiled), 640×480 MJPG |
| success labels | 6/6 successes — implausible; `s`/`f` cannot express "sloppy but succeeded" |

> ### 🐞 BUG — `IL_PLAN.md` storage estimate is wrong by ~50×
> `IL_PLAN.md` "Data Volume Estimate" table claims **~1 MB/frame** and **60 GB per 200 episodes** (also "60K frames → ~60 GB"). Measured is **20.7 KiB/frame**, **5.85 MiB/episode** — i.e. 48× smaller per frame, ~50× smaller per episode. Corrected: 200 eps ≈ **1.2 GiB**, 400 eps ≈ **2.3 GiB**. The same file's "Quick Win Strategy" says "100 demos × 2 min/episode = ~3.5 hours", which is also off (should be ~1.7 h at the measured 1 min/ep, or ~3.3 h at 2 min/ep incl. resets). **Storage was never the constraint and the plan should stop treating it as one.** Recommend fixing the table (out of scope for this read-only pass).

---

## A. Intervention / corrective HITL

### A.1 What the literature actually is (and what each needs from the recorder)

| method | mechanism | what the recorder must know |
|---|---|---|
| **DAgger** (Ross et al., 2011) | run current policy, human supplies action, relabel states with expert action, aggregate datasets, mix policies βₖ→0 | per-frame: `policy_action` *and* `human_action` on the same state, plus which one was executed |
| **HG-DAgger / HG-DAgger+** (Mandlekar et al., 2021, "What Matters in Imitation Learning for Robotic Manipulation") | human grabs a **dead-man switch / 6-DoF pose device** only when the policy misbehaves; multi-modal demonstrations from near-optimal *suboptimal* rollouts | binary `intervention` per frame + the human's action on those frames; also needs to tolerate **suboptimal** expert demos |
| **Intervention-gated learning / intervention training** (Jean et al. 2022 "Learning from Intervention"; Bucker et al. 2023 "Beyond DAgger"); also "robust IL from human interventions" (Lynch & Sermanet 2020, *supervision* token) | same signal, used to gate *whose* label trains which frames; often with a learned intervention predictor | exactly HG-DAgger's signal, plus the **proportion** of human frames as a diagnostic |
| **VIPER** (Bharadhwaj 2023) | video-pretrained **reward model** scores trajectory segments → filter/select demos and label per-frame reward | **no** per-frame human signal; needs a reward model over recorded video + a success definition |
| **Shared autonomy / corrective-action supervision** | blend human and policy actions (gated by a trigger), train BC on blended or on corrections only | the blend weight or trigger state per frame |
| **Trigger-based human↔policy switching (RaC, Recovery-and-Correction)** | explicit state machine: AUTONOMOUS ↔ PAUSED ↔ CORRECTING; policy runs, human corrects on demand | phase per frame + `intervention` flag; corrections recorded either as windows (each = episode) or interleaved (sentry-style) |

**Common denominator: three bits per frame/segment are enough** — `(a) who acted (human vs policy)`, `(b) what the human wanted (their action, even when the policy's was executed)`, `(c) did the episode succeed`. Two of those (`c`) already exist as a *sidecar*. `(a)`/`(b)` do not exist anywhere in this repo today.

### A.2 What LeRobot **0.6.1** actually supports — read from `.venv`, not from docs

> Verified by reading installed source at `.venv/lib/python3.12/site-packages/lerobot/`. Everything below is **confirmed in code**, with file:line.

1. **A full DAgger/RaC HITL recorder ships in LeRobot 0.6.1.** `lerobot/rollout/strategies/dagger.py` (741 lines) implements a three-phase state machine (`DAggerPhase.AUTONOMOUS / PAUSED / CORRECTING`), driven by **keyboard or foot pedal** (`input_device: "keyboard" | "pedal"`, `lerobot/rollout/configs.py:162-210`).
   - Modes: `record_autonomous=True` → sentry-like continuous recording, *both* autonomous and correction frames, corrections tagged; `record_autonomous=False` (**default**) → **only correction windows**, each correction becoming its own episode.
   - Frames are written as `"intervention": np.array([True], dtype=bool)` vs `False` (`dagger.py:401, 429, 583`).
   - The `intervention` feature is added to the **new** dataset's schema at creation: `lerobot/rollout/context.py:418-423` (`dataset_features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}`) — only for `DAggerStrategyConfig`.
   - Repo id for rollout datasets must start with `rollout_` (`context.py:426-431`); `lerobot-record` rejects `eval_` prefixes and tells you to use `lerobot-rollout` (`lerobot_record.py:431-436`).
   - **SO-101 is explicitly named** as an actuated teleop with `send_feedback`, so on AUTONOMOUS→PAUSED the leader is smoothly driven to the follower's last position — no handover jerk (`dagger.py` module docstring, `configs.py:205-210`). This is *the* mechanism that makes a leader-arm rig usable for correction.
   - CLI: `lerobot-rollout --strategy.type=dagger` (`rollout/configs.py:230`), strategies `base|sentry|highlight|dagger|episodic`.
2. **Intervention processor steps exist.** `lerobot/processor/hil_processor.py`:
   - `InterventionActionProcessorStep` (`:449`) — overrides the policy action with the teleop action when `info[TeleopEvents.IS_INTERVENTION]`, sets `done`/`reward` from `success`.
   - `AddTeleopActionAsComplimentaryDataStep` (`:94`) — stashes the raw human action under `teleop_action` → **this is DAgger's "human action on policy state"**, already plumbed.
   - `AddTeleopEventsAsInfoStep` (`:131`) — `is_intervention`, `terminate_episode`, `success`, `rerecord_episode`.
   - `RewardClassifierProcessorStep` (`:551`) — runs a pretrained image reward classifier (`lerobot/rewards/classifier/modeling_classifier.py`, HF `AutoModel` vision encoder + head) to auto-emit `reward=1` / terminate on success. **Directly relevant to auto success detection (B).**
   - `GymHILAdapterProcessorStep` (`:315`) — glue for the `gym-hil` env.
3. **`lerobot.rewards` ships pluggable reward models**: `rewards/{classifier,robometer,sarm,topreward}` + `rewards/pretrained.py` (`PreTrainedRewardModel`, `from_pretrained`), `rewards/factory.py`.
4. **VLM annotation is built in.** `scripts/lerobot_annotate.py` (`lerobot-annotate`) populates `language_persistent` and `language_events` columns **directly in `data/chunk-*/file-*.parquet`**, default model `Qwen/Qwen2.5-VL-7B-Instruct`, via `lerobot/annotations/steerable_pipeline/` (config, executor, frames, prompts, reader, staging, validator, vlm_client, writer). Supports `--job.target=h200` HF Jobs. **This is a stock path for machine-episode labelling** (section B/C/F).
5. **Dataset surgery is stock.** `scripts/lerobot_edit_dataset.py` + `datasets/dataset_tools.py`: `delete_episodes`, `split_dataset`, `merge_datasets`, `add_features`, `remove_feature`, `modify_features`, `modify_tasks`, `recompute_stats`. **`add_features` / `modify_features` is how a per-frame or per-episode `outcome`/`intervention` column can be folded into the dataset without a custom writer** — the feature is declared in `meta/info.json`, so the dataset stays loadable by stock LeRobot.
6. **Evaluation script is sim-only.** `scripts/lerobot_eval.py` runs rollouts against `gym` envs and computes metrics; it does **not** do real-robot human-labelled eval (section F needs its own thing).
7. **Nothing at training time consumes `intervention`.** `grep -rn "intervention" policies/` → **no matches**. So today the flag is *recorded and preserved*, not *used*. Same for `success` in policies (`rewards/` is the only consumer path, via `RewardClassifierProcessorStep`, and it is inference-time, not training-time).

### A.3 What SmolVLA / ACT / GR00T support **today** (installed versions)

- **ACT / Diffusion / VQ-BeT / SmolVLA / GR00T / Pi0** are all plain **behaviour-cloning** learners in `lerobot/policies/`: one `observation.*` + `action` pair per frame, `MEAN_STD` (SmolVLA: `VISUAL=IDENTITY`, `STATE/ACTION=MEAN_STD`) normalisation. **No policy config has an intervention input, a value head, or a "who acted" token.** (Verified: no `intervention` references anywhere in `policies/`.)
- **Consequence:** in LeRobot 0.6.1 you can *record* DAgger data with an `intervention` column, but **you cannot train ACT/SmolVLA/GR00T on it in a DAgger-correct way out of the box**. What you *can* do with zero policy code changes is **filter**: train BC on `intervention == False` (or on corrections only, which is the `record_autonomous=False` default — that is just ordinary expert BC on a different state distribution, i.e. DAgger-lite without the aggregation).
- Real DAgger (minimise loss on *executed-action* labels regardless of who acted, i.e. labels = what actually happened, states = policy's own distribution) is the **one-line variant**: drop the `intervention` filter entirely and train on everything, because the recorded `action` column already *is* the executed action. That is the cheapest correct thing, and it needs **no policy changes** — only a decision about filtering. This is the single most important practical finding of section A.
- SmolVLA has `empty_cameras` and language conditioning only (`policies/smolvla/configuration_smolvla.py`); no auxiliary heads. GR00T (`policies/groot/`, `vla/gr00t_prepare_dataset.py`) likewise exposes only obs/state/action/task.

### A.4 Cost of adding intervention recording *to this repo's* teleop path

⚠️ **Unverified — needs a code read before any commitment:** `record_so101.py` currently calls `lerobot_record.main()` (pure teleop, **no policy in the loop**, by design: `lerobot_record.py:17,197` says "no policy inference … use lerobot-rollout"). So **intervention data cannot be recorded by `record_so101.sh` at all** — there is no policy to intervene on. Corrective HITL requires one of:

1. **Use LeRobot's own `lerobot-rollout --strategy.type=dagger`** with `--policy.path=<checkpoint>` and the SO-101 leader as teleop. Zero new recorder code. Cost: it creates a *separate* `rollout_*` dataset with its own schema (`intervention` column) — needs merging back (or just training separately) to stay compatible with the `thakk100_pink_duck_in_a_box` dataset. **Recommend this over writing anything.**
2. Put a policy inside `record_so101.py` — rejected: fights the "stock `lerobot_record`" invariant and duplicates rollout.
3. Record *policy-free* "corrected demonstrations": not DAgger at all, just more demos.

**Dependency:** option 1 needs a **first trained checkpoint** to intervene on. So the ordering is: collect → train → then DAgger. DAgger cannot be P0.

### A.5 Verdict for A

- **Available today, no code:** `lerobot-rollout --strategy.type=dagger` with `intervention` tagging and SO-101 leader handover. **Nothing in this repo uses it yet.**
- **Available today, no code:** plain BC on everything recorded (DAgger-without-aggregation) because the `action` column is the executed action.
- **Not available today, any code:** training-time use of `intervention` by ACT/SmolVLA/GR00T (no policy consumes it), and an intervention flag inside `record_so101.sh`'s teleop-only path.
- **Not worth building:** a custom intervention-trigger overlay on `record_so101.py`. It would need a policy in the loop that `lerobot_record` explicitly does not run — that is `lerobot-rollout`'s job.

---

## B. Automatic QC at collection time

### B.0 Current on-disk state (re-measured 2026-09-26, differs from the brief)

| | |
|---|---|
| `data/thakk100_pink_duck_in_a_box` | **12 episodes, 3180 frames**, task `push the duck into the square tape`, fps 15, 640×480 AV1 |
| disk | **36 MB total** (35 MB videos + 156 KB parquet) → **~3.0 MB/episode, ~11.6 KB/frame** |
| labels in `meta/outcomes.json` | **11 entries**: 8 `success` (0–5, 7, 10), 2 `failure` (8, 9). **Episodes 6 and 11 have no label at all.** |
| episode 11 | **19 frames (1.27 s)** — an aborted take that got saved anyway |
| episode lengths | 10 of 12 are exactly **300 frames = 20.0 s** (hit the `episode_time_s` cap) |
| wall-clock cycle | label timestamps are 22–42 s apart → **~30–40 s per episode incl. reset**, faster than the "~1 min/episode" planning number |

> The brief said "6 real episodes, all `success`". On disk it is now **12 episodes, 8 success / 2 failure / 2 unlabeled**. Either the brief is stale or collection continued. **The 6/6-success problem is therefore already broken — 2 of the recorded outcomes are failures.** Everything below uses the on-disk numbers. Storage conclusion is unchanged by ~2× either way: **storage is not the constraint.**

> ### 🐞 BUG (minor) — `IL_PLAN.md` size table also wrong by ~2× against today's disk
> Measured 3.0 MB/ep vs claimed 300 MB/ep (60 GB ÷ 200). The ~50× factor in section 0b still stands.

### B.1 What is computable **today, from what is already recorded** — no new capture code

All of this reads `meta/episodes/chunk-000/*.parquet` and `data/chunk-000/*.parquet`, both of which already carry per-episode aggregates:

| validator | source | how |
|---|---|---|
| **speed / smoothness / jerk** | `data/chunk-000/file-*.parquet` → `action` (6 f32/frame) + `observation.state` | Δ, Δ², Δ³ per frame per joint; report q99 of |Δ²|. Whole dataset is **3180 rows — microseconds** |
| **joint-limit / workspace violations** | `stats/action/min,max,q01,q99` per episode vs `robots/so101/calibration/mnri_arms_1_follower.json` `range_min/range_max` | pure arithmetic |
| **gripper sanity** | `action.gripper.pos` (0–100) | count limit-to-limit transitions/episode → gripper oscillation. LeRobot already formalises this as `GripperPenaltyProcessorStep` (`processor/hil_processor.py:351`, thresholds `open<0.1`, `close>0.9`, `max_gripper_pos=30` **— its default is wrong for SO-101's 0–100 range, re-check if reused**) |
| **exposure / washed-out** | `stats/observation.images.{global,wrist}/std` per episode | `PREPROCESSING.md` already establishes the calibration: **`std≈3` = YUYV failure, `std≈90` = good**. Threshold on episode-min `std` |
| **near-duplicate episodes** | decode first + last frame of each episode, downsample to e.g. 32×32 grayscale, cosine distance; or DTW on the 6-D `action` trajectory | ~12×3180 numbers, trivial |
| **duration / truncation** | `length` + `episode_time_s` | flag `length == 300` (hit the cap) and `length < ~50` (aborted — **episode 11 would trip this**) |
| **label coverage** | join `meta/outcomes.json` against `episode_index` | flags episodes 6 and 11 today |

### B.2 What is **NOT** computable — two real gaps

1. **Dropped frames / true loop rate leave no trace in the dataset.** Verified: `timestamp` is *synthesised* as `frame_index / fps` — episode 0's last frame is `19.799999` s = `297/15` exactly. So a frame that arrived 40 ms late or was skipped stores identically to one on time. **`meta/` cannot tell you the loop ran at 12.6 Hz instead of 15.** The only diagnostics today are LeRobot's runtime `WARNING … (12.6 Hz)` and `scripts/record_loop_profile.py --profile=N`. If loop-rate QC matters, it needs a **wall-clock sidecar written at record time** (see P1) — it cannot be recovered afterwards.
2. **`frame_index` is always contiguous 0..N−1.** No gap detection possible either.

Blur (Laplacian variance) and occlusion **are** computable but require decoding the AV1 videos once — cache the result; do not re-decode per validator.

### B.3 Automatic success detection for "duck inside the square" — ranked

**1. Classical CV — pink-duck colour segmentation + black-tape quadrilateral fit. Recommended.**
- Pink chroma is a strong, cheap target (HSV hue band or ΔE from a sampled reference patch). Tape is dark-on-light-table → adaptive threshold + `minAreaRect`/Hough. **Duck rotation is a non-issue** (the decision is centroid-inside-quad, not pose). Use the **global camera only** — the wrist view will not reliably contain the square.
- Failure modes, honestly: (a) **`yuv420p` 4:2:0 chroma subsampling** — `PREPROCESSING.md` already warns chroma is quarter-resolution, and both the tape outline and the duck's pink edge *are* chroma features; (b) auto-exposure moving between takes (`auto_exposure=3` is live, not locked); (c) gripper occluding the duck exactly when it matters; (d) tape lifted or dragged → quad fit degenerates; (e) duck overhanging the tape edge → partial containment, needs an explicit "≥50% of footprint inside" rule.
- Cost: minutes of code, no GPU, runs offline on the last ~2 s of each episode. **Sits naturally in a new `scripts/episode_qc.py`** next to `scripts/episode_outcomes.py`.

**2. Small trained detector — only if (1) fails.** YOLOv8-n / RT-DETR-1, or zero-shot Grounding-DINO/OWLv2 on "pink rubber duck" + "tape square". Robust to occlusion and exposure; costs 50–200 boxed frames plus a model artifact to keep in sync with the rig. **Do not start here.**

**3. VLM judge — good auditor, bad gate.** Two stock routes already reachable:
- `lerobot-annotate` (LeRobot 0.6.1, `scripts/lerobot_annotate.py`) with `Qwen/Qwen2.5-VL-7B-Instruct`, writing `language_events` columns **into the parquet** — no custom writer.
- DreamGen Bench's `eval_sr_qwen_whole`: Qwen2.5-VL gives a binary success score per clip; the DreamGen paper reports **>90% Pearson correlation between this auto-judge and human raters** (App. H.3).
- Weakness: VLMs are poor at "is the centroid inside a 10 cm square" — feed a cropped/zoomed frame and an explicit criterion. **Use it to audit the human labels (section C/F), never to auto-discard.**

**4. LeRobot's own reward classifier.** `RewardClassifierProcessorStep` (`processor/hil_processor.py:551`) runs a pretrained `lerobot.rewards.classifier.Classifier` over the observation images and emits `reward=1` / `done`. ⚠️ **Unverified:** I found `rewards/{classifier,robometer,sarm,topreward}` and `rewards/pretrained.py` but **no training entry point** in `lerobot/scripts/` (`lerobot_train.py` is policies only). Whether you can train such a classifier on this repo's data needs a check before it is planned.

### B.4 Auto-label / auto-discard so the human need not press `s`/`f` every take

**Do not auto-discard at record time.** `d` already exists for that, and mid-session deletion interacts with the every-10-episodes `meta/episodes/` flush — a second path to the "killed run = unloadable dataset" trap.

Correct design: **record freely, label offline.**
- `scripts/episode_outcomes.py` already tolerates unlabeled episodes (6 and 11 prove it) and its `record(root, episode_index, outcome, **meta)` takes **arbitrary extra metadata** — so `outcome_source: "auto"`, `confidence`, `duck_centroid_px`, `quad_px` slot in **without a schema change**.
- New `scripts/episode_qc.py`: (a) run the B.1 numeric validators, (b) run the B.3(1) detector on the final frames, (c) write **only missing** entries into `meta/outcomes.json` (human labels always win; `--force` to overwrite). Human then sweeps the report in seconds and overrides with `episode_outcomes.py` instead of pressing a key per take.
- Keeps the dataset stock-loadable: everything stays in the sidecar, which is the established pattern.

---

## C. Using the labels and failures

### C.1 Three things you can do with a label — and what each needs

| | needs in the dataset | available today? |
|---|---|---|
| **Filter** | an episode index list | ✅ **stock**: `.venv/bin/lerobot-edit-dataset --operation.type delete_episodes --operation.episode_indices "[8,9]"` (+ `--new_root` to keep the original, then `recompute_stats`). `datasets/dataset_tools.py:107 delete_episodes` handles video + parquet + `meta/episodes` re-indexing |
| **Relabel** | an outcome source + a writer | ✅ **stock**: `scripts/episode_outcomes.py` (its `record(root, episode_index, outcome, **meta)` accepts extra keys) |
| **Train against the label** (outcome-conditioned / value head / success token) | the label as a **feature the policy reads** | ❌ **not available for any LeRobot policy — verified below** |

### C.2 Verified: no LeRobot policy can consume a success/outcome signal

Read from `.venv/lib/python3.12/site-packages/lerobot/`:

- `reward_delta_indices` returns **`None`** in **every** policy config (`act`, `diffusion`, `vqbet`, `smolvla`, `pi0`, `pi05`, `pi0_fast`, `groot`, `xvla`, `evo1`, `tdmpc`, `eo1`, `fastwam`, `gaussian_actor`, `lingbot_va`, `molmoact2`, `multi_task_dit`, `vla_jepa`, `wall_x`). The plumbing exists (`datasets/factory.py:56` would load a `reward` column if a config asked for it) but **nobody asks**.
- `grep -rn "intervention" policies/` → **no matches**. `grep -rn "success" policies/` → only unrelated hits.
- `language_persistent` / `language_events` columns *are* implemented (`datasets/language.py`, `annotations/steerable_pipeline/`) and a `RenderMessagesStep` exists (`processor/render_messages_processor.py`) — but **`processor/factory.py` and `policies/` never reference it**, so it is not on SmolVLA's or ACT's default pipeline. ⚠️ Unverified: which policy (if any) opts in.

**So: filtering and relabelling are P0-available; outcome-conditioned training requires new model code.**

### C.3 Verdict on the three label-consumption strategies

1. **Filter (drop failures) — do this, but do it last.** Correct default for BC. With 12 episodes and 2 failures, filtering throws away 17% of the *most informative* data; at 300 episodes with a realistic 20–40% failure rate it throws away 60–120 episodes' worth of *states the policy will visit*. **Filtering is right for the first checkpoint, wrong as the end state.**
2. **Relabel / grade — cheap and worth it.** See C.4.
3. **Outcome-conditioned training (success token, RVT/RVLA-style value heads, RL fine-tuning from a value head) — research-only, not worth it here.** It means writing a new policy head and a new training loop, for a **single-task, 300-episode** project where the label set is 2 classes. Reject. If a value signal is ever needed, use LeRobot's **reward-model training** instead (C.5).
   - ⚠️ Do **not** fake it by editing the task string. `lerobot-edit-dataset --operation.type modify_tasks` (`dataset_tools.modify_tasks`) *can* rewrite `tasks.parquet`, but conditioning a single-task policy on a phrasing that encodes the outcome changes the deploy-time prompt for no control benefit.

### C.4 Label noise: the real problem, and how to measure it cheaply

**What's wrong with the labels today (verified on disk):**
- 12 episodes, **8 success / 2 failure / 2 unlabeled** (indices 6 and 11). The brief's "6/6 success" is stale; the current ratio is 80% success, which is more plausible but still unverified by anyone re-watching.
- **`s` and `f` are two different measurements collapsed into one field.** Pressed *during* the episode → labels and **truncates**. Pressed *during* the reset → **re**labels with hindsight over the whole take and **skips the rest of the reset**. Both write `outcome`. Nothing records which happened, so a hindsight label and an in-flight label are indistinguishable in `meta/outcomes.json` and in `--export` CSV.
- **`success | failure | discarded` cannot express "sloppy but succeeded"** — which is exactly the information you want when deciding whether to keep a take. All three of the `VALID` outcomes are also load-bearing for filtering, so adding a fourth outcome would break `--only` semantics; **a separate optional key (`quality`, 0–2) is the schema-safe way** (see B.4: `record()` already takes `**meta`).

**Cheap measurement protocol (≈1 hour, no code):**
1. **Blind re-label**: 24 h later, watch the 12 episodes in random order (use `lerobot-dataset-viz` or a plain `ffmpeg` trim) and label them again **without looking at `outcomes.json`**. Write to a scratch CSV.
2. **Inter-rater agreement**: Cohen's κ between pass-1 and pass-2 (one rater, two passes = test–retest; κ here measures *label reliability*, not validity). Add a second rater if one exists; two raters + the auto-detector from B.3 gives a 3-way κ.
3. **Decision rule**: κ ≥ 0.8 → labels are usable, proceed to 300 episodes. κ 0.6–0.8 → keep labels but treat as noisy (soft: use for *ranking*, not for hard filtering). κ < 0.6 → the `s`/`f` protocol itself is the bug; fix the *criterion* ("duck centroid fully inside the tape quad, arm retracted") before collecting more.
4. **Sampled audit forever after**: re-audit a fixed random 10% of each new batch (30 episodes out of 300 ≈ 10 min of viewing). Report `auto-vs-human` agreement in the same table.

**Relabelling mechanics:** regenerate `meta/outcomes.json` in place with `episode_outcomes.py` (human wins), and let `episode_qc.py` (B.4) fill only the gaps with `outcome_source: "auto"`. Nothing in the dataset changes; filtering still happens by index list.

### C.5 Learning from suboptimal data — supported today vs research-only

| | status |
|---|---|
| **ACT / Diffusion / SmolVLA on imperfect demos** | ✅ supported — they are plain BC, they will happily fit noisy actions. The limitation is quality, not the tool |
| **Filtered / weighted BC** | ⚠️ **half-supported**: filtering yes (C.1); **per-sample weighting no** — no loss-weighting knob in any policy config found. Weighting would mean editing `vla/finetune.py`'s training loop |
| **DAgger-style correction (train on the executed action, whatever the source)** | ✅ needs no policy change — the `action` column already *is* what was executed (section A.5) |
| **Real DAgger with aggregation βₖ→0, reward-weighted regression, IC-GAIL, advantage-weighted BC** | ❌ research-only; new training loop |
| **Value-/reward-conditioned BC** | ❌ see C.3 |
| **Reward model training** | ✅ **entry point exists**: `lerobot-train` with `TrainPipelineConfig.reward_model` (`configs/train.py:83,146-154`, `scripts/lerobot_train.py:307-390`) supports `reward_classifier` (ResNet-10, 2-class, `num_cameras=2`), `sarm`, `robometer`, `topreward` (`rewards/factory.py`). ⚠️ **Unverified:** how labels reach it from an offline LeRobot dataset with no `reward` column — check before planning on it |

---

## D. Collecting less / choosing what to collect

### D.1 Evidence-based answer to "how many episodes?"

| source | number | note |
|---|---|---|
| **LeRobot SmolVLA docs** (authoritative for *this* model) | **"We recommend recording ~50 episodes of your task as a starting point"**; the SmolVLA paper's reference dataset = **50 episodes across 5 distinct cube positions (10 each)**; **"We tried similar dataset with 25 episodes, and it was not enough"** | https://huggingface.co/docs/lerobot/en/smolvla |
| **Lin et al., ICLR 2025 Oral**, *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (40k demos, 15k rollouts) | generalisation follows a **power law in #environments and #objects**; **"diversity of environments and objects is far more important than the absolute number of demonstrations; once the number of demonstrations per environment or object reaches a certain threshold, additional demonstrations have minimal effect."** Their operating point: **50 demos per environment** | https://arxiv.org/abs/2410.18647 |
| **DreamGen** (NVIDIA, arXiv:2505.12705) | real-world gains from **10–13 real trajectories/task** + 40–50 neural trajectories: **SO-100 21% → 45.5%** over 2 tasks | see E |
| Practitioner reports on SO-101 pick-and-place | **75 clean episodes** (narrowed workspace); another: 100 extra demos moved it from "touching" to "grasping" | blogs, weak evidence, but same order of magnitude |

**Synthesis for this task.** A single fixed setup (one table, one duck, one square) is **one environment × one object**. By Lin et al., you are on the *flat* part of the curve past ~50–100 episodes **unless you vary the environment or the object**. For this rig the only real diversity axis available is the **duck's start pose** (and secondarily lighting and square position). Therefore:

> **Answer: 50–150 episodes are enough *if* the duck start pose is randomised over ~6–10 zones; 300 episodes of essentially the same start pose buys close to nothing. The plan's "300 configured" is defensible only as 6–10 poses × 10–20 repeats.**

Concretely: **6 episodes so far from ~1 placement → coverage ≈ 0.** Randomise first, count second.

### D.2 How to spend the operator's time (priority order)

1. **Randomise the duck start pose deliberately, not by accident.** Print / mark **6–10 start zones** on the table next to the square (paper dots, no code). Cycle them: zone *i* → 10–15 episodes. Record which zone in the sidecar (`episode_outcomes.py` already accepts `**meta` → `"start_zone": "B3"`). This is the single highest-value change and it costs **zero engineering**.
2. **Do not collect more from a pose the policy already nails.** Decide "already nails" with an eval (D.4), not by feeling.
3. **Curriculum:** if the far/awkward zones are hopeless, collect ~30 easy-zone episodes first to get *any* working checkpoint, then shift effort to the failure zones. (Curriculum on start pose = simply rebalancing the zone counts; no curriculum-learning machinery needed.)
4. **Active / uncertainty-driven selection (DAgger-flavoured).** Train a checkpoint, run it with `lerobot-rollout --strategy.type=dagger`, and let the *corrections* be the next data. This is section A and it needs a checkpoint first — **P1, not P0**.
5. **Near-duplicate culling** (B.1) before adding episodes: if 40 of your 300 are the same start pose and same trajectory, they are 40 copies of one datapoint.

### D.3 Demonstration *perturbation* and the `IL_PLAN.md` augmentation stack

`IL_PLAN.md` proposes crop / colour-jitter / blur / rotation / horizontal-flip on the **global** cam, plus **joint-angle Gaussian noise** and **time warping**.

| proposal | verdict |
|---|---|
| colour jitter, blur, exposure | ✅ do it — this is exactly the auto-exposure variance you actually have (B.3) |
| random crop + resize | ⚠️ reasonable but **the global cam is the only view that sees the square's absolute position**. Cropping removes absolute pixel coordinates; if the tape quad is what tells the policy "where the goal is", heavy cropping can *hurt*. Use a mild scale (0.9–1.0), not 0.8 |
| random rotation ±15° | ⚠️ same argument, weaker. Keep small |
| **random horizontal flip** | ❌ **reject.** `IL_PLAN.md` justifies it as "square is centred", but the arm base, `shoulder_pan` limits, and the cable/rail are **not** left-right symmetric. A flipped image paired with the *unflipped* joint-space action is a **wrong label**. Never flip joint-space data |
| **Gaussian noise on joint angles (state aug)** | ⚠️ safe on `observation.state` (it simulates backlash). **Do not add noise to `action`** — that is a poisoned label |
| **Time warping ×0.9–1.1** | ✅ safe and useful (retiming a trajectory keeps the geometry). Easy: resample `action`/`state` along `timestamp` |
| **"200 raw → 800 augmented episodes"** | ❌ **misleading framing.** Augmentation does not create information; it constrains *invariances*. It can make 200 episodes behave like 200 episodes seen under 4 lighting/camera conditions. It cannot substitute for duck-pose coverage |

**Risk to flag explicitly:** if crop/jitter/blur/rotation are applied **only to the global camera** (`vla/augment.py`, `IL_PLAN.md` §Data Augmentation), the policy can learn that the global stream is unreliable and **route attention to the wrist camera** — the opposite of `IL_PLAN.md`'s own claim that the global cam carries context. **Test it, don't assume:** train once with and once without global-cam augmentation and compare (D.4). Cheap, and it is the kind of thing that silently decides whether your second camera is worth its bandwidth.

### D.4 Answering "enough?" empirically — the only method that works

Diminishing returns must be **measured, not guessed** (section F for the protocol):

- Train checkpoints at **N = 25, 50, 100, 150, 300** (same recipe, same steps/epoch ratio). Operator cost of this experiment: **~0** — the data was collected anyway.
- Evaluate each on the **same fixed 20-episode hold-out protocol** with a confidence interval (F.2).
- Plot success vs N. The knee is your real number. Everything in D.1 is a prior for where that knee is; this is the measurement.

**Do not** create a held-out *dataset* split from 12 episodes — with 12 episodes a split leaves nothing. Hold out **episodes collected for evaluation** (start poses + a protocol), not a random 20%.

---

## E. Sim + synthetic as a complement — **the Cosmos verdict**

### E.1 The claim under test

> `IL_PLAN.md` (decision tree): *"Use Cosmos to generate 1000s of synthetic duck-push frames, then pretrain GR00T on synthetic + finetune on your 200 real demos."*

**Working position:** *image/video generative models cannot supply behaviour-cloning **action** labels, so Cosmos cannot multiply episodes for plain BC.*

**Verdict: the specific proposal is refuted (it does not do what it says), but the absolute form of the working position is also false — action-labelled synthetic data IS obtainable, via two routes that need far more real data and compute than this project has.**

### E.2 What each Cosmos product actually provides (sourced)

| product | input → output | gives you **action labels**? |
|---|---|---|
| **Cosmos-Predict / Predict2 (Video2World)** | text + image(+video) → **future video** | ❌ pixels only. This is what `IL_PLAN.md` is pointing at |
| **Cosmos-Predict2-2B-Sample-Action-Conditioned** | video + **action** → future video (post-trained on the **Bridge** dataset); NVIDIA publishes a *"[Post-training for Action-Conditioned Video Prediction](https://docs.nvidia.com/cosmos/latest/predict2/post-training_video2world/action-conditioned_video_prediction.html)"* recipe | ⚠️ **Yes, paired data exists — but it is inverted.** You must *supply* the actions (policy / planner / script / your demos). It is a **forward model, not an action source** |
| **Cosmos-Transfer1 / 2.5** | sim frame + spatial control maps (segmentation / depth / edge) → photoreal restyle of the *same* frame | ❌ no actions generated — **but actions are preserved**, because Transfer re-renders a frame you already have an action for. This is the *sim→real* tool |
| **Cosmos-Reason1 / 2** | video → physical/embodied reasoning, usable as a **reward model / judge** | ❌ no actions, no frames. **Useful for sections B and F** |
| **GR00T-Dreams / DreamGen** (pipeline, uses Cosmos-Predict2) | (initial frame, language) → video → **IDM/LAPA pseudo-action** → LeRobot-format paired data | ✅ **Yes — this is the route that actually works.** But it needs (a) real demos to fine-tune the world model, (b) real demos to train the inverse-dynamics model, (c) a lot of GPU |

**Precise statement to replace the working position:**

> A video model *by itself* never produces an action label. Action-labelled synthetic data is possible only by (i) **supplying** actions to an action-conditioned model, or (ii) **recovering** actions from generated video with an inverse-dynamics / latent-action model trained on your own real demos. Both routes therefore *consume* real demonstrations rather than replacing them.

### E.3 What DreamGen/GR00T-Dreams actually needs — against this fleet

Source: [arXiv:2505.12705](https://arxiv.org/abs/2505.12705) + [github.com/NVIDIA/GR00T-Dreams](https://github.com/nvidia/gr00t-dreams) (Apache-2.0, full pipeline: world-model fine-tune → video rollout → IDM pseudo-actions in **LeRobot format** → GR00T N1 fine-tune, 20k steps).

**Maturity: real, complete, and it ships `so100` preprocessing + GR00T fine-tune scripts** (`IDM_dump/scripts/preprocess/…so100`, `IDM_dump/scripts/finetune/…so100`). SO-100 ≈ SO-101 — the closest possible embodiment match. A *new* embodiment needs `modality.json` + `stats.json` + a data config, and `scripts/idm_training.py` on your own demos.

**Data requirements (the killer):**
- Paper fine-tuned video world models on **100 (GR1) to 1,200 (RoboCasa)** real trajectories; the SO-100 experiments used **40–50 real trajectories/task** and 10–13 for the low-data setting.
- *"the more ground-truth data we have about a given robot, the more useful [the IDM's] dynamics model can learn"* — the IDM is trained on **your** demos.
- → **With 12 episodes you cannot train the IDM or the world model.** DreamGen amplifies 40–150 real demos; it does not replace them.

**Compute against the fleet:**

| step | requirement | `dl` (4× RTX 6000 Ada, 48 GB each) | DGX Spark (128 GB unified) |
|---|---|---|---|
| Cosmos-Predict2 **2B** Video2World inference (480p/16fps) | **32.54 GB VRAM**, **181.0 s/clip** on RTX 6000 Ada | ✅ 1 GPU, 4 parallel → ~45 s/clip effective | ✅ 344.6 s/clip |
| Cosmos-Predict2 **14B** Video2World inference | **56.38 GB VRAM**, 876.7 s/clip on Ada | ❌ **>48 GB per GPU** (no TP path verified) | ✅ fits, but **1902 s/clip (32 min)** → 300 clips ≈ 160 h ❌ |
| Cosmos **post-training** (NVIDIA doc) | *"4 GPUs with 80 GB"* | ❌ 48 GB/GPU < 80 GB — ⚠️ needs verification | ✅ 128 GB unified, but ⚠️ untested aarch64 post-training |
| DreamGen at research scale | *"240k-sample RoboCasa dataset took 54 hours on 1500 L40 GPUs"* | — | — |

**Sizing a realistic DreamGen-lite here:** 300 neural trajectories × 181 s (2B, `dl`) ≈ **15 GPU-hours ≈ 4 h on 4×`dl`** for generation — *plus* world-model fine-tuning, IDM training, and GR00T fine-tuning. Feasible in raw compute terms; **in engineering terms it is a multi-day project that still starts by collecting the 50–150 demos you were going to collect anyway.**

### E.4 So: is synthetic paired data worth building? — **No, not now.**

1. **Wrong problem.** DreamGen's headline gains are *behaviour generalisation* (22 new verbs, 0% → 43.2%) and *environment generalisation* (0% → 28.5% in unseen rooms). **Your task has neither.** One duck, one square, one table. Synthetic data buys distribution breadth; you need **pose coverage** (D) and **precision**.
2. **Wrong cost curve.** ~200 more real episodes at **30–40 s/episode ≈ 2–3 hours of operator time**, 0 engineering. Every synthetic route costs days of engineering before producing one frame.
3. **Wrong bootstrap.** Both routes need real demos you don't have yet (world-model fine-tune + IDM).
4. **The `IL_PLAN.md` Cosmos option specifically is broken** — plain Cosmos output has no actions, so "pretrain GR00T on synthetic frames" is not a defined operation for a BC policy.
5. **Storage was never the issue** (0b). The plan's implicit premise (data is scarce because it is big) is false.

> ### 🐞 BUG — `IL_PLAN.md` decision tree should be rewritten
> Replace the Cosmos branch with: *"Synthetic paired data (DreamGen/IDM) needs ≥40 real demos + a fine-tuned video world model + an IDM + 14B-class VRAM. Defer until (a) 100+ real demos exist and (b) a real generalisation requirement appears."* The `NO` branch of that same tree — "just use real demos + standard augmentations" — is the correct one.

### E.5 Sim: worth it, but **as a teacher, not as BC data**

**Build the duck-push env only if you want one of these three things** (which are all real):

1. **Unlimited episodes with exact ground-truth success labels.** Duck pose is known in MuJoCo → the "is it in the square" test is exact. This is the cheapest way to **validate the section-B detector and the section-C filter pipeline** on data where the answer is certain. High value, low effort.
2. **A scripted / RL solution policy to drive HITL intervention data.** Pushing is simple: in sim the duck pose is free, so *"approach → get behind the duck → push toward square centroid → stop"* is a **0.5–1.5 day scripted controller** — **no RL required** (`rl/` exists if you want it, but don't start there).
3. **The one that survives sim-to-real: sim as a *teacher* for intervention data.** Train any policy in sim, then run it on the **real** arm with `lerobot-rollout --strategy.type=dagger` (section A) and correct it by hand. **Every recorded frame is a real camera frame**, so the visual reality gap disappears; DAgger's whole job is to fix the state-distribution gap. This converts "sim" into exactly the data LeRobot already knows how to tag (`intervention=True`). **This is the recommended use of sim in this repo.**

**Why sim is NOT worth it as direct BC data here** (be specific):

| gap | why it bites |
|---|---|
| **action semantics** | the dataset's `action` is the **leader arm's commanded position**, while `observation.state` is the **follower's motor feedback** (`PREPROCESSING.md` Stage 3). A sim env gives you one clean quantity. Emulating teleop noise/lag to match is guesswork |
| **camera extrinsics / FOV / height** | the real global cam is a specific tripod spot; the sim cameras are not calibrated to it. `scripts/real_to_sim_check.py` exists — ⚠️ unverified whether it covers camera pose |
| **appearance** | textures, lighting, the actual pink duck, tape colour + `yuv420p` chroma loss, MJPEG→AV1 artefacts, `auto_exposure=3` drift. No image-based gap compensation is in the repo, and **Cosmos-Transfer restyling would be another project** |
| **latency / rate** | real loop runs at **12.6 Hz against a 15 Hz target** with a 30 fps camera; sim steps are exact. A policy trained on clean 15 Hz sim timing meets real jitter it has never seen |
| **contact physics** | a light, non-convex, slightly deformable rubber duck skids and rolls; MuJoCo friction/contact will not match. Getting *pushing* right is the hard part of this env |
| **no verified scripted solution** | `so101_single_arm_push_t` / `_cube_push_bridge` / `_cube_push_ramp` have **no duck, no square, and no verified solution policy** — a *new* asset, a *new* reward/success test, and a *new* controller, all unvalidated |

**Cost/bottom line for sim:** env + duck asset + tape texture + scripted controller + LeRobot export path ≈ **2–5 days**, with the LeRobot-from-gym export path **unverified** (I did not find a sim→`LeRobotDataset` recorder in this tree; `lerobot-eval` evaluates, it does not record). Against 2–3 hours of teleop. **P2, and only for reasons (1)–(3) above.**

**Cosmos-Transfer1/2.5 is the correct Cosmos product if you ever do want sim→real pixels** — it preserves your actions and restyles the frame. But `Cosmos-Transfer2.5-2B` needs **65.4 GB** → fits a Spark (128 GB), **not** one 48 GB Ada; Transfer1 14B needs ~80 GB. Budget it as another project.

### E.6 Where Cosmos/Reasoning *is* immediately useful: labelling, not generation

`Cosmos-Reason1-7B` (≈18.5 GB BF16, ≥24 GB VRAM; also a 56B variant) runs comfortably on **any** GPU in the fleet and is designed as an **embodied-reasoning judge / reward model** ([cookbook: Cosmos Reason as Reward Model](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/evaluation/reason_as_reward.html)). For *this* project that is worth more than 1,000 generated frames: feed it the last frames of each episode and ask **"is the pink duck inside the taped square?"**, then reconcile against the human labels (C.4). Same role as the Qwen2.5-VL judge in B.3(3) — **pick one, don't build both.**

---

## F. Online / post-hoc evaluation with humans

### F.1 Evaluation is a *separate dataset*, and the tooling already exists

Verified in `.venv`:

- **`lerobot-rollout --strategy.type=episodic`** runs a trained policy on the real arm and records it like `lerobot-record` does (`rollout/configs.py:124`, "mirrors the behavior of `lerobot-record`": `num_episodes`, `episode_time_s`, `reset_time_s`, right/left arrow = end/discard).
- Rollout datasets **must** be named `rollout_*` (`rollout/context.py:426-431`), and `lerobot-record` **rejects** the `eval_` prefix telling you to use `lerobot-rollout` (`lerobot_record.py:431-436`).
- **The existing sidecar works unchanged on a rollout dataset** — `scripts/episode_outcomes.py --root data/<rollout_repo>` only needs a `meta/` dir. So the *entire* human-labelling pipeline from sections B–C applies to evaluation with **zero new code**.
- `lerobot-eval` is **sim-only** (gym envs); it does not help on the real arm.
- `lerobot-dataset-viz` is the viewer for reviewing takes.

**Naming convention to adopt:** `rollout_<model>_<seed>` per evaluated checkpoint, each with its own `meta/outcomes.json`. Nothing in this repo does this yet.

### F.2 How many eval rollouts? (the CI math, for real)

Per-run success rate at 95% (Wilson / normal approx, worst case p = 0.5):

| rollouts n | 95% CI half-width | time at ~40 s/rollout |
|---|---|---|
| 20 | **±20 pp** | ~13 min |
| **50** | **±14 pp** | ~35 min |
| **100** | **±10 pp** | ~70 min |
| 200 | ±7 pp | ~2.2 h |

**Comparing two checkpoints A vs B** (two-proportion, α = 0.05, power 0.8):

- detect **30 pp** (e.g. 50% → 80%): **~36 rollouts per arm** (~25 min each) — *cheap, do this*
- detect **20 pp**: **~80 rollouts per arm** (~55 min each)
- detect **10 pp**: **~315 rollouts per arm** — *not worth it at this stage*

> **Recommendation:** run **50 rollouts per arm** for checkpoint comparisons. That resolves ~±14 pp per arm and reliably catches any *large* regression or improvement, in ~35 min of operator time. Do **not** chase <15 pp differences before you have 100+ training episodes.

### F.3 Interleaved A/B with blinded labels (cheap, removes the two big biases)

Two failure modes kill naive eval here: **environment drift** (tape lifts, lighting shifts, the duck wears) and **operator expectation** (you know which policy you're watching).

Protocol, using only existing tooling:
1. Alternate **A, B, A, B …** in one session (drift hits both arms equally).
2. Record into two datasets `rollout_A` / `rollout_B` (or one, if you keep an `arm` column in a sidecar).
3. **Blind:** export both with `episode_outcomes.py --export`, concatenate, **shuffle, strip the policy identity**, watch, label.
4. Unblind, compute the two proportions + CI (F.2).

Total: no new software beyond a ~50-line report script (P1, §roadmap).

### F.4 Make the evaluator do the watching — the real scalability win

Human labels are the *ground truth for calibrating* the auto-detector (B.3); once `κ(auto, human)` ≥ 0.8 on ~30 episodes, **switch eval to the auto-detector**. Then a 100-episode eval costs **0 minutes of watching** instead of ~70, which changes what experiments you can afford — it makes the D.4 scaling curve (train at N = 25/50/100/150/300) actually runnable.

Two-stage, explicitly:
- **Stage 1 (now):** human labels on every rollout, auto-detector runs *alongside* silently → builds the confusion matrix for free.
- **Stage 2 (after calibration):** auto-detector is primary, human audits a fixed random 10% (C.4 step 4).

### F.5 Held-out human labels as a *training* signal, not just an eval

- Hold out **start zones**, not random episodes (D.4). Zones used for eval are never recorded into `thakk100_pink_duck_in_a_box`.
- A fixed **20–30 episode "golden set" protocol** (fixed start poses, fixed order) is what makes two checkpoints comparable across weeks — otherwise you are comparing different ducks on different days.
- ⚠️ Do **not** create a held-out split out of the 12 existing training episodes.

---

## Roadmap (P0 / P1 / P2)

Effort = my estimate for *this* repo. Files touched are concrete.

### P0 — do before recording more episodes (~0.5 day total, mostly operator time)

| # | item | effort | depends on | files |
|---|---|---|---|---|
| P0.1 | **Mark 6–10 duck start zones on the table and cycle them.** This is the actual fix for "state coverage ≈ 0" (D.1). Log the zone per episode. | 30 min physical + ~20 lines | — | new: none required. optional: `scripts/episode_outcomes.py` `--set-meta` (`record()` already takes `**meta`); `scripts/so101_env.sh` for a `START_ZONES` note |
| P0.2 | **Label episodes 6 and 11, and audit all 12 blind.** Episode 11 (19 frames) should almost certainly become `discarded`. | 30–45 min human, **0 code** | — | none — `scripts/episode_outcomes.py` |
| P0.3 | **Train the first baseline and run the first eval.** Everything else (DAgger, A/B, scaling curve, augmentation ablation) needs a checkpoint. | ~0.5 day (training is the long pole) | — | `vla/finetune.py`, `lerobot-rollout --strategy.type=episodic`, `scripts/episode_outcomes.py` on the `rollout_*` dataset |
| P0.4 | **Fix the two `IL_PLAN.md` errors** (storage table ~50× wrong; Cosmos branch not a defined operation). | 20 min | — | `IL_PLAN.md` — ⚠️ **I did not modify it** (read-only mandate) |
| P0.5 | **Decide the success criterion in writing** ("centroid inside" / "fully inside" / "inside for 1 s") — needed by the detector, the relabel pass and the eval protocol, and they must agree. | 10 min | — | `docs/SO101_COLLECT_HANDOFF.md` (convention) |

### P1 — makes collection measurably better (1–3 days)

| # | item | effort | depends on | files |
|---|---|---|---|---|
| P1.1 | **`scripts/episode_qc.py`** — B.1 numeric validators from `meta/episodes`+`data` parquet, plus B.3(1) pink-duck/tape detector on the final frames; **auto-fills only missing** entries in `meta/outcomes.json` with `outcome_source:"auto"` + `confidence`. | **1–2 days** | P0.5 | new `scripts/episode_qc.py`; reads `data/thakk100_pink_duck_in_a_box/`; reuses `episode_outcomes.record()` untouched |
| P1.2 | **`labeled_when` (`in_episode`\|`reset`) and `quality` (0–2) optional sidecar keys** — fixes C.4's two-measurements-in-one-field and "sloppy but succeeded". | 2–3 h + tests | — | `scripts/episode_outcome_keys.py`, `scripts/episode_outcomes.py`, `scripts/test_episode_outcomes.py` |
| P1.3 | **Blind audit + inter-rater κ report.** | 1 h human + ~60 lines | P1.2 | extend `scripts/episode_outcomes.py --audit CSV` or new `scripts/label_agreement.py` |
| P1.4 | **Wall-clock timing sidecar** — record real per-frame `time.time()` deltas so dropped/late frames stop being invisible (B.2). **Only P1 item that is time-critical: after the 300-episode collection the information is gone forever.** Pattern already exists (`record_loop_profile.py` wraps methods). | 0.5 day | — | new `scripts/record_timing.py`, installed from `scripts/record_so101.py` |
| P1.5 | **Filtered-dataset export** with stock `lerobot-edit-dataset --operation.type delete_episodes` + `recompute_stats`. | 1 h | P1.1/P1.3 | thin wrapper (e.g. `scripts/make_filtered_dataset.sh`) or a documented command in `docs/PREPROCESSING.md` |
| P1.6 | **`scripts/eval_report.py`** — Wilson CI + two-proportion A/B over `rollout_*` datasets and their sidecars (F.2/F.3). | 0.5–1 day | P0.3 | new `scripts/eval_report.py` |
| P1.7 | **First DAgger session: `lerobot-rollout --strategy.type=dagger`** with the SO-101 leader as teleop (A). ⚠️ **Untested on this rig** — verify the SO-101 `send_feedback` handover and that the `intervention` column lands. | 0.5 day, **no new policy code** | P0.3 | ideally a thin `scripts/record_dagger.sh` mirroring `record_so101.sh`; **do not touch `record_so101.py`** |
| P1.8 | **Global-camera augmentation ablation** — train with and without the `IL_PLAN.md` global-cam aug stack; compare on the same eval protocol (D.3). Includes dropping horizontal flip. | 2 training runs + eval | P0.3 | `vla/augment.py` (flag only), `vla/finetune.py` |

### P2 — only if the trigger fires

| # | item | trigger | effort | files |
|---|---|---|---|---|
| P2.1 | **Duck-push sim env + scripted push controller** (no RL) | you want *exact* ground-truth labels to validate P1.1's detector, **or** a teacher policy to feed P1.7 | 2–5 days | new `envs/mujoco/so101_duck_push/`; ⚠️ sim→`LeRobotDataset` export path **unverified in this tree** |
| P2.2 | **Sim policy → real DAgger** (sim as *teacher* of real intervention data, E.5) | P2.1 exists **and** P1.7 validated | 1–2 days | `lerobot-rollout --strategy.type=dagger` |
| P2.3 | **DreamGen / GR00T-Dreams synthetic paired data** | **both** (a) ≥100 real demos **and** (b) a real generalisation requirement (new objects/duck/table) | multi-day + `dl`; **14B blocked at 48 GB/GPU**; 2B path fits (32.54 GB, 181 s/clip on RTX 6000 Ada) | external (`NVIDIA/GR00T-Dreams`, ships `so100` scripts) |
| P2.4 | **Cosmos-Transfer sim→real restyling** | P2.1 exists *and* you actually train on sim images | another project; **65.4 GB → Spark only, not `dl`** | external |
| P2.5 | **Reward model via `lerobot-train --reward_model.*`** | filtered BC plateaus and you need a value/success signal | half a day **after** verifying how labels reach it ⚠️ | `lerobot/rewards/…`; ⚠️ label plumbing **unverified** |
| P2.6 | **VLM audit pass** (`lerobot-annotate` Qwen2.5-VL-7B *or* Cosmos-Reason1-7B) over 10% of episodes | P1.1 detector disputed by human labels | 0.5 day | `lerobot-annotate` CLI (writes parquet columns) — **choose one of the two judges, not both** |

### Explicitly NOT worth it (short list, by design)

1. **Cosmos-as-`IL_PLAN.md`-describes-it** — no action labels; and even the working variants need the demos you don't have (E.4).
2. **Any custom intervention trigger inside `record_so101.py`** — fights the "stock `lerobot_record`" invariant; `lerobot-rollout --strategy.type=dagger` is exactly this feature (A.5).
3. **Outcome-conditioned / value-head policy** — no LeRobot policy reads a reward channel; it means writing a model for a 2-class, single-task project (C.3).
4. **Horizontal-flip augmentation on joint-space actions** — produces wrong labels (D.3).
5. **Auto-discarding episodes at record time** — duplicates `d`, risks the flush trap (B.4).
6. **Chasing <15 pp eval differences** — needs ~315 rollouts/arm (F.2).
7. **A random held-out split of the training dataset** — with 12 (or even 300) episodes it costs more than it gives (D.4/F.5).
8. **More episodes from one start pose** — Lin et al.: diversity beats count past ~50 per setting (D.1).
9. **Cosmos 14B on `dl`** — 56.38 GB > 48 GB per GPU (E.3).
10. **Encoding success into the task string** — no control benefit, breaks the deploy prompt (C.3).

---

## Open questions needing the user's decision

1. **Success criterion, verbatim.** Centroid inside? Fully inside? Held for N frames? Arm retracted? Needed by P1.1, P1.2 and F.5 simultaneously — without it all three disagree.
2. **Is the square position fixed for the whole campaign?** If yes, the global camera's *absolute* pixel geometry is meaningful → keep global-cam augmentation mild. If you move it, augmentation matters much more (and D.3's crop risk flips).
3. **Do you accept machine-written labels in `meta/outcomes.json`** next to human ones, with `outcome_source: "auto"`? (P1.1) Alternative: a separate `meta/qc.json`. Both are stock-compatible; pick one now, it is hard to change later.
4. **Is generalisation ever a goal** (different duck, table, lighting, or a second task)? If **no**, DreamGen/synthetic data is permanently off the table — answer is "build sim only for label validation, or not at all". If **yes**, revisit P2.3 after 100 real demos.
5. **Wall-clock budget for the planned ~300 episodes.** Measured cycle is **30–40 s/episode**, so 300 ≈ **3–4 h**, not the "1 min/episode" in the brief. Is the 300 target driven by a time budget or by a belief about model needs?
6. **P1.4 timing sidecar: yes/no before the mass collection?** It is the only decision that becomes unrecoverable later.
7. **Should I fix `IL_PLAN.md`** (storage table, Cosmos branch, the 100-demo/2-min arithmetic) in a follow-up? Not done here — read-only mandate.
8. **Where does training run?** Laptop (8 GB) vs `dl` (4×48 GB) vs Spark — decides whether `--remote-host` in `vla/finetune.py` is the default path and whether any future Cosmos work is even reachable.
9. **Label taxonomy:** keep `success|failure|discarded` binary + separate `quality` (recommended, schema-safe), or move to a 3- or 4-class outcome? Affects `--only` semantics in `episode_outcomes.py`.
10. **The `eval_` vs `rollout_` prefix split** — `lerobot-record` reserves `eval_`, `lerobot-rollout` demands `rollout_`. Adopt `rollout_<model>_<seed>` as the convention? (F.1)

---

## Bibliography

**Intervention / corrective HITL**
- Ross, Gordon & Bagnell, *Efficient Reduction of Imitation Learning* (DAgger), AISTATS 2011 — http://proceedings.mlr.press/v15/ross11a/ross11a.pdf
- Kelly, Sidrane, Driggs-Campbell & Kochenderfer, *HG-DAgger: Interactive Imitation Learning with Human Experts*, ICRA 2019, pp. 8077–8083 — https://www.semanticscholar.org/paper/HG-DAgger%3A-Interactive-Imitation-Learning-with-Kelly-Sidrane/c8482a7b09a13d08bed1da2348304125ccb1a30a
- Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*, CoRL 2021 — https://arxiv.org/abs/2108.03298
- Spencer et al., *Learning from Interventions: Human-Robot Interaction as Both Explicit and Implicit Feedback*, 2020 — https://personalrobotics.cs.washington.edu/publications/spencer2020eil.pdf
- Saunders et al., *Learning from Interventions* (LfI origin), 2018
- Escontrela et al., *Video Prediction Models as Rewards for Reinforcement Learning* (VIPER), NeurIPS 2023 — https://arxiv.org/abs/2305.14343
- *Robot-Gated Interactive Imitation Learning with Adaptive Intervention Policy* — https://arxiv.org/html/2506.09176v1
- *Compliant Residual DAgger: Improving Real-World Robot Learning* (practical DAgger design) — https://arxiv.org/html/2506.16685v1
- HIL-SERL (human-in-the-loop RL for manipulation) — https://hil-serl.github.io/static/hil-serl-paper.pdf

**LeRobot 0.6.1 internals (read from `.venv`, not docs)**
- DAgger/RaC HITL rollout strategy — `.venv/lib/python3.12/site-packages/lerobot/rollout/strategies/dagger.py`, configs `lerobot/rollout/configs.py:162`, schema hook `lerobot/rollout/context.py:418`
- HIL processor steps (intervention override, teleop-action capture, reward classifier) — `lerobot/processor/hil_processor.py`
- Reward models + training entry point — `lerobot/rewards/factory.py`, `lerobot/configs/train.py:83,146`, `lerobot/scripts/lerobot_train.py:307`
- VLM annotation into parquet — `lerobot/scripts/lerobot_annotate.py`, `lerobot/annotations/steerable_pipeline/`
- Dataset surgery — `lerobot/scripts/lerobot_edit_dataset.py`, `lerobot/datasets/dataset_tools.py`
- LeRobot SmolVLA docs (~50 episodes; 25 not enough) — https://huggingface.co/docs/lerobot/en/smolvla
- LeRobot rollout CLI reference (used above) — `lerobot-rollout --help`

**Synthetic data / Cosmos / DreamGen**
- Jang et al., *DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories* — https://arxiv.org/abs/2505.12705
- NVIDIA, *GR00T-Dreams* (Apache-2.0 full pipeline, `so100` support) — https://github.com/nvidia/gr00t-dreams ; project page https://research.nvidia.com/labs/gear/dreamgen
- NVIDIA Cosmos **Model Matrix** (VRAM + per-GPU timings incl. RTX 6000 Ada, DGX Spark; `…-2B-Sample-Action-Conditioned`) — https://docs.nvidia.com/cosmos/latest/predict2/model_matrix.html
- NVIDIA, *Post-training for Action-Conditioned Video Prediction* — https://docs.nvidia.com/cosmos/latest/predict2/post-training_video2world/action-conditioned_video_prediction.html
- *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control* — https://arxiv.org/abs/2503.14492 ; code https://github.com/nvidia-cosmos/cosmos-transfer1
- *Cosmos-Reason1: From Physical Common Sense To Embodied Multimodal Reasoning* — https://arxiv.org/abs/2503.15558 ; https://research.nvidia.com/labs/cosmos-lab/cosmos-reason1/
- Cosmos cookbook, *Cosmos Reason as Reward Model* — https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/evaluation/reason_as_reward.html
- NVIDIA dev blog, *Enhance Robot Learning with Synthetic Trajectory Data Generated by World Foundation Models* — https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/

**How many demos / scaling**
- Lin et al., *Data Scaling Laws in Imitation Learning for Robotic Manipulation*, ICLR 2025 Oral (40k demos; diversity ≫ count; ~50 demos per setting) — https://arxiv.org/abs/2410.18647 ; https://data-scaling-laws.github.io/
- *SmolVLA: An Open-Source Vision-Language-Action Model* (50 episodes / 11,939 frames on SO-100) — https://arxiv.org/abs/2506.01844
- Chi et al., *ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — https://arxiv.org/abs/2304.13705

---

*Unverified items are marked ⚠️ inline. Nothing outside `docs/HITL_DATA_COLLECTION.md` was created or modified.*




