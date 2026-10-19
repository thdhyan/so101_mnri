# SO-101 preprocessing — record, train, deploy

Everything that happens to a frame and a joint value between the camera sensor
and the policy's input tensor. A training run and a deployment run must agree
on **all** of it, or the policy silently degrades.

Prose here is orientation. The authoritative numbers are written by LeRobot into
`meta/info.json` and `meta/stats.json` at record time — read them with:

```bash
python scripts/dump_preprocessing_spec.py
python scripts/dump_preprocessing_spec.py --root data/thakk100_so101_single_arm_6dof --json > spec.json
```

Paths below are into the installed LeRobot (`lerobot 0.6.1`).

---

## Stage 0 — sensor, before LeRobot sees anything

This is the stage that quietly ruins datasets, because it happens in the V4L2
driver rather than in Python.

**Always open the cameras with an explicit `fourcc`.** Both collection cameras
default to YUYV, which returns a near-featureless blown-out image. MJPG makes the
camera do the decode and gives back a real picture.

| open | `/dev/video4` mean | std | verdict |
|---|---|---|---|
| default → YUYV | 246.7 | 3.2 | washed out, unusable |
| **MJPG 1280×720** | **126.3** | **90.6** | good |

`std` is the honest health check here, not `mean`. A blank white table reads
`std ≈ 2`; a real scene reads `std ≈ 90`. When the wrist camera is pointed at a
white table, low `std` is expected and fine — it stops being low as soon as the
arm, gripper and objects are in frame.

`exposure_time_absolute` is **not** the knob and should be left alone:

- it saturates — 2500, 3500 and 5000 all returned `mean=247.0`
- on the wrist camera it has no effect at all
- forcing `auto_exposure=1` by hand on `/dev/video4` left it returning uniform
  grey (`std=0.2`) until it was re-opened with an explicit MJPG fourcc

Both cameras therefore run `auto_exposure=3` (Aperture Priority, the UVC
default). `scripts/tune_collection_cameras.py` can change that, but the default
plus MJPG is the known-good combination.

Control values *do* persist across close/open, so anything `--apply` writes
sticks until something resets it.

## Stage 1 — camera read

`lerobot/cameras/opencv/camera_opencv.py`

| step | what happens |
|---|---|
| open | `cv2.VideoCapture(index, cv2.CAP_V4L2)` |
| format | `CAP_PROP_FOURCC = MJPG` — validated, raises if the device refuses (`_validate_fourcc`) |
| size | `CAP_PROP_FRAME_WIDTH/HEIGHT` — validated (`_validate_width_and_height`) |
| rate | `CAP_PROP_FPS` — validated (`_validate_fps`) |
| colour | `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` when `color_mode=rgb` (the default) — line 447 |
| rotation | applied only if `rotation` is set; **not set here** |
| dtype | `uint8`, HWC, exactly `height × width × 3` |

The reader hard-errors if the frame is not 3 channels or does not match the
configured size, so a silent resolution mismatch is not possible here.

**Training and deployment must use the same `width`/`height`/`fps`/`fourcc` and
the same `color_mode=rgb`.** There is no resize anywhere in the pipeline — a
policy trained at 1280×720 fed 640×480 frames will not error, it will just be
wrong.

## Stage 2 — the record loop

`lerobot/scripts/lerobot_record.py`

Per tick, at `dataset.fps` (30 Hz):

```
leader.get_action()  ->  teleop action   (radians-> degrees, see Stage 3)
follower.send_action()
follower.get_observation()
dataset.add_frame({**observation_frame, **action_frame, "task": task})
```

`make_default_processors()` (`lerobot/processor/factory.py:77`) is
**identity only** — three `IdentityProcessorStep` pipelines. Nothing is
normalized, rescaled or reordered while recording. The dataset stores raw
`degrees` / `0-100` / `uint8 RGB`.

Rerun's JPEG compression is display-only and never touches the dataset.

## Stage 3 — joint units and order

`lerobot/robots/so_follower/so_follower.py` and the matching leader file.

| joint | motor id | norm mode |
|---|---|---|
| `shoulder_pan` | 1 | `DEGREES` |
| `shoulder_lift` | 2 | `DEGREES` |
| `elbow_flex` | 3 | `DEGREES` |
| `wrist_flex` | 4 | `DEGREES` |
| `wrist_roll` | 5 | `DEGREES` |
| `gripper` | 6 | `RANGE_0_100` |

Order is the dict order above and is a hard contract. `use_degrees=True` is the
default; setting it `False` switches the body to `RANGE_M100_100` and silently
changes the scale of five of the six channels.

The dataset records the **follower's own motor feedback** as
`observation.state` and the **leader's commanded position** as `action`. They
are not the same quantity — `observation.state` includes tracking error and
follower lag, which is the point.

Calibration lives in `robots/so101/calibration/mnri_arms_1_{leader,follower}.json`
and is passed in with `--robot.calibration_dir` / `--teleop.calibration_dir`.
Raw counts become degrees through `homing_offset` and `range_min`/`range_max`, so
**the same calibration files must be used at train, record and deploy time**.

## Stage 4 — dataset encoding

Images are stored as video, not as raw arrays.

`lerobot/configs/video.py`, `rgb_encoder_defaults()`:

| field | value |
|---|---|
| `vcodec` | `libsvtav1` (AV1) |
| `pix_fmt` | `yuv420p` |
| `crf` | `30` |
| `g` | `2` |
| `preset` | `12` |
| `video_backend` | `pyav` |

Two consequences worth knowing:

- **`yuv420p` is 4:2:0 chroma subsampled.** Luma is full resolution, colour is
  quarter resolution. For thin coloured edges — tape outlines, a coloured gripper
  — that matters. Lower `crf` (e.g. 20) if edge colour fidelity turns out to
  matter for the task; it costs disk and encode time.
- `observation.state` and `action` go to parquet, unnormalized.

`meta/stats.json` (per-feature mean/std/min/max/q01/q99) is computed at the
**end** of recording. It does not exist mid-session.

## Stage 5 — dataset load, at training time

`lerobot/datasets/factory.py:resolve_delta_timestamps` derives the temporal
window from the policy config, not from the dataset:

| key | source | value |
|---|---|---|
| `observation_delta_indices` | `ACTConfig` | `None` → current frame only |
| `action_delta_indices` | `ACTConfig.action_delta_indices` | `range(chunk_size)` = `0..99` |
| `reward_delta_indices` | `None` | not used |

At 30 fps that is an action horizon of `[0.0000 … 3.3000] s`, i.e. **3.33 s of
future actions per sample**, with `n_obs_steps=1` and `n_action_steps=100`.

The consequence for inference: the policy emits a 100-step chunk and you
execute at most 100 of them (or `n_action_steps`) before re-observing. Decoding
3.33 s of future for a task that finishes in 1 s is normal, not a bug.

Images are decoded (`torchcodec` if available, else `pyav`) and returned as
`uint8` by default (`return_uint8=False` gives float).

## Stage 6 — policy preprocessing

`lerobot/processor/factory.py:make_default_pre_post_processors`, order is the
serialized contract:

```
pre:  RenameObservations -> AddBatchDimension -> ToDevice -> Normalize
post: Unnormalize -> ToDevice(cpu)
```

`RenameObservations` has an empty rename map by default, so the dataset's
`observation.*` keys pass through unchanged. This is the hook to use if a
deployment rig names its inputs differently.

Normalization is `MEAN_STD` for all three feature groups (ACT default):

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.MEAN_STD,   # per channel
    "STATE":  NormalizationMode.MEAN_STD,   # per joint
    "ACTION": NormalizationMode.MEAN_STD,   # per joint
}
```

Stats come from `meta/stats.json`. **Recompute them if you re-record or append
episodes** — stale stats are the single most common cause of a policy that
trains to a good loss and behaves badly on the robot.

Note this repo has an 8 GB GPU and 16 GB RAM: MEAN_STD on two 1280×720 camera
streams is the dominant memory cost in training, so expect to reduce image size
or the number of cameras before reaching for a smaller batch.

## Deploy

Deploy must reproduce stages 0–3 and invert stage 6:

1. open both cameras with `fourcc=MJPG`, same `width`/`height`/`fps`, `color_mode=rgb`
2. connect the follower with the **same calibration file** used at record time
3. `get_observation()` → the same `observation.state` key layout
4. run the saved **preprocessor** (not a hand-rolled one) — it carries the
   rename map, the batch-dim step, device placement and the normalization stats
5. `unnormalize` the predicted action, then `send_action()` in degrees

Use the pre/postprocessor that LeRobot saved with the checkpoint
(`POLICY_PREPROCESSOR_DEFAULT_NAME` / `..._POSTPROCESSOR...` JSON files) rather
than reconstructing it — step order is a Hub-serialized contract and easy to get
subtly wrong.

## Quick reference

| | value |
|---|---|
| capture fourcc | `MJPG` |
| resolution | 1280×720 |
| rate | 30 fps |
| colour | `uint8` RGB, HWC (`BGR2RGB`) |
| body joints | degrees |
| gripper | 0–100 |
| joint order | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| record-time processing | identity |
| video codec | `libsvtav1`, `yuv420p`, crf 30, g 2, preset 12 |
| action horizon | 100 steps = 3.33 s @ 30 fps |
| obs steps | 1 |
| normalization | `MEAN_STD` on visual / state / action |
| preprocessing order | Rename → Batch → Device → Normalize |

## The old dataset is not compatible

`data/thakk100_so101_duck_push/` — 22 episodes, 8.7 GB, from the old
`scripts/teleop_record_real.py` path:

- custom `.npz` per episode, **not** a `LeRobotDataset` (no `meta/`, no video)
- **5-DOF** — `observation.state` and `action` are `(N, 5)`, no `wrist_roll`
- 640×480, not 1280×720

It cannot be trained on directly against the 6-DOF setup. Converting it means
adding a `wrist_roll` channel (no source data for it — it would have to be
interpolated or dropped, both of which degrade the data) and re-encoding the
video. It is left untouched; new recordings go to
`thakk100/so101_single_arm_6dof` so the two never share a directory.
