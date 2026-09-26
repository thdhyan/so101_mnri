# SO-101 single-arm data collection — setup

Leader arm → follower arm + 2 cameras → `LeRobotDataset` (v3.0), with a live
view in Rerun's native desktop viewer. This is the real-hardware path; the
MuJoCo/Isaac envs in this repo are separate.

## The two terminals

```bash
cd /Storage/Projects/so101_mnri
source .venv/bin/activate
```

```bash
bash scripts/record_so101.sh
```

That is the whole thing. The live view is Rerun's own native viewer, spawned
automatically because `--display_data` is passed with no `display_ip`/
`display_port`, which makes LeRobot call `rr.spawn()`. No proxy, no second
process, no URL to copy.

```
 leader   /dev/ttyACM0  (serial 5AAF263743, id=mnri_arms_1_leader)
 ...
 live view native Rerun window (spawned automatically)
 labels    s=success+end  f=failure+end  d=discard+re-record  n=end+keep  q=quit
```

### Stopping safely — quit with `q`, never Ctrl+C twice

LeRobot buffers the per-episode index (`meta/episodes/`) and flushes it only
every 10 episodes or in `finalize()`; the frame parquet gets its footer only
when its writer closes. `q` runs `finalize()`. A second Ctrl+C (or a kill)
during finalize leaves `info.json` claiming N episodes with no
`meta/episodes/` and a footer-less parquet. The next resume then falls back to
the Hub and dies with `RepositoryNotFoundError: 404`. That dataset is
unrecoverable -- this happened on 2026-09-25 and cost 8 episodes. Press `q`,
wait for `Exiting`.

### Profiling the loop

```bash
bash scripts/record_so101.sh --profile=12
```

Records ~12 s (timed from the first frame, not launch) then prints a per-call
breakdown. It writes to a separate `<REPO_ID>_profile` dataset because LeRobot
saves the cut-off take -- delete that folder afterwards.

### Other live-view modes

| | |
|---|---|
| `DISPLAY_DATA=false` | no visualisation at all — cheapest, and the data is unaffected |
| `RERUN_WEB=1` | browser view; also run `python scripts/so101_debug_server.py` first |

`so101_debug_server.py` is only needed for the browser mode, which is for
headless / SSH sessions where no desktop window can open. Over SSH it also needs
**both** ports forwarded, because the browser needs the HTML on 9090 *and* the
gRPC sink on 9876:

```bash
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 thakk100@dhyan-LOQ
```

Then open the exact URL the server prints — the `?url=` part is not optional
(see below).

Ports are env-overridable via `scripts/so101_env.sh` — edit that file, not the
scripts. USB enumeration order is not stable across reboots.

### `--dataset.root` is the dataset directory, not a parent

This bites on the first run if you get it wrong, and the error misleads:

```
FileExistsError: [Errno 17] File exists: 'data'
```

`LeRobotDatasetMetadata.create()` does `obj.root.mkdir(parents=True,
exist_ok=False)` (`lerobot/datasets/dataset_metadata.py:832`). So `root` is the
full path to the dataset directory and must **not** already exist. Passing the
bare parent `data` makes LeRobot try to create `data/` itself, which of course
already exists.

`record_so101.sh` computes the full path from `REPO_ID`:

```bash
DATASET_ROOT="${DATA_PARENT}/$(printf '%s' "$REPO_ID" | tr '/' '_')"
# -> data/thakk100_pink_duck_in_a_box
```

On a second run that directory *does* exist, so the script auto-switches to
`--resume=true`. If it exists but has no `meta/info.json`, the script refuses to
start rather than writing into whatever is there.

Verified directly: create → ok, second `create` → `FileExistsError`, `resume` →
ok with 0 episodes.

## Success / failure / retry

LeRobot's dataset schema (v3.0) has no outcome field, and its keyboard
controls only cover flow control — `n`/right = end episode, `r`/left = discard
and re-record, `q`/esc = quit. There is no way to say whether a take was
actually any good.

Recording now adds:

| key | meaning |
|---|---|
| key | during the episode | during the reset |
|---|---|---|
| `s` | label **success** and end it now | (re)label success, skip rest of reset |
| `f` | label **failure** and end it now | (re)label failure, skip rest of reset |
| `d` | **discard** this take, re-record | discard the episode just recorded, re-record |
| `n` / right | end now, keep **unlabeled** (LeRobot's own) | skip rest of reset |
| `r` / left | discard and re-record (LeRobot's own) | same |
| `q` / esc | stop, finalize, exit (LeRobot's own) | same |

LeRobot calls `save_episode()` only *after* the reset phase, so a label pressed
during the episode or its reset lands on that episode; one pressed after the
`[outcome] unlabeled` line applies to the *next* one. A discard also clears any
label pressed during the dropped take. The banner at startup repeats the keys.

Keys come from the **terminal** (pynput is not installed, so LeRobot uses its
TTY reader): keep the recording terminal focused, not the Rerun window.

Typical cycle: do the task → `s`/`f` → reset the scene → `n` to skip the rest
of the 60 s reset.

These hook into LeRobot's real key pipeline — `create_key_listener` is the single
seam both backends (pynput and the terminal reader) funnel through — rather than
prompting with `input()`, which would fight the key listener for the TTY.

### Where the labels live

`meta/outcomes.json` beside the dataset, **not** in the schema:

```json
{"version": 1,
 "episodes": {"0": {"outcome": "success", "n_frames": 300,
                    "recorded_at": "2026-09-25T22:04:11+0000"}}}
```

The dataset stays loadable by stock LeRobot and the labels survive a LeRobot
upgrade. Failed episodes stay on disk; filter at train time. A discarded (`d`)
take is never saved -- LeRobot drops its buffer. `n_frames` is that episode's
own length.

```bash
python scripts/episode_outcomes.py --root data/thakk100_pink_duck_in_a_box
python scripts/episode_outcomes.py --root ... --export successes.csv --only success
```

```
episodes in dataset: 3
labeled:            2
  success        1  #
  failure        1  #
  discarded      0
  unlabeled      1  (no s/f pressed for these)
  1/3 usable for success-only training (33%)
```

`episode_outcomes.py` writes atomically (temp file + rename) so a crash mid-write
cannot leave a corrupt label file, and rejects outcomes outside
`success|failure|discarded`.

Offline check, no hardware: `python scripts/test_episode_outcomes.py`.

## The optional web debug server

`scripts/so101_debug_server.py` runs two things:

| | |
|---|---|
| **gRPC sink** on `:9876` | buffers every logged frame server-side |
| **Web viewer** on `:9090` | the browser UI, served over HTTP |

`lerobot-record --display_data --display_ip 127.0.0.1 --display_port 9876`
connects to the sink and streams into it; the browser pulls from the sink and
renders it. Because the sink **buffers**, you can open the page late or
reconnect and still scrub back through the whole session — which you cannot do
with `rerun.spawn()`, the native-viewer path that plain `lerobot-record` uses.

Two gotchas that are easy to hit:

- `rr.init()` must run **before** `rr.serve_grpc()`. Otherwise `serve_grpc()`
  silently returns the string `"[rerun is disabled]"` and logs nothing useful.
- Detecting a connected producer with a `connect()` to the gRPC port always
  succeeds, because the sink is itself listening there. Read the ESTABLISHED
  entries in `/proc/net/tcp` instead (state `01`).

### The `?url=` parameter is not optional

Opening bare `http://localhost:9090` always shows the **"Welcome to rerun"**
screen with an empty Sources panel, no matter how much data is streaming. That
is not a broken connection — the served HTML is completely static, and it has
no idea where the sink is.

The sink address is passed to the page as a query parameter. Rerun builds it in
`crates/top/re_sdk/src/web_viewer.rs`:

```rust
let viewer_url = format!(
    "{http_web_viewer_url}?url=rerun%2Bhttp://localhost:{grpc_port}/proxy"
);
```

and percent-encodes the value with `NON_ALPHANUMERIC`, which is why `+`, `:` and
`/` all appear as `%2B`, `%3A`, `%2F`. `serve_web_viewer(connect_to=...)` only
feeds `webbrowser::open()` — it does not change what is served, and it has no
effect at all when `open_browser=False`. So the debug server prints the finished
URL and you open that.

Verified: with the producer connected, `?url=...` returns HTTP 200, the wasm
loads (41 MB `re_viewer_bg.wasm`, 227 KB `re_viewer.js`), and `/proc/net/tcp`
shows state `01` on port 9876.

Note both ports must be reachable from wherever the browser runs. Over SSH that
means two `-L` forwards, not one.

## Hardware as found

| device | what | notes |
|---|---|---|
| arm, USB serial `5AAF263743` | **leader** — the arm in your hand | |
| arm, USB serial `5AAF263600` | **follower** — the arm on the table | |
| `/dev/video2` | `Innomaker-U20CAM-720P` | **wrist**, 1280x720 |
| `/dev/video4` | `USB Camera` | **global**, 1920x1080 native |
| `/dev/video0` | laptop `Integrated Camera` | unused — black frame |

Odd `/dev/videoN` indices are UVC *metadata* nodes, not capture devices. Only
even indices stream.

### Identify arms by USB serial, never by `/dev/ttyACM*`

`/dev/ttyACM*` order follows USB enumeration and **reverses unpredictably**.
On 2026-09-25, simply moving to a different USB port flipped both arms:

| serial | role | before | after |
|---|---|---|---|
| `5AAF263743` | leader | ttyACM1 | **ttyACM0** |
| `5AAF263600` | follower | ttyACM0 | **ttyACM1** |

The hardcoded config was then exactly backwards, and the run wrote the
**follower's calibration into the leader's servos**. Serilog recovery is in
`scripts/repair_calibration.py`; the durable fix is that `record_so101.sh` now
resolves serials → ports at launch via `scripts/resolve_arm_ports.sh` and aborts
if either is missing.

```bash
bash scripts/resolve_arm_ports.sh     # show the current mapping
```

Never hardcode a `ttyACM` path for an arm.

### The two arms run different rails

| arm | rail | `Present_Voltage` (raw, 0.1 V) |
|---|---|---|
| leader | 5 V | `49` → 4.9 V |
| follower | 12 V | `119` → 11.9 V |

The leader only reads positions, so 5 V is correct for it. Do not flag the
leader as low-voltage against a 12 V threshold.

## Cameras: the exposure trap

Both cameras ship with `auto_exposure=3` (Aperture Priority). The real fix is
not an exposure value, it is the **pixel format**.

**Open with `fourcc="MJPG"`.** A default open negotiates YUYV, and both cameras
then return near-featureless blown-out frames:

| open | `/dev/video4` mean | std | verdict |
|---|---|---|---|
| default (YUYV) | 246.7 | 3.2 | washed out |
| **MJPG 1280x720** | **126.3** | **90.6** | good |

A flat wall reads `std=2.2`; a real scene reads `std=90`. If `std` is near
zero the frame is useless, regardless of what `mean` says.

`sixteen`-bit detail: `exposure_time_absolute` is not the knob. It saturates —
values 2500/3500/5000 all returned `mean=247.0` — and on the wrist cam it has no
effect at all. Setting `auto_exposure=1` by hand on `/dev/video4` actually left
it returning **uniform grey** (`std=0.2`) until it was re-opened with an explicit
MJPG fourcc. The control values do persist across closes, so `--apply` in
`scripts/tune_collection_cameras.py` sticks, but MJPG is what actually fixes it.

`scripts/tune_collection_cameras.py --sweep` saves a frame per step and reports
mean / std / blown-% if you need to re-verify after a lighting change.

## Identifying the leader and follower

Both arms are electrically identical 6-motor Feetech STS3215 buses, so there is
no way to tell them apart by reading IDs. The calibration ranges are not
decisive either — both ports matched 5–6 of 6 joints against both files.

So: move an arm and watch which port moves. The probe is read-only — it never
writes calibration, never enables torque, and never sends a target.

```bash
# already running in tmux session `so101id`
tmux attach -t so101id          # watch the numbers live
```

Move **one** arm — preferably the leader, the one in your hand. Then:

```bash
python scripts/port_id_report.py
```

**Result on 2026-09-25: `/dev/ttyACM1` is the leader, `/dev/ttyACM0` is the
follower.** Already written into `scripts/so101_env.sh`. Re-run the probe if the
arms get replugged — USB enumeration order is not stable.

Watch out when reading the report early: the probe keeps running, so if you
move an arm and immediately run the report you can read a file that predates the
movement and get a confident "no arm moved". If that happens, wait a couple of
seconds and run it again.

If you moved the arm in your hand:

```bash
python scripts/port_id_report.py --write-env --moved-is leader
```

If you moved the arm on the table:

```bash
python scripts/port_id_report.py --write-env --moved-is follower
```

That rewrites `LEADER_PORT` / `FOLLOWER_PORT` in `scripts/so101_env.sh`.

To restart the probe from scratch:

```bash
tmux kill-session -t so101id 2>/dev/null
tmux new-session -d -s so101id \
  '.venv/bin/python scripts/probe_arm_ports.py --port /dev/ttyACM0 --port /dev/ttyACM1 --out data/port_id/probe.jsonl'
```

## Checking the config without moving anything

```bash
bash scripts/record_so101.sh --dry-run   # exact argv LeRobot would get; no hardware
python scripts/check_record_config.py          # ports, calibration, camera modes
python scripts/check_record_config.py --deep   # also really opens both cameras
```

`--dry-run` installs the torque guard, has LeRobot parse the real argv, prints
the resolved robot/teleop/cameras/dataset, and exits. Use it after any change to
`record_so101.sh` or `so101_env.sh` — it catches quoting and arg-parsing bugs
before they become a confusing `unrecognized arguments:` dump on the terminal.

`check_record_config.py` resolves the arms by USB serial, verifies the ports
exist and differ, that both calibration files are present and 6-motor, that the
requested camera mode is supported by the device, and that draccus accepts the
config. `--deep` additionally opens both cameras through LeRobot's
`OpenCVCamera` in the recorder's order, which catches warmup/timeout failures
here rather than after the arms are live.

Note: the parse resolves `--robot.type=so101_follower` to `so100_follower` in the
echoed config. That is cosmetic — both names are registered to the same
`SOFollowerRobotConfig` class:

```python
RobotConfig.get_choice_class('so100_follower') is RobotConfig.get_choice_class('so101_follower')  # True
```

## Recording shape

- 6 joints: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- Gripper is `RANGE_0_100`; the other five are `DEGREES`.
- Dataset lands in `data/<repo_id with / → _>/`, 30 fps, `so101_follower`'s
  cameras as `observation.images.*` and the follower's own motor feedback as
  `observation.state`.

`scripts/teleop_record_real.py` is the older hand-rolled path and should not be
used for new data: it is 5-DOF (drops `wrist_roll`) and writes ad-hoc `.npz`
files rather than a real `LeRobotDataset`.

## The torque guard — why connect() was failing

`SOFollower.connect()` ends with `configure()`, which is:

```python
with self.bus.torque_disabled():
    ...write PID gains...
# context exit ->
    self.enable_torque(motors)
```

Two things go wrong on this rig, and they compound:

**1. Stale `Goal_Position`.** That register lives in EEPROM and survives power
cycles. On this follower all six read **0** while `Present_Position` read ~2000.
Energising a position-mode servo with `Goal_Position=0` commands it to drive to
0 — far outside its calibrated `range_min` (6..742). It slams into the mechanical
stop, draws a stall-current spike, and the rail sags. That is a real hazard, not
just a connectivity nuisance.

**2. No retries, all six at once.** `FeetechMotorsBus.enable_torque` writes
`Torque_Enable=1` then `Lock=1` per motor in one tight loop with `num_retry=0`.
The sag from (1) drops packets:

```
ConnectionError: Failed to write 'Torque_Enable' on id_=5 with '1' after 1 tries.
                 [TxRxResult] There is no status packet!
ConnectionError: Failed to write 'Lock' on id_=5 with '1' after 1 tries.
                 [TxRxResult] Incorrect status packet!
```

**"after 1 tries" is the tell** — there is no second attempt, because
`enable_torque` passes `num_retry=0`. A transient packet loss becomes a fatal
error, and it happens *after* both arms are live and torque is on.

`scripts/so101_torque_guard.py` replaces `enable_torque` to:

1. sync `Goal_Position := Present_Position` first (raw counts, `normalize=False`
   on both sides, so nothing round-trips the calibration tables), and report
   the residual — verified **0 counts** on all six motors
2. enable motors one at a time, with 4 attempts each and a 0.25 s settle gap
   between motors

`record_so101.sh` now goes through `scripts/record_so101.py`, which installs the
patch and then calls `lerobot-record`'s own `main()` with the same argv. The
recording loop, dataset writer and visualisation are untouched. If the patch
fails to install, the wrapper refuses to record rather than risk an unguarded
connect.

Verified against the real follower: goal sync residual 0/0/0/0/0/0, and a bare
`Goal_Position` write does **not** itself enable torque (checked 0 → write → 0),
so the sync is safe to do while the arm is limp.

Tune with `GUARD_RETRIES`, `GUARD_MOTOR_DELAY`, `GUARD_RETRY_DELAY` in
`so101_env.sh`.

## Camera warmup

`warmup_s` defaults to 1 second, and LeRobot's connect loop requires a **fresh**
frame on every pass because `async_read` clears `new_frame_event` after each
read. The wrist camera measured **0.85 s** for its first frame at 1280x720 MJPG,
so `warmup_s=1` leaves roughly 150 ms of margin and intermittently fails with:

```
TimeoutError: Timed out waiting for frame from camera OpenCVCamera(2) after 1000 ms.
Read thread alive: True.
```

`connect()` always burns the full warmup window, so `WARMUP_S=5` in
`scripts/so101_env.sh` costs a few seconds of one-time startup and removes the
race. It is passed per camera in the `--robot.cameras` JSON.

If a camera does still fail, the process aborts with `FATAL: exception not
rethrown` / `Aborted (core dumped)` during teardown. That is a LeRobot/driver
cleanup bug, not the original fault — read the traceback above it.

## Capture rate — why 15 Hz and 640x480

The first real recording attempt ran the loop at **5 Hz** against a 30 Hz target,
and LeRobot's warning is the correct response to ignore:

```
Record loop is running slower (5.0 Hz) than the target FPS (30 Hz). Dataset
frames might be dropped and robot control might be unstable.
```

Recording at a rate the loop cannot hold is not a compromise, it is corruption:
frames get dropped and the control loop goes unstable. Measured on this laptop:

| component | 1280x720 @ 30 | 640x480 @ 15 + streaming |
|---|---|---|
| both cameras, sequential read | 32 ms | 33 ms |
| rerun JPEG, 2 images | 37 ms | 13 ms |
| **`dataset.add_frame`** | **>6600 ms** | **0.9 ms** |
| loop budget | unusable | **47.5 ms → 21 Hz** |
| serial (leader read) | 0.9 ms | 0.9 ms |

Two findings worth keeping:

**Camera read cost is resolution-independent.** It is 32.1–32.3 ms at
1280x720, 960x540, 800x600, 640x480 and 640x360 alike — because `cap.read()`
blocks until the camera's next frame, and the hardware delivers at ~30 fps.
Dropping resolution does not speed up the *read*. It does cut everything
downstream, where the pixels actually cost something.

**Reading both cameras together costs the same as reading one** (32.2 ms vs
32.4 ms) — the second read returns an already-buffered frame. So two cameras are
not a bandwidth problem, and moving them to separate USB ports changed nothing.
`SOFollower.get_observation()` uses `cam.read_latest()` anyway, which is
non-blocking and served from each camera's background thread.

**`add_frame` was the real bottleneck**, and it is not the camera or the serial
bus. With `streaming_encoding=False` it timed out doing 45 frames in 300 s.
`--dataset.streaming_encoding=true --dataset.encoder_threads=2` moves AV1
encoding off the critical path and took it to 0.9 ms/frame.

### Camera fps and dataset fps are different knobs

```
RuntimeError: OpenCVCamera(4) failed to set fps=15 (actual_fps=30.0).
```

Both cameras expose exactly **one** frame interval at 640x480:

```
Size: Discrete 640x480
    Interval: Discrete 0.033s (30.000 fps)
```

and `OpenCVCamera._validate_fps` compares with `math.isclose(..., rel_tol=1e-3)`,
so requesting anything but 30 raises at connect time. There is no 15 fps mode to
select.

So the two rates are configured separately:

| setting | value | meaning |
|---|---|---|
| `CAMERA_FPS` | 30 | what the camera is asked to stream (hardware-limited) |
| `FPS` | 15 | dataset + control rate; `--dataset.fps`, baked into `meta/info.json` |

The camera delivers 30 fps and the 15 Hz loop takes the latest frame from it via
`read_latest()`, yielding 15 evenly spaced distinct frames per second. LeRobot's
`sanity_check_dataset_robot_compatibility` compares the existing dataset's fps
against the *config* fps, not against the camera's, so the two need not match.

`check_record_config.py` queries `v4l2-ctl --list-formats-ext` for the real
intervals and fails if `CAMERA_FPS` is not among them, so this is caught before
the arms are live rather than after `connect()`.

Also note: because the dataset's `fps` and camera shape are baked into
`meta/info.json` and drive `delta_timestamps` at train time, resuming a dataset
with different values produces episodes that disagree about their own timing.
`record_so101.sh` refuses unless `ALLOW_MIXED_CAPTURE_SETTINGS=true`.

## Align the leader to the follower BEFORE recording

This is the single most important operational step, and it is not a software
problem.

`record_loop` calls `robot.send_action()` on its **very first tick**, with
whatever the leader currently reads. If the two arms are not near each other,
that first command asks six servos to travel a long way against gravity. They
stall, draw a large current, the supply rail sags, and the Feetech bus starts
dropping packets. The whole thing presents as a flaky cable:

```
WARNING  Record loop is running slower (5.3 Hz) than the target FPS (15 Hz)
ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6]
                 after 3 tries. [TxRxResult] There is no status packet!
ConnectionError: Failed to write 'Torque_Enable' on id_=1 with '0'
                 after 6 tries. [TxRxResult] There is no status packet!
FATAL: exception not rethrown
Aborted (core dumped)
```

The second `ConnectionError` is only a consequence — the bus is already dead by
the time `disconnect()` tries to release torque.

Measured on 2026-09-25, which is what a 99° mismatch looks like:

```
joint             leader   follower     delta   move leader
shoulder_pan       -15.7        1.5     -17.1   #####
shoulder_lift     -104.7       -9.5     -95.3   ##############################
elbow_flex         98.3       -0.9     +99.2   ##############################
wrist_flex         64.4       52.1     +12.3   ####
wrist_roll          6.2       10.2      -4.0   #
gripper             1.4        1.7      -0.3
```

Check before every session:

```bash
python scripts/check_pose_sync.py --watch    # live, until aligned
```

`record_so101.sh` runs it as a preflight gate and **refuses to start** (exit 2)
while any joint is more than `MAX_POSE_DELTA` (default 10°) off. It is
read-only. Override with `SKIP_POSE_GATE=true` if you really mean it.

Move the arm **in your hand**, not the one on the table.

## Bus health

```bash
python scripts/check_bus_health.py --port /dev/ttyACM0 --arm leader
python scripts/check_bus_health.py --port /dev/ttyACM1 --arm follower
```

Read-only. Reports which motors answer and their `Present_Voltage` / `Present_
Current`. Connects with `handshake=False` on purpose — the bus handshake refuses
to connect when motors are missing, which is exactly the state you want to
inspect.

Current state: **6/6 on both arms, stable across rounds, ~0 mA** (torque off).
Both were healthy once the arm was moved off the shared USB lane.

If motors drop off mid-session it is power, not software. In order: supply
current rating; an **unpowered** USB hub stealing bus current (a port sharing a
lane with the cameras is the usual culprit); power-connector seating at both
ends of any daisy-chain extension; then voltage sag.

**Never run two arm checks in parallel** — serial ports are exclusive and the
second open produces `SerialException: ... multiple access on port`, which looks
like a dead bus but is not.

## venv fixes that were needed

Two blockers, both already applied:

1. **`datasets` was not installed** → `LeRobotDataset` import failed, so
   `lerobot-record` could not run at all. Installed `lerobot[dataset]`.
2. **`transformers` 4.57.6 conflicted with `huggingface-hub` 1.28.0**
   (`transformers` wanted `hf-hub<1.0`) → every `lerobot-record` import died in
   `tokenizer_processor`. Upgraded to `transformers==5.5.4`, which is what
   `lerobot 0.6.1` actually pins (`>=5.4.0,<5.6.0`).

```
av 15.1.0        datasets 4.8.5       dill 0.4.1        jsonlines 4.0.0
multiprocess 0.70.19  pandas 2.3.3    torchcodec 0.11.1  tzdata 2026.4
xxhash 4.0.1     transformers 5.5.4 (was 4.57.6)      shellingham 1.5.4
typer 0.27.2
```

`rerun-sdk` is 0.36.3rc1, which is **outside** LeRobot's `viz` pin
(`>=0.24.0,<0.34.0`). It works, and the web-viewer API used here is current,
but a LeRobot upgrade may pin it back.


## Scripts added

| script | what |
|---|---|
| `scripts/so101_debug_server.py` | Rerun gRPC sink + Web UI on `:9090` |
| `scripts/record_so101.sh` | the record launcher (reads `so101_env.sh`) |
| `scripts/record_so101.py` | guarded wrapper: installs the torque guard, calls lerobot-record |
| `scripts/so101_torque_guard.py` | safe `enable_torque` (goal sync + per-motor retries) |
| `scripts/so101_env.sh` | hardware assignment — edit this |
| `scripts/resolve_arm_ports.sh` | USB serial → current `/dev/ttyACM*` |
| `scripts/repair_calibration.py` | rewrite motor-side calibration, serial-resolved |
| `scripts/check_record_config.py` | validate the whole config, no hardware |
| `scripts/check_bus_health.py` | per-motor V / mA / presence, read-only |
| `scripts/probe_arm_ports.py` | read-only joint-position stream per port |
| `scripts/port_id_report.py` | which arm is on which port |
| `scripts/tune_collection_cameras.py` | V4L2 exposure/format tuning + verify |
| `scripts/check_pose_sync.py` | leader-vs-follower alignment gate (read-only) |
| `scripts/episode_outcomes.py` | read/report/export per-episode success labels |
| `scripts/episode_outcome_keys.py` | s/f/d key hooks (installed by the recorder) |
| `scripts/test_episode_outcomes.py` | offline test for the above, no hardware |
| `scripts/dump_preprocessing_spec.py` | read the real spec out of a recorded dataset |

## Still to do before useful data

- [x] Identify leader vs follower — `ttyACM1` leader, `ttyACM0` follower.
- [ ] Frame the global camera on the actual workspace.
- [ ] Verify the follower moves when the leader moves, at a safe height, with
      the gripper clear.

The wrist camera is aimed at a white table by design and is left where it is.
Low image `std` is expected while the table fills the frame; it recovers as soon
as the arm and objects are in shot. No camera setting is needed for it.

## Preprocessing

The full capture -> train -> deploy pipeline is in
**[PREPROCESSING.md](PREPROCESSING.md)**. It is the document to read before
training anything on this data, because every stage in it has to be reproduced
at deployment time.
