# SO-101 single-arm data collection — setup

Leader arm → follower arm + 2 cameras → `LeRobotDataset`, with a live view in
a browser Web UI. This is the real-hardware path; the MuJoCo/Isaac envs in this
repo are separate.

## The two terminals

```bash
cd /Storage/Projects/so101_mnri
source .venv/bin/activate
```

```bash
# terminal 1 — debug server (the Web UI). Leave running.
python scripts/so101_debug_server.py
```

It prints the exact URL to open. **Use the one it prints** — the `?url=` part is
not optional:

```
http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy
```

```bash
# terminal 2 — record
bash scripts/record_so101.sh
```

From another machine, forward **both** ports — the browser needs the HTML on
9090 *and* the gRPC sink on 9876:

```bash
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 thakk100@dhyan-LOQ
```

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
# -> data/thakk100_so101_single_arm_6dof
```

On a second run that directory *does* exist, so the script auto-switches to
`--resume=true`. If it exists but has no `meta/info.json`, the script refuses to
start rather than writing into whatever is there.

Verified directly: create → ok, second `create` → `FileExistsError`, `resume` →
ok with 0 episodes.

## What the debug server is

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
