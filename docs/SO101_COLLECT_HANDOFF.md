# HANDOFF — SO-101 real-robot data collection

Scope: the single-arm leader→follower LeRobot recording path. Written
2026-09-26. For the sim/RL/VLA side of the repo see [HANDOFF.md](../HANDOFF.md);
for the full capture guide see [SO101_RECORD_SETUP.md](SO101_RECORD_SETUP.md).

## The command

```bash
cd /Storage/Projects/so101_mnri
source .venv/bin/activate
bash scripts/record_so101.sh
```

That is the whole thing. It resolves the arms by USB serial, gates on
leader/follower pose alignment, installs three patches, and hands its argv to
stock `lerobot.scripts.lerobot_record.main()`. Output is a stock LeRobot
**v3.0** dataset in `data/thakk100_pink_duck_in_a_box`.

**Quit with `q`, never Ctrl+C twice.** `meta/episodes/` is flushed only every
10 episodes or in `finalize()`; a kill leaves `info.json` claiming N episodes
with no episode index and a footer-less parquet, and the dataset will not load
again. That already cost 8 episodes on 2026-09-25.

Keys (the **recording terminal** must have focus, not the Rerun window):

| key | during the episode | during the reset |
|---|---|---|
| `s` | label **success**, end now | (re)label success, skip the rest of the reset |
| `f` | label **failure**, end now | (re)label failure, skip the rest of the reset |
| `d` | **discard** this take, re-record | discard the episode just recorded |
| `n` / right | end now, keep unlabeled | skip rest of reset |
| `r` / left | discard, re-record | same |
| `q` / esc | finalize and exit | same |

With turn-based recording on (the default) `n` and `r` no longer cut the reset
short — see the loop below.

## The loop, turn by turn

Recording is **turn-based**: nothing is written until you say the scene is ready.
`SPACE` is the turn key.

```
  [waiting]   arm is holding its pose, nothing recorded
              place the duck in a zone
              press 1-9        -> that zone applies to the next episode
              press SPACE      -> recording starts
  [recording] do the task
              press s / f      -> label it AND end it
              press d          -> discard, retake from the same zone
  [waiting]   back to the top
              press q at any point -> finalize and exit
```

### Why this exists

LeRobot alternates two calls to the same `record_loop` — one with a `dataset`
(the episode) and one without (the reset). Two consequences that used to bite:

- The reset call never reaches `add_frame`, so **your frames of moving the duck
  are not in the dataset.** That part was already fine.
- But the reset call *does* keep calling `robot.send_action()`, so the follower
  tracks the leader while you reset, and it ends on a timer **or on `n`**.
  Pressing `n` to skip the reset made the next episode start recording while
  your hand was still in frame and the duck still moving. Those frames *are*
  recorded, and they teach the policy that a hand at t=0 is normal.

`scripts/episode_trigger.py` gates that boundary. The reset gets an unbounded
`control_time_s` and an events proxy whose `exit_early` is true only after the
turn key, so `n` cannot cut it short. `q` still quits — the proxy passes every
other key through, and the arm keeps tracking the leader so it returns to rest.
The first episode waits too, so a session never starts before you are ready.

While waiting, `s`/`f`/`d`/`r` are **inert** — a stray keypress mid-reset cannot
label or discard an episode that was never recorded. The zone digits stay live,
because that is when you set them.

Escape hatch: `--no-turn` restores LeRobot's timer reset. `TURN_KEY=0` in the
environment does the same.

## The five prompts, and the key layout

This is a **multi-task** dataset. LeRobot v3.0 supports per-frame task strings
natively, so each episode records the prompt chosen for it as its own `task`
value, `meta/tasks.parquet` accumulates the distinct prompts, and the result is
a stock multi-task `LeRobotDataset` that stock LeRobot trains on unchanged. No
schema patch, no custom writer.

| key | selects |
|---|---|
| `1` | pick the duck |
| `2` | push the duck into the square tape |
| `3` | pick the duck and put it in the square |
| `4` | pick the duck and put it in the red cup |
| `5` | pick the duck and put it in the brown cup |
| `i` `o` `k` `l` | duck start zone 1 / 2 / 3 / 4 |
| `z` | clear the pending zone |
| `SPACE` | turn key: "duck is placed, start recording" |
| `s` / `f` | label success / failure, and end the episode |
| `d` | discard this take, re-record |
| `q` | finalize and exit |

Press the prompt number and the zone letter **before** the take, while you are
placing the duck. Both apply to the next episode saved.

Defined in `scripts/so101_env.sh` as `TASK_PROMPTS` (`|`-separated, position N
is what key N selects) and `ZONE_KEYS` (space-separated, the 1st key is zone
1). Re-tape more zones or add a sixth prompt there — no code change.

**Prompt 2 is deliberately the exact string your first 12 episodes were
recorded with**, so resuming does not orphan them as a separate task.

### What the prompt changes about labelling

Prompt 2 is the only *planar push* — gripper open. Prompts 3–5 need the gripper
to close, lift, move and release. Prompt 1 is a pick with no place, so "success"
means lifted clear of the table.

So `s` means "the duck is in the target named by the **active** prompt", which
makes pressing the right number part of labelling. The save line echoes what it
recorded so you can catch a mismatch:

```
  [outcome] episode 42 -> success, prompt 3, zone 2
```

The prompt is **frozen** on the first frame of an episode, so a keypress
mid-take cannot split one episode across two task strings. It unfreezes at save
and on a `d` re-record — and like the zone, the *pending* prompt survives the
re-record, because a retake is the same task being attempted again.

Multi-task coverage has its own failure mode, so the report counts prompts:

```
  prompts:
       38  #####...  pick the duck and put it in the red cup
       31  #####...  pick the duck
        9  #...      push the duck into the square tape
    -> 3 distinct prompt(s) recorded
```

A dataset can look healthy by episode count while one prompt holds 95% of it,
which trains a policy that ignores the instruction.

### The scene, as the global camera sees it

Verified from a live frame on 2026-09-27:

- White table, well exposed (`mean 145`, `std 68` — textured, not blown)
- The taped **2×2 grid** fully in frame, centre-left
- **The pink duck sitting inside one of the cells** — see the caveat below
- SO-101 follower on the right: red shoulder link, grey links, cables, tan
  wooden base plate
- A tan cup bottom-left and a white dish with a red rim bottom-centre, both
  clipped by the frame edge — these are the prompt 4 and 5 targets, so check
  the arm can comfortably reach them and that they stay in frame
- Two black cables cross the frame. Harmless if they never move; nudge one
  between sessions and it becomes a spurious visual cue.

**Caveat worth acting on: the duck starts inside a cell.** If "any cell counts"
then prompts 2 and 3 are satisfied at t=0 and `s` stops meaning anything. The
2×2 is taped as the *start-zone* grid, so the goal needs to be unambiguous and
distinguishable from a start zone — either a separate square outside the grid,
or one designated cell marked distinctly (thicker tape, second colour) and
excluded from the zone rotation.

## Duck start zones — the coverage control

| key | meaning |
|---|---|
| `1`–`9` | set the start zone for the **next** episode to be saved |
| `0` | zone 10 |
| `z` | clear the pending zone |

Press the zone digit **while you are placing the duck**, before the take. It is
written to `meta/outcomes.json` as `"zone": N` and consumed by exactly one
saved episode.

This is the only thing that fixes state coverage, and coverage is what decides
whether the dataset is usable: 100 episodes from one duck placement is one
demonstration recorded 100 times. The first 12 episodes here all came from a
single placement, which is the reason they are worth 12 and not 120.

Two behaviours worth knowing:

- **A zone survives `d`.** The duck has not moved, so a re-recorded take is the
  same condition and is logged as such. Only the *outcome* is cleared on a
  re-record; clearing the zone too would mark every retake as unzoned.
- **A zoned take with no `s`/`f` is kept**, stored as `"outcome": "unlabeled"`,
  so the zone is not lost. That is different from an episode with no entry at
  all, which means nobody reviewed it. `report()` counts the two separately.

After 3 consecutive takes from one zone it nudges you to cycle. Not a hard
limit — some zones are genuinely harder and deserve more takes.

### The taped grid (2x2, zones 1-4)

As laid out on 2026-09-27 — four zones, numbered in reading order:

```
   far from the square
   ┌───────────┬───────────┐
   │     1     │     2     │
   ├───────────┼───────────┤
   │     3     │     4     │
   └───────────┴───────────┘
                  ▲
          SO-101 follower
             (goal = the black tape square, kept fixed)
```

Digits `5`-`9` and `0` are wired up but unused. If you later tape more zones,
no code change is needed — just press the new number.

Four zones is a reasonable start. The cheap way to widen coverage without
taping more is to **rotate the duck within each zone**: 4 positions × 4
orientations is 16 distinct start conditions, and heading-to-square direction
varies for free as you move around the grid.

Suggested cadence: cycle 1→2→3→4 before repeating any, ~20 takes per zone for
the first 80 episodes. The 3-in-a-row nudge fires only if you repeat without
cycling, and is a hint rather than a limit.

### Checking coverage

```bash
.venv/bin/python scripts/episode_outcomes.py --root data/thakk100_pink_duck_in_a_box
```

```
  duck start zones:
    zone 1      24  ############
    zone 2      21  ###########
    zone 3      19  #########
    zone 4      18  #########
    none        12  (episode not attributed to a zone)
    -> 4 of 4 in use (zones 1-4); 12 episodes unzoned
```

Episodes recorded before zones existed show under `none`, which is expected for
the first 12. The denominator is the range you have actually used, so a 2x2 rig
does not look like it is short six zones.

## The ending position, and whether to add a box

**You do not need to mark the ending position, and you should not add a box.**
Both are worth a clear reason rather than a shrug.

The ending position is not a free variable — it is *defined* by the task. A take
either ends with the duck inside the black tape square or it does not, and `s`
vs `f` already records exactly that. A separate "end zone" would be a second
encoding of the same fact, free to disagree with the first.

What is genuinely missing is *verification*: right now `s` means "I believe it
went in". The tape square is a black quadrilateral and the duck is pink, on a
fixed table under a fixed camera — a classical CV check (colour-segment the
pink, fit the black quad, test containment) is cheap and would catch missed
takes and optimistic labelling. That is `scripts/episode_qc.py` in the HITL
roadmap, and it is the right answer to "is my `s` trustworthy", not another
number to press.

A cardboard box would be a **different task**, not a better marker:

- Pushing a duck *into* a box needs the duck to clear a wall, so the motion
  changes from planar pushing to a lift-and-place. Every demonstration would no
  longer match the task you have been recording.
- It occludes the duck exactly when containment matters, which is the moment
  you most want to see it.
- It adds a wall-collision failure mode that has nothing to do with the
  behaviour you are trying to teach.
- The goal is currently one fixed square, which is what makes "success"
  checkable at all. Moving the goal per episode would make the success
  criterion episode-dependent and much harder to label consistently.

Tape the grid, keep the square fixed, and put the effort into verifying `s`
instead. If you do want a container task later, that is a new task string and a
new dataset — not a change to this one.

## The one thing that gets worse with delay

`timestamp` in the dataset is synthesised as `frame_index / fps`, so dropped and
late frames leave no trace. The 12.6-vs-15 Hz record-loop question therefore
cannot be answered from data already collected. See the HITL roadmap's P1.4
(wall-clock timing sidecar); after a large collection that information is gone.

## Status, 2026-09-26

| | |
|---|---|
| arms | both present, resolved by serial: leader `5AAF263743`→`/dev/ttyACM0`, follower `5AAF263600`→`/dev/ttyACM1` |
| cameras | both stream: global `/dev/video4`, wrist `/dev/video2` |
| calibration | `mnri_arms_1_{leader,follower}.json`, 6 motors each |
| config check | `record_so101.sh --dry-run` passes; argv accepted by LeRobot |
| offline tests | `test_episode_outcomes.py` and `test_camera_flip.py` both pass |
| dataset | `data/thakk100_pink_duck_in_a_box` did not exist before today — every episode in it is 6-DOF, 640×480, 15 fps, upright |
| legacy data | `data/thakk100_so101_duck_push` (8.7 GB, 22 episodes) is the old hand-rolled `.npz` path: 5-DOF, no `wrist_roll`, not a `LeRobotDataset`. Not trainable with this pipeline. Left alone, deliberately. |

Not yet verified on hardware, and both are operational, not software:

- **Which camera is physically inverted.** See the orientation section — the
  setting is in place and follows the name, but confirm it against the live
  view on the first take.
- **The follower has not been confirmed to track the leader.** Move the arm in
  your hand at a safe height, gripper clear, and watch the table arm before you
  trust a take. Everything up to `send_action()` is tested; the motion itself is
  not.
- **The global camera is not framed on the workspace.** It sees the table, the
  arm and (in the frame checked on 2026-09-26) the duck and the tape square, but
  the framing has not been checked against where the arm actually works. After
  the flip, confirm the duck and the square stay in frame; if not, that is a
  physical fix (move the camera), not a config one.

## Record rate

`CAMERA_FPS=30` and `FPS=15` are different things, and the warning

```
WARNING  Record loop is running slower (12.6 Hz) than the target FPS (15 Hz).
```

is about the second one — the achieved rate of the Python record loop against
its own target. The camera still streams at 30.

- The **camera** rate is hardware-limited to 30 fps; both cameras expose exactly
  one interval at 640×480, and OpenCV's check uses
  `math.isclose(rel_tol=1e-3)`, so asking for 15 raises at connect time.
- The **dataset/control** rate is 15 Hz, chosen on 2026-09-25 (commit `fb2a53b`)
  because the loop could not hold 30. At 720p it managed 5 Hz. This path has
  never run a 30 Hz loop.
- The only 30 fps dataset in `data/` is `thakk100_so101_duck_push`, which is the
  old hand-rolled path, not this one.

The camera flip is **not** the cause of a slowdown: it is 0.026 ms on the
camera's own thread, which the record loop only peeks at non-blockingly.

To find out what actually costs, profile the real loop rather than reasoning
about it:

```bash
bash scripts/record_so101.sh --profile=12
```

It prints a per-call breakdown and writes to `<REPO_ID>_profile`, never the real
dataset. **The arm will move**, and the pose gate runs first.

## Camera orientation — changed 2026-09-26

The global camera is mounted **inverted**, so its frames arrive upside down.
Two obvious fixes do not work here:

- **No V4L2 flip control.** `v4l2-ctl -d /dev/video2 -L` and `-d /dev/video4
  -L` list brightness/contrast/saturation/hue/gamma/gain/exposure and nothing
  else. No flip, no mirror, no rotate.
- **`rotation` is not a flip.** `OpenCVCameraConfig.rotation` accepts only
  0/90/180/−90, and 180 is a vertical *and* horizontal mirror — the wrong
  transform when only the vertical axis is backwards.

So the flip is software, in `scripts/so101_camera_flip.py`, which wraps
`OpenCVCamera._postprocess_image`. Config is `VFLIP_CAMS` in
`scripts/so101_env.sh`, as camera **names**:

```bash
VFLIP_CAMS=global   # current setting: flip the global cam only
VFLIP_CAMS=both     # or wrist / none / all
```

An unknown token is a hard error, because a typo here would quietly record an
upside-down dataset. The names are resolved to indices at launch, so this
setting follows the camera even when the indices swap again.

Why that seam:

- It runs in the camera's own read thread (`_read_loop`), which the recorder
  only peeks at via `read_latest()`, so the cost never lands on the record
  loop. Measured `cv2.flip` on 640×480×3: **0.026 ms/frame**, against a
  47.5 ms loop budget, and 4× cheaper than the BGR→RGB convert already in that
  same function.
- Every consumer of LeRobot's camera sees the corrected frame — the dataset,
  the Rerun live view, and any later reader. Flipping in the recorder instead
  would leave the live view inverted.
- The shape check inside `_postprocess_image` runs before the flip and a flip
  does not change the shape, so the video encoder cannot tell.

### The cameras swapped indices, not roles

They were re-plugged on 2026-09-26. What each one *is* was confirmed from its
own frame, and the USB device names agree:

| role | device | index now | index before |
|---|---|---|---|
| global (whole table: duck, tape square, arm) | `USB Camera` 32e6:9221 | **video2** | video4 |
| wrist (gripper, close up) | `Innomaker-U20CAM-720P` 0c45:6367 | **video4** | video2 |

`scripts/so101_env.sh` had this backwards, so a session recorded on 2026-09-26
stored wrist footage under `observation.images.global`. That dataset was
deleted rather than kept. Indices are not stable across replugs — re-check if
the views look swapped:

```bash
v4l2-ctl -d /dev/video2 --info | grep 'Card type'
v4l2-ctl -d /dev/video4 --info | grep 'Card type'
```

### Unverified: which camera is actually inverted

`VFLIP_CAMS=global` follows the operator's read of the mount, but **this was
not confirmed from a still frame**, and it is worth 10 seconds to check. The
workspace view is top-down and its features — the duck, the tape square — lie
along the horizontal axis, so a vertical flip barely changes how they read. A
vertical flip is also *undetectable* from mean/std, because it only reorders
pixels.

Look at the live view (or the as-recorded snapshots below) before the first
take. If the global view is the wrong way up, it is one word:

```bash
VFLIP_CAMS=wrist    # in scripts/so101_env.sh
```

Check without touching the arms — the flip set comes from the config, so these
PNGs are the frames **as they would be recorded**:

```bash
.venv/bin/python scripts/so101_camera_flip.py --snap 2,4 --snap-out /tmp/flip
```

### Inference must use the same orientation

`vla/deploy.py:RealCamera` (used by deployment *and* `vla/record_video.py`) and
`vla/record_demos.py` open cameras with **raw `cv2.VideoCapture`**, not
LeRobot's `OpenCVCamera`, so the patch above does not reach them. They call
`so101_camera_flip.apply_to_frame()`, which reads the same `VFLIP_CAMS`, so
there is one source of truth for orientation across recording and inference.
Without that the policy trains on upright frames and is served inverted ones,
which fails silently — it just gets worse, with no error anywhere.

If you add another capture path, route it through `apply_to_frame()`.

**`VFLIP_CAMS` is part of the dataset's meaning**, like fps and resolution.
Flipping one camera after episode 40 leaves 40 episodes one way up and the rest
the other. Set it before the first episode, or start a new `REPO_ID`.

## Before every session

```bash
bash scripts/record_so101.sh --dry-run        # argv LeRobot would get; no hardware
.venv/bin/python scripts/check_record_config.py --deep   # really opens both cameras
.venv/bin/python scripts/check_joint_range.py --arm leader --compare   # motor posture
.venv/bin/python scripts/check_pose_sync.py --watch     # live, until aligned
```

The launcher runs the pose gate itself and **refuses to start** (exit 2) while
any joint is more than `MAX_POSE_DELTA` (default 10°) off. This matters:
`record_loop` calls `send_action()` on its very first tick, so a 99° mismatch
stalls six servos against gravity, sags the rail, and kills the Feetech bus —
which presents as a flaky cable, not as a software bug. Move the arm **in your
hand**. Override with `SKIP_POSE_GATE=true` only if you really mean it.

### Which check answers which question

Four checks, all read-only, and they are not interchangeable:

| script | question it answers | when |
|---|---|---|
| `scripts/check_pose_sync.py` | **the arm delta**: how far the leader pose is from the follower pose, joint by joint, in degrees | before every take — this is the `MAX_POSE_DELTA` gate the launcher enforces |
| `scripts/check_joint_range.py` | the **motor posture**: each joint's live travel as a % of its calibrated range, against the raw 0..4095 counts the servo reports | when a joint reads pinned at 0 % or 100 %, i.e. the arm is at a limit it should not be at |
| `scripts/check_bus_health.py` | per-motor voltage and current, and whether the bus is alive | when the bus drops packets, or a motor is not moving at all |
| `scripts/check_record_config.py` | the whole config: ports by USB serial, calibration files, camera modes, the argv LeRobot will get | once per session, and after any change to `so101_env.sh` |

`check_pose_sync.py` prints a `delta` column per joint and the worst one:

```
joint           leader   follower      delta   move leader
shoulder_lift     44.0      45.0       -1.0
elbow_flex        95.0      51.0      +44.0   <-- too far
```

`check_joint_range.py --compare` is the one that catches a **bad calibration**
rather than a bad pose: if a joint sits at 1 % or 99 % of its range on every
read, the calibration records more travel than the joint has, and no amount of
re-aligning will help. That is what it found on this rig before the
2026-09-25 re-calibration — the follower's `shoulder_lift` was recorded over a
0..4093 span, i.e. the full raw count range, when the joint really travels
756..3097. Both files were re-calibrated that evening and the previous ones
are kept in `robots/so101/calibration/backup-20260925-215302/`.

The **printed base** is a different thing entirely: `SO101-ref/check_base_M5.py`
checks the M5 base CAD (seat floors, countersink angle, the wall left around
each seat). It is a geometry check on a mesh, it needs no arms and no cameras,
and it has nothing to do with a recording session. Run it with
`/usr/bin/python3 SO101-ref/check_base_M5.py` from `SO101-ref/`.

## Known traps

- **Never hardcode `/dev/ttyACM*`.** Enumeration order reverses on replug; it
  did on 2026-09-25 and the run wrote the follower's calibration into the
  leader's servos. `scripts/so101_env.sh` holds serials;
  `record_so101.sh` resolves them at launch and aborts if either is missing.
- **Open the cameras with `fourcc=MJPG`.** A default open negotiates YUYV and
  both cameras return blown-out near-featureless frames (`std≈3` instead of
  `≈90`).
- **`CAMERA_FPS` (30) and `FPS` (15) are different knobs.** Both cameras expose
  exactly one interval at 640×480 and OpenCV's fps check uses
  `math.isclose(rel_tol=1e-3)`, so asking the camera for 15 raises at connect
  time. The camera streams 30; the 15 Hz loop takes the latest frame.
- **640×480, not 720p.** At 720p the loop ran at 5 Hz, which LeRobot correctly
  warns drops frames and destabilises control.
- **`--profile` writes to `<REPO_ID>_profile`**, never the real dataset, because
  it stops mid-episode and LeRobot saves that take. Delete it afterwards.
- **One arm check at a time.** Serial ports are exclusive; a second open gives
  `SerialException: multiple access on port`, which looks like a dead bus.
- The two arms run different rails (leader 5 V, follower 12 V). Do not flag the
  leader as low-voltage against a 12 V threshold.

## Second, non-canonical recorder

`vla/record_demos.py` (referenced by [IL_PLAN.md](../IL_PLAN.md)) is a separate
hand-rolled recorder that opens cameras with raw `cv2.VideoCapture` and writes
its own writer. It does **not** get the torque guard or the `s`/`f`/`d` outcome
keys, and it does not apply the camera flip. Use `scripts/record_so101.sh` for
new data.

## Where the labels are

`meta/outcomes.json` beside the dataset — not in the LeRobot schema, so the
dataset stays loadable by stock LeRobot.

```bash
.venv/bin/python scripts/episode_outcomes.py --root data/thakk100_pink_duck_in_a_box
.venv/bin/python scripts/episode_outcomes.py --root ... --export successes.csv --only success
```

Then train on it following [PREPROCESSING.md](PREPROCESSING.md).
