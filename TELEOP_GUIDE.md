# TELEOP_GUIDE — demo-day master document

Real-arm ↔ sim bridging for the SO-101 arms. **Read sections 1–3 on demo
morning; run the 30-second sanity check first.** All commands assume:

```bash
cd /Storage/Projects/so101_mnri && source .venv/bin/activate
```

---

## 1. Overview — four teleop modes

| # | Mode | Script | Status | Hardware needed |
|---|------|--------|--------|-----------------|
| 1 | Gamepad → sim (MuJoCo IK) | `scripts/teleop_gamepad_ik.py` | dry-run validated; live path in use | DS4 |
| 2 | Gamepad → sim (Isaac) | `scripts/isaac_teleop_gamepad.py` | null-device validated (single + dual); live path needs DS4 check | DS4 + Isaac |
| 3 | Leader arm → sim (**real2sim**) | `scripts/real_to_sim_check.py` (sanity), `scripts/teleop_leader_lerobot.py` (lerobot calib), `scripts/teleop_leader_arm.py` (raw scservo_sdk) | dry-run validated | leader arm |
| 4 | Sim → real follower (**sim2real**) | `scripts/sim_to_real.py` (`--source script`, `gamepad`, or `checkpoint`) | dry-run validated; live needs hardware check | follower arm (+ optional DS4/checkpoint) |

Bonus combos that already work:

- **Leader → sim AND real at once** ("shadow teleop"): `teleop_leader_lerobot.py --mirror-real`
- **Gamepad IK → sim AND real at once**: `teleop_gamepad_ik.py --real-follower-port ... --real-follower-id ...`

> Note on mode 2: wired via `scripts/isaac_teleop_gamepad.py` (Aug 2026) —
> pygame DS4 → DifferentialIKController → joint actions, validated headless
> with `--null-device` on the single- and dual-arm tasks. `isaacteleop` (Quest
> 3 XR→SO-101, inside lerobot source) remains the planned VR route —
> HANDOFF.md "Open work" item 3.

Every script has a `--dry-run` that runs hardware-free and exits 0. The table's
"hardware needed" column is what still must be exercised physically before showtime.

## 2. Hardware setup

### DS4 gamepad

- USB: plug in micro-USB; shows up as `/dev/input/js0`.
- Bluetooth: hold **PS + SHARE** until the light bar double-flashes, then pair
  with `bluetoothctl` (`scan on`, `pair <MAC>`, `trust <MAC>`, `connect <MAC>`).
- Permissions: pygame opens `/dev/input/js*`, so
  `sudo usermod -aG input $USER` and **log out/in**. Check: `groups | grep input`.
- Verify detection + watch axes/buttons move live (expect `Wireless Controller`):

```bash
python scripts/check_gamepad.py --list-only   # enumerate pads, exit 0 even with none
python scripts/check_gamepad.py               # live 5-second view of axes/buttons
```

No pad listed? `sudo modprobe joydev`, then replug / re-pair.

### SO-101 arms (Feetech STS3215, servo IDs 1–6, 1 Mbaud)

- Find ports: `ls /dev/ttyACM*` (unplug/replug to identify), or
  `lerobot-find-port`. Typical layout: leader `/dev/ttyACM0`, follower `/dev/ttyACM1`
  — **verify on the day**, USB order is not stable.
- Serial permission: `sudo usermod -aG dialout $USER` + relogin.
- Calibration (lerobot convention, shared by all lerobot-based scripts here):
  follower: `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower`
  leader:   `lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader`
  Files land under `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`
  (and `.../teleoperators/so_leader/<id>.json`). Scripts pass `--follower-id` /
  `--leader-id`; if omitted, id=None uses `<None>.json` — prefer explicit ids.
- Raw-SDK path only (`teleop_leader_arm.py`): its own tick-offset calibration at
  `scripts/leader_calib.json` via `python scripts/teleop_leader_arm.py --calibrate --port /dev/ttyACM0`.

### Quest 3

Not needed for any mode below. VR teleop goes through isaacteleop/XR (see note
on mode 2). Keep the headset charged if you want it as a backup demo.

---

## 3. Per-mode quickstart

### Mode 1 — gamepad → sim (MuJoCo IK)

```bash
# headless CI check first (no DS4 required)
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run
# live (desktop session, NOT with MUJOCO_GL=egl, so the viewer can open)
python scripts/teleop_gamepad_ik.py --env single            # add --device 0 if several pads
python scripts/teleop_gamepad_ik.py --env dual --arm left   # Share toggles arms
```

Mapping (printed at startup): left stick = EE x/y · right stick vert = EE z ·
right stick horiz = wrist roll · **L1/R1** = pitch · **L2/R2** = gripper
close/open · **Share** = toggle arm (dual) · **Options** = re-center IK target
· **Square** = safe quit · **PS** = E-STOP latch (freeze; Options resumes).
Deadzone 0.15. Safety: sim-only — nothing physical moves.

### Mode 2 — gamepad → sim (Isaac)

Wired: `scripts/isaac_teleop_gamepad.py` (`scripts/README.md` §5). Same DS4
mapping as the MuJoCo gamepad script.

```bash
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless  # dry-run, no hardware
python scripts/isaac_teleop_gamepad.py --task SO101-PickLift-Single-v0        # live (windowed Kit)
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --arm both
```

Needs GPU + ~4 GB headroom; check `nvidia-smi` and `free -g` first — silent
deaths are resource starvation. (isaacteleop/XR route remains an alternative.)

### Mode 3 — leader arm → sim (real2sim)

**30-second sanity check FIRST on demo morning:**

```bash
MUJOCO_GL=egl python scripts/real_to_sim_check.py --dry-run              # no hardware at all
python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id leader           # print 5 s of positions
python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id leader --mirror-sim  # + drive MuJoCo sim
```

Prints live per-joint degrees, a mean/std/min/max table, and (with
`--mirror-sim`) the sim EE position. All readings should track your hand
smoothly; jitter means bus/calibration trouble (section 5).

**Full teleop sessions:**

```bash
# lerobot-calibration path (recommended; shares calib with follower tooling)
MUJOCO_GL=egl python scripts/teleop_leader_lerobot.py --env single \
    --leader-port /dev/ttyACM0 --leader-id leader --dry-run        # fake leader, no hardware
python scripts/teleop_leader_lerobot.py --env single \
    --leader-port /dev/ttyACM0 --leader-id leader                  # live, headless loop
# shadow teleop: leader drives SIM and REAL follower together
python scripts/teleop_leader_lerobot.py --env single --leader-port /dev/ttyACM0 \
    --leader-id leader --mirror-real --follower-port /dev/ttyACM1 --follower-id follower
```

```bash
# raw scservo_sdk path (tick-offset calib in scripts/leader_calib.json)
MUJOCO_GL=egl python scripts/teleop_leader_arm.py --dry-run --env single
python scripts/teleop_leader_arm.py --env single --port /dev/ttyACM0
python scripts/teleop_leader_arm.py --calibrate --port /dev/ttyACM0   # interactive re-calib
```

Safety: the LEADER is torque-free by design (you move it by hand); only
`--mirror-real` variants move physical hardware — clear the follower's
workspace before starting those.

### Mode 4 — sim → real follower (sim2real)

```bash
# ALWAYS dry-run the exact source first (100 ticks, prints commands, exit 0)
MUJOCO_GL=egl python scripts/sim_to_real.py --source script     --dry-run
MUJOCO_GL=egl python scripts/sim_to_real.py --source gamepad    --dry-run
MUJOCO_GL=egl python scripts/sim_to_real.py --source checkpoint --dry-run   # untrained policy OK for plumbing test

# LIVE: scripted demo-safe home<->reach trajectory (needs nothing else)
python scripts/sim_to_real.py --source script \
    --follower-port /dev/ttyACM1 --follower-id follower --torque-limit 40

# LIVE: drive the real arm with the DS4 through sim IK (viewer shows the sim)
# (add --no-viewer when running over SSH / without a display)
python scripts/sim_to_real.py --source gamepad \
    --follower-port /dev/ttyACM1 --follower-id follower --torque-limit 40 --max-jump-rad 0.10

# LIVE: policy checkpoint drives the real arm (see section 4)
python scripts/sim_to_real.py --source checkpoint --task pick_lift \
    --checkpoint rl/runs/mujoco/pick_lift_skrl/<stamp>/agent.pt \
    --follower-port /dev/ttyACM1 --follower-id follower --max-jump-rad 0.05
```

Built-in safety layers:

| Layer | Mechanism |
|-------|-----------|
| `--max-jump-rad` (default 0.15) | per-tick clamp between streamed targets, applied in the sink AND mirrored into lerobot's `max_relative_target` (degrees) |
| `--torque-limit <pct>` | writes STS3215 `Max_Torque_Limit` to all 6 servos at connect (e.g. 40 → 40%) |
| Ctrl-C | ramps smoothly back to HOME pose at the clamp rate, then disables torque, then disconnects |
| startup banner | big red **REAL HARDWARE** box naming the port whenever a live run starts |

Safety notes: start with the gripper away from objects/people; first live run
of any source uses `--max-jump-rad 0.05` and `--torque-limit 30`; keep one hand
near the power switch (or USB hub) as the true e-stop — Ctrl-C is graceful,
power cut is instant.

---

## 4. Sim2real policy deployment (checkpoint source)

Flow: `rl.train` (mujoco backend) → `agent.pt` → `sim_to_real.py --source checkpoint`.

```bash
# 1. train (or reuse an existing run under rl/runs/mujoco/...)
python -m rl.train --backend mujoco --task pick_lift --algo skrl --max-iterations 200 --device cpu

# 2. sanity-check the checkpoint in pure sim first
#    WARNING: rl/play.py still calls the OLD skrl API (act signature / step
#    tuple changed in our skrl fork) and currently TypeErrors on the mujoco
#    path. Until it's fixed, validate checkpoints with step 3's --dry-run
#    (it loads the real checkpoint and rolls the policy headless):
python -m rl.play --backend mujoco --task pick_lift --checkpoint <path>/agent.pt  # KNOWN BROKEN

# 3. dry-run the sim2real bridge with THAT checkpoint
MUJOCO_GL=egl python scripts/sim_to_real.py --source checkpoint --task pick_lift \
    --checkpoint <path>/agent.pt --dry-run

# 4. live, conservative limits
python scripts/sim_to_real.py --source checkpoint --task pick_lift \
    --checkpoint <path>/agent.pt \
    --follower-port /dev/ttyACM1 --follower-id follower \
    --max-jump-rad 0.05 --torque-limit 30 --hz 30
```

Mechanics: the script rebuilds the exact PPO actor from `rl/train.py`
(`SkrlPolicy`/`SkrlValue`, skrl-fork dataclass API), rolls the policy in the
MuJoCo task env at `--hz`, and streams the env's joint-position targets
(radians) to the follower as `<joint>.pos` degrees. Episodes auto-reset until
Ctrl-C. Single-arm tasks only (`action_space.shape == (6,)`).

Caveats — read before trusting it on hardware:

- **Sim calibration vs real offsets.** Sim targets are radians in the MJCF
  convention; the real arm maps degrees through lerobot homing offsets. The two
  line up only if the follower's calibration zero matches the sim zero pose.
  First live run: watch whether the arm starts near its home pose; if there's a
  constant offset, re-run `lerobot-calibrate` before anything else.
- **Joint ordering contract** is `shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper` end-to-end (MJCF, lerobot motor map, and the
  sink all use this order) — servo IDs 1–6 must match too (`lerobot-setup-motors`).
- **Start with a low `--max-jump-rad`** (0.05 rad ≈ 2.9°/tick at 30 Hz). Raise
  only after watching a full episode.
- Policies are state-obs only today (images are stripped by training design);
  camera rendering is skipped in the loop for speed.
- Domain gap: cube position randomization in sim ≠ real table; expect the arm
  to track joint targets faithfully but the *task* may not transfer visually.

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Permission denied: /dev/ttyACM*` | Not in `dialout`: `sudo usermod -aG dialout $USER`, log out/in; quick unblock `sudo chmod 666 /dev/ttyACM1`. |
| Port exists but connect fails / wrong device | USB enumeration order changed: re-check with `ls /dev/ttyACM*` + unplug/replug; use `lerobot-find-port`. |
| Feetech scan fails / "Incorrect status packet!" | Baud must be 1 Mbaud; only ONE program can hold the port (kill stray teleop procs); check servo IDs 1–6 with `lerobot-setup-motors`; retry — occasional corrupt status packets are normal, lerobot retries twice. |
| Arm jumps or sits at constant offset vs sim | Calibration mismatch: re-run `lerobot-calibrate` for that id; verify IDs match joint order (section 4). |
| pygame sees no joystick ("No joystick found") | DS4 not paired/awake; BT: re-pair (PS+SHARE); USB: try another cable/port; user not in `input` group (log out/in); Steam Input grabbing the pad — disable it; check `ls /dev/input/js*`. |
| Viewer window fails / EGL errors on desktop | Don't set `MUJOCO_GL=egl` for live viewer runs; use it only for headless/dry-run commands as written above. |
| Isaac task dies silently (exit 0, short log) | GPU/RAM starvation (~90%): free memory, close sims, retry — see HANDOFF.md sharp edges. |
| `--dry-run` passes but live errors instantly | Expected split: dry-run never imports lerobot/pygame hardware paths; live needs port + calibration file present (`~/.cache/huggingface/lerobot/calibration/...`). |
| Leader readings frozen/jumpy in `real_to_sim_check.py` | Cable/connector seating; another process holding the bus; drop `--hz` to 15–20 if the laptop CPU is loaded. |

Final pre-demo checklist:

```bash
MUJOCO_GL=egl python scripts/sim_to_real.py     --source script     --dry-run   # exit 0
MUJOCO_GL=egl python scripts/sim_to_real.py     --source gamepad    --dry-run   # exit 0
MUJOCO_GL=egl python scripts/sim_to_real.py     --source checkpoint --dry-run   # exit 0
MUJOCO_GL=egl python scripts/real_to_sim_check.py --dry-run --mirror-sim          # exit 0
```
