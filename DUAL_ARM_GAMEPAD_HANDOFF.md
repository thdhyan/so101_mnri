# DUAL-ARM GAMEPAD HANDOFF — next agent

## Goal

Rewrite the DS4 gamepad teleop so **both arms move simultaneously** in
`--arm both` mode, with independent per-arm joystick + trigger control.
No Share-toggle needed. Update both MuJoCo and Isaac scripts to the same
mapping. Keep single-arm mode unchanged (left stick XY, right stick Z,
Circle/Cross gripper, L1/R1 safe-home).

## New dual-arm mapping (`--arm both`)

| Input | Action |
|---|---|
| **Right stick X/Y** | right arm EE target X/Y velocity |
| **Left stick X/Y** | left arm EE target X/Y velocity |
| **R1 held** | right arm +Z velocity (up) |
| **R2 held** | right arm −Z velocity (down) |
| **L1 held** | left arm +Z velocity (up) |
| **L2 held** | left arm −Z velocity (down) |
| **Circle (held)** | right gripper CLOSE at fixed rate |
| **Cross (held)** | right gripper OPEN at fixed rate |
| **Triangle (held)** | left gripper CLOSE at fixed rate |
| **Square (held)** | left gripper OPEN at fixed rate |
| **Options** | re-center BOTH arms' IK targets at current EE poses |
| **PS** | e-stop BOTH arms (freeze + neutral gripper; Options resumes) |

**Removed in `--arm both` mode**: Share (no toggle needed), L1/R1 safe-home
(shoulders now drive Z). Safe-home is still available in single-arm mode.

### Single-arm mode (`--arm left` or `--arm right`, unchanged)

Keep the existing mapping from the current scripts:

| Input | Action |
|---|---|
| Left stick X/Y | active arm EE target XY velocity |
| Right stick Y | active arm EE target Z velocity |
| Right stick X | DEAD |
| Circle/Cross (held) | active gripper close/open |
| L1/R1 (press) | safe-home: ramp to HOME_POSE, re-center |
| Options | re-center at current EE |
| Share | toggle arm (single-arm dual tasks) |
| PS | e-stop (freeze; Options resumes) |
| Square | safe quit (MuJoCo only) |

L2/R2 are DEAD in single-arm mode.

## Architecture changes

### MuJoCo (`scripts/teleop_gamepad_ik.py`)

Current limitation (line 55-61): `--arm both` is explicitly unsupported
because "a single DS4 has only 4 stick DoF while driving two arms'
XY+Z simultaneously needs 6." The NEW mapping uses 6 DoF: left stick
XY (2) + right stick XY (2) + L1/L2 shoulder buttons (1+1) = 6. This
is now solvable.

Changes needed:

1. **`run_gamepad()` function** (line 649): when `env == "dual"` and
   `arm == "both"`, drive BOTH `ik_by_arm["left"]` and
   `ik_by_arm["right"]` every tick with separate commands from the
   SAME pad state. Currently only `cur_ik = ik_by_arm[active_arm]`
   moves; the other arm holds still. Instead:

   ```
   right_ik = ik_by_arm["right"]
   left_ik  = ik_by_arm["left"]

   # Right arm: right stick XY, R1/R2 for Z
   r_vx = -st.right_stick[1] * LINEAR_SCALE
   r_vy = -st.right_stick[0] * LINEAR_SCALE
   r_vz = (float(st.buttons["r1"]) - float(st.buttons["r2"])) * LINEAR_SCALE
   right_ik.set_ee_target(right_ik.ee_target + [r_vx, r_vy, r_vz] * dt)

   # Left arm: left stick XY, L1/L2 for Z
   l_vx = -st.left_stick[1] * LINEAR_SCALE
   l_vy = -st.left_stick[0] * LINEAR_SCALE
   l_vz = (float(st.buttons["l1"]) - float(st.buttons["l2"])) * LINEAR_SCALE
   left_ik.set_ee_target(left_ik.ee_target + [l_vx, l_vy, l_vz] * dt)

   # Grippers: Circle/Cross = right, Triangle/Square = left
   r_grip = (btn("cross") - btn("circle")) * GRIP_SCALE
   l_grip = (btn("square") - btn("triangle")) * GRIP_SCALE
   right_ik.set_gripper(right_ik.gripper_openness + r_grip * dt / (hi-lo))
   left_ik.set_gripper(left_ik.gripper_openness + l_grip * dt / (hi-lo))
   ```

2. **Remove Share toggle** from `--arm both` mode (no-op). Keep it in
   single-arm mode.

3. **Update `print_mapping_table()`** to show the dual mapping when
   `--arm both`.

4. **Debug sphere**: draw TWO spheres — one per arm. Current
   `_draw_target_marker()` writes exactly one geom. Extend to write
   two geoms: red for right arm, blue for left arm (or just draw both
   red at different positions). The user_scn can hold multiple geoms.

5. **E-stop/Options**: apply to BOTH arms (already does via `ik_by_arm.values()`).

6. **Update `--arm` help text** to remove the "intentionally NOT offered"
   comment (line 887-889). `--arm both` is now supported.

### Isaac (`scripts/isaac_teleop_gamepad.py`)

The Isaac script already supports `--arm both` (line 543-544):
```python
elif args.arm == "both":
    active = list(arm_names_all)
```

But the DS4Device produces ONE `Cmd` for ONE active arm. The current
`assemble_actions` loops over `arm_names_all` but only the active ones
get the command; inactive ones get `home_row`.

Changes needed:

1. **`DS4Device`** or the main loop: produce TWO `Cmd` objects — one
   per arm. The simplest approach: parse the pad state in the main loop
   and build per-arm Cmds, rather than having DS4Device return a single
   Cmd. Alternatively, extend `Cmd` to carry `dpos_left`, `dpos_right`,
   etc.

2. **`assemble_actions`**: when `args.arm == "both"`, fill both arm
   slots from the per-arm Cmd objects.

3. **ArmIK markers**: draw per-arm markers (already does —
   `marker_positions()` stacks all active arms). The left/right
   marker distinction comes from the order in `active`.

4. **Remove arm toggle** from `--arm both` mode (it currently only
   toggles in single mode — line 645 condition already handles this).

5. **Update `print_mapping()`** for dual-arm mode.

### Both scripts — mock-button test updates

The `_MockPad` / `NullDevice` need new test sequences for dual-arm:

- Left stick drives left arm, right stick drives right arm
- R1/R2 affect right arm Z, L1/L2 affect left arm Z
- Circle/Cross affect right gripper, Triangle/Square affect left gripper
- Both arms' markers move independently

## Key files

- `scripts/teleop_gamepad_ik.py` — MuJoCo (DS4Gamepad, IKTeleop, run_gamepad, mock tests)
- `scripts/isaac_teleop_gamepad.py` — Isaac (DS4Device, ArmIK, assemble_actions, NullDevice, mock tests)
- `TELEOP_GUIDE.md` — update mapping table (both modes)
- `scripts/check_gamepad.py` — no changes needed (already shows all axes/buttons)

## Validation

### MuJoCo
```bash
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run  # unchanged, must pass
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env dual --arm both --dry-run  # NEW
MUJOCO_GL=egl python scripts/verify_manual.py --env dual --headless-check  # must still pass
```

### Isaac
```bash
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --arm both --null-device --headless
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless  # unchanged
```

### Both
- All existing single-arm dry-runs must pass unchanged
- New dual-arm dry-runs must pass with per-arm assertions:
  - left stick moves left arm, right stick moves right arm
  - R1/R2 change right arm Z, L1/L2 change left arm Z
  - Circle/Cross change right gripper, Triangle/Square change left gripper
  - Both debug spheres visible and tracking independently
  - PS e-stop freezes both arms; Options re-centers both
  - L1/L2 and R1/R2 DO NOT trigger safe-home in `--arm both` mode

## Gotchas

1. **Button conflicts in `--arm both`**: L1/R1 now drive Z, NOT safe-home.
   Safe-home is removed in dual mode. Make sure the code does NOT also
   trigger homing when L1/R1 are pressed in `--arm both` mode.

2. **L2/R2 are analog triggers** on DS4 (rest at 0 or -1 depending on
   SDL layer). They need thresholding: treat as a button (pressed if
   value > 0.5) for Z velocity, NOT as a continuous rate. The existing
   `ds4_trigger_norm()` function converts raw [-1,1] → [0,1]; use a
   deadzone threshold (e.g. > 0.3 = pressed).

3. **Wrist joints**: in `--arm both` mode, both arms' wrist_flex/wrist_roll
   are driven by IK nullspace (no direct user input). Pin them to current
   qpos like the single-arm code does (line 814-817).

4. **Rate scaling for Z via triggers**: triggers give a binary pressed/not,
   so Z velocity at full trigger = LINEAR_SCALE (same as stick full
   deflection). This is intentional — user said "R1/R2 for +z and -z".

5. **Existing `_draw_target_marker`** uses `scn.geoms[0]` — extend to
   `scn.geoms[0]` and `scn.geoms[1]` (set `scn.ngeom = 2`). Verify
   MAX_GEOM in the viewer is large enough (mujoco default is 100+).
