#!/usr/bin/env python
"""DualShock 4 gamepad Cartesian teleop for the SO-101 Isaac Lab tasks.

Mirrors scripts/teleop_gamepad_ik.py (MuJoCo) but drives Isaac Lab
ManagerBasedRLEnvs through the official controller stack:

    DS4Device (pygame) -> per-tick velocity command
        -> DifferentialIKController (isaaclab.controllers, position/relative/dls)
        -> joint-position targets (clipped to soft limits)
        -> env.step(action)   [absolute joint-position actions, 6/arm]

The EE is the ``gripper_frame_link`` body (USD twin of the MuJoCo
``gripperframe`` site). Pose feedback and the geometric Jacobian come straight
from articulation data (``body_link_jacobian_w``, root/body poses), following
isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction. Position-only
IK drives the 5 non-gripper joints; the gripper is an openness scalar mapped
onto its joint range — neither passes through IK.

Mapping (2026-08 rework):
    left stick x/y          EE target x/y velocity
    right stick y           EE target z velocity (right stick x is DEAD)
    Circle (held)           gripper CLOSE at fixed rate
    Cross (held)            gripper OPEN at the same fixed rate
    L2 / R2                 DEAD
    L1 or R1 (press)        safe-home: rate-limited (~1.5 s) ramp of ALL
                            joints to HOME_JOINT_POS (envs/isaac/so101.py),
                            then re-center the IK target at the home EE pose;
                            interrupted by any stick deflection past deadzone
    Share                   toggle active arm (dual tasks)
    Options                 re-center EE target on current EE pose
    PS                      quit, ramping home first

A semi-transparent red sphere marker (isaaclab.markers.VisualizationMarkers,
r = 1.2 cm) is drawn at each active arm's IK target every step — including
during homing — in both windowed and headless runs.

Usage:
    # headless pipeline validation + mock-button test suite (no pygame):
    python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless
    python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --null-device --arm both
    # live demo (windowed):
    python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --arm both
"""

import argparse
import importlib
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# task_id -> (cfg module short name, cfg class name, n_arms)
TASKS = {
    "SO101-PickLift-Single-v0": ("pick_lift", "PickLiftEnvCfg", 1),
    "SO101-PickPlace-Single-v0": ("pick_place", "PickPlaceEnvCfg", 1),
    "SO101-CylReach-Single-v0": ("cylinder_reach_single", "CylReachSingleEnvCfg", 1),
    "SO101-CylGrasp-Dual-v0": ("cylinder_grasp", "CylGraspEnvCfg", 2),
    "SO101-CylReach-Dual-v0": ("cylinder_reach", "CylReachEnvCfg", 2),
}

# ---- DS4 conventions --------------------------------------------------------
DEADZONE = 0.15
LINEAR_SCALE = 0.15  # m/s at full stick deflection (x --pos-sensitivity)
GRIP_RATE = 1.0      # fixed gripper openness/s while Circle/Cross is held
HOME_RAMP_TIME = 1.5  # s — safe-home ramp duration from any reachable pose
HOME_TOL = 0.02       # rad — joint feedback tolerance for "homed"

# Raw hid-sony layout (what the MuJoCo script uses on this machine): axes
# LX,LY,RX,RY,L2,R2; buttons square,cross,circle,triangle,L1,R1,share,
# options,L3,R3,PS,... If SDL exposes the pad through the gamecontroller API,
# button indices shift (see _BUTTONS_GC); axes are the same.
AX_LX, AX_LY, AX_RX, AX_RY = range(4)
_BUTTONS_RAW = {"square": 0, "cross": 1, "circle": 2, "triangle": 3,
                "l1": 4, "r1": 5, "share": 6, "options": 7, "ps": 10}
_BUTTONS_GC = {"cross": 0, "circle": 1, "square": 2, "triangle": 3,
               "l1": 9, "r1": 10, "share": 4, "options": 6, "ps": 5}


def apply_deadzone(x, dz=DEADZONE):
    if abs(x) < dz:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    return sign * (abs(x) - dz) / (1.0 - dz)


class Cmd:
    """Per-tick device command."""

    __slots__ = ("dpos", "grip_rate", "rate_mode",
                 "quit", "recentre", "toggle_arm", "home", "stick_active")

    def __init__(self):
        self.dpos = None           # rates (m/s) if rate_mode else per-tick displacements
        self.grip_rate = 0.0       # openness/s if rate_mode else per-tick openness delta
        self.rate_mode = True      # DS4: rates scaled by measured dt; null device: False
        self.quit = False          # edge: exit teleop (with home ramp)
        self.recentre = False      # edge: snap EE target onto current EE pose
        self.toggle_arm = False    # edge: switch active arm (dual tasks)
        self.home = False          # edge: start safe-home ramp (L1/R1)
        self.stick_active = False  # any stick beyond deadzone (cancels homing)


class DS4Device:
    """pygame DualShock 4 reader producing per-tick velocity commands.

    Mapping: left stick = EE x/y, right stick Y = EE z (right stick X dead),
    Circle/Cross held = gripper close/open at GRIP_RATE, L1/R1 press =
    safe-home ramp, Share toggles the active arm (dual), Options re-centers
    the EE target, PS quits with a home ramp. L2/R2 are dead.
    """

    def __init__(self, index=0, pos_sensitivity=1.0):
        import pygame

        self._pos_scale = LINEAR_SCALE * pos_sensitivity
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick found. Connect the DS4 or use --null-device.", flush=True)
            raise SystemExit(1)
        self._pygame = pygame
        self._joy = pygame.joystick.Joystick(index)
        self._joy.init()
        self._prev_btns = {}
        # Button layout depends on whether SDL applied a gamecontroller mapping.
        self._btn = _BUTTONS_RAW
        self._layout = "raw DS4"
        try:
            import pygame._sdl2 as sdl2

            if sdl2.controller.is_controller(index):
                self._btn = _BUTTONS_GC
                self._layout = "gamecontroller"
        except Exception:
            pass
        print(f"Using joystick: {self._joy.get_name()} "
              f"(axes={self._joy.get_numaxes()}, buttons={self._joy.get_numbuttons()}, "
              f"layout={self._layout})", flush=True)

    def advance(self) -> Cmd:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cmd = Cmd()
                cmd.quit = True
                return cmd

        joy = self._joy
        lx_raw = joy.get_axis(AX_LX)
        ly_raw = joy.get_axis(AX_LY)
        rx_raw = joy.get_axis(AX_RX) if joy.get_numaxes() > AX_RX else 0.0
        ry_raw = joy.get_axis(AX_RY) if joy.get_numaxes() > AX_RY else 0.0
        lx = apply_deadzone(lx_raw)
        ly = apply_deadzone(ly_raw)
        ry = apply_deadzone(ry_raw)  # rx intentionally unused: right stick X is DEAD

        def btn(name):
            i = self._btn[name]
            return bool(joy.get_button(i)) if i < joy.get_numbuttons() else False

        cmd = Cmd()
        # MuJoCo-script sign conventions: stick up/left = negative axis value.
        cmd.dpos = [-ly * self._pos_scale, -lx * self._pos_scale, -ry * self._pos_scale]
        cmd.grip_rate = (btn("circle") - btn("cross")) * GRIP_RATE  # close -, open +
        now = {n: btn(n) for n in ("share", "options", "ps", "l1", "r1")}
        cmd.toggle_arm = now["share"] and not self._prev_btns.get("share")
        cmd.recentre = now["options"] and not self._prev_btns.get("options")
        cmd.quit = now["ps"] and not self._prev_btns.get("ps")
        cmd.home = (now["l1"] or now["r1"]) and not (
            self._prev_btns.get("l1") or self._prev_btns.get("r1"))
        cmd.stick_active = any(abs(a) > DEADZONE for a in (lx_raw, ly_raw, rx_raw, ry_raw))
        self._prev_btns = now
        return cmd

    @staticmethod
    def reseed():
        pass

    def close(self):
        try:
            self._pygame.joystick.quit()
            self._pygame.quit()
        except Exception:
            pass


class NullDevice:
    """Scripted device proving the IK->action pipeline headlessly.

    Phase schedule (see advance()):
      track : circular EE trajectory + sinusoidal gripper sweep (tracking metric)
      close : Circle held     -> gripper closes at GRIP_RATE
      open  : Cross held      -> gripper opens at GRIP_RATE
      press : single L1 press -> triggers the safe-home ramp
      wait  : zero commands while the run loop homes (until notify_homed())
      rx    : right stick X fully deflected -> must produce NO motion
    """

    DT_MOCK = 1.0 / 60.0  # nominal dt for per-tick deltas (rate_mode=False)
    TRACK_STEPS = 40
    GRIP_STEPS = 20
    HOME_WAIT_CAP = 400
    RX_STEPS = 10

    def __init__(self, radius=0.03, steps_per_rev=100):
        self.radius = radius
        self.steps_per_rev = steps_per_rev
        self.i = 0
        self.phase = "track"
        self.finished = False
        self._homed = False
        self._wait = 0
        self._rx = 0
        self._n = 0  # ticks spent in current phase
        # prev starts at ORIGIN so the first delta jumps out to T(0)=(r,0,0):
        # the run then shows genuine convergence onto the moving circle
        self._prev_t = [0.0, 0.0, 0.0]
        self._prev_grip = 0.0

    def _local_target(self, i):
        theta = 2 * math.pi * i / self.steps_per_rev
        return [self.radius * math.cos(theta), self.radius * math.sin(theta), 0.0]

    def notify_homed(self):
        self._homed = True

    def _end_phase(self, limit, nxt):
        self._n += 1
        if self._n >= limit:
            self.phase = nxt
            self._n = 0

    def advance(self) -> Cmd:
        cmd = Cmd()
        cmd.rate_mode = False
        cmd.dpos = [0.0, 0.0, 0.0]
        p = self.phase
        if p == "track":
            t = self._local_target(self.i)
            grip = 0.5 * (1 + math.sin(2 * math.pi * self.i / self.steps_per_rev))
            cmd.dpos = [a - b for a, b in zip(t, self._prev_t)]
            cmd.grip_rate = grip - self._prev_grip
            self._prev_t = t
            self._prev_grip = grip
            self._end_phase(self.TRACK_STEPS, "close")
        elif p == "close":
            cmd.dpos = [0.0, 0.0, 0.0]
            cmd.grip_rate = -GRIP_RATE * self.DT_MOCK  # Circle held -> CLOSE
            self._end_phase(self.GRIP_STEPS, "open")
        elif p == "open":
            cmd.dpos = [0.0, 0.0, 0.0]
            cmd.grip_rate = GRIP_RATE * self.DT_MOCK   # Cross held -> OPEN
            self._end_phase(self.GRIP_STEPS, "press")
        elif p == "press":
            cmd.home = True
            self._end_phase(1, "wait")
        elif p == "wait":
            if self._homed or self._wait >= self.HOME_WAIT_CAP:
                self._end_phase(1, "rx")
            else:
                self._wait += 1
        elif p == "rx":
            cmd.stick_active = True  # right stick X fully deflected: must be DEAD
            self._rx += 1
            if self._rx >= self.RX_STEPS:
                self.finished = True
        self.i += 1
        return cmd

    def reseed(self):
        self.i = 0
        self.phase = "track"
        self.finished = False
        self._homed = False
        self._wait = 0
        self._rx = 0
        self._n = 0
        self._prev_t = [0.0, 0.0, 0.0]
        self._prev_grip = 0.0

    def close(self):
        pass


class EETargetMarkers:
    """Debug overlay: semi-transparent red sphere at each arm's IK target.

    Spawned visual-only under /Visuals via VisualizationMarkers (a
    UsdGeom.PointInstancer — no physics, harmless headless). ``visualize()``
    is called every step with the world-frame EE targets; the last written
    translations are recorded so headless runs can assert updates happened.
    """

    RADIUS = 0.012  # m (~1.2 cm)

    def __init__(self, prim_path="/Visuals/so101_ee_targets"):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "ee_target": sim_utils.SphereCfg(
                    radius=self.RADIUS,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        opacity=0.5,
                    ),
                )
            },
        )
        self.viz = VisualizationMarkers(cfg)
        self.n_updates = 0
        self.first_pos = None  # (M, 3) world translations, first write
        self.last_pos = None   # (M, 3) world translations, latest write

    def update(self, positions_world):
        pos = positions_world.detach().cpu()
        self.last_pos = pos.clone()
        if self.first_pos is None:
            self.first_pos = pos.clone()
        self.n_updates += 1
        self.viz.visualize(translations=pos)


class ArmIK:
    """Differential-IK wrapper for one SO-101 arm inside a ManagerBasedRLEnv.

    Uses isaaclab.controllers.DifferentialIKController (position command,
    relative mode, damped least squares). Feedback pose = EE body pose in the
    arm's root frame via subtract_frame_transforms; Jacobian columns restricted
    to the 5 non-gripper joints of body_link_jacobian_w — the same recipe as
    isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction.
    """

    IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    ALL_JOINT_NAMES = IK_JOINT_NAMES + ["gripper"]
    EE_BODY_CANDIDATES = ["gripper_frame_link", "gripper_frame"]

    def __init__(self, robot, num_envs, device="cpu", lambda_val=0.05):
        from isaaclab.controllers.differential_ik import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

        self.robot = robot
        # controller buffers must match the sim-data tensors handed to it
        self._dev = str(device)
        self.joint_ids, _ = robot.find_joints(self.IK_JOINT_NAMES, preserve_order=True)
        self.all_ids, _ = robot.find_joints(self.ALL_JOINT_NAMES, preserve_order=True)
        body_ids, body_names = [], []
        for cand in self.EE_BODY_CANDIDATES:
            body_ids, body_names = robot.find_bodies(cand)
            if len(body_ids) == 1:
                break
        if len(body_ids) != 1:
            raise RuntimeError(f"EE body not found uniquely: matched {body_names}")
        self.ee_body_idx = body_ids[0]
        self._jac_body_idx = self.ee_body_idx - 1 if robot.is_fixed_base else self.ee_body_idx
        self._jac_col_ids = [j + robot.num_base_dofs for j in self.joint_ids]

        cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": lambda_val},
        )
        self.ik = DifferentialIKController(cfg, num_envs=num_envs, device=self._dev)

        self.home_row = None  # (6,) torch cpu, filled by caller
        self.reset_state()

    def reset_state(self):
        self.ee_target = None
        self.openness = 0.0
        self.homing = False
        self._home_q = None      # (N, 6) ramped joint target during homing
        self._home_rates = None  # (N, 6) rad/s toward home
        self._open0 = 0.0        # openness when homing started

    def ee_pose_b(self):
        from isaaclab.utils.math import subtract_frame_transforms

        d = self.robot.data
        return subtract_frame_transforms(
            d.root_pos_w.torch, d.root_quat_w.torch,
            d.body_pos_w.torch[:, self.ee_body_idx], d.body_quat_w.torch[:, self.ee_body_idx],
        )

    def seed_target(self):
        pos, _ = self.ee_pose_b()
        self.ee_target = pos.clone()

    def apply_command(self, cmd: Cmd, mirror_y: bool, dt: float):
        dpos = list(cmd.dpos)
        if mirror_y:
            dpos[1] = -dpos[1]
        if cmd.rate_mode:
            self.ee_target = self.ee_target + dt * self.ee_target.new_tensor(dpos)
            self.openness += cmd.grip_rate * dt
        else:
            self.ee_target = self.ee_target + self.ee_target.new_tensor(dpos)
            self.openness += cmd.grip_rate
        self.openness = max(0.0, min(1.0, self.openness))

    def start_homing(self):
        """Begin the rate-limited joint-space ramp to the home pose."""
        import torch

        q = self.robot.data.joint_pos.torch[:, self.all_ids].cpu().to(torch.float32)  # (N, 6)
        delta = self.home_row.to(q.dtype) - q                                        # (N, 6)
        self._home_q = q.clone()
        self._home_rates = (delta.abs() / HOME_RAMP_TIME).clamp(min=1e-3, max=5.0)
        self._open0 = self.openness
        self.homing = True

    def cancel_homing(self):
        self.homing = False
        self.openness = self._home_openness()
        self.seed_target()

    def _home_openness(self):
        """Openness scalar whose gripper mapping equals the home joint angle."""
        import torch

        limits = self.robot.data.soft_joint_pos_limits.cpu().to(torch.float32)[0, self.all_ids[5]]
        lo, hi = float(limits[0]), float(limits[1])
        return max(0.0, min(1.0, (float(self.home_row[5]) - lo) / max(hi - lo, 1e-6)))

    def home_step(self, dt: float):
        """Advance the ramp one tick; return absolute joint targets (N, 6).

        All six joints (gripper included) ramp toward ``home_row``; the IK
        target is dragged along with the actual EE pose so the debug sphere
        tracks the arm throughout the homing motion.
        """
        import torch

        step = torch.clamp(self.home_row.to(self._home_q.dtype) - self._home_q,
                           -self._home_rates * dt, self._home_rates * dt)
        self._home_q = self._home_q + step
        self.openness = max(0.0, self.openness - abs(self._open0) / HOME_RAMP_TIME * dt)
        self.ee_target = self.ee_pose_b()[0].clone()

        limits = self.robot.data.soft_joint_pos_limits.cpu().to(torch.float32)[:, self.all_ids]
        return torch.clamp(self._home_q, limits[..., 0], limits[..., 1])  # (N, 6)

    def homed(self):
        """True once actual joints sit at home within HOME_TOL."""
        import torch

        q = self.robot.data.joint_pos.torch[:, self.all_ids].cpu().to(torch.float32)
        return bool(((q - self.home_row.to(q.dtype)).abs() < HOME_TOL).all())

    def action_row(self, debug=False):
        """Solve IK and return absolute joint-position targets, shape (N, 6)."""
        import torch

        ee_pos_b, ee_quat_b = self.ee_pose_b()
        joint_pos = self.robot.data.joint_pos.torch[:, self.joint_ids]       # (N, 5)
        jacobian = self.robot.data.body_link_jacobian_w.torch[
            :, self._jac_body_idx, :, self._jac_col_ids
        ]                                                                     # (N, 6, 5)
        self.ik.set_command(self.ee_target - ee_pos_b, ee_pos_b, ee_quat_b)
        q_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)     # (N, 5)
        if debug:
            print(f"    [dbg] ee_pos_b={ee_pos_b[0].cpu().numpy().round(4)} "
                  f"tgt={self.ee_target[0].cpu().numpy().round(4)} "
                  f"|jac|={float(jacobian[0].norm()):.4f} "
                  f"q={joint_pos[0].cpu().numpy().round(3)} "
                  f"q_des={q_des[0].cpu().numpy().round(3)}", flush=True)

        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.all_ids]  # (N, 6, 2)
        q_des = torch.clamp(q_des, limits[:, :5, 0], limits[:, :5, 1])
        gripper_lo, gripper_hi = limits[:, 5, 0], limits[:, 5, 1]
        gripper = gripper_lo + self.openness * (gripper_hi - gripper_lo)
        return torch.cat([q_des, gripper.unsqueeze(1)], dim=1)                 # (N, 6)


def print_mapping():
    rows = [
        ("Left stick (x/y)", "EE target x/y velocity (0.15 m/s at full deflection)"),
        ("Right stick vertical", "EE target z velocity"),
        ("Right stick horizontal", "DEAD"),
        ("Circle (held)", f"gripper CLOSE at fixed rate ({GRIP_RATE:.1f} openness/s)"),
        ("Cross (held)", f"gripper OPEN at fixed rate ({GRIP_RATE:.1f} openness/s)"),
        ("L2 / R2 (triggers)", "DEAD"),
        ("L1 or R1 (press)", f"safe-home: ~{HOME_RAMP_TIME:.1f} s ramp of all joints to "
                             f"HOME_JOINT_POS, then re-center IK target; sticks cancel"),
        ("Share", "toggle active arm (dual-arm tasks)"),
        ("Options", "re-center EE target on current EE pose"),
        ("PS", "quit (ramps home first)"),
    ]
    print("\nDS4 mapping:", flush=True)
    for k, v in rows:
        print(f"  {k:<24} {v}", flush=True)
    print("", flush=True)


def strip_cameras(scene_cfg):
    """Drop camera sensors from the scene cfg — state-based teleop never reads them.

    Same trick training uses to avoid headless RTX renders (test.md: flaky
    under memory pressure). enable_cameras=True is still required because the
    task scenes are authored with camera sensors.
    """
    removed = []
    for f in fields(scene_cfg):
        if type(getattr(scene_cfg, f.name, None)).__name__.endswith("CameraCfg"):
            setattr(scene_cfg, f.name, None)
            removed.append(f.name)
    if removed:
        print(f"[teleop] stripped camera sensors: {removed}", flush=True)


def build_env(args, task_entry):
    from isaaclab.envs import ManagerBasedRLEnv

    short, cfg_cls_name, n_arms = task_entry
    module = importlib.import_module(f"envs.isaac.tasks.{short}.env_cfg")
    cfg = getattr(module, cfg_cls_name)()
    cfg.scene.num_envs = args.num_envs
    strip_cameras(cfg.scene)
    env = ManagerBasedRLEnv(cfg=cfg)
    assert env.action_manager.total_action_dim == 6 * n_arms
    return env, n_arms


def run(args, simulation_app):
    import torch

    from envs.isaac.so101 import HOME_JOINT_POS, JOINT_NAMES
    assert ArmIK.ALL_JOINT_NAMES == JOINT_NAMES

    task_entry = TASKS[args.task]
    n_arms_task = task_entry[2]
    arm_names_all = ["robot_left", "robot_right"] if n_arms_task == 2 else ["robot"]
    if n_arms_task == 1:
        active = ["robot"]
    elif args.arm == "both":
        active = list(arm_names_all)
    else:
        active = [f"robot_{args.arm}"]

    env, n_arms = build_env(args, task_entry)
    obs, _ = env.reset()

    controllers = {}
    for name in arm_names_all:
        aik = ArmIK(env.scene[name], env.num_envs, device=env.device)
        aik.seed_target()
        ids, _ = env.scene[name].find_joints(JOINT_NAMES, preserve_order=True)
        aik.home_row = env.scene[name].data.default_joint_pos.torch[:, ids][0].cpu().to(torch.float32)
        expected_home = torch.tensor([HOME_JOINT_POS[n] for n in JOINT_NAMES])
        assert torch.allclose(aik.home_row, expected_home, atol=1e-6), \
            f"{name}: default_joint_pos != HOME_JOINT_POS"
        controllers[name] = aik

    markers = EETargetMarkers()

    def marker_positions():
        from isaaclab.utils.math import combine_frame_transforms

        pts = []
        for name in active:
            d = env.scene[name].data
            pos_w, _ = combine_frame_transforms(
                d.root_pos_w.torch, d.root_quat_w.torch, controllers[name].ee_target)
            pts.append(pos_w[0].cpu())
        return torch.stack(pts)  # (M, 3) world

    if args.null_device:
        device = NullDevice(radius=args.circle_radius)
        max_steps = args.steps if args.steps else (
            NullDevice.TRACK_STEPS + 2 * NullDevice.GRIP_STEPS + 1
            + NullDevice.HOME_WAIT_CAP + NullDevice.RX_STEPS + 1)
        print(f"[null-device] task={args.task} arms={active} circle r={args.circle_radius} m "
              f"for <= {max_steps} steps (incl. mock-button suite)", flush=True)
    else:
        device = DS4Device(args.device, args.pos_sensitivity)
        max_steps = args.steps if args.steps else 10 ** 9
        print_mapping()
        print(f"[teleop] task={args.task} active={active} — PS quits (home ramp first)",
              flush=True)

    homing_on = False
    homing_start = 0
    quit_pending = False

    # mock-button assertion bookkeeping (null-device only)
    mock_results = {}
    mock_snap = {}
    last_phase = None
    rows_hist = {}  # name -> latest action row (cpu)

    def mrecord(key, ok, msg):
        if key not in mock_results:
            mock_results[key] = (ok, msg)
            print(f"[mock-test] {'PASS' if ok else 'FAIL'} {key}: {msg}", flush=True)

    def assemble_actions(cmd, dt):
        act = torch.zeros(env.num_envs, 6 * n_arms, dtype=torch.float32)
        for slot, name in enumerate(arm_names_all):
            aik = controllers[name]
            if name in active:
                if aik.homing:
                    act[0, slot * 6:(slot + 1) * 6] = aik.home_step(dt)[0].cpu()
                else:
                    mirror_y = (n_arms == 2 and args.arm == "both" and name == "robot_right")
                    aik.apply_command(cmd, mirror_y, dt)
                    act[0, slot * 6:(slot + 1) * 6] = aik.action_row(debug=args.debug and step < 4)[0].cpu()
            else:
                act[0, slot * 6:(slot + 1) * 6] = aik.home_row
            rows_hist[name] = act[0, slot * 6:(slot + 1) * 6].cpu().clone()
        return act.to(env.device)

    last_t = time.time() - 0.02
    errors_window = []
    err_history = []
    all_finite = True
    step = 0

    try:
        while step < max_steps and simulation_app.is_running():
            dt = min(0.05, max(1e-3, time.time() - last_t))
            last_t = time.time()

            cmd = device.advance()
            if cmd.quit:
                quit_pending = True
                if not homing_on:
                    for aik in controllers.values():
                        aik.start_homing()
                    homing_on = True
                    homing_start = step
                    print(f"[teleop] PS quit — ramping home first (~{HOME_RAMP_TIME:.1f} s)",
                          flush=True)
            if cmd.recentre:
                for name in active:
                    controllers[name].seed_target()
                print("[teleop] re-centered", flush=True)
            if cmd.toggle_arm and n_arms == 2 and args.arm != "both":
                other = "robot_right" if active[0] == "robot_left" else "robot_left"
                active = [other]
                print(f"[teleop] active arm -> {'right' if other.endswith('right') else 'left'}",
                      flush=True)
            if cmd.home and not homing_on and not quit_pending:
                for aik in controllers.values():
                    aik.start_homing()
                homing_on = True
                homing_start = step
                if args.null_device:
                    mock_snap["home_marker_pos"] = markers.last_pos.clone() \
                        if markers.last_pos is not None else None
                print(f"[teleop] safe-home engaged (~{HOME_RAMP_TIME:.1f} s ramp; "
                      f"stick input cancels)", flush=True)
            if homing_on and cmd.stick_active and not quit_pending:
                for aik in controllers.values():
                    if aik.homing:
                        aik.cancel_homing()
                homing_on = False
                print("[teleop] safe-home cancelled (stick input)", flush=True)

            if quit_pending and not homing_on:
                break

            actions = assemble_actions(cmd, dt)
            obs, rew, term, trunc, _extras = env.step(actions)
            step += 1

            # debug sphere follows the (possibly homing-dragged) IK target every step
            markers.update(marker_positions())

            finite = bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
            all_finite &= finite
            if not finite:
                print(f"[step {step}] NON-FINITE obs/reward", flush=True)

            if term.any() or trunc.any():
                obs, _ = env.reset()
                for aik in controllers.values():
                    aik.reset_state()
                    aik.seed_target()
                    if homing_on:
                        aik.start_homing()
                device.reseed()

            if homing_on:
                if all(aik.homed() for aik in controllers.values()):
                    for aik in controllers.values():
                        if aik.homing:
                            aik.cancel_homing()  # clears flag + seeds target at home EE pose
                    homing_on = False
                    print("[teleop] homed — IK target re-centered at home EE pose", flush=True)
                    if args.null_device:
                        device.notify_homed()
                        mrecord("home_ramp_reaches_home_then_clears", True,
                                f"joints at HOME_JOINT_POS (+/-{HOME_TOL}) after "
                                f"{step - homing_start} steps; homing flags cleared")
                        ee0 = controllers[active[0]].ee_target[0].cpu()
                        eep = controllers[active[0]].ee_pose_b()[0][0].cpu()
                        mrecord("home_recentre_at_home_ee", bool((ee0 - eep).norm() < 1e-4),
                                f"|ee_target - ee_pose| = {float((ee0 - eep).norm()):.2e} m")
                        snap = mock_snap.get("home_marker_pos")
                        moved_home = (float((markers.last_pos - snap).abs().max())
                                      if snap is not None and markers.last_pos is not None else 0.0)
                        mrecord("marker_tracks_during_homing", moved_home > 1e-3,
                                f"sphere moved {moved_home * 1000:.1f} mm while ramping home")
                elif step - homing_start > 2 * NullDevice.HOME_WAIT_CAP:
                    for aik in controllers.values():
                        aik.cancel_homing()
                    homing_on = False
                    mrecord("home_ramp_reaches_home_then_clears", False,
                            "timeout — joints never reached HOME_JOINT_POS")
                    device.notify_homed()

            # null-device mock-button assertions
            if args.null_device:
                phase = device.phase
                if phase != last_phase:
                    if phase == "close":
                        mock_snap["close_grip"] = rows_hist[active[0]][5].item()
                        mock_snap["close_open"] = controllers[active[0]].openness
                    elif phase == "open":
                        g0 = mock_snap.pop("close_grip")
                        o0 = mock_snap.pop("close_open")
                        g1 = rows_hist[active[0]][5].item()
                        o1 = controllers[active[0]].openness
                        mrecord("circle_closes_gripper", g1 < g0 - 1e-3,
                                f"gripper joint target {g0:+.4f} -> {g1:+.4f} rad "
                                f"(openness {o0:.3f} -> {o1:.3f})")
                        mock_snap["open_grip"], mock_snap["open_open"] = g1, o1
                    elif phase == "press":
                        g2 = rows_hist[active[0]][5].item()
                        o2 = controllers[active[0]].openness
                        mrecord("cross_opens_gripper",
                                g2 > mock_snap["open_grip"] + 1e-3,
                                f"gripper joint target {mock_snap['open_grip']:+.4f} -> "
                                f"{g2:+.4f} rad (openness {mock_snap['open_open']:.3f} -> {o2:.3f})")
                    elif phase == "rx":
                        mock_snap["rx_tgt"] = {n: controllers[n].ee_target.clone()
                                               for n in active}
                        mock_snap["rx_roll"] = {n: rows_hist[n][4].item() for n in active}
                    last_phase = phase

                if last_phase == "rx" and device.finished \
                        and "right_stick_x_dead" not in mock_results:
                    ok, msg = True, []
                    for n in active:
                        dtgt = float((controllers[n].ee_target - mock_snap["rx_tgt"][n]).abs().max())
                        droll = abs(rows_hist[n][4].item() - mock_snap["rx_roll"][n])
                        ok &= dtgt == 0.0 and droll < 1e-3  # ee frozen exactly; IK chatter tol on roll
                        msg.append(f"{n}: |d tgt|={dtgt:.1e} d|roll_tgt|={droll:.1e}")
                    mrecord("right_stick_x_dead", ok,
                            "right-stick X fully deflected -> " + "; ".join(msg))

                # tracking metric only over the scripted-circle phase
                if phase == "track":
                    errs = []
                    for name in active:
                        aik = controllers[name]
                        ee_pos_b, _ = aik.ee_pose_b()
                        errs.append(float(torch.linalg.norm(ee_pos_b - aik.ee_target).item()))
                    errors_window.append(max(errs))
                    if args.debug and step <= 15:
                        print(f"    [dbg-metric] step {step}: ee={ee_pos_b[0].cpu().numpy().round(4)} "
                              f"tgt={aik.ee_target[0].cpu().numpy().round(4)}", flush=True)
                    if step % 10 == 0 and errors_window:
                        wmean = sum(errors_window) / len(errors_window)
                        err_history.append(wmean)
                        print(f"[null-device] step {step:3d}: mean EE-target error "
                              f"{wmean * 100:6.2f} cm   reward(sum)={float(rew.sum()):8.3f}",
                              flush=True)
                        errors_window = []

            if device.finished:
                break
    except KeyboardInterrupt:
        print("\n[teleop] interrupted - restoring home pose", flush=True)
    finally:
        try:  # graceful exit: hold home pose briefly, then close
            act = torch.zeros(env.num_envs, 6 * n_arms, dtype=torch.float32)
            for slot, name in enumerate(arm_names_all):
                act[0, slot * 6:(slot + 1) * 6] = controllers[name].home_row
            act = act.to(env.device)
            for _ in range(50):
                env.step(act)
        except Exception as e:
            print(f"[teleop] home-restore skipped ({e})", flush=True)
        env.close()
        device.close()

    if args.null_device:
        ok_track = (len(err_history) >= 2 and err_history[-1] < err_history[0]
                    and err_history[-1] < 0.03)
        print(f"[null-device] final tracking-window mean error "
              f"{err_history[-1] * 100 if err_history else float('nan'):.2f} cm, "
              f"all-finite={all_finite}", flush=True)

        mp = markers.last_pos
        mk_ok = (mp is not None and markers.n_updates >= 10 and bool(torch.isfinite(mp).all()))
        moved = 0.0
        if markers.first_pos is not None and mp is not None:
            moved = float((mp - markers.first_pos).abs().max())
        mk_moving = moved > 1e-3
        print(f"[mock-test] {'PASS' if mk_ok and mk_moving else 'FAIL'} marker_positions_written: "
              f"{markers.n_updates} visualize() writes, last={mp.tolist() if mp is not None else None}, "
              f"max drift {moved * 1000:.1f} mm (incl. homing)", flush=True)

        expected_keys = {"circle_closes_gripper", "cross_opens_gripper",
                         "home_ramp_reaches_home_then_clears", "home_recentre_at_home_ee",
                         "right_stick_x_dead", "marker_tracks_during_homing"}
        missing = expected_keys - set(mock_results)
        for k in sorted(missing):
            mrecord(k, False, "never ran")

        ok = (ok_track and all_finite and mk_ok and mk_moving
              and all(v[0] for v in mock_results.values()))
        print(f"[null-device/mock] -> {'OK' if ok else 'FAIL'} "
              f"({sum(v[0] for v in mock_results.values())}/{len(mock_results)} mock checks passed)",
              flush=True)
        return 0 if ok else 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="DS4 gamepad Cartesian teleop for SO-101 Isaac Lab tasks")
    parser.add_argument("--task", default="SO101-CylReach-Single-v0", choices=sorted(TASKS))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--arm", choices=["left", "right", "both"], default="left",
                        help="which arm(s) the sticks drive (dual-arm tasks)")
    parser.add_argument("--device", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--pos-sensitivity", "--mouse-sensitivity", dest="pos_sensitivity",
                        type=float, default=1.0, help="linear velocity multiplier")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0,
                        help="(deprecated — gripper rate is fixed; kept for compatibility)")
    parser.add_argument("--null-device", action="store_true",
                        help="headless dry-run: scripted circle + mock-button suite, no pygame")
    parser.add_argument("--steps", type=int, default=None,
                        help="step limit (default: full null-device suite, unlimited live)")
    parser.add_argument("--circle-radius", type=float, default=0.03,
                        help="null-device circle radius [m]")
    parser.add_argument("--headless", action="store_true",
                        help="run Kit headless (default opens a window for demos)")
    parser.add_argument("--debug", action="store_true",
                        help="print IK internals for the first few steps")
    return parser.parse_args()


def main():
    args = parse_args()

    # AppLauncher FIRST, before any isaaclab/envs imports (test.md landmine).
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = launcher.app

    exit_code = 0
    try:
        exit_code = run(args, simulation_app)
    except BaseException:
        import traceback

        print("[FATAL] traceback follows", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
