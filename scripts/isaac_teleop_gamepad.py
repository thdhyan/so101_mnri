#!/usr/bin/env python
"""DualShock 4 gamepad Cartesian teleop for the SO-101 Isaac Lab tasks.

Mirrors scripts/teleop_gamepad_ik.py (MuJoCo) but drives Isaac Lab
ManagerBasedRLEnvs through the official controller stack:

    DS4Device (pygame) -> per-tick delta pose command
        -> DifferentialIKController (isaaclab.controllers, position/relative/dls)
        -> joint-position targets (clipped to soft limits)
        -> env.step(action)   [absolute joint-position actions, 6/arm]

The EE is the ``gripper_frame_link`` body (USD twin of the MuJoCo
``gripperframe`` site). Pose feedback and the geometric Jacobian come straight
from articulation data (``body_link_jacobian_w``, root/body poses), following
isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction. wrist_flex /
wrist_roll are direct joint offsets on top of the IK solution (position-only
IK, same rationale as the MuJoCo script); the gripper is an openness scalar
mapped onto its joint range — neither passes through IK.

Usage:
    # headless pipeline validation (no pygame needed):
    python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless
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

# ---- DS4 conventions copied from scripts/teleop_gamepad_ik.py --------------
DEADZONE = 0.15
LINEAR_SCALE = 0.15  # m/s at full stick deflection (x --pos-sensitivity)
ROLL_SCALE = 1.5     # rad/s at full stick deflection (x --rot-sensitivity)
PITCH_SCALE = 1.0    # rad/s via L1/R1 buttons
GRIPPER_SCALE = 1.5  # openness/s from triggers

# Raw hid-sony layout (what the MuJoCo script uses on this machine): axes
# LX,LY,RX,RY,L2,R2; buttons square,cross,circle,triangle,L1,R1,share,
# options,L3,R3,PS,... If SDL exposes the pad through the gamecontroller API,
# button indices shift (see _BUTTONS_GC); axes are the same.
AX_LX, AX_LY, AX_RX, AX_RY, AX_L2, AX_R2 = range(6)
_BUTTONS_RAW = {"l1": 4, "r1": 5, "share": 6, "options": 7, "ps": 10}
_BUTTONS_GC = {"l1": 9, "r1": 10, "share": 4, "options": 6, "ps": 5}

ROLL_LIMIT = 1.0   # rad clamp on the direct wrist_roll offset
PITCH_LIMIT = 1.0  # rad clamp on the direct wrist_flex offset


def apply_deadzone(x, dz=DEADZONE):
    if abs(x) < dz:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    return sign * (abs(x) - dz) / (1.0 - dz)


class Cmd:
    """Per-tick device command."""

    __slots__ = ("dpos", "roll_rate", "pitch_rate", "grip_rate", "rate_mode",
                 "quit", "recentre", "toggle_arm")

    def __init__(self):
        self.dpos = None           # rates (m/s, rad/s) if rate_mode else per-tick displacements
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.grip_rate = 0.0       # openness/s if rate_mode else per-tick openness delta
        self.rate_mode = True      # DS4: rates scaled by measured dt; null device: False
        self.quit = False          # edge: exit teleop
        self.recentre = False      # edge: snap EE target onto current EE pose
        self.toggle_arm = False    # edge: switch active arm (dual tasks)


class DS4Device:
    """pygame DualShock 4 reader producing per-tick delta-pose commands.

    Same mapping as scripts/teleop_gamepad_ik.py: left stick = EE x/y, right
    stick Y = EE z, right stick X = wrist roll, L1/R1 = wrist pitch, L2/R2 =
    gripper close/open. Share toggles the active arm (dual), Options
    re-centers the EE target, PS quits.
    """

    def __init__(self, index=0, pos_sensitivity=1.0, rot_sensitivity=1.0):
        import pygame

        self._pos_scale = LINEAR_SCALE * pos_sensitivity
        self._rot_scale = ROLL_SCALE * rot_sensitivity
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
        lx = apply_deadzone(joy.get_axis(AX_LX))
        ly = apply_deadzone(joy.get_axis(AX_LY))
        rx = apply_deadzone(joy.get_axis(AX_RX)) if joy.get_numaxes() > AX_RX else 0.0
        ry = apply_deadzone(joy.get_axis(AX_RY)) if joy.get_numaxes() > AX_RY else 0.0
        trig_close = max(0.0, apply_deadzone(joy.get_axis(AX_L2))) if joy.get_numaxes() > AX_L2 else 0.0
        trig_open = max(0.0, apply_deadzone(joy.get_axis(AX_R2))) if joy.get_numaxes() > AX_R2 else 0.0

        def btn(name):
            i = self._btn[name]
            return bool(joy.get_button(i)) if i < joy.get_numbuttons() else False

        cmd = Cmd()
        # MuJoCo-script sign conventions: stick up/left = negative axis value.
        cmd.dpos = [-ly * self._pos_scale, -lx * self._pos_scale, -ry * self._pos_scale]
        cmd.roll_rate = rx * self._rot_scale
        cmd.pitch_rate = (btn("r1") - btn("l1")) * PITCH_SCALE
        cmd.grip_rate = (trig_open - trig_close) * GRIPPER_SCALE
        now = {n: btn(n) for n in ("share", "options", "ps")}
        cmd.toggle_arm = now["share"] and not self._prev_btns.get("share")
        cmd.recentre = now["options"] and not self._prev_btns.get("options")
        cmd.quit = now["ps"] and not self._prev_btns.get("ps")
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
    """Scripted circular EE trajectory — proves the IK->action pipeline headless.

    Emits per-tick deltas integrating to a circle of radius ``radius`` around
    wherever the target was seeded, plus a sinusoidal gripper sweep.
    """

    def __init__(self, radius=0.03, steps_per_rev=100):
        self.radius = radius
        self.steps_per_rev = steps_per_rev
        self.i = 0
        # prev starts at ORIGIN so the first delta jumps out to T(0)=(r,0,0):
        # the run then shows genuine convergence onto the moving circle
        self._prev_t = [0.0, 0.0, 0.0]
        self._prev_grip = 0.0

    def _local_target(self, i):
        theta = 2 * math.pi * i / self.steps_per_rev
        return [self.radius * math.cos(theta), self.radius * math.sin(theta), 0.0]

    def advance(self) -> Cmd:
        t = self._local_target(self.i)
        grip = 0.5 * (1 + math.sin(2 * math.pi * self.i / self.steps_per_rev))
        cmd = Cmd()
        cmd.rate_mode = False
        cmd.dpos = [a - b for a, b in zip(t, self._prev_t)]
        cmd.grip_rate = grip - self._prev_grip
        self._prev_t = t
        self._prev_grip = grip
        self.i += 1
        return cmd

    def reseed(self):
        self.i = 0
        self._prev_t = [0.0, 0.0, 0.0]
        self._prev_grip = 0.0

    def close(self):
        pass


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
        self.roll_off = 0.0
        self.pitch_off = 0.0
        self.openness = 0.0

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
        import torch

        dpos = list(cmd.dpos)
        if mirror_y:
            dpos[1] = -dpos[1]
        if cmd.rate_mode:
            self.ee_target = self.ee_target + dt * self.ee_target.new_tensor(dpos)
            self.roll_off += cmd.roll_rate * dt
            self.pitch_off += cmd.pitch_rate * dt
            self.openness += cmd.grip_rate * dt
        else:
            self.ee_target = self.ee_target + self.ee_target.new_tensor(dpos)
            self.openness += cmd.grip_rate
        self.roll_off = max(-ROLL_LIMIT, min(ROLL_LIMIT, self.roll_off))
        self.pitch_off = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch_off))
        self.openness = max(0.0, min(1.0, self.openness))

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
        # direct-driven wrist offsets layered on the IK solution
        q_des[:, 3] = torch.clamp(q_des[:, 3] + self.pitch_off, limits[:, 3, 0], limits[:, 3, 1])
        q_des[:, 4] = torch.clamp(q_des[:, 4] + self.roll_off, limits[:, 4, 0], limits[:, 4, 1])
        gripper_lo, gripper_hi = limits[:, 5, 0], limits[:, 5, 1]
        gripper = gripper_lo + self.openness * (gripper_hi - gripper_lo)
        return torch.cat([q_des, gripper.unsqueeze(1)], dim=1)                 # (N, 6)


def print_mapping():
    rows = [
        ("Left stick (x/y)", "EE x/y velocity (0.15 m/s at full deflection)"),
        ("Right stick vertical", "EE z velocity"),
        ("Right stick horizontal", "wrist_roll rate (direct joint offset)"),
        ("L1 / R1", "wrist_flex (pitch) rate, opposite directions"),
        ("L2 / R2 (triggers)", "gripper close / open"),
        ("Share", "toggle active arm (dual-arm tasks)"),
        ("Options", "re-center EE target on current EE pose"),
        ("PS", "quit (restores home pose)"),
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

    from envs.isaac.so101 import JOINT_NAMES
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
        controllers[name] = aik

    if args.null_device:
        steps = args.steps if args.steps else 100
        device = NullDevice(radius=args.circle_radius)
        max_steps = steps
        print(f"[null-device] task={args.task} arms={active} circle r={args.circle_radius} m "
              f"for {max_steps} steps", flush=True)
    else:
        device = DS4Device(args.device, args.pos_sensitivity, args.rot_sensitivity)
        max_steps = args.steps if args.steps else 10 ** 9
        print_mapping()
        print(f"[teleop] task={args.task} active={active} — PS quits", flush=True)

    def assemble_actions(cmd, dt):
        act = torch.zeros(env.num_envs, 6 * n_arms, dtype=torch.float32)
        for slot, name in enumerate(arm_names_all):
            if name in active:
                mirror_y = (n_arms == 2 and args.arm == "both" and name == "robot_right")
                controllers[name].apply_command(cmd, mirror_y, dt)
                act[0, slot * 6:(slot + 1) * 6] = controllers[name].action_row(debug=args.debug and step < 4)[0].cpu()
            else:
                act[0, slot * 6:(slot + 1) * 6] = controllers[name].home_row
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
                print("[teleop] quit requested", flush=True)
                break
            if cmd.recentre:
                for name in active:
                    controllers[name].seed_target()
                print("[teleop] re-centered", flush=True)
            if cmd.toggle_arm and n_arms == 2 and args.arm != "both":
                other = "robot_right" if active[0] == "robot_left" else "robot_left"
                active = [other]
                print(f"[teleop] active arm -> {'right' if other.endswith('right') else 'left'}",
                      flush=True)

            actions = assemble_actions(cmd, dt)
            obs, rew, term, trunc, _extras = env.step(actions)
            step += 1

            finite = bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
            all_finite &= finite
            if not finite:
                print(f"[step {step}] NON-FINITE obs/reward", flush=True)

            if term.any() or trunc.any():
                obs, _ = env.reset()
                for aik in controllers.values():
                    aik.reset_state()
                    aik.seed_target()
                device.reseed()

            # null-device metric: ACTUAL EE position error vs the moving target
            # (the integrated ee_target traces the scripted circle; this is the
            # Isaac twin of teleop_gamepad_ik.py's ik.ee_error())
            if args.null_device:
                errs = []
                for name in active:
                    aik = controllers[name]
                    ee_pos_b, _ = aik.ee_pose_b()
                    errs.append(float(torch.linalg.norm(ee_pos_b - aik.ee_target).item()))
                errors_window.append(max(errs))
                if args.debug and step <= 15:
                    print(f"    [dbg-metric] step {step}: ee={ee_pos_b[0].cpu().numpy().round(4)} "
                          f"tgt={aik.ee_target[0].cpu().numpy().round(4)}", flush=True)
                if step % 10 == 0:
                    wmean = sum(errors_window) / len(errors_window)
                    err_history.append(wmean)
                    print(f"[null-device] step {step:3d}: mean EE-target error "
                          f"{wmean * 100:6.2f} cm   reward(sum)={float(rew.sum()):8.3f}",
                          flush=True)
                    errors_window = []
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
        ok = (len(err_history) >= 2 and err_history[-1] < err_history[0]
              and err_history[-1] < 0.03 and all_finite)
        print(f"[null-device] final window mean error {err_history[-1] * 100:.2f} cm, "
              f"all-finite={all_finite} -> {'OK' if ok else 'FAIL'}", flush=True)
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
                        help="wrist-roll/gripper rate multiplier")
    parser.add_argument("--null-device", action="store_true",
                        help="headless dry-run: scripted circular EE target, no pygame")
    parser.add_argument("--steps", type=int, default=None,
                        help="step limit (default 100 with --null-device, unlimited live)")
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
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
