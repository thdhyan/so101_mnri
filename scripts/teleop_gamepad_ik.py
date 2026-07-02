#!/usr/bin/env python
"""Gamepad Cartesian teleop for SO-101 (single or dual arm) via damped-least-squares IK.

The IK / target-tracking logic lives in the `IKTeleop` class, decoupled from the
gamepad-reading loop. `IKTeleop.set_ee_target(pos, quat=None)` and
`IKTeleop.set_gripper(openness)` are the two entry points a driver needs to call
each control tick -- the gamepad loop below is one such driver. A future driver
based on mujoco-ar-viewer (https://github.com/Improbable-AI/mujoco-ar-viewer) can
call the exact same two methods instead of reading a joystick.

Usage:
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env dual --arm left
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run
"""
import argparse
import sys
import time

import numpy as np
import mujoco

from _env_utils import scene_path, joint_names, gripper_site_name, HOME_POSE, resolve_actuator_ids, resolve_joint_ids

DEADZONE = 0.15
LINEAR_SCALE = 0.15   # m/s at full stick deflection
ROLL_SCALE = 1.5       # rad/s at full stick deflection
PITCH_SCALE = 1.0      # rad/s for shoulder buttons
GRIPPER_SCALE = 1.5    # rad/s
CONTROL_HZ = 60.0


def apply_deadzone(x, dz=DEADZONE):
    if abs(x) < dz:
        return 0.0
    # rescale so output is continuous from 0 at the deadzone edge
    sign = 1.0 if x > 0 else -1.0
    return sign * (abs(x) - dz) / (1.0 - dz)


class IKTeleop:
    """Damped-least-squares Cartesian IK controller for one SO-101 arm.

    Owns a mujoco model/data pair (or a view into a shared dual-arm model),
    tracks a Cartesian EE target + gripper openness, and solves for joint
    position-actuator commands each `step()` call.
    """

    def __init__(self, model, data, joint_ids, actuator_ids, site_id,
                 home_pose=HOME_POSE, damping=1e-2, use_orientation=False):
        self.model = model
        self.data = data
        self.joint_ids = joint_ids          # qpos/qvel address indices for this arm's joints (assumes 1-dof hinge joints)
        self.actuator_ids = actuator_ids
        self.site_id = site_id
        self.home_pose = np.asarray(home_pose, dtype=float)
        self.damping = damping
        self.use_orientation = use_orientation
        self.n = len(joint_ids)

        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.jnt_range = np.array([model.jnt_range[j] for j in joint_ids])
        self.ctrl_range = np.array([model.actuator_ctrlrange[a] for a in actuator_ids])

        # gripper is last joint in the arm's chain by contract
        self._gripper_local_idx = self.n - 1

        init_pos = self.data.xpos[model.site_bodyid[site_id]] if False else None
        self.ee_target = self._current_ee_pos()
        self.ee_target_quat = None
        self.gripper_openness = 0.0  # 0=closed .. 1=open

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---- public interface (this is what an external driver, e.g. AR viewer, calls) ----
    def set_ee_target(self, pos, quat=None):
        self.ee_target = np.asarray(pos, dtype=float).copy()
        if quat is not None:
            self.ee_target_quat = np.asarray(quat, dtype=float).copy()

    def set_gripper(self, openness):
        self.gripper_openness = float(np.clip(openness, 0.0, 1.0))

    def nudge_ee_target(self, delta_pos):
        self.ee_target = self.ee_target + np.asarray(delta_pos, dtype=float)

    # ---- internals ----
    def _current_ee_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def _current_qpos(self):
        return self.data.qpos[self.qpos_adr].copy()

    def solve_step(self):
        """Run one damped-least-squares IK step and write to data.ctrl."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        # columns of J for just this arm's dofs
        Jp = self._jacp[:, self.dof_adr]  # (3, n)

        cur_pos = self._current_ee_pos()
        pos_err = self.ee_target - cur_pos

        if self.use_orientation and self.ee_target_quat is not None:
            Jr = self._jacr[:, self.dof_adr]
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, self.data.site_xmat[self.site_id])
            quat_err = np.zeros(3)
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, cur_quat)
            diff_quat = np.zeros(4)
            mujoco.mju_mulQuat(diff_quat, self.ee_target_quat, neg_cur)
            mujoco.mju_quat2Vel(quat_err, diff_quat, 1.0)
            J = np.vstack([Jp, Jr])
            err = np.concatenate([pos_err, quat_err])
        else:
            J = Jp
            err = pos_err

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        m = J.shape[0]
        lam2 = self.damping ** 2
        JJt = J @ J.T + lam2 * np.eye(m)
        dq = J.T @ np.linalg.solve(JJt, err)

        # nullspace bias toward home pose (excluding gripper dof)
        cur_q = self._current_qpos()
        q_bias = np.zeros(self.n)
        q_bias[:-1] = 0.3 * (self.home_pose[:-1] - cur_q[:-1])
        Jpinv = J.T @ np.linalg.solve(JJt, np.eye(m))
        N = np.eye(self.n) - Jpinv @ J
        dq = dq + N @ q_bias

        target_q = cur_q + dq
        # gripper is driven directly by openness, not by IK
        lo, hi = self.jnt_range[self._gripper_local_idx]
        target_q[self._gripper_local_idx] = lo + self.gripper_openness * (hi - lo)

        target_q = np.clip(target_q, self.jnt_range[:, 0], self.jnt_range[:, 1])
        ctrl = np.clip(target_q, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        self.data.ctrl[self.actuator_ids] = ctrl
        return cur_pos

    def ee_error(self):
        return float(np.linalg.norm(self.ee_target - self._current_ee_pos()))


def build_ik(env, arm):
    path = scene_path(env)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    jnames = joint_names(env, arm)
    site_name = gripper_site_name(env, arm)
    joint_ids = resolve_joint_ids(model, jnames)
    actuator_ids = resolve_actuator_ids(model, jnames)
    site_id = model.site(site_name).id

    # initialize qpos at home pose for this arm so IK starts near a sane config
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    data.qpos[qpos_adr] = HOME_POSE
    data.ctrl[actuator_ids] = HOME_POSE
    mujoco.mj_forward(model, data)

    ik = IKTeleop(model, data, joint_ids, actuator_ids, site_id)
    return model, data, ik


def run_dry_run(env, arm, steps=100):
    model, data, ik = build_ik(env, arm)
    center = ik._current_ee_pos().copy()
    radius = 0.05
    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    for i in range(steps):
        theta = 2 * np.pi * i / steps
        target = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        ik.set_ee_target(target)
        ik.set_gripper(0.5 * (1 + np.sin(theta)))
        ik.solve_step()
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    err = ik.ee_error()
    print(f"[dry-run] env={env} arm={arm} steps={steps} final EE-target error = {err*100:.3f} cm")
    if err > 0.05:
        print("[dry-run] WARNING: error exceeds 5 cm threshold")
        sys.exit(1)
    print("[dry-run] OK")


def run_gamepad(env, arm, device_index, no_viewer=False):
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick found. Connect a gamepad or use --dry-run.")
        sys.exit(1)
    joy = pygame.joystick.Joystick(device_index)
    joy.init()
    print(f"Using joystick: {joy.get_name()}")

    model, data, ik = build_ik(env, arm)

    active_arm = arm  # only meaningful for dual; toggled via button
    ik_by_arm = {arm: ik}
    if env == "dual":
        other = "right" if arm == "left" else "left"
        jnames = joint_names(env, other)
        site_name = gripper_site_name(env, other)
        joint_ids = resolve_joint_ids(model, jnames)
        actuator_ids = resolve_actuator_ids(model, jnames)
        site_id = model.site(site_name).id
        qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        data.qpos[qpos_adr] = HOME_POSE
        data.ctrl[actuator_ids] = HOME_POSE
        ik_by_arm[other] = IKTeleop(model, data, joint_ids, actuator_ids, site_id)
        mujoco.mj_forward(model, data)

    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    toggle_button_prev = False

    viewer = None
    if not no_viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        running = True
        while running:
            t0 = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            lx = apply_deadzone(joy.get_axis(0))
            ly = apply_deadzone(joy.get_axis(1))
            rx = apply_deadzone(joy.get_axis(2)) if joy.get_numaxes() > 2 else 0.0
            ry = apply_deadzone(joy.get_axis(3)) if joy.get_numaxes() > 3 else 0.0

            # triggers: axis 4/5 on many pads, else fall back to buttons
            trig_open = 0.0
            trig_close = 0.0
            if joy.get_numaxes() > 5:
                trig_close = max(0.0, apply_deadzone(joy.get_axis(4)))
                trig_open = max(0.0, apply_deadzone(joy.get_axis(5)))

            nbtn = joy.get_numbuttons()
            btn = lambda i: joy.get_button(i) if i < nbtn else 0

            # shoulder buttons (LB/RB, commonly 4/5) -> pitch (wrist_flex) rate
            pitch_rate = (btn(5) - btn(4)) * PITCH_SCALE

            # toggle active arm (dual only) on button 6 (back/select) rising edge
            if env == "dual":
                toggle_now = bool(btn(6))
                if toggle_now and not toggle_button_prev:
                    active_arm = "right" if active_arm == "left" else "left"
                    print(f"Active arm -> {active_arm}")
                toggle_button_prev = toggle_now

            cur_ik = ik_by_arm[active_arm]

            vx = -ly * LINEAR_SCALE
            vy = -lx * LINEAR_SCALE
            vz = -ry * LINEAR_SCALE
            wrist_roll_rate = rx * ROLL_SCALE
            gripper_rate = (trig_open - trig_close) * GRIPPER_SCALE

            new_target = cur_ik.ee_target + np.array([vx, vy, vz]) * dt_ctrl
            cur_ik.set_ee_target(new_target)

            new_openness = cur_ik.gripper_openness + gripper_rate * dt_ctrl
            cur_ik.set_gripper(new_openness)

            # wrist_roll / pitch driven as direct joint offsets (not through IK nullspace)
            roll_idx = 4  # wrist_roll is index 4 in JOINT_SUFFIXES
            pitch_idx = 3  # wrist_flex is index 3
            qpos_adr = cur_ik.qpos_adr
            data.qpos[qpos_adr[roll_idx]] += wrist_roll_rate * dt_ctrl
            data.qpos[qpos_adr[roll_idx]] = np.clip(
                data.qpos[qpos_adr[roll_idx]], *cur_ik.jnt_range[roll_idx])
            data.qpos[qpos_adr[pitch_idx]] += pitch_rate * dt_ctrl
            data.qpos[qpos_adr[pitch_idx]] = np.clip(
                data.qpos[qpos_adr[pitch_idx]], *cur_ik.jnt_range[pitch_idx])

            for a in ik_by_arm.values():
                a.solve_step()
            # override roll/pitch ctrl to match manually-adjusted qpos targets
            data.ctrl[cur_ik.actuator_ids[roll_idx]] = data.qpos[qpos_adr[roll_idx]]
            data.ctrl[cur_ik.actuator_ids[pitch_idx]] = data.qpos[qpos_adr[pitch_idx]]

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if viewer is not None:
                _draw_target_marker(viewer, cur_ik.ee_target)
                viewer.sync()
                if not viewer.is_running():
                    running = False

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt_ctrl - elapsed))
    finally:
        if viewer is not None:
            viewer.close()


def _draw_target_marker(viewer, pos):
    scn = viewer.user_scn
    scn.ngeom = 0
    mujoco.mjv_initGeom(
        scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0, 0],
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.2, 0.2, 0.8], dtype=float),
    )
    scn.ngeom = 1


def main():
    parser = argparse.ArgumentParser(description="Gamepad Cartesian IK teleop for SO-101")
    parser.add_argument("--env", choices=["single", "dual"], default="single")
    parser.add_argument("--arm", choices=["left", "right"], default="left",
                         help="Initial active arm (dual env only)")
    parser.add_argument("--device", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run headless IK loop with a scripted circular target, 100 steps")
    parser.add_argument("--no-viewer", action="store_true", help="Disable passive viewer window")
    args = parser.parse_args()

    arm = args.arm if args.env == "dual" else None

    if args.dry_run:
        run_dry_run(args.env, arm)
        return

    run_gamepad(args.env, arm if arm is not None else "left", args.device, no_viewer=args.no_viewer)


if __name__ == "__main__":
    main()
