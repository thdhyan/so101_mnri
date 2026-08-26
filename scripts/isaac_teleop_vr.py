#!/usr/bin/env python
"""VR (Meta Quest 3) teleop for the SO-101 Isaac Lab tasks via isaacteleop + CloudXR.

Wired path (see VR_TELEOP_SETUP.md for the full setup):

    Quest 3 browser (WebXR CloudXR client)
      | WebRTC stream + controller poses
      v
    isaacteleop CloudXR runtime (Monado, bundled in the wheel) + WSS proxy :48322
      v
    Isaac Sim Kit OpenXR session (--xr; auto-starts headless)
      v  isaacsim.kit.xr.teleop.bridge handles
    TeleopSession -> retargeting pipeline (everything shipped in isaacteleop):
      ControllersSource(right) -> ControllerTransform(world_T_anchor = base_T_world)
        -> SO101ClutchRetargeter  : absolute EE pose [xyz quat_xyzw], robot base frame
        -> SO101GripperRetargeter : analog jaw closedness c in [0, 1]
      -> TensorReorderer       : flat 8D action [x y z qx qy qz qw c]
      v
    DifferentialIKController (pose, absolute, dls) -> 5 arm joint targets
    closedness affine -> gripper joint target   =>  6D absolute JointPositionAction

Right-controller controls:
    pose            EE target (clutched delta around the engage origin)
    trigger         gripper closedness (analog)
    grip (squeeze)  re-center the clutch origin (arm stays put)
    A button        toggle XR anchor rotation (built into IsaacTeleopDevice)

Usage:
    # dry run: scripted controller trajectory through the real retargeting+IK
    # pipeline; headless GPU; no XR runtime / CloudXR / headset required
    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device

    # VR demo day (see VR_TELEOP_SETUP.md)
    python -m isaacteleop.cloudxr --host-client       # terminal 1: runtime + web client
    python scripts/isaac_teleop_vr.py --headless --cloudxr external   # terminal 2: Isaac
"""

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "SO101-CylReach-Single-v0"
ANCHOR_LEAF = "world_T_anchor"
BASE_PRIM_SUFFIX = "/Robot/Geometry/base_link"
EE_BODY_SUFFIX = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN_RAD = -0.174533  # URDF lower limit
GRIPPER_CLOSE_RAD = 1.74533   # URDF upper limit


def _fail(msg: str) -> None:
    print(f"\n[isaac_teleop_vr] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help=f"registered task id (default {DEFAULT_TASK}; single-arm only)")
    parser.add_argument("--null-device", action="store_true",
                        help="dry run: scripted controller trajectory through the real "
                        "retargeting+IK pipeline, headless, no XR runtime needed")
    parser.add_argument("--steps", type=int, default=100,
                        help="control steps for --null-device (default 100)")
    parser.add_argument("--headless", action="store_true",
                        help="no local Isaac window (the Quest still gets its own stream)")
    parser.add_argument("--cloudxr", choices=["auto", "external", "off"], default="auto",
                        help="auto: launch runtime+WSS from this script (default); external: "
                        "runtime started separately (python -m isaacteleop.cloudxr ...); "
                        "off: no CloudXR (drive Kit's XR panel manually)")
    parser.add_argument("--anchor-pos", type=float, nargs=3, default=(0.35, 0.0, 0.95),
                        metavar=("X", "Y", "Z"),
                        help="world point that appears at the headset origin (default in front of the arm)")
    parser.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RAD)
    parser.add_argument("--gripper-close", type=float, default=GRIPPER_CLOSE_RAD)
    parser.add_argument("--ik-lambda", type=float, default=0.05,
                        help="damped-least-squares damping (default 0.05)")
    return parser.parse_args()


##
# isaacteleop lazy imports (kept out of module scope so --help works without them)
##


def _import_isaacteleop_bits() -> dict:
    try:
        import numpy as np
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import (
            ComputeContext,
            ExecutionEvents,
            ExecutionState,
            OutputCombiner,
            ValueInput,
        )
        from isaacteleop.retargeting_engine.interface.tensor_group import TensorGroup
        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
        from isaacteleop.retargeting_engine.tensor_types import (
            ControllerInput,
            ControllerInputIndex,
            TransformMatrix,
        )
        from isaacteleop.retargeting_engine.utilities.controller_transform import ControllerTransform
        from isaacteleop.retargeters import (
            SO101ClutchRetargeter,
            SO101GripperRetargeter,
            TensorReorderer,
        )
    except ModuleNotFoundError as e:
        _fail(
            f"isaacteleop component missing ({e}). Setup steps:\n"
            "  1. source .venv/bin/activate\n"
            "  2. pip install 'isaacteleop[retargeters-lite]' "
            "--extra-index-url https://pypi.nvidia.com\n"
            "     (already installed in this repo's venv — check you are using it)\n"
            "  3. if SO101ClutchRetargeter is missing, upgrade: pip install -U isaacteleop"
        )
    return dict(
        np=np,
        ControllersSource=ControllersSource,
        ComputeContext=ComputeContext,
        ExecutionEvents=ExecutionEvents,
        ExecutionState=ExecutionState,
        OutputCombiner=OutputCombiner,
        ValueInput=ValueInput,
        TensorGroup=TensorGroup,
        OptionalType=OptionalType,
        ControllerInput=ControllerInput,
        ControllerInputIndex=ControllerInputIndex,
        TransformMatrix=TransformMatrix,
        ControllerTransform=ControllerTransform,
        SO101ClutchRetargeter=SO101ClutchRetargeter,
        SO101GripperRetargeter=SO101GripperRetargeter,
        TensorReorderer=TensorReorderer,
    )


@dataclass
class PipelineBundle:
    combiner: object          # OutputCombiner with an "action" output (flat 8D)
    clutch: object            # SO101ClutchRetargeter instance (re-center handle)
    ctrl_leaf_name: str       # script-mode controller leaf name (None in xr mode)
    source_mode: str          # "xr" | "script"


def build_so101_pipeline(bits: dict, home_base_T_ee, source_mode: str) -> PipelineBundle:
    """Build the SO-101 retargeting graph shared by the VR and null-device paths.

    source_mode="xr" uses a real ControllersSource (TeleopSession polls the
    DeviceIO tracker);     source_mode="script" swaps in a ValueInput leaf fed by
    --null-device. Downstream nodes are identical.
    """
    if source_mode == "xr":
        controllers = bits["ControllersSource"](name="controllers")
        right_src = controllers.output("controller_right")
        ctrl_leaf = None
    elif source_mode == "script":
        ctrl_leaf = "scripted_controllers"
        leaf = bits["ValueInput"](ctrl_leaf, bits["OptionalType"](bits["ControllerInput"]()))
        right_src = leaf.output("value")
    else:
        raise ValueError(f"unknown source_mode {source_mode!r}")

    anchor_in = bits["ValueInput"](ANCHOR_LEAF, bits["TransformMatrix"]())
    xform = bits["ControllerTransform"]("controller_xform")
    transformed = xform.connect({
        "controller_right": right_src,
        "transform": anchor_in.output("value"),
    })

    clutch = bits["SO101ClutchRetargeter"](
        name="so101_ee",
        input_device="controller_right",
        home_base_T_ee=home_base_T_ee,
    )
    ee_graph = clutch.connect({"controller_right": transformed.output("controller_right")})

    gripper = bits["SO101GripperRetargeter"](name="so101_gripper", input_device="controller_right")
    grip_graph = gripper.connect({"controller_right": transformed.output("controller_right")})

    reorderer = bits["TensorReorderer"](
        input_config={
            "ee_pose": ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"],
            "gripper_command": ["gripper_closedness"],
        },
        output_order=["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw", "gripper_closedness"],
        name="so101_action_flat",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    flat = reorderer.connect({
        "ee_pose": ee_graph.output("ee_pose"),
        "gripper_command": grip_graph.output("gripper_command"),
    })
    combiner = bits["OutputCombiner"]({"action": flat.output("output")})
    return PipelineBundle(combiner=combiner, clutch=clutch, ctrl_leaf_name=ctrl_leaf, source_mode=source_mode)


def make_controller_tensorgroup(bits: dict, pos, quat_xyzw, trigger: float, valid: bool = True):
    """One fully-populated ControllerInput TensorGroup (null-device feeding)."""
    np = bits["np"]
    tg = bits["TensorGroup"](bits["ControllerInput"]())
    zero3, zero4 = np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_POSITION)] = np.asarray(pos, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_ORIENTATION)] = np.asarray(quat_xyzw, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_IS_VALID)] = bool(valid)
    tg[int(bits["ControllerInputIndex"].AIM_POSITION)] = zero3
    tg[int(bits["ControllerInputIndex"].AIM_ORIENTATION)] = zero4
    tg[int(bits["ControllerInputIndex"].AIM_IS_VALID)] = False
    for field in ("PRIMARY_CLICK", "SECONDARY_CLICK", "THUMBSTICK_X", "THUMBSTICK_Y",
                  "THUMBSTICK_CLICK", "MENU_CLICK", "SQUEEZE_VALUE"):
        tg[int(getattr(bits["ControllerInputIndex"], field))] = 0.0
    tg[int(bits["ControllerInputIndex"].TRIGGER_VALUE)] = float(trigger)
    return tg


def reclutch(bundle: PipelineBundle) -> None:
    """Re-center the clutch at the current controller pose (arm does not move).

    Clearing the latched origin makes the next RUNNING frame re-capture p0 while
    keeping the running home at the last commanded EE pose.
    """
    if getattr(bundle.clutch, "_origin", None) is not None:
        bundle.clutch._origin = None


def step_pipeline_script(bits: dict, bundle: PipelineBundle, base_T_world, ctrl_tg):
    """Execute the graph one step directly; returns the flat 8D action (numpy)."""
    np = bits["np"]
    xform_tg = bits["TensorGroup"](bits["TransformMatrix"]())
    xform_tg[0] = np.asarray(base_T_world, dtype=np.float32)
    ctx = bits["ComputeContext"](
        execution_events=bits["ExecutionEvents"](
            reset=False, execution_state=bits["ExecutionState"].RUNNING
        )
    )
    outputs = bundle.combiner.execute_pipeline(
        {bundle.ctrl_leaf_name: {"value": ctrl_tg}, ANCHOR_LEAF: {"value": xform_tg}},
        context=ctx,
    )
    return np.from_dlpack(outputs["action"][0]).astype(np.float32)


##
# Environment / IK driver
##


def make_env(task_id: str, num_envs: int = 1):
    import gymnasium as gym

    import envs.isaac  # noqa: F401  (task registration side effects)
    from isaaclab.envs import ManagerBasedRLEnv

    entry = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    mod_name, _, cls_name = entry.partition(":")
    import importlib

    cfg = getattr(importlib.import_module(mod_name), cls_name)()
    cfg.scene.num_envs = num_envs
    # XR rendering conflicts with extra camera sensors (isaaclab_teleop docs)
    for cam in ("wrist_cam", "overhead_cam", "front_cam"):
        if hasattr(cfg.scene, cam):
            setattr(cfg.scene, cam, None)
    return ManagerBasedRLEnv(cfg=cfg)


class So101IKDriver:
    """8D retargeter output -> DifferentialIK -> 6D absolute joint-position action."""

    def __init__(self, env, gripper_open: float, gripper_close: float, ik_lambda: float):
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

        self.torch = torch
        self.subtract_frame_transforms = subtract_frame_transforms
        self.matrix_from_quat = matrix_from_quat
        self.quat_inv = quat_inv

        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.data.body_names)
        joint_names = list(robot.data.joint_names)

        def find(names, suffix, what):
            hits = [i for i, n in enumerate(names) if n.endswith(suffix)]
            if len(hits) != 1:
                raise KeyError(f"expected exactly one {what} ending '{suffix}', got {hits} in {names}")
            return hits[0]

        self.robot = robot
        self.ee_body_idx = find(body_names, EE_BODY_SUFFIX, "body")
        self.base_body_idx = find(body_names, "base_link", "body")
        self.ee_jacobi_idx = self.ee_body_idx - 1  # fixed-base jacobian rows drop the root body
        self.arm_joint_ids = [find(joint_names, j, "joint") for j in ARM_JOINTS]
        self.gripper_joint_id = find(joint_names, "gripper", "joint")

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": ik_lambda},
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close

    def reset(self):
        self.ik.reset()

    def base_T_world(self):
        """(4, 4) base_link_T_world float32 — fed to the pipeline as world_T_anchor."""
        pose = self.robot.data.body_pose_w.torch[:, self.base_body_idx]
        t = self.torch.eye(4, device=self.robot.device)
        t[:3, :3] = self.matrix_from_quat(pose[:, 3:7])[0]
        t[:3, 3] = pose[0, :3]
        return t.detach().cpu().numpy().astype("float32")

    def home_base_T_ee(self):
        """(4, 4) base_link_T_ee float64 at the current (reset) pose — clutch seed."""
        pose = self.robot.data.body_pose_w.torch
        ee_w, base_w = pose[:, self.ee_body_idx], pose[:, self.base_body_idx]
        pos_b, quat_b = self.subtract_frame_transforms(
            base_w[:, 0:3], base_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )
        t = self.torch.eye(4, device=self.robot.device, dtype=self.torch.float64)
        t[:3, :3] = self.matrix_from_quat(quat_b)[0].to(self.torch.float64)
        t[:3, 3] = pos_b[0].to(self.torch.float64)
        return t.detach().cpu().numpy()

    def step(self, action8):
        """action8: (8,) [x y z qx qy qz qw closedness] expressed in base frame."""
        torch = self.torch
        cmd = torch.as_tensor(action8, device=self.robot.device, dtype=torch.float32).unsqueeze(0)

        pose = self.robot.data.body_pose_w.torch
        ee_w, base_w = pose[:, self.ee_body_idx], pose[:, self.base_body_idx]
        ee_pos_b, ee_quat_b = self.subtract_frame_transforms(
            base_w[:, 0:3], base_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )

        jacobian = self.robot.data.body_link_jacobian_w.torch[
            :, self.ee_jacobi_idx, :, self.arm_joint_ids
        ]
        rot_to_base = self.matrix_from_quat(self.quat_inv(base_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(rot_to_base, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(rot_to_base, jacobian[:, 3:, :])

        self.ik.set_command(cmd[:, 0:7])
        joint_pos_arm = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        arm_targets = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)
        # keep DLS targets inside physical limits (5-DOF arm cannot hit every
        # orientation; unclamped wind-up would chatter against the limit stops)
        limits = self.robot.data.joint_pos_limits.torch[:, self.arm_joint_ids]
        arm_targets = arm_targets.clamp(limits[..., 0], limits[..., 1])

        closedness = float(cmd[0, 7])
        gripper_target = self.gripper_open + closedness * (self.gripper_close - self.gripper_open)
        action = joint_pos_arm.new_zeros(self.robot.num_instances, len(self.arm_joint_ids) + 1)
        action[:, :-1] = arm_targets
        action[:, -1] = gripper_target
        return action


def build_home_and_bundle(bits: dict, driver: So101IKDriver, source_mode: str) -> PipelineBundle:
    home = driver.home_base_T_ee()
    print(f"[setup] clutch home EE (base frame): {[round(v, 4) for v in home[:3, 3]]}")
    return build_so101_pipeline(bits, home_base_T_ee=home, source_mode=source_mode)


##
# --null-device dry run (headless GPU, no XR anything)
##


def run_null_device(args) -> int:
    import math

    bits = _import_isaacteleop_bits()  # fail fast, before Kit boots
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    rc = 1
    try:
        import numpy as np
        import torch

        print(f"[null-device] creating {args.task} (num_envs=1, cameras stripped) ...")
        env = make_env(args.task)
        env.reset()
        driver = So101IKDriver(env, args.gripper_open, args.gripper_close, args.ik_lambda)
        bundle = build_home_and_bundle(bits, driver, source_mode="script")
        cx, cy, cz = driver.home_base_T_ee()[:3, 3].tolist()
        r = 0.06
        ident = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        finite = True
        reclutch_step = max(args.steps // 2, 2)
        last = None
        for t in range(args.steps):
            frac = t / max(args.steps - 1, 1)
            phase = 2.0 * math.pi * 2.0 * frac
            pos = [
                cx + r * math.sin(phase),
                cy + r * (math.cos(phase) - 1.0),
                cz + 0.04 * math.sin(2.0 * phase),
            ]
            trigger = abs(math.sin(math.pi * frac))
            ctrl = make_controller_tensorgroup(bits, pos, ident, trigger, valid=True)
            action8 = step_pipeline_script(bits, bundle, driver.base_T_world(), ctrl)
            finite &= bool(np.isfinite(action8).all())
            joint_action = driver.step(torch.as_tensor(action8))
            finite &= bool(torch.isfinite(joint_action).all())
            env.step(joint_action)
            if t == reclutch_step:
                reclutch(bundle)
                print(f"[null-device] step {t}: squeeze -> clutch origin re-latched (arm held)")
            last = (action8, joint_action)

        a8, ja = last
        print(f"[null-device] final ee target  : {[round(float(v), 4) for v in a8[:7]]}")
        print(f"[null-device] final closedness : {float(a8[7]):.4f}")
        print(f"[null-device] final joint tgt  : {ja[0].detach().cpu().numpy().round(4).tolist()}")
        ok = finite and abs(float(a8[7])) < 1e-5
        print("[null-device] PASS" if ok else "[null-device] FAIL (non-finite values or bad gripper ramp)")
        env.close()
        rc = 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _fail(str(e))
    finally:
        simulation_app.close()
    return rc


##
# Live VR loop
##


def run_vr(args) -> int:
    if args.cloudxr != "off":
        # Pick up the resolved CloudXR env (XR_RUNTIME_JSON etc.) written by the
        # runtime launcher BEFORE Kit boots, so Kit's OpenXR loader finds the
        # bundled Monado/CloudXR runtime immediately.
        env_file = Path.home() / ".cloudxr" / "run" / "cloudxr.env"
        if env_file.is_file():
            import os

            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("export ") and "=" in line:
                    k, _, v = line[len("export "):].partition("=")
                    val = shlex.split(v)
                    os.environ[k.strip()] = val[0] if val else ""

    bits = _import_isaacteleop_bits()  # fail fast, before Kit boots
    from isaaclab.app import AppLauncher

    # In headless mode, suppress XR display rendering to avoid GPU OOM from
    # the 4096×3584 per-eye swapchain textures. Controller tracking still works.
    if args.headless:
        import os as _os
        _os.environ["OMNI_KIT_HEADLESS"] = "1"

    launcher = AppLauncher(headless=args.headless, xr=True)
    simulation_app = launcher.app

    # Disable XR display pipeline after Kit boots — this frees the GPU memory
    # that would be used for stereo swapchain textures, while keeping the
    # OpenXR session + controller tracking alive for isaacteleop.
    if args.headless:
        try:
            import carb.settings
            s = carb.settings.get_settings()
            # Disable RTX rendering entirely (not needed for state-based teleop)
            s.set("/app/renderer/enabled", False)
            # Disable XR display composition (tracking still works)
            s.set("/xr/profile/display/enabled", False)
            # Reduce render resolution to minimum (fallback if display isn't fully disabled)
            s.set("/app/renderer/resolution/width", 256)
            s.set("/app/renderer/resolution/height", 256)
            print("[vr] headless: disabled XR display pipeline (tracking-only mode)", flush=True)
        except Exception as e:
            print(f"[vr] warning: could not disable XR display ({e})", flush=True)

    # isaaclab_teleop depends on 'carb' (Kit SDK) which is only available
    # AFTER AppLauncher boots Kit — must import here, not before.
    try:
        from isaaclab_teleop import IsaacTeleopCfg
        from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_JS_ENV
        from isaaclab_teleop.isaac_teleop_device import create_isaac_teleop_device
        from isaaclab_teleop.xr_cfg import XrCfg
    except ModuleNotFoundError as e:
        _fail(
            f"isaaclab_teleop unavailable ({e}). Setup steps:\n"
            "  pip install 'isaaclab-teleop' --extra-index-url https://pypi.nvidia.com\n"
            "  (it ships inside the isaaclab 3.0.0b2 wheel already present in this venv)"
        )
    rc = 1
    try:
        import torch

        print(f"[vr] creating {args.task} ...")
        env = make_env(args.task)
        env.reset()
        driver = So101IKDriver(env, args.gripper_open, args.gripper_close, args.ik_lambda)
        bundle = build_home_and_bundle(bits, driver, source_mode="xr")

        base_prim = "/World/envs/env_0" + BASE_PRIM_SUFFIX
        teleop_cfg = IsaacTeleopCfg(
            xr_cfg=XrCfg(anchor_pos=tuple(args.anchor_pos)),
            pipeline_builder=lambda: bundle.combiner,
            target_frame_prim_path=base_prim,
            app_name="SO101VRTeleop",
        )

        def on_reset():
            env.reset()
            driver.reset()

        if args.cloudxr == "off":
            cloudxr_env_file, auto_launch = None, False
        else:
            cloudxr_env_file, auto_launch = CLOUDXR_JS_ENV, args.cloudxr == "auto"
            if args.cloudxr == "external":
                import os

                os.environ.setdefault("ISAACLAB_CXR_SKIP_AUTOLAUNCH", "1")

        device = create_isaac_teleop_device(
            teleop_cfg,
            sim_device=str(env.device),
            callbacks={"RESET": on_reset},
            cloudxr_env_file=cloudxr_env_file,
            auto_launch_cloudxr=auto_launch,
        )
        CII = bits["ControllerInputIndex"]

        print(
            "\n[vr] waiting for the Quest ...\n"
            "  1. headset browser -> WebXR client (see VR_TELEOP_SETUP.md for the URL)\n"
            f"  2. server IP = this laptop's LAN IP, port 48322\n"
            "  3. accept the self-signed cert, Enter VR, then press Play/Start\n"
            "  (advance()==None means still waiting; tensors flow once connected)\n"
        )

        with device:
            squeeze_prev = False
            while simulation_app.is_running():
                action8 = device.advance()
                if action8 is None:
                    # Still waiting for the OpenXR session / client: keep the Kit
                    # event loop ticking so the XR session can come up.
                    simulation_app.update()
                    continue
                right = device._session_lifecycle.last_right_controller
                squeeze_now = bool(
                    right is not None
                    and not right.is_none
                    and float(right[int(CII.SQUEEZE_VALUE)]) > 0.5
                )
                if squeeze_now and not squeeze_prev:
                    reclutch(bundle)
                    print("[vr] grip press -> clutch re-centered")
                squeeze_prev = squeeze_now
                env.step(driver.step(action8))
        env.close()
        rc = 0
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _fail(str(e))
    finally:
        simulation_app.close()
    return rc


def main():
    args = parse_args()
    return run_null_device(args) if args.null_device else run_vr(args)


if __name__ == "__main__":
    raise SystemExit(main())
