"""
Shared SO-101 definitions for the Isaac Lab port.

Single source of robot truth remains `robots/so101/` (URDF). The USD in
`assets/` is generated from `robots/so101/so101.urdf` by `convert_urdf.py` —
do not hand-edit the USD.

Dual-arm tasks spawn TWO instances of the single-arm USD (the arms are
identical chains, no mirroring — same as `robots/so101/so101_dual.urdf`):

    left_arm  at (0.18, +0.2286, 0.82)   ┐ bases 18 in (0.4572 m) apart in Y,
    right_arm at (0.18, -0.2286, 0.82)   ┘ Z and X axes parallel (identity rot)
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ISAAC_DIR = Path(__file__).parent
SO101_USD = str(ISAAC_DIR / "assets" / "so101" / "so101.usda")

# Joint names — stable contract shared with the MuJoCo envs and teleop scripts.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Prim path of the TCP link inside the converted USD (kept as its own rigid
# body via merge_fixed_joints=False in convert_urdf.py). EE frames and wrist
# cameras attach here — the USD twin of the MuJoCo `gripperframe` site.
# The path is relative to the articulation spawn prim; use
# `gripper_frame_prim()` to build the absolute (env-namespaced) path.
GRIPPER_FRAME_REL = (
    "Geometry/base_link/shoulder_link/upper_arm_link/"
    "lower_arm_link/wrist_link/gripper_link/gripper_frame_link"
)


def gripper_frame_prim(spawn_prim: str = "{ENV_REGEX_NS}/Robot") -> str:
    """Absolute prim path of the TCP link for an arm spawned at ``spawn_prim``."""
    return f"{spawn_prim}/{GRIPPER_FRAME_REL}"

# Home pose, radians (matches DEFAULT_QPOS_EACH in the MuJoCo envs).
HOME_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.5,
    "elbow_flex": 0.8,
    "wrist_flex": 0.4,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# Real-rig geometry: bases 18 in apart in Y, parallel Z/X axes, on a 0.82 m table.
BASE_POS = (0.18, 0.0, 0.82)
LEFT_BASE_POS = (0.18, 0.2286, 0.82)
RIGHT_BASE_POS = (0.18, -0.2286, 0.82)

# Table (matches the MuJoCo scenes: top surface at z = 0.82).
TABLE_TOP_POS = (0.35, 0.0, 0.80)
TABLE_TOP_SIZE = (0.80, 1.10, 0.04)  # full extents; top surface lands at z = 0.82

# Manipulation objects (match the MuJoCo scenes).
CUBE_INIT_POS = (0.45, 0.0, 0.845)
CUBE_SIZE = (0.05, 0.05, 0.05)
CUBE_MASS = 0.08

CYLINDER_INIT_POS = (0.45, 0.0, 0.83)
CYLINDER_RADIUS = 0.015
CYLINDER_HEIGHT = 0.15  # cylindrical section; end sites at ±0.075 along local Y
CYLINDER_MASS = 0.05


def so101_articulation_cfg(prim_path: str, init_pos=BASE_POS) -> ArticulationCfg:
    """Articulation config for one SO-101 follower arm.

    Implicit PD actuators approximate the STS3215 position servos modelled in
    MuJoCo (kp≈998 clamped at 3.35 N·m); gains here are softened for PhysX
    stability at 200 Hz and clamped by the same effort limit.
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=SO101_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos=dict(HOME_JOINT_POS),
        ),
        actuators={
            "so101_servos": ImplicitActuatorCfg(
                joint_names_expr=JOINT_NAMES,
                effort_limit_sim=3.35,
                velocity_limit_sim=10.0,
                stiffness=100.0,
                damping=2.5,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
