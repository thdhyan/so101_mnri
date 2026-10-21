"""
SO-101 single-arm cylinder reach task.

Task: a thin cylinder is FIXED in place in front of the arm (kinematic body,
no gravity). The gripper must reach a target point ``z_offset`` above ONE end
of the cylinder (end ``"b"``, local +Y side) — no grasping, no lifting.
Success: TCP within 1 cm of the target for ``hold_steps`` consecutive steps.

Single-arm sibling of ``tasks/cylinder_reach`` (dual-arm): same tolerances
(1 cm vs MuJoCo's 3.5 cm), same tanh-kernel reward weights, same hold-counter
success logic reduced to one arm/end pair.

Registration: ``SO101-CylReach-Single-v0``.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from envs.isaac import so101
from envs.isaac.so101 import (
    CYLINDER_HEIGHT,
    CYLINDER_INIT_POS,
    CYLINDER_MASS,
    CYLINDER_RADIUS,
)
from envs.isaac.so101 import BASE_POS, gripper_frame_prim
from envs.isaac.tasks import common, mdp

Z_OFFSET = 0.05  # target height above the cylinder end (MuJoCo env default)
POS_TOL = 0.01  # 1 cm success tolerance (matches the dual-arm task)
HOLD_STEPS = 10
REACH_END = "b"  # which cylinder end (+Y side) the gripper must reach


##
# Scene
##


@configclass
class CylReachSingleSceneCfg(InteractiveSceneCfg):
    """One SO-101 arm with a fixed (kinematic) thin cylinder in front of it."""

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING  # static cylinder (kinematic body)
    ee_frame: FrameTransformerCfg = MISSING
    wrist_cam: CameraCfg = MISSING
    overhead_cam: CameraCfg = MISSING
    front_cam: CameraCfg = MISSING

    table = common.table_cfg()
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    arm_action: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=so101.JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        cylinder_ends = ObsTerm(
            func=mdp.observations.cylinder_ends_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_cylinder_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        # Cylinder pose randomization (targets derive from the cylinder end
        # each step, so they follow automatically).
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    reach_end = RewTerm(
        func=mdp.rewards.cyl_reach_target_distance,
        params={
            "std": 0.05,
            "end": REACH_END,
            "z_offset": Z_OFFSET,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.5,
    )
    hold_bonus = RewTerm(
        func=mdp.terminations.cyl_reach_success_single,
        params={"z_offset": Z_OFFSET, "pos_tol": POS_TOL, "hold_steps": HOLD_STEPS},
        weight=2.0,
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    alive = RewTerm(func=base_mdp.is_alive, weight=-0.05)
    success_bonus = RewTerm(
        func=mdp.terminations.cyl_reach_success_single,
        params={"z_offset": Z_OFFSET, "pos_tol": POS_TOL, "hold_steps": HOLD_STEPS},
        weight=10.0,
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.terminations.cyl_reach_success_single,
        params={"z_offset": Z_OFFSET, "pos_tol": POS_TOL, "hold_steps": HOLD_STEPS},
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class CylReachSingleEnvCfg(ManagerBasedRLEnvCfg):
    scene: CylReachSingleSceneCfg = CylReachSingleSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Small-GPU-friendly PhysX buffers (8 GB laptop GPU; defaults OOM when
        # other jobs share the card).
        from isaaclab_physx.physics import PhysxCfg

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            gpu_found_lost_pairs_capacity=1024,
            gpu_found_lost_aggregate_pairs_capacity=262144,
            gpu_total_aggregate_pairs_capacity=8192,
            gpu_heap_capacity=33554432,
            gpu_temp_buffer_capacity=16777216,
        )

        self.scene.robot = so101.so101_articulation_cfg("{ENV_REGEX_NS}/Robot")
        # Kinematic body: fixed in place (no gravity effect) but still
        # teleporteable at reset for pose randomization. Mass is irrelevant
        # to this reach-only task — no mass DR here.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_INIT_POS),
            spawn=sim_utils.CapsuleCfg(
                radius=CYLINDER_RADIUS,
                height=CYLINDER_HEIGHT,
                axis="Y",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=CYLINDER_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.2)),
            ),
        )
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path=gripper_frame_prim(), name="ee")
            ],
        )
        self.scene.wrist_cam = common.camera_cfg(
            gripper_frame_prim(), pos=common.WRIST_CAM_POS, rot_wxyz=common.WRIST_CAM_ROT_SINGLE
        )
        self.scene.overhead_cam = common.camera_cfg(
            "{ENV_REGEX_NS}/overhead_cam",
            pos=common.OVERHEAD_CAM_POS,
            euler=common.OVERHEAD_CAM_EULER,
        )
        self.scene.front_cam = common.camera_cfg(
            "{ENV_REGEX_NS}/front_cam",
            pos=common.FRONT_CAM_POS,
            euler=common.FRONT_CAM_EULER,
        )


@configclass
class CylReachSingleEnvCfg_PLAY(CylReachSingleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
