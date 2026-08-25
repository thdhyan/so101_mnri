"""
SO-101 dual-arm cooperative cylinder grasp-and-lift task (Isaac Lab port of
`envs/mujoco/so101_dual_arm_cylinder_grasp/`).

Task: the left arm grasps one end of a thin free cylinder, the right arm the
opposite end; both arms then lift it together above ``lift_threshold``.
Termination: success (both ends grasped within tolerance AND lifted) or
timeout.

Dual-arm registration: ``SO101-CylGrasp-Dual-v0``. The two follower bases sit
18 in (0.4572 m) apart in Y with parallel Z and X axes — the real rig.
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
    LEFT_BASE_POS,
    RIGHT_BASE_POS,
)
from envs.isaac.so101 import gripper_frame_prim
from envs.isaac.tasks import common, mdp

LIFT_THRESHOLD = 0.05
GRASP_DIST_THRESHOLD = 0.035


##
# Scene
##


@configclass
class CylGraspSceneCfg(InteractiveSceneCfg):
    """Two SO-101 arms (18 in apart, parallel axes) with a free thin cylinder."""

    robot_left: ArticulationCfg = MISSING
    robot_right: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING  # the thin cylinder
    ee_frame_left: FrameTransformerCfg = MISSING
    ee_frame_right: FrameTransformerCfg = MISSING
    wrist_cam_left: CameraCfg = MISSING
    wrist_cam_right: CameraCfg = MISSING
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
    """(12,) = left 6 joint targets + right 6 joint targets, radians (absolute)."""

    left_arm_action: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot_left",
        joint_names=so101.JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    right_arm_action: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot_right",
        joint_names=so101.JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_left")}
        )
        left_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_left")}
        )
        right_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_right")}
        )
        right_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_right")}
        )
        cylinder_pose = ObsTerm(
            func=mdp.observations.object_pose_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_left")},
        )
        cylinder_ends = ObsTerm(
            func=mdp.observations.cylinder_ends_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_left")},
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
        params={
            # x noise only — the cylinder must stay centred between the arms
            "pose_range": {"x": (-0.04, 0.04), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    randomize_cylinder_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (CYLINDER_MASS, 0.25),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


@configclass
class RewardsCfg:
    reach_left = RewTerm(
        func=mdp.rewards.cyl_end_ee_distance,
        params={"std": 0.1, "end": "a", "ee_frame_cfg": SceneEntityCfg("ee_frame_left")},
        weight=1.0,
    )
    reach_right = RewTerm(
        func=mdp.rewards.cyl_end_ee_distance,
        params={"std": 0.1, "end": "b", "ee_frame_cfg": SceneEntityCfg("ee_frame_right")},
        weight=1.0,
    )
    grasp_left = RewTerm(
        func=mdp.rewards.cyl_grasp_bonus,
        params={"threshold": GRASP_DIST_THRESHOLD, "end": "a", "ee_frame_cfg": SceneEntityCfg("ee_frame_left")},
        weight=2.5,
    )
    grasp_right = RewTerm(
        func=mdp.rewards.cyl_grasp_bonus,
        params={"threshold": GRASP_DIST_THRESHOLD, "end": "b", "ee_frame_cfg": SceneEntityCfg("ee_frame_right")},
        weight=2.5,
    )
    lift_progress = RewTerm(
        func=mdp.rewards.cyl_lift_progress,
        params={"lift_threshold": LIFT_THRESHOLD},
        weight=1.5,
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    alive = RewTerm(func=base_mdp.is_alive, weight=-0.05)
    success_bonus = RewTerm(
        func=mdp.terminations.cyl_grasp_success,
        params={"lift_threshold": LIFT_THRESHOLD},
        weight=10.0,
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.terminations.cyl_grasp_success, params={"lift_threshold": LIFT_THRESHOLD}
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class CylGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: CylGraspSceneCfg = CylGraspSceneCfg(num_envs=2048, env_spacing=3.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0  # 1000 steps @ 50 Hz control
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

        # Two identical arms — bases 18 in apart in Y, parallel Z/X axes.
        self.scene.robot_left = so101.so101_articulation_cfg(
            "{ENV_REGEX_NS}/LeftArm", init_pos=LEFT_BASE_POS
        )
        self.scene.robot_right = so101.so101_articulation_cfg(
            "{ENV_REGEX_NS}/RightArm", init_pos=RIGHT_BASE_POS
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_INIT_POS),
            spawn=sim_utils.CapsuleCfg(
                radius=CYLINDER_RADIUS,
                height=CYLINDER_HEIGHT,
                axis="Y",  # thin cylinder lying across, ends facing each arm
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=CYLINDER_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.2)),
            ),
        )
        self.scene.ee_frame_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/LeftArm/Geometry/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=gripper_frame_prim("{ENV_REGEX_NS}/LeftArm"),
                    name="ee_left",
                )
            ],
        )
        self.scene.ee_frame_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RightArm/Geometry/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=gripper_frame_prim("{ENV_REGEX_NS}/RightArm"),
                    name="ee_right",
                )
            ],
        )
        # Wrist cameras on each follower's gripper + global workspace cameras.
        left_tcp = gripper_frame_prim("{ENV_REGEX_NS}/LeftArm")
        right_tcp = gripper_frame_prim("{ENV_REGEX_NS}/RightArm")
        self.scene.wrist_cam_left = common.camera_cfg(
            left_tcp, pos=common.WRIST_CAM_POS, rot_wxyz=common.WRIST_CAM_ROT_LEFT
        )
        self.scene.wrist_cam_right = common.camera_cfg(
            right_tcp, pos=common.WRIST_CAM_POS, rot_wxyz=common.WRIST_CAM_ROT_RIGHT
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
class CylGraspEnvCfg_PLAY(CylGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
