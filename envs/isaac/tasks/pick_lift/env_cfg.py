"""
SO-101 single-arm pick-lift task (Isaac Lab port of
`envs/mujoco/so101_single_arm_pick_lift/`).

Task: grasp the cube and lift it ``lift_threshold`` above its initial height.
Reward: staged reach→grasp→lift dense terms + dominant per-step alive penalty
+ sparse success bonus. Termination: success (grasp tolerance, height gain >
threshold) or timeout.

Single-arm registration: ``SO101-PickLift-Single-v0``.
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
from envs.isaac.so101 import CUBE_INIT_POS, CUBE_MASS, CUBE_SIZE
from envs.isaac.so101 import gripper_frame_prim
from envs.isaac.tasks import common, mdp

LIFT_THRESHOLD = 0.05  # metres above init height to count as lifted
GRASP_DIST_THRESHOLD = 0.035


##
# Scene
##


@configclass
class PickLiftSceneCfg(InteractiveSceneCfg):
    """Single SO-101 arm on the table with a cube."""

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
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
        use_default_offset=False,  # absolute radian targets — parity with MuJoCo envs
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """State-based policy observations (cameras render separately)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.rewards.object_position_in_robot_root_frame)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    randomize_object_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (CUBE_MASS, 0.35),  # ~1x-4x nominal, per MJLAB_INTEGRATION.md
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


@configclass
class RewardsCfg:
    reach_object = RewTerm(
        func=mdp.rewards.ee_object_distance, params={"std": 0.1}, weight=1.0
    )
    grasp_bonus = RewTerm(
        func=mdp.rewards.grasp_bonus, params={"threshold": GRASP_DIST_THRESHOLD}, weight=2.5
    )
    lift_progress = RewTerm(
        func=mdp.rewards.lift_progress, params={"lift_threshold": LIFT_THRESHOLD}, weight=1.5
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    alive = RewTerm(func=base_mdp.is_alive, weight=-0.05)  # dominant over action_rate
    success_bonus = RewTerm(
        func=mdp.terminations.pick_lift_success,
        params={"lift_threshold": LIFT_THRESHOLD},
        weight=10.0,
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.terminations.pick_lift_success, params={"lift_threshold": LIFT_THRESHOLD}
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class PickLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickLiftSceneCfg = PickLiftSceneCfg(num_envs=4096, env_spacing=2.5)
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

        self.scene.robot = so101.so101_articulation_cfg("{ENV_REGEX_NS}/Robot")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_INIT_POS),
            spawn=sim_utils.CuboidCfg(
                size=CUBE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
            ),
        )
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path=gripper_frame_prim(), name="ee")
            ],
        )
        # Cameras — all poses customizable (defaults in tasks/common.py).
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
class PickLiftEnvCfg_PLAY(PickLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
