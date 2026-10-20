import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_scene_to_default
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp as pick_lift_mdp
from .scene import ROBOT_XML, add_table, get_cube_spec

GRIPPER_SITE = ("gripperframe",)
GRASP_DIST_THRESHOLD = 0.01  # 1cm tolerance, tightened from env.py's 3.5cm
GRIPPER_CLOSED_THRESHOLD = 0.27  # rad, matches env.py's GRIPPER_CLOSED_THRESHOLD * 0.3
LIFT_THRESHOLD = 0.05

_robot_asset_cfg = SceneEntityCfg("robot", site_names=GRIPPER_SITE)
_gripper_joint_cfg = SceneEntityCfg("robot", joint_names=("gripper",))


def _get_robot_spec():
  return mujoco.MjSpec.from_file(ROBOT_XML)


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.18, 0.0, 0.82),
      joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.5,
        "elbow_flex": 0.8,
        "wrist_flex": 0.4,
        "wrist_roll": 0.0,
        "gripper": 0.0,
      },
    ),
    spec_fn=_get_robot_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
    ),
  )


def make_pick_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_to_object": ObservationTermCfg(
      func=pick_lift_mdp.ee_to_object,
      params={"object_name": "cube", "asset_cfg": _robot_asset_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_pos": ObservationTermCfg(
      func=pick_lift_mdp.object_pos_b,
      params={"object_name": "cube", "asset_cfg": _robot_asset_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "is_grasped": ObservationTermCfg(
      func=pick_lift_mdp.is_grasped,
      params={
        "object_name": "cube",
        "grasp_dist_threshold": GRASP_DIST_THRESHOLD,
        "gripper_joint_threshold": GRIPPER_CLOSED_THRESHOLD,
        "asset_cfg": _robot_asset_cfg,
        "gripper_joint_cfg": _gripper_joint_cfg,
      },
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
  }
  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "lift": pick_lift_mdp.LiftCommandCfg(
      entity_name="cube",
      lift_threshold=LIFT_THRESHOLD,
      nominal_pos=(0.45, 0.0, 0.83),
      resampling_time_range=(1e9, 1e9),  # resample only on episode reset
      object_pose_range=pick_lift_mdp.LiftCommandCfg.ObjectPoseRangeCfg(
        x=(-0.06, 0.06),
        y=(-0.06, 0.06),
        z=(0.0, 0.0),
      ),
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "cube_mass": EventTermCfg(
      # mass/inertia scale by e^(2*alpha); alpha=0.738 -> ~4.375x -> 0.08-0.35kg.
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube"),
        "alpha_range": (0.0, 0.738),
      },
    ),
    # No gripper-friction DR: so101_follower.xml's collision geoms are
    # unnamed (no name= attribute), so they can't be targeted by
    # dr.geom_friction's name-regex matching without editing robots/so101/
    # (forbidden — see robots/so101/README.md). Deferred per
    # MJLAB_INTEGRATION.md's "TBD, validate empirically" note.
  }

  rewards = {
    "reach_ee_object": RewardTermCfg(
      func=pick_lift_mdp.reach_ee_object,
      weight=1.0,
      params={"object_name": "cube", "asset_cfg": _robot_asset_cfg},
    ),
    "grasp_bonus": RewardTermCfg(
      func=pick_lift_mdp.grasp_bonus,
      weight=1.0,
      params={
        "object_name": "cube",
        "grasp_dist_threshold": GRASP_DIST_THRESHOLD,
        "gripper_joint_threshold": GRIPPER_CLOSED_THRESHOLD,
        "asset_cfg": _robot_asset_cfg,
        "gripper_joint_cfg": _gripper_joint_cfg,
      },
    ),
    "lift_progress": RewardTermCfg(
      func=pick_lift_mdp.lift_progress,
      weight=3.0,
      params={
        "command_name": "lift",
        "object_name": "cube",
        "grasp_dist_threshold": GRASP_DIST_THRESHOLD,
        "gripper_joint_threshold": GRIPPER_CLOSED_THRESHOLD,
        "asset_cfg": _robot_asset_cfg,
        "gripper_joint_cfg": _gripper_joint_cfg,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
    "alive_penalty": RewardTermCfg(func=pick_lift_mdp.alive_penalty, weight=-0.05),
    "success_bonus": RewardTermCfg(
      func=pick_lift_mdp.success_bonus,
      weight=10.0,
      params={
        "command_name": "lift",
        "object_name": "cube",
        "grasp_dist_threshold": GRASP_DIST_THRESHOLD,
        "gripper_joint_threshold": GRIPPER_CLOSED_THRESHOLD,
        "asset_cfg": _robot_asset_cfg,
        "gripper_joint_cfg": _gripper_joint_cfg,
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "success": TerminationTermCfg(
      func=pick_lift_mdp.success,
      params={
        "command_name": "lift",
        "object_name": "cube",
        "grasp_dist_threshold": GRASP_DIST_THRESHOLD,
        "gripper_joint_threshold": GRIPPER_CLOSED_THRESHOLD,
        "asset_cfg": _robot_asset_cfg,
        "gripper_joint_cfg": _gripper_joint_cfg,
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.5,
      entities={
        "robot": get_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
      },
      spec_fn=add_table,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=1.2,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=55,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=15,  # 0.002 * 15 = 30ms ctrl step, matches env.py's 30Hz control_freq_hz
    episode_length_s=34.0,
  )

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
