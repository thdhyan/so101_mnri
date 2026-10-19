"""
SO-101 Single Arm MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Cameras:
  - wrist         : attached to gripper body
  - outside_left  : world-fixed, configurable position
  - outside_right : world-fixed, configurable position
  - overhead_cam  : world-fixed, top-down over table center
  - front_cam     : world-fixed, front view at eye height

Tasks:
  - "none"  : pure data collection, no object reward
  - "push"  : push cube to goal_pos
  - "pull"  : pull cube toward robot from far side
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSETS_DIR = Path(__file__).parent / "assets"
SCENE_XML = str(ASSETS_DIR / "scene.xml")

# Joint names in order (matches actuator order in scene.xml)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = 6

# Default joint position (relaxed upright pose)
DEFAULT_QPOS = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """
    Camera with absolute position (metres) and orientation (roll/pitch/yaw degrees).
    For world-fixed cameras: pos/euler in world frame.
    For body-attached cameras (wrist): pos/euler in body frame.
    """
    name: str
    pos: tuple = (0.0, 0.0, 1.0)        # xyz metres
    euler: tuple = (0.0, 0.0, 0.0)      # roll pitch yaw degrees (XYZ intrinsic)
    width: int = 640
    height: int = 480
    fov: float = 75.0                   # vertical FOV degrees


@dataclass
class ObjectConfig:
    """Manipulation object properties."""
    size: tuple = (0.025, 0.025, 0.025)     # box half-extents
    mass: float = 0.08
    rgba: tuple = (0.85, 0.2, 0.2, 1.0)
    # Random init range relative to table centre (0.45, 0.0, 0.83)
    init_pos_noise: tuple = (0.08, 0.08, 0.0)    # ±xyz


@dataclass
class GoalConfig:
    target_pos: tuple = (0.55, 0.15, 0.83)
    target_quat: tuple = (1.0, 0.0, 0.0, 0.0)   # wxyz
    pos_threshold: float = 0.04
    rot_threshold: float = 0.3            # radians
    visualize: bool = True


@dataclass
class SingleArmEnvConfig:
    episode_length_s: float = 25.0
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"       # "rgb_array" | "human"

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("outside_left",  pos=(0.462, -0.110, 1.273), euler=(0.5,  3.5,  91.0), fov=65.0),
        CameraConfig("outside_right", pos=(0.382, -0.304, 1.066), euler=(65.0, 1.0,  -3.0), fov=65.0),
        CameraConfig("overhead_cam",  pos=(0.35, 0.0, 1.35), euler=(0.0, 0.0, 90.0), fov=60.0),
        CameraConfig("front_cam",     pos=(0.40, -0.95, 1.05), euler=(75.0, 0.0, 0.0), fov=60.0),
    ])

    task: str = "push"                   # "push" | "pull" | "none"
    object: ObjectConfig = field(default_factory=ObjectConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """Quaternion wxyz from roll-pitch-yaw in degrees (XYZ intrinsic)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y_ = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y_, z])




def _quat_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (radians) between two unit quaternions (wxyz)."""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * math.acos(dot)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101SingleArmEnv(gym.Env):
    """
    Single SO-101 follower arm + wrist camera + 2 configurable outside cameras.

    Observations (dict):
        joint_pos          : (6,) radians
        joint_vel          : (6,) rad/s
        ee_pos             : (3,) gripper site world position
        object_pos         : (3,) [if task != "none"]
        object_quat        : (4,) wxyz [if task != "none"]
        object_vel         : (3,) [if task != "none"]
        goal_pos           : (3,) [if task != "none"]
        goal_quat          : (4,) [if task != "none"]
        images             : dict {cam_name: (H,W,3) uint8}

    Actions:
        (6,) target joint positions in radians, clipped to joint limits
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: SingleArmEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or SingleArmEnvConfig()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self._max_steps = int(self.cfg.episode_length_s * self.cfg.control_freq_hz)
        self._step_count = 0
        self._physics_steps_per_ctrl = max(1, int(
            (1.0 / self.cfg.control_freq_hz) / self.model.opt.timestep
        ))

        # Cache body / joint ids
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in JOINT_NAMES
        ]
        self._gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )
        # qpos address for cube freejoint (pos=3, quat=4)
        self._cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_qvel_addr = self.model.jnt_dofadr[self._cube_joint_id]

        # Apply camera configs from dataclass
        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        # Build gym spaces
        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "joint_pos": spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_vel": spaces.Box(-50.0,  50.0,  (N_JOINTS,), np.float32),
            "ee_pos":    spaces.Box(-5.0,   5.0,   (3,),        np.float32),
        }
        if self.cfg.task != "none":
            obs_dict.update({
                "object_pos":  spaces.Box(-5.0, 5.0,  (3,), np.float32),
                "object_quat": spaces.Box(-1.0, 1.0,  (4,), np.float32),
                "object_vel":  spaces.Box(-10., 10.,  (3,), np.float32),
                "goal_pos":    spaces.Box(-5.0, 5.0,  (3,), np.float32),
                "goal_quat":   spaces.Box(-1.0, 1.0,  (4,), np.float32),
            })
        obs_dict["images"] = spaces.Dict({
            cam.name: spaces.Box(0, 255, (cam.height, cam.width, 3), np.uint8)
            for cam in self._all_cameras
        })
        self.observation_space = spaces.Dict(obs_dict)

        # Renderer (shared across cameras via resize)
        max_h = max(c.height for c in self._all_cameras)
        max_w = max(c.width  for c in self._all_cameras)
        self._renderer = mujoco.Renderer(self.model, max_h, max_w)

        self._viewer = None  # lazy-init for human render mode

    # ------------------------------------------------------------------
    # Camera config
    # ------------------------------------------------------------------

    def _apply_camera_configs(self):
        """Override camera pos/orientation via cam_pos + cam_quat in mjModel."""
        for cam_cfg in self._all_cameras:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg.name
            )
            if cam_id < 0:
                continue
            pos = np.array(cam_cfg.pos, dtype=np.float64)
            self.model.cam_pos[cam_id] = pos
            quat = _euler_deg_to_quat(*cam_cfg.euler)
            self.model.cam_quat[cam_id] = quat
            self.model.cam_fovy[cam_id] = cam_cfg.fov

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Set default arm pose
        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS[i]
            self.data.ctrl[self._actuator_ids[i]] = DEFAULT_QPOS[i]

        # Randomise cube position
        if self.cfg.task != "none":
            rng = self.np_random
            noise = self.cfg.object.init_pos_noise
            cx = 0.45 + rng.uniform(-noise[0], noise[0])
            cy = 0.00 + rng.uniform(-noise[1], noise[1])
            cz = 0.83
            self.data.qpos[self._cube_qpos_addr:self._cube_qpos_addr+3] = [cx, cy, cz]
            self.data.qpos[self._cube_qpos_addr+3:self._cube_qpos_addr+7] = [1, 0, 0, 0]

            # Set goal marker position
            goal_pos = np.array(self.cfg.goal.target_pos, dtype=np.float64)
            self.model.body_pos[self._goal_body_id] = goal_pos

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i, act_id in enumerate(self._actuator_ids):
            self.data.ctrl[act_id] = action[i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_success()
        truncated = self._step_count >= self._max_steps
        info = {"success": terminated}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.cfg.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        # Default: return main scene from first outside camera
        cam = self.cfg.outside_cameras[0]
        return self._render_camera(cam.name, cam.width, cam.height)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        ee_pos = self.data.site_xpos[self._gripper_site_id].astype(np.float32)

        obs = {"joint_pos": joint_pos, "joint_vel": joint_vel, "ee_pos": ee_pos}

        if self.cfg.task != "none":
            obj_pos  = self.data.xpos[self._cube_body_id].astype(np.float32)
            obj_quat = self.data.xquat[self._cube_body_id].astype(np.float32)
            obj_vel  = self.data.qvel[self._cube_qvel_addr:self._cube_qvel_addr+3].astype(np.float32)
            goal_pos  = np.array(self.cfg.goal.target_pos,  dtype=np.float32)
            goal_quat = np.array(self.cfg.goal.target_quat, dtype=np.float32)
            obs.update({
                "object_pos":  obj_pos,
                "object_quat": obj_quat,
                "object_vel":  obj_vel,
                "goal_pos":    goal_pos,
                "goal_quat":   goal_quat,
            })

        obs["images"] = {
            cam.name: self._render_camera(cam.name, cam.width, cam.height)
            for cam in self._all_cameras
        }
        return obs

    def _render_camera(self, cam_name: str, width: int, height: int) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_name)
        img = self._renderer.render()
        if img.shape[:2] != (height, width):
            img = img[:height, :width]
        return img.astype(np.uint8)

    def _compute_reward(self) -> float:
        if self.cfg.task == "none":
            return 0.0
        obj_pos  = self.data.xpos[self._cube_body_id]
        obj_quat = self.data.xquat[self._cube_body_id]
        goal_pos  = np.array(self.cfg.goal.target_pos)
        goal_quat = np.array(self.cfg.goal.target_quat)
        dist     = float(np.linalg.norm(obj_pos - goal_pos))
        rot_err  = _quat_angle_diff(obj_quat / (np.linalg.norm(obj_quat) + 1e-9),
                                    goal_quat)
        reward   = -dist - 0.05 * rot_err
        if dist < self.cfg.goal.pos_threshold and rot_err < self.cfg.goal.rot_threshold:
            reward += 10.0
        return float(reward)

    def _is_success(self) -> bool:
        if self.cfg.task == "none":
            return False
        obj_pos  = self.data.xpos[self._cube_body_id]
        obj_quat = self.data.xquat[self._cube_body_id]
        goal_pos  = np.array(self.cfg.goal.target_pos)
        goal_quat = np.array(self.cfg.goal.target_quat)
        dist    = float(np.linalg.norm(obj_pos - goal_pos))
        rot_err = _quat_angle_diff(obj_quat / (np.linalg.norm(obj_quat) + 1e-9),
                                   goal_quat)
        return dist < self.cfg.goal.pos_threshold and rot_err < self.cfg.goal.rot_threshold


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: SingleArmEnvConfig | None = None,
):
    """
    EnvHub-compatible factory.

    Args:
        n_envs:          number of parallel envs
        use_async_envs:  use AsyncVectorEnv (multi-process) instead of Sync
        cfg:             SingleArmEnvConfig — override cameras, task, etc.

    Returns:
        gym.vector.VectorEnv
    """
    if cfg is None:
        cfg = SingleArmEnvConfig()

    def _make():
        return SO101SingleArmEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
