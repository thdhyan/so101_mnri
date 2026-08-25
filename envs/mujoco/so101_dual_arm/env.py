"""
SO-101 Dual Arm MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Cameras:
  - left_wrist      : attached to left gripper
  - right_wrist     : attached to right gripper
  - overhead_left   : world-fixed, configurable
  - overhead_right  : world-fixed, configurable
  - overhead_cam    : world-fixed, top-down over table center
  - front_cam       : world-fixed, front view at eye height

Tasks:
  - "none"  : pure data collection
  - "push"  : push cube to goal_pos
  - "pull"  : pull cube toward centre from far side
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSETS_DIR = Path(__file__).parent / "assets"
SCENE_XML = str(ASSETS_DIR / "scene.xml")

LEFT_JOINT_NAMES  = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                     "left_wrist_flex", "left_wrist_roll", "left_gripper"]
RIGHT_JOINT_NAMES = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                     "right_wrist_flex", "right_wrist_roll", "right_gripper"]
ALL_JOINT_NAMES   = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
N_JOINTS_EACH     = 6
N_JOINTS_TOTAL    = 12

DEFAULT_QPOS_EACH = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Config dataclasses (same pattern as single arm)
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """
    Camera with absolute position (metres) and orientation (roll/pitch/yaw degrees).
    For world-fixed cameras: pos/euler in world frame.
    For body-attached cameras (wrist): pos/euler in body frame.
    """
    name: str
    pos: tuple = (0.0, 0.0, 1.0)
    euler: tuple = (0.0, 0.0, 0.0)    # roll pitch yaw degrees (XYZ intrinsic)
    width: int = 640
    height: int = 480
    fov: float = 75.0


@dataclass
class ObjectConfig:
    size: tuple = (0.025, 0.025, 0.025)
    mass: float = 0.08
    rgba: tuple = (0.85, 0.2, 0.2, 1.0)
    init_pos_noise: tuple = (0.06, 0.06, 0.0)


@dataclass
class GoalConfig:
    target_pos: tuple = (0.55, 0.0, 0.83)
    target_quat: tuple = (1.0, 0.0, 0.0, 0.0)
    pos_threshold: float = 0.04
    rot_threshold: float = 0.3
    visualize: bool = True


@dataclass
class DualArmEnvConfig:
    episode_length_s: float = 25.0
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    left_wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="left_wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    right_wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="right_wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 6.0),
        fov=75.0,
    ))
    overhead_cameras: list = field(default_factory=lambda: [
        CameraConfig("overhead_left",  pos=(0.637,  0.027, 1.248), euler=(-26.0,  0.5, -90.5), fov=80.0),
        CameraConfig("overhead_right", pos=(0.530, -0.491, 1.061), euler=(-67.0, -5.0,-159.5), fov=80.0),
        CameraConfig("overhead_cam",   pos=(0.35, 0.0, 1.45), euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",      pos=(0.40, -1.05, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
    ])

    task: str = "push"
    object: ObjectConfig = field(default_factory=ObjectConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)


# ---------------------------------------------------------------------------
# Helpers (same as single arm)
# ---------------------------------------------------------------------------

def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
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
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * math.acos(dot)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101DualArmEnv(gym.Env):
    """
    Two SO-101 follower arms + 2 wrist cameras + 2 configurable overhead cameras.

    Observations (dict):
        left_joint_pos    : (6,)
        left_joint_vel    : (6,)
        left_ee_pos       : (3,)
        right_joint_pos   : (6,)
        right_joint_vel   : (6,)
        right_ee_pos      : (3,)
        object_pos        : (3,) [if task != "none"]
        object_quat       : (4,) [if task != "none"]
        object_vel        : (3,) [if task != "none"]
        goal_pos          : (3,) [if task != "none"]
        goal_quat         : (4,) [if task != "none"]
        images            : dict {cam_name: (H,W,3) uint8}

    Actions:
        (12,) = left 6 + right 6 target joint positions in radians
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: DualArmEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or DualArmEnvConfig()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self._max_steps = int(self.cfg.episode_length_s * self.cfg.control_freq_hz)
        self._step_count = 0
        self._physics_steps_per_ctrl = max(1, int(
            (1.0 / self.cfg.control_freq_hz) / self.model.opt.timestep
        ))

        # Cache ids
        self._left_jids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINT_NAMES]
        self._right_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINT_NAMES]
        self._left_act   = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEFT_JOINT_NAMES]
        self._right_act  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_JOINT_NAMES]

        self._left_ee_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
        self._right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripperframe")

        self._cube_body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._goal_body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self._cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_qvel_addr = self.model.jnt_dofadr[self._cube_joint_id]

        # All cameras
        self._all_cameras = [
            self.cfg.left_wrist_camera,
            self.cfg.right_wrist_camera,
        ] + self.cfg.overhead_cameras
        self._apply_camera_configs()

        # Action space: 12D
        all_act_ids = self._left_act + self._right_act
        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in all_act_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in all_act_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "left_joint_pos":  spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "left_joint_vel":  spaces.Box(-50.0,  50.0,  (6,), np.float32),
            "left_ee_pos":     spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "right_joint_pos": spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "right_joint_vel": spaces.Box(-50.0,  50.0,  (6,), np.float32),
            "right_ee_pos":    spaces.Box(-5.0,   5.0,   (3,), np.float32),
        }
        if self.cfg.task != "none":
            obs_dict.update({
                "object_pos":  spaces.Box(-5.0, 5.0, (3,), np.float32),
                "object_quat": spaces.Box(-1.0, 1.0, (4,), np.float32),
                "object_vel":  spaces.Box(-10., 10., (3,), np.float32),
                "goal_pos":    spaces.Box(-5.0, 5.0, (3,), np.float32),
                "goal_quat":   spaces.Box(-1.0, 1.0, (4,), np.float32),
            })
        obs_dict["images"] = spaces.Dict({
            cam.name: spaces.Box(0, 255, (cam.height, cam.width, 3), np.uint8)
            for cam in self._all_cameras
        })
        self.observation_space = spaces.Dict(obs_dict)

        max_h = max(c.height for c in self._all_cameras)
        max_w = max(c.width  for c in self._all_cameras)
        self._renderer = mujoco.Renderer(self.model, max_h, max_w)
        self._viewer = None

    def _apply_camera_configs(self):
        for cam_cfg in self._all_cameras:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg.name)
            if cam_id < 0:
                continue
            pos = np.array(cam_cfg.pos, dtype=np.float64)
            self.model.cam_pos[cam_id] = pos
            quat = _euler_deg_to_quat(*cam_cfg.euler)
            self.model.cam_quat[cam_id] = quat
            self.model.cam_fovy[cam_id] = cam_cfg.fov

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        for i, jid in enumerate(self._left_jids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS_EACH[i]
            self.data.ctrl[self._left_act[i]] = DEFAULT_QPOS_EACH[i]
        for i, jid in enumerate(self._right_jids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS_EACH[i]
            self.data.ctrl[self._right_act[i]] = DEFAULT_QPOS_EACH[i]

        if self.cfg.task != "none":
            rng = self.np_random
            noise = self.cfg.object.init_pos_noise
            cx = 0.45 + rng.uniform(-noise[0], noise[0])
            cy = 0.00 + rng.uniform(-noise[1], noise[1])
            self.data.qpos[self._cube_qpos_addr:self._cube_qpos_addr+3] = [cx, cy, 0.83]
            self.data.qpos[self._cube_qpos_addr+3:self._cube_qpos_addr+7] = [1, 0, 0, 0]
            self.model.body_pos[self._goal_body_id] = np.array(self.cfg.goal.target_pos)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i, act_id in enumerate(self._left_act):
            self.data.ctrl[act_id] = action[i]
        for i, act_id in enumerate(self._right_act):
            self.data.ctrl[act_id] = action[N_JOINTS_EACH + i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_success()
        truncated = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {"success": terminated}

    def render(self):
        if self.cfg.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        cam = self.cfg.overhead_cameras[0]
        return self._render_camera(cam.name, cam.width, cam.height)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer.close()

    def _get_obs(self) -> dict:
        def _jpos(jids):
            return np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in jids], np.float32)
        def _jvel(jids):
            return np.array([self.data.qvel[self.model.jnt_dofadr[j]]  for j in jids], np.float32)

        obs = {
            "left_joint_pos":  _jpos(self._left_jids),
            "left_joint_vel":  _jvel(self._left_jids),
            "left_ee_pos":     self.data.site_xpos[self._left_ee_id].astype(np.float32),
            "right_joint_pos": _jpos(self._right_jids),
            "right_joint_vel": _jvel(self._right_jids),
            "right_ee_pos":    self.data.site_xpos[self._right_ee_id].astype(np.float32),
        }

        if self.cfg.task != "none":
            obs.update({
                "object_pos":  self.data.xpos[self._cube_body_id].astype(np.float32),
                "object_quat": self.data.xquat[self._cube_body_id].astype(np.float32),
                "object_vel":  self.data.qvel[self._cube_qvel_addr:self._cube_qvel_addr+3].astype(np.float32),
                "goal_pos":    np.array(self.cfg.goal.target_pos,  dtype=np.float32),
                "goal_quat":   np.array(self.cfg.goal.target_quat, dtype=np.float32),
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
        goal_pos = np.array(self.cfg.goal.target_pos)
        dist     = float(np.linalg.norm(obj_pos - goal_pos))
        obj_quat  = self.data.xquat[self._cube_body_id]
        goal_quat = np.array(self.cfg.goal.target_quat)
        rot_err   = _quat_angle_diff(obj_quat / (np.linalg.norm(obj_quat) + 1e-9), goal_quat)
        reward = -dist - 0.05 * rot_err
        if dist < self.cfg.goal.pos_threshold and rot_err < self.cfg.goal.rot_threshold:
            reward += 10.0
        return float(reward)

    def _is_success(self) -> bool:
        if self.cfg.task == "none":
            return False
        obj_pos  = self.data.xpos[self._cube_body_id]
        goal_pos = np.array(self.cfg.goal.target_pos)
        dist = float(np.linalg.norm(obj_pos - goal_pos))
        obj_quat  = self.data.xquat[self._cube_body_id]
        goal_quat = np.array(self.cfg.goal.target_quat)
        rot_err   = _quat_angle_diff(obj_quat / (np.linalg.norm(obj_quat) + 1e-9), goal_quat)
        return dist < self.cfg.goal.pos_threshold and rot_err < self.cfg.goal.rot_threshold


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: DualArmEnvConfig | None = None,
):
    if cfg is None:
        cfg = DualArmEnvConfig()

    def _make():
        return SO101DualArmEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
