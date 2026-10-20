"""
SO-101 Dual Arm Cylinder Reach MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: A thin cylinder is fixed in place between the two arms (static, no
freejoint). Left arm must reach a target point above the cylinder's left
end-face at a positive z offset; right arm does the same for the right end.
No grasping or lifting — pure reach task.

Cameras:
  - left_wrist      : attached to left gripper
  - right_wrist     : attached to right gripper
  - overhead_left   : world-fixed, configurable
  - overhead_right  : world-fixed, configurable
  - overhead_cam    : world-fixed top-down over the entire workspace, configurable
  - front_cam       : world-fixed front view of the entire workspace, configurable

Success: both gripper sites within pos_threshold of their respective
target points (cylinder end-face position + z_offset).
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
N_JOINTS_EACH = 6

DEFAULT_QPOS_EACH = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    name: str
    pos: tuple = (0.0, 0.0, 1.0)
    euler: tuple = (0.0, 0.0, 0.0)
    width: int = 640
    height: int = 480
    fov: float = 75.0


@dataclass
class CylinderReachConfig:
    z_offset: float = 0.20               # metres above each end-face for the reach target
    pos_threshold: float = 0.035         # gripper-to-target distance to count as reached
    hold_steps: int = 15                 # consecutive steps both targets must be held


@dataclass
class DualArmCylinderReachConfig:
    episode_length_s: float = 25.0
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    left_wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="left_wrist", pos=(0.000, -0.043, -0.042), euler=(29.0, -6.0, 0.5), fov=75.0,
    ))
    right_wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="right_wrist", pos=(0.000, -0.043, -0.042), euler=(29.0, -6.0, 6.0), fov=75.0,
    ))
    overhead_cameras: list = field(default_factory=lambda: [
        CameraConfig("overhead_left",  pos=(0.637,  0.027, 1.248), euler=(-26.0,  0.5, -90.5), fov=80.0),
        CameraConfig("overhead_right", pos=(0.530, -0.491, 1.061), euler=(-67.0, -5.0,-159.5), fov=80.0),
        # Global cameras covering the entire workspace
        CameraConfig("overhead_cam",   pos=(0.40, 0.0, 1.45), euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",      pos=(0.40, -1.0, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
    ])

    task: CylinderReachConfig = field(default_factory=CylinderReachConfig)


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101DualArmCylinderReachEnv(gym.Env):
    """
    Two SO-101 follower arms: reach target points above a fixed cylinder's ends.

    Observations (dict):
        left_joint_pos    : (6,)
        left_joint_vel    : (6,)
        left_ee_pos       : (3,)
        right_joint_pos   : (6,)
        right_joint_vel   : (6,)
        right_ee_pos      : (3,)
        cyl_end_a_pos     : (3,) world position of end A site (fixed)
        cyl_end_b_pos     : (3,) world position of end B site (fixed)
        left_target_pos   : (3,) end A + z_offset
        right_target_pos  : (3,) end B + z_offset
        images            : dict {cam_name: (H,W,3) uint8}

    Actions:
        (12,) = left 6 + right 6 target joint positions in radians
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: DualArmCylinderReachConfig | None = None):
        super().__init__()
        self.cfg = cfg or DualArmCylinderReachConfig()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self._max_steps = int(self.cfg.episode_length_s * self.cfg.control_freq_hz)
        self._step_count = 0
        self._hold_count = 0
        self._physics_steps_per_ctrl = max(1, int(
            (1.0 / self.cfg.control_freq_hz) / self.model.opt.timestep
        ))

        self._left_jids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINT_NAMES]
        self._right_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINT_NAMES]
        self._left_act   = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEFT_JOINT_NAMES]
        self._right_act  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_JOINT_NAMES]

        self._left_ee_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
        self._right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripperframe")

        self._end_a_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cyl_end_a")
        self._end_b_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cyl_end_b")
        self._goal_a_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_a")
        self._goal_b_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_b")

        self._all_cameras = [
            self.cfg.left_wrist_camera,
            self.cfg.right_wrist_camera,
        ] + self.cfg.overhead_cameras
        self._apply_camera_configs()

        all_act_ids = self._left_act + self._right_act
        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in all_act_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in all_act_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "left_joint_pos":   spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "left_joint_vel":   spaces.Box(-50.0,  50.0,  (6,), np.float32),
            "left_ee_pos":      spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "right_joint_pos":  spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "right_joint_vel":  spaces.Box(-50.0,  50.0,  (6,), np.float32),
            "right_ee_pos":     spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "cyl_end_a_pos":    spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "cyl_end_b_pos":    spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "left_target_pos":  spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "right_target_pos": spaces.Box(-5.0,   5.0,   (3,), np.float32),
        }
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
        self._hold_count = 0

        for i, jid in enumerate(self._left_jids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS_EACH[i]
            self.data.ctrl[self._left_act[i]] = DEFAULT_QPOS_EACH[i]
        for i, jid in enumerate(self._right_jids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS_EACH[i]
            self.data.ctrl[self._right_act[i]] = DEFAULT_QPOS_EACH[i]

        mujoco.mj_forward(self.model, self.data)

        # Cylinder is static/fixed: end-face sites are constant world positions.
        end_a = self.data.site_xpos[self._end_a_site_id].copy()
        end_b = self.data.site_xpos[self._end_b_site_id].copy()
        self._left_target = end_a + np.array([0.0, 0.0, self.cfg.task.z_offset])
        self._right_target = end_b + np.array([0.0, 0.0, self.cfg.task.z_offset])
        self.model.body_pos[self._goal_a_body_id] = self._left_target
        self.model.body_pos[self._goal_b_body_id] = self._right_target

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

        if self._both_at_target():
            self._hold_count += 1
        else:
            self._hold_count = 0

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _both_at_target(self) -> bool:
        left_ee  = self.data.site_xpos[self._left_ee_id]
        right_ee = self.data.site_xpos[self._right_ee_id]
        left_dist  = float(np.linalg.norm(left_ee - self._left_target))
        right_dist = float(np.linalg.norm(right_ee - self._right_target))
        return left_dist < self.cfg.task.pos_threshold and right_dist < self.cfg.task.pos_threshold

    def _get_obs(self) -> dict:
        def _jpos(jids):
            return np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in jids], np.float32)
        def _jvel(jids):
            return np.array([self.data.qvel[self.model.jnt_dofadr[j]]  for j in jids], np.float32)

        return {
            "left_joint_pos":   _jpos(self._left_jids),
            "left_joint_vel":   _jvel(self._left_jids),
            "left_ee_pos":      self.data.site_xpos[self._left_ee_id].astype(np.float32),
            "right_joint_pos":  _jpos(self._right_jids),
            "right_joint_vel":  _jvel(self._right_jids),
            "right_ee_pos":     self.data.site_xpos[self._right_ee_id].astype(np.float32),
            "cyl_end_a_pos":    self.data.site_xpos[self._end_a_site_id].astype(np.float32),
            "cyl_end_b_pos":    self.data.site_xpos[self._end_b_site_id].astype(np.float32),
            "left_target_pos":  self._left_target.astype(np.float32),
            "right_target_pos": self._right_target.astype(np.float32),
            "images": {
                cam.name: self._render_camera(cam.name, cam.width, cam.height)
                for cam in self._all_cameras
            }
        }

    def _render_camera(self, cam_name: str, width: int, height: int) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_name)
        img = self._renderer.render()
        if img.shape[:2] != (height, width):
            img = img[:height, :width]
        return img.astype(np.uint8)

    def _compute_reward(self) -> float:
        left_ee  = self.data.site_xpos[self._left_ee_id]
        right_ee = self.data.site_xpos[self._right_ee_id]
        left_dist  = float(np.linalg.norm(left_ee - self._left_target))
        right_dist = float(np.linalg.norm(right_ee - self._right_target))

        reward = -left_dist - right_dist
        if self._both_at_target():
            reward += 2.0
        if self._is_success():
            reward += 10.0
        return float(reward)

    def _is_success(self) -> bool:
        return self._hold_count >= self.cfg.task.hold_steps


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: DualArmCylinderReachConfig | None = None,
):
    if cfg is None:
        cfg = DualArmCylinderReachConfig()

    def _make():
        return SO101DualArmCylinderReachEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
