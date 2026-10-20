"""
SO-101 Single Arm Cylinder Reach MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: A thin cylinder is fixed in place on the table (static, no freejoint).
The arm must reach a target point above the cylinder's end-face A at a
positive z offset. No grasping or lifting — pure reach task.

Cameras:
  - wrist         : attached to gripper body
  - outside_left  : world-fixed, configurable
  - outside_right : world-fixed, configurable
  - overhead_cam  : world-fixed top-down over workspace
  - front_cam     : world-fixed front view

Success: gripper site within pos_threshold of target for hold_steps
consecutive steps.
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

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = 6

DEFAULT_QPOS = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])


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
    z_offset: float = 0.20               # metres above end-face for the reach target
    pos_threshold: float = 0.035         # gripper-to-target distance to count as reached
    hold_steps: int = 15                 # consecutive steps target must be held


@dataclass
class SingleArmCylinderReachConfig:
    episode_length_s: float = 25.0
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist", pos=(0.000, -0.043, -0.042), euler=(29.0, -6.0, 0.5), fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("outside_left",  pos=(0.462, -0.110, 1.273), euler=(0.5,  3.5,  91.0), fov=65.0),
        CameraConfig("outside_right", pos=(0.382, -0.304, 1.066), euler=(65.0, 1.0,  -3.0), fov=65.0),
        CameraConfig("overhead_cam",  pos=(0.40, 0.0, 1.45), euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",     pos=(0.40, -1.0, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
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

class SO101SingleArmCylinderReachEnv(gym.Env):
    """
    Single SO-101 follower arm: reach target point above a fixed cylinder's end A.

    Observations (dict):
        joint_pos      : (6,)
        joint_vel      : (6,)
        ee_pos         : (3,)
        cyl_end_a_pos  : (3,) world position of end A site (fixed)
        cyl_end_b_pos  : (3,) world position of end B site (fixed)
        target_pos     : (3,) end A + z_offset
        images         : dict {cam_name: (H,W,3) uint8}

    Actions:
        (6,) target joint positions in radians
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: SingleArmCylinderReachConfig | None = None):
        super().__init__()
        self.cfg = cfg or SingleArmCylinderReachConfig()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self._max_steps = int(self.cfg.episode_length_s * self.cfg.control_freq_hz)
        self._step_count = 0
        self._hold_count = 0
        self._physics_steps_per_ctrl = max(1, int(
            (1.0 / self.cfg.control_freq_hz) / self.model.opt.timestep
        ))

        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in JOINT_NAMES
        ]
        self._ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._end_a_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cyl_end_a")
        self._end_b_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cyl_end_b")
        self._goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")

        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "joint_pos":     spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_vel":     spaces.Box(-50.0,  50.0,  (N_JOINTS,), np.float32),
            "ee_pos":        spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "cyl_end_a_pos": spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "cyl_end_b_pos": spaces.Box(-5.0,   5.0,   (3,), np.float32),
            "target_pos":    spaces.Box(-5.0,   5.0,   (3,), np.float32),
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
            self.model.cam_pos[cam_id] = np.array(cam_cfg.pos, dtype=np.float64)
            self.model.cam_quat[cam_id] = _euler_deg_to_quat(*cam_cfg.euler)
            self.model.cam_fovy[cam_id] = cam_cfg.fov

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._hold_count = 0

        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS[i]
            self.data.ctrl[self._actuator_ids[i]] = DEFAULT_QPOS[i]

        mujoco.mj_forward(self.model, self.data)

        # Cylinder is static: end-face sites are constant world positions.
        end_a = self.data.site_xpos[self._end_a_site_id].copy()
        self._target = end_a + np.array([0.0, 0.0, self.cfg.task.z_offset])
        self.model.body_pos[self._goal_body_id] = self._target

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i, act_id in enumerate(self._actuator_ids):
            self.data.ctrl[act_id] = action[i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        if self._at_target():
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

    def _at_target(self) -> bool:
        ee = self.data.site_xpos[self._ee_id]
        dist = float(np.linalg.norm(ee - self._target))
        return dist < self.cfg.task.pos_threshold

    def _get_obs(self) -> dict:
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[j]] for j in self._joint_ids
        ], np.float32)
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[j]] for j in self._joint_ids
        ], np.float32)

        return {
            "joint_pos":     joint_pos,
            "joint_vel":     joint_vel,
            "ee_pos":        self.data.site_xpos[self._ee_id].astype(np.float32),
            "cyl_end_a_pos": self.data.site_xpos[self._end_a_site_id].astype(np.float32),
            "cyl_end_b_pos": self.data.site_xpos[self._end_b_site_id].astype(np.float32),
            "target_pos":    self._target.astype(np.float32),
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
        ee = self.data.site_xpos[self._ee_id]
        dist = float(np.linalg.norm(ee - self._target))
        reward = -dist
        if self._at_target():
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
    cfg: SingleArmCylinderReachConfig | None = None,
):
    if cfg is None:
        cfg = SingleArmCylinderReachConfig()

    def _make():
        return SO101SingleArmCylinderReachEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
