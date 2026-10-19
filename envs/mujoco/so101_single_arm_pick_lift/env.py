"""
SO-101 Single Arm Pick-Lift MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: "Pick up the cube." Grasp cube and lift it above lift_threshold height.

Cameras:
  - wrist         : attached to gripper body
  - outside_left  : world-fixed, configurable position
  - outside_right : world-fixed, configurable position

Success: is_grasped AND (cube height - init height) > lift_threshold
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

CUBE_INIT_POS = (0.45, 0.0, 0.83)
GRIPPER_CLOSED_THRESHOLD = 0.9   # rad; gripper joint above this ~ closed


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
    pos: tuple = (0.0, 0.0, 1.0)
    euler: tuple = (0.0, 0.0, 0.0)
    width: int = 640
    height: int = 480
    fov: float = 75.0


@dataclass
class ObjectConfig:
    """Manipulation object properties."""
    size: tuple = (0.025, 0.025, 0.025)
    mass: float = 0.08
    rgba: tuple = (0.85, 0.2, 0.2, 1.0)
    init_pos_noise: tuple = (0.06, 0.06, 0.0)


@dataclass
class PickLiftConfig:
    """Pick-lift task parameters."""
    lift_threshold: float = 0.05        # metres above init height counts as lifted
    grasp_dist_threshold: float = 0.035  # gripper-to-cube distance to count as grasped
    n_distractors: int = 0


@dataclass
class SingleArmPickLiftEnvConfig:
    episode_length_s: float = 34.0       # ~1024 steps @ 30hz
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("outside_left",  pos=(0.462, -0.110, 1.273), euler=(0.5,  3.5,  91.0), fov=65.0),
        CameraConfig("outside_right", pos=(0.382, -0.304, 1.066), euler=(65.0, 1.0,  -3.0), fov=65.0),
        # Global cameras covering the entire workspace
        CameraConfig("overhead_cam",  pos=(0.40, 0.0, 1.45), euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",     pos=(0.40, -1.0, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
    ])

    object: ObjectConfig = field(default_factory=ObjectConfig)
    task: PickLiftConfig = field(default_factory=PickLiftConfig)


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

class SO101SingleArmPickLiftEnv(gym.Env):
    """
    Single SO-101 follower arm: pick up cube and lift above threshold height.

    Observations (dict):
        joint_pos    : (6,) radians
        joint_vel    : (6,) rad/s
        tcp_pos      : (3,) gripper site world position
        tcp_quat     : (4,) gripper site orientation, wxyz
        is_grasped   : (1,) 0/1 float
        obj_pos      : (3,) cube position
        obj_quat     : (4,) cube orientation, wxyz
        tcp_to_obj   : (3,) tcp_pos - obj_pos
        images       : dict {cam_name: (H,W,3) uint8}

    Actions:
        (6,) target joint positions in radians, clipped to joint limits
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: SingleArmPickLiftEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or SingleArmPickLiftEnvConfig()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self._max_steps = int(self.cfg.episode_length_s * self.cfg.control_freq_hz)
        self._step_count = 0
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
        self._gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
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
        self._cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_qvel_addr = self.model.jnt_dofadr[self._cube_joint_id]

        self._init_cube_z = CUBE_INIT_POS[2]

        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "joint_pos":  spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_vel":  spaces.Box(-50.0,  50.0,  (N_JOINTS,), np.float32),
            "tcp_pos":    spaces.Box(-5.0,   5.0,   (3,),        np.float32),
            "tcp_quat":   spaces.Box(-1.0,   1.0,   (4,),        np.float32),
            "is_grasped": spaces.Box(0.0,    1.0,   (1,),        np.float32),
            "obj_pos":    spaces.Box(-5.0,   5.0,   (3,),        np.float32),
            "obj_quat":   spaces.Box(-1.0,   1.0,   (4,),        np.float32),
            "tcp_to_obj": spaces.Box(-5.0,   5.0,   (3,),        np.float32),
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

    # ------------------------------------------------------------------
    # Camera config
    # ------------------------------------------------------------------

    def _apply_camera_configs(self):
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

        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = DEFAULT_QPOS[i]
            self.data.ctrl[self._actuator_ids[i]] = DEFAULT_QPOS[i]

        rng = self.np_random
        noise = self.cfg.object.init_pos_noise
        cx = CUBE_INIT_POS[0] + rng.uniform(-noise[0], noise[0])
        cy = CUBE_INIT_POS[1] + rng.uniform(-noise[1], noise[1])
        cz = CUBE_INIT_POS[2]
        self.data.qpos[self._cube_qpos_addr:self._cube_qpos_addr+3] = [cx, cy, cz]
        self.data.qpos[self._cube_qpos_addr+3:self._cube_qpos_addr+7] = [1, 0, 0, 0]
        self._init_cube_z = cz

        # Lift-height marker floats above cube's actual (randomised) xy
        goal_pos = np.array([cx, cy, cz + self.cfg.task.lift_threshold + 0.15])
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

    def _is_grasped(self) -> bool:
        """Grasped if gripper is closed and near cube."""
        tcp_pos = self.data.site_xpos[self._gripper_site_id]
        obj_pos = self.data.xpos[self._cube_body_id]
        dist = float(np.linalg.norm(tcp_pos - obj_pos))
        gripper_qpos = self.data.qpos[self.model.jnt_qposadr[self._gripper_joint_id]]
        return dist < self.cfg.task.grasp_dist_threshold and gripper_qpos > GRIPPER_CLOSED_THRESHOLD * 0.3

    def _get_obs(self) -> dict:
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        tcp_pos = self.data.site_xpos[self._gripper_site_id].astype(np.float32)
        tcp_quat = self.data.site_xmat[self._gripper_site_id].reshape(3, 3)
        tcp_quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quat_wxyz, tcp_quat.flatten())
        tcp_quat_wxyz = tcp_quat_wxyz.astype(np.float32)

        obj_pos  = self.data.xpos[self._cube_body_id].astype(np.float32)
        obj_quat = self.data.xquat[self._cube_body_id].astype(np.float32)

        is_grasped = np.array([1.0 if self._is_grasped() else 0.0], dtype=np.float32)
        tcp_to_obj = (tcp_pos - obj_pos).astype(np.float32)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "tcp_pos": tcp_pos,
            "tcp_quat": tcp_quat_wxyz,
            "is_grasped": is_grasped,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "tcp_to_obj": tcp_to_obj,
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
        tcp_pos = self.data.site_xpos[self._gripper_site_id]
        obj_pos = self.data.xpos[self._cube_body_id]

        reach_dist = float(np.linalg.norm(tcp_pos - obj_pos))
        reach_reward = -reach_dist

        grasped = self._is_grasped()
        grasp_reward = 1.0 if grasped else 0.0

        height_gain = obj_pos[2] - self._init_cube_z
        lift_reward = max(0.0, height_gain) / self.cfg.task.lift_threshold if grasped else 0.0
        lift_reward = min(lift_reward, 1.0) * 3.0

        reward = reach_reward + grasp_reward + lift_reward
        if self._is_success():
            reward += 10.0
        return float(reward)

    def _is_success(self) -> bool:
        obj_pos = self.data.xpos[self._cube_body_id]
        height_gain = obj_pos[2] - self._init_cube_z
        return self._is_grasped() and height_gain > self.cfg.task.lift_threshold


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: SingleArmPickLiftEnvConfig | None = None,
):
    """
    EnvHub-compatible factory.

    Args:
        n_envs:          number of parallel envs
        use_async_envs:  use AsyncVectorEnv (multi-process) instead of Sync
        cfg:             SingleArmPickLiftEnvConfig — override cameras, task, etc.

    Returns:
        gym.vector.VectorEnv
    """
    if cfg is None:
        cfg = SingleArmPickLiftEnvConfig()

    def _make():
        return SO101SingleArmPickLiftEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
