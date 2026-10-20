"""
SO-101 Single Arm Push-T MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: "Push the T-shaped block onto the target T outline on the table."
Non-prehensile: the gripper is held closed and used as a pusher (no grasp).

Cameras:
  - wrist    : attached to gripper body
  - overhead : world-fixed, top-down over the workspace
  - front    : world-fixed, front view at eye height

Success: T-block centre within 2.5 cm of the target centre AND
|wrapped yaw error| < 15 deg.
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

# Default joint position (relaxed upright pose)
DEFAULT_QPOS = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])

# Gripper held closed: pinned actuator target (pusher mode)
GRIPPER_CLOSED_POS = 1.5   # rad; within ctrlrange [-0.17453, 1.74533]
GRIPPER_ACTION_IDX = 5     # action channel overridden with the value above

# T-block spawn / target poses (world frame). Block z fixed: slides on table.
# Spawn box sits beside the arm (clear of the home-pose gripper at (0.467, 0)).
BLOCK_SPAWN_CENTER = np.array([0.36, 0.12])
BLOCK_Z = 0.8325           # 2.5 cm thick block resting on 0.82 m table top
TARGET_POS = np.array([0.45, 0.0])
TARGET_YAW = 0.0


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
class TBlockConfig:
    """T-block spawn randomization."""
    init_pos_noise: tuple = (0.04, 0.04)   # ±xy half-extent of spawn box (~8 cm box)
    random_yaw: bool = True                # uniform yaw in [0, 2pi)


@dataclass
class PushTTaskConfig:
    """Push-T task parameters."""
    pos_tol: float = 0.025          # m; centre XY distance for success
    yaw_tol_deg: float = 15.0       # deg; wrapped yaw error for success
    sigma_pos: float = 0.10         # m; exp-kernel width of the position term
    sigma_yaw_deg: float = 60.0     # deg; exp-kernel width of the yaw term
    w_pos: float = 1.0              # weight, exp(-d_xy^2 / (2 sigma_pos^2))
    w_yaw: float = 0.5              # weight, exp(-(dyaw / sigma_yaw)^2)
    w_action_rate: float = 0.01     # weight on ||a_t - a_{t-1}||^2
    w_alive: float = 0.05           # per-step alive penalty (dominant)
    success_bonus: float = 10.0


@dataclass
class SingleArmPushTEnvConfig:
    episode_length_s: float = 20.0       # 600 steps @ 30 Hz -> truncated
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("overhead", pos=(0.41, 0.05, 1.35), euler=(0.0,  0.0, 90.0), fov=65.0),
        CameraConfig("front",    pos=(0.40, -0.95,  1.05), euler=(75.0, 0.0,  0.0), fov=60.0),
    ])

    block: TBlockConfig = field(default_factory=TBlockConfig)
    task: PushTTaskConfig = field(default_factory=PushTTaskConfig)


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


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Quaternion wxyz for a pure z-axis rotation."""
    return np.array([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)])


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101SingleArmPushTEnv(gym.Env):
    """
    Single SO-101 follower arm pushes a T-shaped block (gripper held closed,
    non-prehensile) onto a target T outline drawn on the table.

    Observations (dict):
        joint_pos    : (6,) radians
        joint_vel    : (6,) rad/s
        tcp_pos      : (3,) gripper site world position
        t_pose       : (3,) T-block pose in robot root frame (x, y, yaw); z fixed
        target_pose  : (3,) target pose in robot root frame (x, y, yaw)
        last_action  : (6,) previous executed joint targets
        images       : dict {cam_name: (H,W,3) uint8}

    Flat policy vector: 6+6+3+3+3+6 = (27,).

    Actions:
        (6,) absolute target joint positions in radians, clipped to joint
        limits. The gripper channel is overridden with GRIPPER_CLOSED_POS
        (the gripper stays closed and acts as a pusher).
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: SingleArmPushTEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or SingleArmPushTEnvConfig()

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
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self._block_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "t_block"
        )
        self._block_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "t_block_joint"
        )
        self._block_qpos_addr = self.model.jnt_qposadr[self._block_joint_id]
        self._block_qvel_addr = self.model.jnt_dofadr[self._block_joint_id]

        self._last_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float64)

        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dict = {
            "joint_pos":   spaces.Box(-np.pi,  np.pi,  (N_JOINTS,), np.float32),
            "joint_vel":   spaces.Box(-50.0,   50.0,  (N_JOINTS,), np.float32),
            "tcp_pos":     spaces.Box(-5.0,    5.0,   (3,),        np.float32),
            "t_pose":      spaces.Box(np.array([-5.0, -5.0, -np.pi], dtype=np.float32),
                                      np.array([5.0,  5.0,  np.pi], dtype=np.float32), dtype=np.float32),
            "target_pose": spaces.Box(np.array([-5.0, -5.0, -np.pi], dtype=np.float32),
                                      np.array([5.0,  5.0,  np.pi], dtype=np.float32), dtype=np.float32),
            "last_action": spaces.Box(ctrl_low.astype(np.float32), ctrl_high.astype(np.float32), dtype=np.float32),
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
        self._last_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float64)

        # Arm to home pose, gripper held closed
        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            qpos = GRIPPER_CLOSED_POS if JOINT_NAMES[i] == "gripper" else DEFAULT_QPOS[i]
            self.data.qpos[addr] = qpos
            self.data.ctrl[self._actuator_ids[i]] = qpos

        # T-block: randomized xy in spawn box, uniform yaw, z fixed on table
        rng = self.np_random
        noise = self.cfg.block.init_pos_noise
        bx = BLOCK_SPAWN_CENTER[0] + rng.uniform(-noise[0], noise[0])
        by = BLOCK_SPAWN_CENTER[1] + rng.uniform(-noise[1], noise[1])
        byaw = rng.uniform(0.0, 2.0 * math.pi) if self.cfg.block.random_yaw else 0.0
        self.data.qpos[self._block_qpos_addr:self._block_qpos_addr+2] = [bx, by]
        self.data.qpos[self._block_qpos_addr+2] = BLOCK_Z
        self.data.qpos[self._block_qpos_addr+3:self._block_qpos_addr+7] = _yaw_to_quat_wxyz(byaw)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(
            np.asarray(action, dtype=np.float64).flatten(),
            self.action_space.low, self.action_space.high
        )
        executed = action.copy()
        executed[GRIPPER_ACTION_IDX] = GRIPPER_CLOSED_POS  # pusher mode
        for i, act_id in enumerate(self._actuator_ids):
            self.data.ctrl[act_id] = executed[i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = self._last_action
        self._last_action = executed
        self._step_count += 1
        obs = self._get_obs()
        terms = self._reward_terms()
        terminated = self._is_success()
        truncated = self._step_count >= self._max_steps
        reward = terms["total"]
        info = {"success": terminated, **terms}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.cfg.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        cam = self.cfg.outside_cameras[-1]
        return self._render_camera(cam.name, cam.width, cam.height)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _block_yaw(self) -> float:
        """T-block yaw from its body x-axis projected onto the world XY plane."""
        R = self.data.xmat[self._block_body_id].reshape(3, 3)
        return float(math.atan2(R[1, 0], R[0, 0]))

    def _get_obs(self) -> dict:
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        tcp_pos = self.data.site_xpos[self._gripper_site_id].astype(np.float32)

        base_xpos = self.data.xpos[self._base_body_id]
        block_xpos = self.data.xpos[self._block_body_id]
        t_pose = np.array([
            block_xpos[0] - base_xpos[0],
            block_xpos[1] - base_xpos[1],
            self._block_yaw(),
        ], dtype=np.float32)
        target_pose = np.array([
            TARGET_POS[0] - base_xpos[0],
            TARGET_POS[1] - base_xpos[1],
            TARGET_YAW,
        ], dtype=np.float32)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "tcp_pos": tcp_pos,
            "t_pose": t_pose,
            "target_pose": target_pose,
            "last_action": self._last_action.astype(np.float32),
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

    def _reward_terms(self) -> dict:
        block_xpos = self.data.xpos[self._block_body_id]
        d_xy = float(np.linalg.norm(block_xpos[:2] - TARGET_POS))
        yaw_err = abs(_wrap_angle(self._block_yaw() - TARGET_YAW))

        r_pos = math.exp(-(d_xy ** 2) / (2 * self.cfg.task.sigma_pos ** 2))
        sigma_yaw = math.radians(self.cfg.task.sigma_yaw_deg)
        r_yaw = math.exp(-(yaw_err / sigma_yaw) ** 2)
        r_action_rate = -self.cfg.task.w_action_rate * float(
            np.sum((self._last_action - self._prev_action) ** 2)
        )
        r_alive = -self.cfg.task.w_alive
        bonus = self.cfg.task.success_bonus if self._is_success() else 0.0

        total = (self.cfg.task.w_pos * r_pos + self.cfg.task.w_yaw * r_yaw
                 + r_action_rate + r_alive + bonus)

        return {
            "r_pos": r_pos,
            "r_yaw": r_yaw,
            "r_action_rate": r_action_rate,
            "r_alive": r_alive,
            "success_bonus": bonus,
            "yaw_err_deg": math.degrees(yaw_err),
            "d_xy": d_xy,
            "total": float(total),
        }

    def _is_success(self) -> bool:
        block_xpos = self.data.xpos[self._block_body_id]
        d_xy = float(np.linalg.norm(block_xpos[:2] - TARGET_POS))
        yaw_err = abs(_wrap_angle(self._block_yaw() - TARGET_YAW))
        return (d_xy < self.cfg.task.pos_tol
                and yaw_err < math.radians(self.cfg.task.yaw_tol_deg))


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: SingleArmPushTEnvConfig | None = None,
):
    """
    EnvHub-compatible factory.

    Args:
        n_envs:          number of parallel envs
        use_async_envs:  use AsyncVectorEnv (multi-process) instead of Sync
        cfg:             SingleArmPushTEnvConfig — override cameras, task, etc.

    Returns:
        gym.vector.VectorEnv
    """
    if cfg is None:
        cfg = SingleArmPushTEnvConfig()

    def _make():
        return SO101SingleArmPushTEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
