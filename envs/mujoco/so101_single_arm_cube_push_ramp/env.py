"""
SO-101 Single Arm Cube-Push-Ramp MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: "Push the cube up the ramp into the green goal patch." The arm (jaws
closed, non-prehensile pusher) must shove a red cube up a 15-degree incline so
its centre reaches the goal patch near the ramp top. Gravity + low friction
pull the cube back down whenever pushing stops.

MDP/reward structure adapted from the Franka cube-push project
(Belief-based-pushing/source/franka_cube_push_project, Isaac Lab):
see MDP.md for the provenance map and adaptations.

Cameras:
  - wrist        : attached to gripper body
  - overhead_cam : world-fixed, configurable position
  - front_cam    : world-fixed, configurable position

Success: planar cube-centre-to-goal-centre distance < success_threshold
(default 0.010 m, taken from the source project's GOAL_SUCCESS_THRESHOLD).

Observation modes (cfg.obs_mode):
  - "full"   : cube pose visible (MDP)
  - "belief" : cube pose hidden; known initial pose + finite action/proprio
               history instead (POMDP proxy)
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

# Home pose: jaws closed (pusher) parked low just behind the cube's resting
# spot, acting as a backstop at the ramp base. Extending the arm shoves the
# cube up-slope; stopping lets gravity pull the cube back onto the pusher.
DEFAULT_QPOS = np.array([0.0, -0.85, 1.68, 0.68, 0.0, 1.74533])

# ---------------------------------------------------------------------------
# Scene geometry (must stay consistent with assets/scene.xml)
# ---------------------------------------------------------------------------

SLOPE_DEG = 15.0                                   # ramp inclination
_SIN = math.sin(math.radians(SLOPE_DEG))
_COS = math.cos(math.radians(SLOPE_DEG))
RAMP_LOW_EDGE = np.array([0.28, 0.0, 0.82])        # top surface, low edge (world)
CUBE_HALF = 0.0225                                 # 4.5 cm cube
CUBE_REST_S = 0.0587                               # cube centre along-slope dist from low edge
CUBE_Y_JITTER = 0.015                              # lateral spawn jitter (+- m)
GOAL_PATCH_RADIUS = 0.07                           # visual disc radius
GOAL_S = 0.24                                      # goal centre along-slope dist from low edge
GOAL_SUCCESS_THRESHOLD = 0.010                     # from Franka project GOAL_SUCCESS_THRESHOLD


def _slope_dir() -> np.ndarray:
    """Unit vector pointing up-slope along the ramp surface."""
    return np.array([_COS, 0.0, _SIN])


def _slope_normal() -> np.ndarray:
    """Ramp surface normal."""
    return np.array([-_SIN, 0.0, _COS])


def _pitch_quat() -> np.ndarray:
    """wxyz quaternion pitching about +y by -SLOPE_DEG (matches ramp/cube/goal)."""
    half = math.radians(SLOPE_DEG) / 2.0
    return np.array([math.cos(half), 0.0, -math.sin(half), 0.0])


def _cube_rest_pose(y: float) -> tuple[np.ndarray, np.ndarray]:
    """Cube centre (world) resting flush on the ramp at lateral offset y."""
    centre = RAMP_LOW_EDGE + _slope_dir() * CUBE_REST_S + _slope_normal() * CUBE_HALF
    centre[1] = y
    return centre, _pitch_quat()


def _goal_world_pos() -> np.ndarray:
    return RAMP_LOW_EDGE + _slope_dir() * GOAL_S + _slope_normal() * 0.0032


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
class CubePushRampEnvConfig:
    """Cube-push-ramp task parameters."""
    obs_mode: str = "full"                # "full" | "belief"
    episode_length_s: float = 25.0        # timeout; ramp pushing is slower than flat pushes
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"

    history_k: int = 5                    # belief-mode finite-history length

    # Reward weights (see MDP.md; adapted from the Franka cube-push project)
    proximity_weight: float = 2.0         # source cube_goal_distance_tanh weight
    proximity_std: float = 0.10           # source tanh kernel std
    action_rate_weight: float = -0.01     # ||a_t - a_{t-1}||_2^2 penalty
    alive_weight: float = -0.05           # per-step alive penalty (dominant)
    success_bonus: float = 10.0
    success_threshold: float = GOAL_SUCCESS_THRESHOLD

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("overhead_cam", pos=(0.43, 0.0, 1.45), euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",    pos=(0.43, -1.0, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
    ])


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


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Yaw (rotation about world z) of a wxyz quaternion."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=np.float64))
    return float(math.atan2(m[3], m[0]))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101SingleArmCubePushRampEnv(gym.Env):
    """
    Single SO-101 follower arm pushes a cube up a 15-deg ramp into a green
    goal patch. Non-prehensile: gripper jaws closed act as a blunt pusher.

    Observations (dict), mode selected by cfg.obs_mode:

    "full" (MDP):
        joint_pos    : (6,) radians
        joint_vel    : (6,) rad/s
        cube_pos     : (3,) cube position, robot-root frame
        cube_yaw     : (1,) cube yaw, robot-root frame
        goal_pos     : (2,) goal patch xy, robot-root frame
        last_action  : (6,)
        images       : dict {cam_name: (H,W,3) uint8}
        (concatenated policy vector: 24)

    "belief" (POMDP proxy — cube pose hidden):
        joint_pos    : (6,)
        joint_vel    : (6,)
        goal_pos     : (2,)
        last_action  : (6,)
        cube_init_pos: (3,) cube pose AT RESET, robot-root frame (known)
        cube_init_yaw: (1,)
        history      : (90,) last K=5 steps of (joint_pos, joint_vel,
                       last_action) concatenated, oldest -> newest
        images       : dict
        (concatenated policy vector: 114)

    Actions:
        (6,) target joint positions in radians, clipped to joint limits.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: CubePushRampEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or CubePushRampEnvConfig()
        if self.cfg.obs_mode not in ("full", "belief"):
            raise ValueError(f"obs_mode must be 'full' or 'belief', got {self.cfg.obs_mode!r}")
        if self.cfg.history_k < 1:
            raise ValueError("history_k must be >= 1")

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
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_patch"
        )
        self._cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]

        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        self._k = self.cfg.history_k
        self._hist_dim = 3 * N_JOINTS
        self._history = np.zeros((self._k, self._hist_dim), dtype=np.float32)
        self._last_action = np.zeros(N_JOINTS, dtype=np.float32)

        common = {
            "joint_pos":   spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_vel":   spaces.Box(-50.0,  50.0,  (N_JOINTS,), np.float32),
            "goal_pos":    spaces.Box(-5.0,   5.0,   (2,),        np.float32),
            "last_action": spaces.Box(ctrl_low.copy(), ctrl_high.copy(), (N_JOINTS,), np.float32),
        }
        if self.cfg.obs_mode == "full":
            fields = {
                **common,
                "cube_pos": spaces.Box(-5.0, 5.0, (3,), np.float32),
                "cube_yaw": spaces.Box(-np.pi, np.pi, (1,), np.float32),
            }
        else:
            fields = {
                **common,
                "cube_init_pos": spaces.Box(-5.0,   5.0,   (3,),              np.float32),
                "cube_init_yaw": spaces.Box(-np.pi, np.pi, (1,),              np.float32),
                "history":       spaces.Box(-50.0,  50.0,  (self._k * self._hist_dim,), np.float32),
            }
        fields["images"] = spaces.Dict({
            cam.name: spaces.Box(0, 255, (cam.height, cam.width, 3), np.uint8)
            for cam in self._all_cameras
        })
        self.observation_space = spaces.Dict(fields)

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
    # Frames
    # ------------------------------------------------------------------

    def _to_root_frame(self, p_world: np.ndarray) -> np.ndarray:
        root_pos = self.data.xpos[self._base_body_id]
        root_mat = self.data.xmat[self._base_body_id].reshape(3, 3)
        return root_mat.T @ (p_world - root_pos)

    def _root_yaw(self) -> float:
        root_mat = self.data.xmat[self._base_body_id].reshape(3, 3)
        return math.atan2(root_mat[1, 0], root_mat[0, 0])

    def _cube_pose_root(self) -> tuple[np.ndarray, float]:
        pos = self._to_root_frame(self.data.xpos[self._cube_body_id])
        yaw = _yaw_from_quat_wxyz(self.data.xquat[self._cube_body_id]) - self._root_yaw()
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        return pos, yaw

    def _goal_xy_root(self) -> np.ndarray:
        return self._to_root_frame(self.data.xpos[self._goal_body_id])[:2]

    def _cube_goal_dist(self) -> float:
        cube_xy = self.data.xpos[self._cube_body_id][:2]
        goal_xy = self.data.xpos[self._goal_body_id][:2]
        return float(np.linalg.norm(cube_xy - goal_xy))

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
        self._last_action = np.zeros(N_JOINTS, dtype=np.float32)

        rng = self.np_random
        y = float(rng.uniform(-CUBE_Y_JITTER, CUBE_Y_JITTER))
        centre, quat = _cube_rest_pose(y)
        # Spawn a hair above the surface and let contacts settle so the recorded
        # initial pose is a true resting pose (settling drift is negligible).
        self.data.qpos[self._cube_qpos_addr:self._cube_qpos_addr+3] = centre \
            + _slope_normal() * 0.001
        self.data.qpos[self._cube_qpos_addr+3:self._cube_qpos_addr+7] = quat

        mujoco.mj_forward(self.model, self.data)
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)

        # Known initial cube pose for belief mode (recorded AFTER jitter/settle).
        init_pos, init_yaw = self._cube_pose_root()
        self._init_cube_pos = init_pos.astype(np.float32)
        self._init_cube_yaw = np.array([init_yaw], dtype=np.float32)

        home_vel = np.zeros(N_JOINTS, dtype=np.float32)
        sample = np.concatenate([DEFAULT_QPOS.astype(np.float32), home_vel, self._last_action])
        self._history[:] = sample

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64),
                         self.action_space.low, self.action_space.high)
        for i, act_id in enumerate(self._actuator_ids):
            self.data.ctrl[act_id] = action[i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        joint_pos = self._get_joint_pos()
        joint_vel = self._get_joint_vel()
        sample = np.concatenate([joint_pos, joint_vel,
                                 action.astype(np.float32)])
        self._history[:-1] = self._history[1:]
        self._history[-1] = sample
        prev_action = self._last_action.copy()
        self._last_action = action.astype(np.float32)

        obs = self._get_obs()
        reward = self._compute_reward(prev_action)
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

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        return np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)

    def _get_obs(self) -> dict:
        joint_pos = self._get_joint_pos()
        joint_vel = self._get_joint_vel()
        goal_xy = self._goal_xy_root().astype(np.float32)

        obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "goal_pos": goal_xy,
            "last_action": self._last_action.copy(),
        }
        if self.cfg.obs_mode == "full":
            cube_pos, cube_yaw = self._cube_pose_root()
            obs["cube_pos"] = cube_pos.astype(np.float32)
            obs["cube_yaw"] = np.array([cube_yaw], dtype=np.float32)
        else:
            obs["cube_init_pos"] = self._init_cube_pos.copy()
            obs["cube_init_yaw"] = self._init_cube_yaw.copy()
            obs["history"] = self._history.flatten().copy()
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

    def _compute_reward(self, prev_action: np.ndarray) -> float:
        d = self._cube_goal_dist()
        proximity = self.cfg.proximity_weight * (1.0 - math.tanh(d / self.cfg.proximity_std))
        action_rate = self.cfg.action_rate_weight * float(np.sum((self._last_action - prev_action) ** 2))
        reward = proximity + action_rate + self.cfg.alive_weight
        if self._is_success():
            reward += self.cfg.success_bonus
        return float(reward)

    def _is_success(self) -> bool:
        return self._cube_goal_dist() < self.cfg.success_threshold


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: CubePushRampEnvConfig | None = None,
):
    """
    EnvHub-compatible factory.

    Args:
        n_envs:          number of parallel envs
        use_async_envs:  use AsyncVectorEnv (multi-process) instead of Sync
        cfg:             CubePushRampEnvConfig — set obs_mode="belief" for the
                         POMDP variant, override cameras/reward weights, etc.

    Returns:
        gym.vector.VectorEnv
    """
    if cfg is None:
        cfg = CubePushRampEnvConfig()

    def _make():
        return SO101SingleArmCubePushRampEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
