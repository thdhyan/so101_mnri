"""
SO-101 Single Arm Cube-Push-Bridge MuJoCo Environment
EnvHub-compatible: exposes make_env(n_envs, use_async_envs, cfg)

Task: "Push the red cube across the narrow bridge onto the far table and into
the green goal patch." Non-prehensile: the closed gripper is used as a pusher.

Scene: two table sections at the same height separated by a 13 cm gap, spanned
by a 7 cm-wide bridge (top flush with the tables). The cube starts on the near
table; careless pushing knocks it into the gap (failure termination).

Adapted from the Franka belief-based cube-push project
(Belief-based-pushing/source/franka_cube_push_project, RSS'20-style bridge
benchmark): see MDP.md for provenance of reward terms and success criterion.

Observation modes (cfg.obs_mode):
  - "full"   : cube pose visible (MDP)
  - "belief" : cube pose hidden; finite proprioceptive history instead (POMDP)

Success: cube centre within `success_threshold` (default 5 cm, XY) of the goal.
Failure: cube falls below table-top - 5 cm (into the gap / off any edge).
"""

import math
from collections import deque
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

CUBE_INIT_POS = (0.36, 0.0, 0.844)   # near table, roughly aligned with the bridge
TABLE_TOP_Z = 0.82                    # shared top surface of tables + bridge
FALL_MARGIN = 0.05                    # failure when cube z < TABLE_TOP_Z - FALL_MARGIN
HISTORY_K = 5                         # belief-mode proprio history length


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
    """Manipulated cube properties."""
    size: tuple = (0.0225, 0.0225, 0.0225)     # half-extents -> 4.5 cm cube
    mass: float = 0.08
    rgba: tuple = (0.85, 0.2, 0.2, 1.0)
    init_pos_noise: tuple = (0.02, 0.02, 0.0)  # +-2 cm XY jitter on near table


@dataclass
class TaskConfig:
    """Bridge-push task parameters."""
    success_threshold: float = 0.05   # cube-centre to goal-centre XY distance [m]


@dataclass
class CubePushBridgeEnvConfig:
    episode_length_s: float = 25.0       # 750 steps @ 30 Hz
    control_freq_hz: float = 30.0
    render_mode: str = "rgb_array"
    obs_mode: str = "full"               # "full" | "belief"

    wrist_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ))
    outside_cameras: list = field(default_factory=lambda: [
        CameraConfig("overhead_cam", pos=(0.30, 0.0, 1.50),  euler=(0.0, 0.0, 90.0), fov=65.0),
        CameraConfig("front_cam",    pos=(0.35, -1.05, 1.05), euler=(75.0, 0.0, 0.0), fov=65.0),
    ])

    object: ObjectConfig = field(default_factory=ObjectConfig)
    task: TaskConfig = field(default_factory=TaskConfig)


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


def _yaw_from_quat_wxyz(q) -> float:
    """Yaw angle from a wxyz quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))


def _wrap_angle(a) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SO101SingleArmCubePushBridgeEnv(gym.Env):
    """
    Single SO-101 follower arm pushes a red cube across a narrow bridge onto a
    far table section, into a fixed green goal patch.

    Observation dict — cfg.obs_mode == "full" (MDP):
        joint_pos       : (6,) radians
        joint_vel       : (6,) rad/s
        last_action     : (6,) previous applied action (home at reset)
        cube_pos_root   : (3,) cube position in robot root frame, m
        cube_yaw_root   : (1,) cube yaw in robot root frame, rad
        goal_pos_root   : (2,) goal XY in robot root frame, m
        images          : dict {cam_name: (H,W,3) uint8}

    Observation dict — cfg.obs_mode == "belief" (POMDP):
        joint_pos, joint_vel, last_action, goal_pos_root, images as above, plus
        cube_init_pos_root : (3,) cube position AT RESET in robot root frame
        cube_init_yaw_root : (1,) cube yaw AT RESET in robot root frame
        history            : (90,) last K=5 steps of (joint_pos, joint_vel,
                             last_action) concatenated — finite-history proxy
                             for the source project's particle-filter belief

    Actions:
        (6,) target joint positions in radians, clipped to actuator ctrlrange
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: CubePushBridgeEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or CubePushBridgeEnvConfig()
        assert self.cfg.obs_mode in ("full", "belief"), \
            f"obs_mode must be 'full' or 'belief', got {self.cfg.obs_mode!r}"

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
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._cube_qpos_addr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_qvel_addr = self.model.jnt_dofadr[self._cube_joint_id]

        self._fall_z = TABLE_TOP_Z - FALL_MARGIN
        self._success_threshold = self.cfg.task.success_threshold

        # Belief-mode finite-history buffer: entries are (joint_pos, joint_vel,
        # last_action); holds the K steps preceding the current observation.
        self._history_len = HISTORY_K * 3 * N_JOINTS
        self._history = deque(maxlen=HISTORY_K)

        self._all_cameras = [self.cfg.wrist_camera] + self.cfg.outside_cameras
        self._apply_camera_configs()

        ctrl_low  = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids])
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids])
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        self.observation_space = self._build_observation_space()

        self._renderer = mujoco.Renderer(self.model,
                                         max(c.height for c in self._all_cameras),
                                         max(c.width for c in self._all_cameras))
        self._viewer = None

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        d = {
            "joint_pos":   spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_vel":   spaces.Box(-50.0, 50.0, (N_JOINTS,), np.float32),
            "last_action": spaces.Box(self.action_space.low, self.action_space.high,
                                      (N_JOINTS,), np.float32),
            "goal_pos_root": spaces.Box(-5.0, 5.0, (2,), np.float32),
        }
        if self.cfg.obs_mode == "full":
            d["cube_pos_root"] = spaces.Box(-5.0, 5.0, (3,), np.float32)
            d["cube_yaw_root"] = spaces.Box(-np.pi, np.pi, (1,), np.float32)
        else:  # belief
            d["cube_init_pos_root"] = spaces.Box(-5.0, 5.0, (3,), np.float32)
            d["cube_init_yaw_root"] = spaces.Box(-np.pi, np.pi, (1,), np.float32)
            entry_lo = np.concatenate([np.full(N_JOINTS, -np.pi),
                                       np.full(N_JOINTS, -50.0), self.action_space.low])
            entry_hi = np.concatenate([np.full(N_JOINTS, np.pi),
                                       np.full(N_JOINTS, 50.0), self.action_space.high])
            d["history"] = spaces.Box(
                np.tile(entry_lo, HISTORY_K).astype(np.float32),
                np.tile(entry_hi, HISTORY_K).astype(np.float32),
                (self._history_len,), np.float32)
        d["images"] = spaces.Dict({
            cam.name: spaces.Box(0, 255, (cam.height, cam.width, 3), np.uint8)
            for cam in self._all_cameras
        })
        return spaces.Dict(d)

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
            self.model.cam_pos[cam_id] = np.array(cam_cfg.pos, dtype=np.float64)
            self.model.cam_quat[cam_id] = _euler_deg_to_quat(*cam_cfg.euler)
            self.model.cam_fovy[cam_id] = cam_cfg.fov

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def _to_root_frame(self, pos_w: np.ndarray):
        """World position -> robot root frame; returns (pos_root, yaw_root)."""
        base_xpos = self.data.xpos[self._base_body_id]
        base_mat = self.data.xmat[self._base_body_id].reshape(3, 3)
        p_root = base_mat.T @ (np.asarray(pos_w, dtype=np.float64) - base_xpos)
        yaw_base = math.atan2(base_mat[1, 0], base_mat[0, 0])
        return p_root, yaw_base

    def _cube_pose(self):
        """Cube pose as ((pos_w, yaw_w), (pos_root, yaw_root))."""
        pos_w = self.data.xpos[self._cube_body_id].copy()
        quat_w = self.data.xquat[self._cube_body_id].copy()
        yaw_w = _yaw_from_quat_wxyz(quat_w)
        pos_r, yaw_base = self._to_root_frame(pos_w)
        return (pos_w, yaw_w), (pos_r, _wrap_angle(yaw_w - yaw_base))

    def _goal_xy(self):
        """Goal marker body pos in world (parent is world) and root frames."""
        goal_w = self.model.body_pos[self._goal_body_id].copy()
        goal_r, _ = self._to_root_frame(goal_w)
        return goal_w[:2], goal_r[:2]

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

        mujoco.mj_forward(self.model, self.data)

        self._last_action = DEFAULT_QPOS.copy()
        (_, _), (self._cube_init_pos_root, self._cube_init_yaw_root) = self._cube_pose()
        _, goal_root = self._goal_xy()
        self._goal_pos_root = goal_root
        self._prev_goal_dist = self._cube_goal_dist()
        self._history.clear()
        home_entry = np.concatenate([DEFAULT_QPOS, np.zeros(N_JOINTS), DEFAULT_QPOS])
        for _ in range(HISTORY_K):
            self._history.append(home_entry.astype(np.float32))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64),
                         self.action_space.low, self.action_space.high)
        prev_action = self._last_action.copy()
        for i, act_id in enumerate(self._actuator_ids):
            self.data.ctrl[act_id] = action[i]

        for _ in range(self._physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._last_action = action.astype(np.float32)
        self._step_count += 1

        fell = self._has_fallen()
        success = (not fell) and self._is_success()
        obs = self._get_obs()
        reward = self._compute_reward(prev_action, success)
        terminated = bool(fell or success)
        truncated = bool(self._step_count >= self._max_steps and not terminated)
        info = {"success": success, "fell_off_bridge": fell}

        # History stores the just-finished step, so the NEXT observation sees
        # it as history (history excludes the current proprio instant).
        joint_pos, joint_vel = self._read_joints()
        self._history.append(np.concatenate(
            [joint_pos, joint_vel, self._last_action]).astype(np.float32))

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

    def _read_joints(self):
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids
        ], dtype=np.float32)
        return joint_pos, joint_vel

    def _get_obs(self) -> dict:
        joint_pos, joint_vel = self._read_joints()
        (_, _), (cube_pos_root, yaw_root) = self._cube_pose()
        obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "last_action": self._last_action.astype(np.float32),
            "goal_pos_root": self._goal_pos_root.astype(np.float32),
            "images": {
                cam.name: self._render_camera(cam.name, cam.width, cam.height)
                for cam in self._all_cameras
            }
        }
        if self.cfg.obs_mode == "full":
            obs["cube_pos_root"] = cube_pos_root.astype(np.float32)
            obs["cube_yaw_root"] = np.array([yaw_root], dtype=np.float32)
        else:
            obs["cube_init_pos_root"] = self._cube_init_pos_root.astype(np.float32)
            obs["cube_init_yaw_root"] = np.array([self._cube_init_yaw_root], dtype=np.float32)
            obs["history"] = np.concatenate(list(self._history)).astype(np.float32)
        return obs

    def _cube_goal_dist(self) -> float:
        goal_w, _ = self._goal_xy()
        pos_w, _ = self._cube_pose()[0]
        return float(np.linalg.norm(pos_w[:2] - goal_w))

    def _is_success(self) -> bool:
        return self._cube_goal_dist() < self._success_threshold

    def _has_fallen(self) -> bool:
        pos_w = self.data.xpos[self._cube_body_id]
        return bool(pos_w[2] < self._fall_z)

    def _compute_reward(self, prev_action: np.ndarray, success: bool) -> float:
        pos_w, _ = self._cube_pose()[0]
        goal_w, _ = self._goal_xy()
        dist = float(np.linalg.norm(pos_w[:2] - goal_w))

        # Dense shaping adapted from the Franka cube-push project (see MDP.md).
        progress = self._prev_goal_dist - dist          # signed per-step progress
        self._prev_goal_dist = dist

        direction = goal_w - pos_w[:2]
        norm = max(float(np.linalg.norm(direction)), 1e-6)
        vel_xy = self.data.qvel[self._cube_qvel_addr:self._cube_qvel_addr+2]
        speed_to_goal = float(np.dot(vel_xy, direction / norm))
        vel_term = float(np.clip(speed_to_goal / 0.20, -1.0, 1.0))

        tcp_pos = self.data.site_xpos[self._gripper_site_id]
        ee_cube_dist = float(np.linalg.norm(tcp_pos[:2] - pos_w[:2]))

        action_rate_sq = float(np.sum((self._last_action - prev_action) ** 2))
        reward = (
            -0.05                                        # alive penalty
            - 0.01 * action_rate_sq                      # action-rate penalty
            - 0.6 * dist                                 # cube-goal L2
            + 2.0 * (1.0 - math.tanh(dist / 0.10))       # cube-goal tanh kernel
            + 80.0 * progress                            # cube-goal signed progress
            + 3.0 * vel_term                             # cube velocity toward goal
            + 0.15 * (1.0 - math.tanh(ee_cube_dist / 0.16))  # ee-cube proximity
        )
        if success:
            reward += 10.0                               # success bonus (no bonus on fall)
        return float(reward)

    def _render_camera(self, cam_name: str, width: int, height: int) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_name)
        img = self._renderer.render()
        if img.shape[:2] != (height, width):
            img = img[:height, :width]
        return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# EnvHub entry point
# ---------------------------------------------------------------------------

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: CubePushBridgeEnvConfig | None = None,
):
    """
    EnvHub-compatible factory.

    Args:
        n_envs:          number of parallel envs
        use_async_envs:  use AsyncVectorEnv (multi-process) instead of Sync
        cfg:             CubePushBridgeEnvConfig — set obs_mode="belief" for the
                         POMDP variant, override cameras/task/etc.

    Returns:
        gym.vector.VectorEnv
    """
    if cfg is None:
        cfg = CubePushBridgeEnvConfig()

    def _make():
        return SO101SingleArmCubePushBridgeEnv(cfg)

    vec_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vec_cls([_make for _ in range(n_envs)])
