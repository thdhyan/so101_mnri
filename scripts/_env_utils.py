"""Shared helpers for locating SO-101 MuJoCo scene files and resolving
environment-specific names (joints/actuators/sites) dynamically from the
compiled model rather than hardcoding indices.
"""
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

SCENE_PATHS = {
    "single": os.path.join(
        _REPO_ROOT, "envs/mujoco/so101_single_arm/assets/scene.xml"
    ),
    "dual": os.path.join(
        _REPO_ROOT, "envs/mujoco/so101_dual_arm/assets/scene.xml"
    ),
}

# Joint order matches the physical SO-101 leader/follower arm (6 DoF incl. gripper).
JOINT_SUFFIXES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

HOME_POSE = np.array([0.0, -0.5, 0.8, 0.4, 0.0, 0.0])


def scene_path(env):
    if env not in SCENE_PATHS:
        raise ValueError(f"Unknown env '{env}', expected one of {list(SCENE_PATHS)}")
    path = SCENE_PATHS[env]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene file not found: {path}")
    return path


def arm_prefix(env, arm):
    """Return the joint/actuator name prefix for the given env/arm selection."""
    if env == "single":
        return ""
    if arm not in ("left", "right"):
        raise ValueError("--arm must be 'left' or 'right' for env='dual'")
    return f"{arm}_"


def joint_names(env, arm=None):
    prefix = arm_prefix(env, arm)
    return [f"{prefix}{s}" for s in JOINT_SUFFIXES]


def gripper_site_name(env, arm=None):
    prefix = arm_prefix(env, arm)
    return f"{prefix}gripperframe"


def resolve_actuator_ids(model, names):
    """Look up actuator ids by name, discovered dynamically from the model."""
    ids = []
    for n in names:
        aid = model.actuator(n).id
        ids.append(aid)
    return np.array(ids, dtype=int)


def resolve_joint_ids(model, names):
    ids = []
    for n in names:
        jid = model.joint(n).id
        ids.append(jid)
    return np.array(ids, dtype=int)
