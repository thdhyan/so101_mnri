"""The env layer's names are a contract other scripts depend on.

`AGENT_TASKS.md` is explicit about it: the joint, actuator and site names are
a stable contract, and renaming one without updating every consumer breaks
scripts that have nothing to do with the rename. The names below are the
ones `scripts/`, `envs/isaac/` and the real-robot code use.

These tests build the real MuJoCo env -- no GPU, no arms, no display -- and
check that the contract still holds and that the env actually runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SITE_NAMES = ["gripperframe"]

ENVS = ["single", "dual"]


def _make_env(env: str):
    """Build the env by name, the way the scripts do."""
    if env == "single":
        from envs.mujoco.so101_single_arm.env import make_env
    elif env == "dual":
        from envs.mujoco.so101_dual_arm.env import make_env
    else:
        raise ValueError(env)
    return make_env(n_envs=1, cfg=None)


def _inner_env(vec_env):
    """The single MuJoCo env behind the vector wrapper."""
    for attr in ("single_env", "unwrapped"):
        inner = getattr(vec_env, attr, None)
        if inner is not None and hasattr(inner, "model"):
            return inner
    envs = getattr(vec_env, "envs", None)
    if envs:
        return envs[0]
    raise AssertionError("cannot find the inner MuJoCo env")


def _arrays(obs):
    """Flatten an observation dict into (name, array) pairs."""
    for key, value in obs.items():
        if isinstance(value, dict):
            for sub, arr in value.items():
                yield f"{key}.{sub}", np.asarray(arr)
        else:
            yield key, np.asarray(value)


@pytest.fixture(scope="module", params=ENVS)
def env(request):
    e = _make_env(request.param)
    yield e
    close = getattr(e, "close", None)
    if callable(close):
        close()


def test_joint_names_are_the_contract(env):
    """The contract names, optionally `left_`/`right_` prefixed for the dual rig."""
    import mujoco

    model = _inner_env(env).model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

    for joint in JOINT_NAMES:
        matches = [n for n in names if n == joint or n.endswith(f"_{joint}")]
        assert matches, f"{joint} is missing from the model ({names})"


def test_gripper_site_exists(env):
    """`gripperframe`, or `left_gripperframe` / `right_gripperframe`."""
    import mujoco

    model = _inner_env(env).model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]

    for site in SITE_NAMES:
        matches = [n for n in names if n == site or n.endswith(f"_{site}")]
        assert matches, f"site {site!r} is missing from the model ({names})"


def test_action_space_matches_the_joint_count(env):
    """6 actuators per arm, and the action space says so."""
    action_space = env.action_space
    shape = action_space.shape
    n_arms = 1 if shape[-1] == 6 else 2
    assert shape[-1] == 6 * n_arms, f"unexpected action shape {shape}"
    assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high))


def test_reset_and_step_are_finite(env):
    obs, _ = env.reset()
    assert obs is not None
    for name, arr in _arrays(obs):
        assert np.all(np.isfinite(arr)), f"observation {name!r} is not finite after reset"

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        for name, arr in _arrays(obs):
            assert np.all(np.isfinite(arr)), f"observation {name!r} went non-finite"
        assert np.all(np.isfinite(np.asarray(reward))), "reward went non-finite"
