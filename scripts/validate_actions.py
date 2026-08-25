"""
Random-action and zero-action validation for the SO-101 tasks (MuJoCo + Isaac).

Verifies for each task: reset works, obs shapes are finite, and episodes run
to completion under (a) zero actions and (b) uniform-random actions without
NaNs/blowups.

    python -m scripts.validate_actions --backend mujoco
    python -m envs.isaac.scripts.validate_actions_isaac   # Isaac variant
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MUJOCO_TASKS = {
    "single": ("envs.mujoco.so101_single_arm.env", "make_env"),
    "dual": ("envs.mujoco.so101_dual_arm.env", "make_env"),
    "pick_lift": ("envs.mujoco.so101_single_arm_pick_lift.env", "make_env"),
    "pick_place": ("envs.mujoco.so101_single_arm_pick_place.env", "make_env"),
    "cyl_grasp": ("envs.mujoco.so101_dual_arm_cylinder_grasp.env", "make_env"),
    "cyl_reach": ("envs.mujoco.so101_dual_arm_cylinder_reach.env", "make_env"),
}


def check_env(name, make_env, steps=120, seed=0):
    results = {}
    for mode, act_fn in [
        ("zero", lambda space, rng, i: np.zeros(space.shape, dtype=np.float32)),
        ("random", lambda space, rng, i: rng.uniform(space.low, space.high).astype(np.float32)),
    ]:
        env = make_env(n_envs=1)
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        finite = True
        for i in range(steps):
            action = act_fn(env.action_space, rng, i)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(np.sum(reward))
            flat = np.concatenate([np.asarray(v).ravel() for k, v in obs.items() if k != "images"])
            finite &= bool(np.isfinite(flat).all()) and np.isfinite(reward).all()
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        results[mode] = {"return": round(total, 2), "finite": finite}
        status = "OK" if finite else "NON-FINITE"
        print(f"  [{name}] {mode:6s}: return={results[mode]['return']:.2f} finite={finite} ({status})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    print("MuJoCo random/zero-action validation:")
    all_ok = True
    for name, (mod, fn) in MUJOCO_TASKS.items():
        make_env = getattr(importlib.import_module(mod), fn)
        res = check_env(name, make_env, args.steps)
        all_ok &= all(r["finite"] for r in res.values())
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
