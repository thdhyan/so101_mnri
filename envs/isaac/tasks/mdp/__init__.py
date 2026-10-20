"""
Custom MDP terms for the SO-101 Isaac Lab tasks.

Ported from the reward/termination logic of the MuJoCo gymnasium envs
(`envs/mujoco/*/env.py`) and the design in MJLAB_INTEGRATION.md:
staged reach→grasp→(lift|place) dense rewards, a dominant per-step
``alive`` penalty (reward-hack resistance), and two termination paths —
success (within tolerance) or timeout.

Conventions:
    - Position tolerances: 1 cm; orientation tolerance: 10 deg.
    - Success terms are also exposed as termination terms so episodes end
      on success (``terminated``) or timeout (``truncated``).
"""

from . import observations, rewards, terminations  # noqa: F401
