"""Regression test: run_teleop() must get past the model load.

`import mujoco.viewer` inside a function binds the name `mujoco` as a local
for the *whole* function, so the `mujoco.MjModel.from_xml_path(...)` call
earlier in run_teleop() raised

    UnboundLocalError: cannot access local variable 'mujoco' where it is not
    associated with a value

before the viewer was ever reached. The script looked fine at import time and
died on its first line of work, which is exactly the kind of failure that only
shows up when someone plugs in the arm.

The function still exits when it cannot open the serial port, so the test
asserts on *which* error comes out: not UnboundLocalError.
"""

from __future__ import annotations

import pytest

import teleop_leader_arm as T


def test_run_teleop_passes_the_model_load():
    """It must fail on the serial port, not on the mujoco name binding."""
    with pytest.raises((SystemExit, Exception)) as excinfo:
        T.run_teleop("single", {None: "/dev/null"}, "no-such-calibration.json")

    assert not isinstance(excinfo.value, UnboundLocalError), (
        "run_teleop hit the local-import bug: `import mujoco.viewer` inside "
        "the function makes `mujoco` a local name for the whole function"
    )


def test_module_level_mujoco_is_not_shadowed():
    """The module-level `import mujoco` must still be reachable."""
    assert T.mujoco.__name__ == "mujoco"
    assert hasattr(T.mujoco, "MjModel")
