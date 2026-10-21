"""The calibration files must stay exactly as they are.

`robots/so101/calibration/mnri_arms_1_{leader,follower}.json` are the
**verified** calibration for this rig: the leader and the follower are
compared in degrees derived from these files, and the pose gate refuses to
start unless they agree within 10 degrees.

Re-calibrating one arm moves its zero point, so the arms stop agreeing and
the gate either blocks every session or lets a 99-degree mismatch through --
and `record_loop` calls `send_action()` on its first tick, so six servos
lunge against gravity, the rail sags and the Feetech bus starts dropping
packets. It presents as a flaky cable, not as a bug.

`scripts/calibrate_arm.py` overwrites these files **and the arm's EEPROM**,
and there is no undo except a backup directory. So this test is the tripwire:
if someone runs the calibrator, CI goes red instead of the rig going quiet.

The values below are the ones recorded on 2026-09-25 and checked against the
live arms. The follower's `shoulder_lift` is the one that was recorded over
the full 0..4095 raw span when the joint really travels 756..3097.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# (range_min, range_max, homing_offset) per joint, as verified on 2026-09-25.
VERIFIED = {
    "leader": {
        "shoulder_pan": (822, 3529, 1469),
        "shoulder_lift": (877, 3327, -1650),
        "elbow_flex": (800, 3072, 1151),
        "wrist_flex": (765, 3136, -1934),
        "wrist_roll": (0, 4095, 1281),
        "gripper": (1818, 3186, -1477),
    },
    "follower": {
        "shoulder_pan": (782, 3495, -1263),
        "shoulder_lift": (756, 3097, -1983),
        "elbow_flex": (817, 3095, 1613),
        "wrist_flex": (833, 3213, -2032),
        "wrist_roll": (0, 4095, 1623),
        "gripper": (2005, 3541, 860),
    },
}

ARMS = list(VERIFIED)


def _load(arm: str) -> dict:
    path = CALIB_DIR / f"mnri_arms_1_{arm}.json"
    assert path.exists(), f"missing calibration file: {path}"
    with path.open() as fh:
        return json.load(fh)


@pytest.mark.parametrize("arm", ARMS)
def test_calibration_file_exists(arm):
    assert (CALIB_DIR / f"mnri_arms_1_{arm}.json").is_file()


@pytest.mark.parametrize("arm", ARMS)
def test_calibration_has_every_joint(arm):
    data = _load(arm)
    assert sorted(data) == sorted(JOINTS), (
        f"{arm}: the calibration must cover exactly {JOINTS}, got {sorted(data)}"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_joint_ranges_are_unchanged(arm):
    """The tripwire: a recalibration shows up here."""
    data = _load(arm)
    for joint, (rmin, rmax, _) in VERIFIED[arm].items():
        entry = data[joint]
        assert (entry["range_min"], entry["range_max"]) == (rmin, rmax), (
            f"{arm}/{joint}: range is now "
            f"({entry['range_min']}, {entry['range_max']}), the verified range is "
            f"({rmin}, {rmax}). If a joint really is at a limit it should not be "
            f"at, READ it with `scripts/check_joint_range.py --arm {arm} "
            f"--compare` and correct range_min / range_max by hand -- do not "
            f"recalibrate."
        )


@pytest.mark.parametrize("arm", ARMS)
def test_homing_offsets_are_unchanged(arm):
    """The zero point. Moving one is what makes the arms disagree in degrees."""
    data = _load(arm)
    for joint, (_, _, homing) in VERIFIED[arm].items():
        assert data[joint]["homing_offset"] == homing, (
            f"{arm}/{joint}: homing_offset is now {data[joint]['homing_offset']}, "
            f"the verified value is {homing}. A different zero point is what a "
            f"recalibration does -- do not recalibrate."
        )


@pytest.mark.parametrize("arm", ARMS)
def test_ranges_are_inside_the_raw_span(arm):
    """A range wider than the real travel is what makes a joint read pinned."""
    data = _load(arm)
    for joint, entry in data.items():
        assert 0 <= entry["range_min"] < entry["range_max"] <= 4095, (
            f"{arm}/{joint}: range ({entry['range_min']}, {entry['range_max']}) "
            f"is not a sane window inside the 0..4095 raw span"
        )


@pytest.mark.parametrize("arm", ARMS)
def test_calibration_is_json_serialisable_and_stable(arm):
    """The file the recorder reads must be plain JSON with ints, not floats."""
    data = _load(arm)
    for joint, entry in data.items():
        for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
            assert isinstance(entry[key], int), (
                f"{arm}/{joint}: {key} is {type(entry[key]).__name__}, expected int"
            )
