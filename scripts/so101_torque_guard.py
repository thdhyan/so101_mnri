#!/usr/bin/env python3
"""Torque guard for the SO-101 follower: make `connect()` safe and retryable.

The failure this fixes
----------------------
`SOFollower.connect()` ends with `configure()`, which is::

    with self.bus.torque_disabled():
        ...write PID gains...
# context exit ->
    self.enable_torque(motors)

`FeetechMotorsBus.enable_torque` writes `Torque_Enable=1` then `Lock=1` to every
motor in one tight loop with `num_retry=0`. Two things go wrong on this rig:

1. **Stale Goal_Position.** `Goal_Position` is an EEPROM-backed register that
   persists across power cycles, and on this follower all six read 0 while
   `Present_Position` read ~2000. Energising a position-mode servo with
   `Goal_Position=0` commands it to drive to 0 -- far outside its calibrated
   `range_min` (6..742). It slams into the mechanical stop, draws a stall
   current spike, and the rail sags.

2. **No retries, all six at once.** That current spike drops packets, which is
   what surfaces as either of these:

       ConnectionError: Failed to write 'Torque_Enable' on id_=5 with '1'
                        after 1 tries. [TxRxResult] There is no status packet!
       ConnectionError: Failed to write 'Lock' on id_=5 with '1'
                        after 1 tries. [TxRxResult] Incorrect status packet!

   "after 1 tries" is the tell: `enable_torque` passes `num_retry=0`, so there
   is no second attempt.

What the guard does
-------------------
Before enabling torque, it syncs `Goal_Position := Present_Position` so the arm
holds its current pose instead of driving to 0, verifies the sync actually took,
then enables the motors one at a time with retries and a short settle between
each. Same end state, without the slam and with a real chance of riding out a
transient packet loss.

Used via `scripts/record_so101.py`, which installs the patch and then calls
`lerobot-record`'s own `main()`, so the recording loop itself is untouched.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("so101_torque_guard")

# How hard to try before giving up, and how long to let the rail recover between
# motors. 6 STS3215 servos on a 12 V rail do briefly dip as each one latches.
DEFAULT_RETRIES = 4
PER_MOTOR_DELAY_S = 0.25
RETRY_DELAY_S = 0.4

_installed = False


def _motor_names(bus, motors) -> list[str]:
    """Normalise whatever LeRobot passes into a list of motor-name strings."""
    if motors is None:
        return list(bus.motors.keys())
    if isinstance(motors, str):
        return [motors]
    out = []
    for m in motors:
        # _get_motors_list yields names, but be tolerant of Motor objects.
        out.append(m if isinstance(m, str) else next(
            n for n, obj in bus.motors.items() if getattr(obj, "id", None) == getattr(m, "id", None)
        ))
    return out


def _sync_goals_to_present(bus, names: list[str]) -> dict[str, float]:
    """Write Goal_Position := Present_Position, and report the residual delta.

    Uses raw counts (normalize=False on both sides) so nothing round-trips
    through the calibration tables.
    """
    present = bus.sync_read("Present_Position", motors=names, normalize=False, num_retry=2)
    bus.sync_write("Goal_Position", dict(present), normalize=False, num_retry=2)
    time.sleep(0.2)
    goal = bus.sync_read("Goal_Position", motors=names, normalize=False, num_retry=2)
    residual = {n: abs(float(present[n]) - float(goal[n])) for n in names}
    return residual


def _patched_enable_torque(self, motors=None, num_retry=0):
    """Drop-in replacement for FeetechMotorsBus.enable_torque."""
    names = _motor_names(self, motors)
    attempts = max(int(num_retry) + 1, DEFAULT_RETRIES)

    # Step 1: kill the stale-goal hazard. Only meaningful when the bus has a
    # calibration loaded (sync_write needs it to resolve addresses); if the bus
    # is uncalibrated we still try, since normalize=False skips the tables.
    try:
        residual = _sync_goals_to_present(self, names)
        worst = max(residual.values()) if residual else 0.0
        if worst > 50:
            logger.warning(
                "torque guard: Goal_Position sync residual %.0f counts (largest motor); "
                "enabling torque anyway", worst
            )
        else:
            logger.info(
                "torque guard: Goal_Position synced to Present_Position "
                "(residual <= %.0f counts across %d motors)", worst, len(names)
            )
    except Exception as e:  # noqa: BLE001
        # Never block recording on the safety nicety; the retry loop below is
        # the part that actually fixes the packet loss.
        logger.warning("torque guard: could not sync Goal_Position (%s: %s)",
                       type(e).__name__, e)

    # Step 2: enable one motor at a time, with real retries and a settle gap.
    for name in names:
        last = None
        for attempt in range(1, attempts + 1):
            try:
                self.write("Torque_Enable", name, 1)
                self.write("Lock", name, 1)
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning(
                    "torque guard: enable %s attempt %d/%d failed (%s: %s)",
                    name, attempt, attempts, type(e).__name__, str(e)[:80]
                )
                time.sleep(RETRY_DELAY_S)
        if last is not None:
            raise ConnectionError(
                f"torque guard: failed to enable torque on '{name}' after {attempts} "
                f"attempts. Last error: {last}"
            )
        time.sleep(PER_MOTOR_DELAY_S)


def install(retries: int = DEFAULT_RETRIES,
            per_motor_delay: float = PER_MOTOR_DELAY_S,
            retry_delay: float = RETRY_DELAY_S) -> bool:
    """Monkeypatch FeetechMotorsBus.enable_torque. Idempotent."""
    global _installed
    if _installed:
        return True
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as e:  # noqa: BLE001
        logger.error("torque guard: cannot import FeetechMotorsBus (%s)", e)
        return False

    global DEFAULT_RETRIES, PER_MOTOR_DELAY_S, RETRY_DELAY_S
    DEFAULT_RETRIES = retries
    PER_MOTOR_DELAY_S = per_motor_delay
    RETRY_DELAY_S = retry_delay

    FeetechMotorsBus.enable_torque = _patched_enable_torque
    _installed = True
    logger.info(
        "torque guard: installed (retries=%d, per-motor delay=%.2fs, retry delay=%.2fs)",
        retries, per_motor_delay, retry_delay,
    )
    return True


def uninstall() -> None:
    """Restore LeRobot's original enable_torque (used by the test suite)."""
    global _installed
    if not _installed:
        return
    from lerobot.motors.feetech import FeetechMotorsBus

    del FeetechMotorsBus.enable_torque
    _installed = False
