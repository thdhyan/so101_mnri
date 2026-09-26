#!/usr/bin/env python3
"""Record single-arm SO-101 demos to a LeRobotDataset, with the torque guard
installed, streaming the live view into the Rerun debug server.

This is a thin wrapper around `lerobot-record`: it installs
`scripts/so101_torque_guard.py` and then hands every argv straight to
`lerobot.scripts.lerobot_record.main()`. The recording loop, dataset writer and
visualisation are all unchanged -- only the follower connect path is made safe.

Use `scripts/record_so101.sh`, which builds the argv; this exists so the patch
can be applied before LeRobot touches the hardware.

    bash scripts/record_so101.sh
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_torque_guard  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("so101_torque_guard").setLevel(logging.INFO)

    retries = 4
    per_motor_delay = 0.25
    retry_delay = 0.4
    argv = sys.argv[1:]

    # Let the launcher tune the guard without a separate flag namespace.
    while "--guard-" in " ".join(argv):
        joined = " ".join(argv)
        if "--guard-retries=" in joined:
            retries = int(joined.split("--guard-retries=")[1].split()[0])
            argv = joined.replace(f"--guard-retries={retries}", "").split()
        elif "--guard-motor-delay=" in joined:
            per_motor_delay = float(joined.split("--guard-motor-delay=")[1].split()[0])
            argv = joined.replace(f"--guard-motor-delay={per_motor_delay}", "").split()
        elif "--guard-retry-delay=" in joined:
            retry_delay = float(joined.split("--guard-retry-delay=")[1].split()[0])
            argv = joined.replace(f"--guard-retry-delay={retry_delay}", "").split()
        else:
            break

    if not so101_torque_guard.install(retries, per_motor_delay, retry_delay):
        print("ERROR: could not install the torque guard; refusing to record, because "
              "an unguarded connect() can slam the follower into its limits.",
              file=sys.stderr)
        return 1

    from lerobot.scripts.lerobot_record import main as record_main

    print(f"[guard] torque guard active: retries={retries}, "
          f"per-motor delay={per_motor_delay}s, retry delay={retry_delay}s",
          flush=True)
    print("[guard] Goal_Position will be synced to Present_Position before torque is "
          "enabled, so the arm holds its current pose instead of driving to 0.",
          flush=True)

    sys.argv = [sys.argv[0], *argv]
    return record_main()


if __name__ == "__main__":
    raise SystemExit(main())
