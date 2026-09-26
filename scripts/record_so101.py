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
    # This MUST operate on the list. Rejoining argv into one string and calling
    # .split() destroys quoting, which turns
    #   --dataset.single_task=push the duck into the square tape
    # from one argument into six, and draccus then rejects the extras.
    guard_opts: dict[str, float] = {
        "--guard-retries": retries,
        "--guard-motor-delay": per_motor_delay,
        "--guard-retry-delay": retry_delay,
    }
    cleaned: list[str] = []
    for arg in argv:
        matched = next((k for k in guard_opts if arg.startswith(k + "=")), None)
        if matched is None:
            cleaned.append(arg)
            continue
        try:
            guard_opts[matched] = float(arg.split("=", 1)[1])
        except ValueError:
            print(f"ERROR: {matched} needs a number, got {arg!r}", file=sys.stderr)
            return 2
    argv = cleaned
    retries = int(guard_opts["--guard-retries"])
    per_motor_delay = guard_opts["--guard-motor-delay"]
    retry_delay = guard_opts["--guard-retry-delay"]

    if not so101_torque_guard.install(retries, per_motor_delay, retry_delay):
        print("ERROR: could not install the torque guard; refusing to record, because "
              "an unguarded connect() can slam the follower into its limits.",
              file=sys.stderr)
        return 1

    print(f"[guard] torque guard active: retries={retries}, "
          f"per-motor delay={per_motor_delay}s, retry delay={retry_delay}s",
          flush=True)
    print("[guard] Goal_Position will be synced to Present_Position before torque is "
          "enabled, so the arm holds its current pose instead of driving to 0.",
          flush=True)

    # --dry-run: prove the guard installed and LeRobot accepts the argv, then
    # stop. Touches no hardware, so this is safe to run any time.
    if "--dry-run" in argv:
        argv = [a for a in argv if a != "--dry-run"]
        try:
            import draccus
            from lerobot.scripts.lerobot_record import RecordConfig

            cfg = draccus.parse(config_class=RecordConfig, args=argv)
        except SystemExit as e:
            print(f"\n[dry-run] LeRobot rejected the config (exit {e.code})", file=sys.stderr)
            return int(e.code or 1)
        except Exception as e:  # noqa: BLE001
            print(f"\n[dry-run] LeRobot rejected the config: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        print(f"\n[dry-run] argv accepted: {len(argv)} args after guard flags stripped")
        print(f"[dry-run] robot  {cfg.robot.type} port={cfg.robot.port} id={cfg.robot.id}")
        print(f"[dry-run] teleop {cfg.teleop.type} port={cfg.teleop.port} id={cfg.teleop.id}")
        print(f"[dry-run] cameras {list(cfg.robot.cameras)}")
        for n, cam in cfg.robot.cameras.items():
            print(f"[dry-run]   {n}: /dev/video{cam.index_or_path} {cam.width}x{cam.height} "
                  f"@{cam.fps} {cam.fourcc} {cam.color_mode.value} warmup={cam.warmup_s}s")
        print(f"[dry-run] dataset {cfg.dataset.repo_id} root={cfg.dataset.root} "
              f"fps={cfg.dataset.fps} task={cfg.dataset.single_task!r}")
        print(f"[dry-run] resume={cfg.resume} display={cfg.display_mode} "
              f"{cfg.display_ip}:{cfg.display_port}")
        print("[dry-run] no hardware touched")
        return 0

    from lerobot.scripts.lerobot_record import main as record_main

    sys.argv = [sys.argv[0], *argv]
    return record_main()


if __name__ == "__main__":
    raise SystemExit(main())
