#!/usr/bin/env python3
"""Validate a LeRobot record configuration WITHOUT touching hardware.

Parses the exact same draccus config `lerobot-record` would, then prints the
resolved robot / teleoperator / camera settings and the per-camera V4L2 modes
that are actually supported by the devices. Catches typos, bad camera names and
unsupported resolutions before you move an arm.

    python scripts/check_record_config.py
    python scripts/check_record_config.py --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "robots" / "so101" / "calibration"
ENV_FILE = Path(__file__).resolve().parent / "so101_env.sh"


def env_defaults() -> dict[str, str]:
    """Read the hardware assignment out of scripts/so101_env.sh.

    Keeps this checker and the recorder pointed at the same hardware instead of
    duplicating the port list in two places.
    """
    if not ENV_FILE.is_file():
        return {}
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        # Strip trailing comments and quotes.
        val = val.split("#", 1)[0].strip().strip("\"'")
        out[key.strip()] = val
    return out


def supported_modes(dev: int, w: int, h: int) -> str:
    """Ask v4l2 whether (w,h) is a real mode for this device."""
    proc = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{dev}", "--list-formats-ext"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return "could not query"
    size = f"Size: Discrete {w}x{h}"
    lines = proc.stdout.splitlines()
    for i, ln in enumerate(lines):
        if size in ln:
            fmt = next((x.strip() for x in lines[i + 1:i + 4] if "Pixel Format" in x), "?")
            return f"supported ({fmt})"
    return f"NOT in supported list (requested {w}x{h})"


def supported_fps(dev: int, w: int, h: int) -> list[float]:
    """Frame rates the device actually offers at (w,h).

    Worth checking separately from the size: both SO-101 cameras expose exactly
    one interval at 640x480 (0.033s / 30 fps), and OpenCVCamera._validate_fps
    compares with math.isclose(..., rel_tol=1e-3), so asking for anything else
    raises at connect time.
    """
    proc = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{dev}", "--list-formats-ext"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return []
    out: list[float] = []
    keep = False
    for ln in proc.stdout.splitlines():
        if f"Size: Discrete {w}x{h}" in ln:
            keep = True
            continue
        if not keep:
            continue
        if "Interval: Discrete" in ln:
            # "Interval: Discrete 0.033s (30.000 fps)" -- the value inside the
            # parens is ALREADY the frame rate, so use it directly.
            try:
                out.append(round(float(ln.split("(")[1].split()[0].rstrip(")")), 3))
            except (IndexError, ValueError):
                pass
        elif "Size: Discrete" in ln:
            break
    return sorted(set(out))


def main() -> int:
    env = env_defaults()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leader-serial", default=env.get("LEADER_SERIAL", "5AAF263743"))
    ap.add_argument("--follower-serial", default=env.get("FOLLOWER_SERIAL", "5AAF263600"))
    ap.add_argument("--leader-port", default=None)
    ap.add_argument("--follower-port", default=None)
    ap.add_argument("--leader-id", default=env.get("LEADER_ID", "mnri_arms_1_leader"))
    ap.add_argument("--follower-id", default=env.get("FOLLOWER_ID", "mnri_arms_1_follower"))
    ap.add_argument("--global-cam", type=int, default=int(env.get("GLOBAL_CAM", 4)))
    ap.add_argument("--wrist-cam", type=int, default=int(env.get("WRIST_CAM", 2)))
    ap.add_argument("--width", type=int, default=int(env.get("WIDTH", 1280)))
    ap.add_argument("--height", type=int, default=int(env.get("HEIGHT", 720)))
    ap.add_argument("--fps", type=int, default=int(env.get("FPS", 15)))
    ap.add_argument("--camera-fps", type=int, default=int(env.get("CAMERA_FPS", 30)))
    ap.add_argument("--fourcc", default=env.get("FOURCC", "MJPG"))
    ap.add_argument("--warmup-s", type=int, default=int(env.get("WARMUP_S", 5)))
    ap.add_argument("--repo-id", default=env.get("REPO_ID", "thakk100/so101_single_arm_6dof"))
    ap.add_argument("--deep", action="store_true",
                    help="actually open both cameras through LeRobot's OpenCVCamera, "
                         "in the same order the recorder uses. Catches warmup/timeout "
                         "failures here instead of after the arms are live.")
    args = ap.parse_args()

    print("=" * 68)
    print(f"LeRobot record config check (no hardware touched)  [{ENV_FILE.name}]")
    print("=" * 68)

    # --- resolve arms by USB serial, never by path ------------------------
    # Enumeration order reverses on replug; a stale path writes one arm's
    # calibration into the other arm's servos.
    def port_for(serial: str) -> str | None:
        for dev in sorted(Path("/dev").glob("ttyACM*")):
            proc = subprocess.run(["udevadm", "info", "--query=property", "--name", str(dev)],
                                  capture_output=True, text=True, check=False)
            got = next((ln.split("=", 1)[1] for ln in proc.stdout.splitlines()
                        if ln.startswith("ID_SERIAL_SHORT=")), None)
            if got == serial:
                return str(dev)
        return None

    lp = args.leader_port or port_for(args.leader_serial)
    fp = args.follower_port or port_for(args.follower_serial)
    args.leader_port, args.follower_port = lp, fp

    if lp and fp:
        print(f"[ok]   leader   serial {args.leader_serial} -> {lp}")
        print(f"[ok]   follower serial {args.follower_serial} -> {fp}")
    else:
        for role, want_ser, got in (("leader", args.leader_serial, lp),
                                    ("follower", args.follower_serial, fp)):
            if got is None:
                print(f"[FAIL] {role}: no /dev/ttyACM* with serial {want_ser}")
                ok = False
        for dev in sorted(Path("/dev").glob("ttyACM*")):
            proc = subprocess.run(["udevadm", "info", "--query=property", "--name", str(dev)],
                                  capture_output=True, text=True, check=False)
            got = next((ln.split("=", 1)[1] for ln in proc.stdout.splitlines()
                        if ln.startswith("ID_SERIAL_SHORT=")), "?")
            print(f"       present: {dev}  serial={got}")
    print()

    # --- ports exist? ---
    ok = True
    for role, port in (("leader", args.leader_port), ("follower", args.follower_port)):
        p = Path(port)
        if p.exists():
            print(f"[ok]   {role:8s} port {port} exists")
        else:
            print(f"[FAIL] {role:8s} port {port} DOES NOT EXIST")
            ok = False
    if args.leader_port == args.follower_port:
        print("[FAIL] leader and follower are the same port")
        ok = False

    # --- calibration files ---
    for role, aid in (("leader", args.leader_id), ("follower", args.follower_id)):
        f = CALIB_DIR / f"{aid}.json"
        if f.is_file():
            n = len(json.load(open(f)))
            print(f"[ok]   {role:8s} calibration {f.relative_to(REPO_ROOT)} ({n} motors)")
        else:
            print(f"[WARN] {role:8s} no calibration for id '{aid}' -> LeRobot will offer to calibrate")
    if not CALIB_DIR.is_dir():
        print(f"[FAIL] calibration dir missing: {CALIB_DIR}")

    # --- cameras ---
    for role, dev in (("global", args.global_cam), ("wrist", args.wrist_cam)):
        p = Path(f"/dev/video{dev}")
        if not p.exists():
            print(f"[FAIL] {role:8s} /dev/video{dev} does not exist")
            ok = False
            continue
        status = supported_modes(dev, args.width, args.height)
        flag = "FAIL" if "NOT in" in status else "ok"
        rates = supported_fps(dev, args.width, args.height)
        if rates:
            if args.camera_fps in rates:
                rate_line = f"{args.camera_fps} fps supported (of {rates})"
            else:
                rate_line = (f"CAMERA_FPS={args.camera_fps} NOT supported; "
                             f"device offers {rates} -> connect() will raise")
                flag = "FAIL"
        else:
            rate_line = "could not query frame intervals"
        print(f"[{flag}]   {role:8s} /dev/video{dev} {args.width}x{args.height} "
              f"{args.fourcc}: {status}; {rate_line}")
        print(f"           (dataset records at {args.fps} fps, sampled from the "
              f"{args.camera_fps} fps stream)")

    # --- does LeRobot accept this exact config? ---
    print("\n--- draccus parse ---")
    # CameraConfig is a draccus ChoiceRegistry, so every entry needs its
    # "type" discriminator ("opencv" is the registered name for OpenCVCameraConfig).
    cameras = {
        role: {
            "type": "opencv",
            "index_or_path": dev,
            "fps": args.camera_fps,
            "width": args.width,
            "height": args.height,
            "fourcc": args.fourcc,
            "warmup_s": args.warmup_s,
        }
        for role, dev in (("global", args.global_cam), ("wrist", args.wrist_cam))
    }
    argv = [
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.id={args.follower_id}",
        f"--robot.calibration_dir={CALIB_DIR}",
        f"--robot.cameras={json.dumps(cameras)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={args.leader_port}",
        f"--teleop.id={args.leader_id}",
        f"--teleop.calibration_dir={CALIB_DIR}",
        "--dataset.repo_id=" + args.repo_id,
        "--dataset.single_task=push the duck into the square tape",
        "--dataset.root=data",
        f"--dataset.fps={args.fps}",
        "--display_data=true",
    ]
    try:
        import draccus
        from lerobot.scripts.lerobot_record import RecordConfig

        cfg = draccus.parse(config_class=RecordConfig, args=argv)
        print("[ok]   LeRobot parsed the config")
        print(f"       robot  type={cfg.robot.type} port={cfg.robot.port} id={cfg.robot.id}")
        print(f"       teleop type={cfg.teleop.type} port={cfg.teleop.port} id={cfg.teleop.id}")
        print(f"       calibration_dir={cfg.robot.calibration_dir}")
        print(f"       cameras: {', '.join(cfg.robot.cameras)}")
        for name, cam in cfg.robot.cameras.items():
            print(f"         {name:8s} index={cam.index_or_path} {cam.width}x{cam.height}"
                  f" @{cam.fps}fps fourcc={cam.fourcc} mode={cam.color_mode.value}")
        print(f"       dataset {cfg.dataset.repo_id} @ {cfg.dataset.fps}fps")
        print(f"       display_data={cfg.display_data} -> {cfg.display_mode} "
              f"ip={cfg.display_ip} port={cfg.display_port}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] LeRobot rejected the config: {type(e).__name__}: {e}")
        print("       argv:", " ".join(argv))
        ok = False

    # --- optional: really open the cameras ---------------------------------
    if args.deep:
        print("\n--- live camera connect (LeRobot OpenCVCamera) ---")
        try:
            import time

            from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        except ImportError as e:
            print(f"[SKIP] cannot import lerobot camera classes: {e}")
            return ok

        opened = []
        # Same order as SOFollower.connect(): dict order, global then wrist.
        for role, dev in (("global", args.global_cam), ("wrist", args.wrist_cam)):
            if not Path(f"/dev/video{dev}").exists():
                print(f"[FAIL] {role}: /dev/video{dev} missing")
                ok = False
                continue
            cfg = OpenCVCameraConfig(
                index_or_path=dev, fps=args.fps, width=args.width, height=args.height,
                fourcc=args.fourcc, warmup_s=args.warmup_s,
            )
            cam = OpenCVCamera(cfg)
            t0 = time.perf_counter()
            try:
                cam.connect()
                dt = time.perf_counter() - t0
                opened.append(cam)
                print(f"[ok]   {role:6s} /dev/video{dev} connected in {dt:.2f}s")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[FAIL] {role:6s} /dev/video{dev} after {time.perf_counter() - t0:.2f}s")
                print(f"       {type(e).__name__}: {e}")
                break
        for cam in opened:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if ok and len(opened) == 2:
            print("       both cameras open simultaneously OK")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
