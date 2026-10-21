#!/usr/bin/env python
"""List and live-test SDL gamepads (DualShock 4 etc.) via pygame 2.x.

Demo-day pairing check: run it, wiggle sticks / pull triggers / mash buttons,
and watch the values move. Exits 0 even with no controller attached (prints
setup guidance instead), so it is safe for scripted validation.

Usage:
    python scripts/check_gamepad.py                 # list devices, then live-view first
    python scripts/check_gamepad.py --device 1      # live-view a specific index
    python scripts/check_gamepad.py --list-only     # just enumerate and exit
    python scripts/check_gamepad.py --seconds 5     # auto-exit after N seconds
"""
import argparse
import os
import sys
import time

# Joystick input needs no display; keep this runnable headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

NO_PAD_HELP = """No gamepad detected. Checklist:
  1. Plug the DualShock 4 in over USB, or pair Bluetooth: hold SHARE + PS
     until the light bar double-flashes, connect "Wireless Controller".
  2. Permission: you must be in the 'input' group to read /dev/input --
     `groups` to check, `sudo usermod -aG input $USER` + re-login to fix.
  3. Kernel driver: `sudo modprobe joydev`, then replug / re-pair.
  4. Still nothing? Check `ls -l /dev/input/js* /dev/input/event*` exist."""

# Raw DS4-on-Linux button names for display fallback (see teleop_gamepad_ik.py)
RAW_BUTTON_NAMES = {
    0: "square", 1: "cross", 2: "circle", 3: "triangle",
    4: "l1", 5: "r1", 8: "share", 9: "options",
    10: "l3", 11: "r3", 13: "ps?", 17: "ps",
}


def describe_devices():
    n = pygame.joystick.get_count()
    print(f"SDL {pygame.get_sdl_version()} | pygame {pygame.version.ver} | "
          f"{n} joystick(s) detected")
    devs = []
    for i in range(n):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        guid = joy.get_guid()
        sony = " [Sony/DS4 vendor]" if "4c05" in (guid or "") else ""
        print(f"  [{i}] {joy.get_name()}{sony}")
        print(f"      guid={guid} axes={joy.get_numaxes()} "
              f"buttons={joy.get_numbuttons()} hats={joy.get_numhats()}")
        devs.append((i, joy))
    return devs


def axis_labels(joy):
    """Labeled axes when the layout matches a raw DS4, else AX0..AXn."""
    if joy.get_numaxes() == 6:
        return ["LX", "LY", "RX", "L2", "R2", "RY"]
    return [f"AX{i}" for i in range(joy.get_numaxes())]


def live_view(index, hz=20.0, seconds=None):
    joy = pygame.joystick.Joystick(index)
    joy.init()
    labels = axis_labels(joy)
    print(f"\nLive view of [{index}] {joy.get_name()} -- move sticks/triggers/"
          f"buttons; Ctrl+C to quit.\n")
    t_end = time.time() + seconds if seconds else None
    try:
        while True:
            pygame.event.pump()
            parts = []
            vals = []
            for lbl, a in zip(labels, range(joy.get_numaxes())):
                v = joy.get_axis(a)
                vals.append(v)
                parts.append(f"{lbl} {v:+.2f}")
            pressed = []
            for b in range(joy.get_numbuttons()):
                if joy.get_button(b):
                    name = RAW_BUTTON_NAMES.get(b, f"B{b}")
                    pressed.append(name)
            for h in range(joy.get_numhats()):
                dx, dy = joy.get_hat(h)
                if (dx, dy) != (0, 0):
                    pressed.append(f"hat{h}{(dx, dy)}")
            line = " | ".join(parts) + f" | btns: {','.join(pressed) or '-':<24}"
            sys.stdout.write("\r" + line + "        ")
            sys.stdout.flush()
            if t_end and time.time() >= t_end:
                break
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    except pygame.error as e:
        print(f"\n[gamepad] device error ({e}) -- unplugged? Re-run to retry.")
        return
    print("\ndone.")


def main():
    parser = argparse.ArgumentParser(description="Gamepad listing / live tester (pygame SDL)")
    parser.add_argument("--device", type=int, default=None,
                        help="joystick index to live-view (default: first found)")
    parser.add_argument("--list-only", action="store_true", help="enumerate devices and exit")
    parser.add_argument("--hz", type=float, default=20.0, help="live refresh rate")
    parser.add_argument("--seconds", type=float, default=None,
                        help="auto-exit after this many seconds (for scripting)")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    devs = describe_devices()

    if not devs:
        print("\n" + NO_PAD_HELP)
        return
    if args.list_only:
        return

    idx = args.device if args.device is not None else devs[0][0]
    if args.device is not None and args.device >= len(devs):
        print(f"error: --device {args.device} out of range (found {len(devs)})")
        sys.exit(1)

    # keep the dummy video driver quiet; joystick events flow without a window
    try:
        pygame.display.init()
    except pygame.error:
        pass
    live_view(idx, hz=args.hz, seconds=args.seconds)


if __name__ == "__main__":
    main()
