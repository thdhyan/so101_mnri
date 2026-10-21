# VR Teleop — Meta Quest 3 → SO-101 Isaac Sim (demo-day guide)

Wired path: **`isaacteleop` + CloudXR (bundled Monado runtime) + WebXR browser
client on the Quest 3**. No APK sideload, no Quest Link cable, no lerobot
source clone. The headset connects over Wi-Fi to this laptop with the stock
Meta Quest Browser.

Script: `scripts/isaac_teleop_vr.py`
Task: `SO101-CylReach-Single-v0` (default, single arm)

---

## Why this path (decision record)

| Option | Status in this venv | Verdict |
|---|---|---|
| **(a) `isaaclab_teleop.IsaacTeleopDevice` + isaacteleop CloudXR** | installed (`isaacteleop==1.3.131[retargeters-lite]`, `isaacsim.kit.xr.teleop.bridge` ext present). Wheel ships SO-101-specific retargeters (`SO101ClutchRetargeter`, `SO101GripperRetargeter`) and the native CloudXR/Monado runtime under `isaacteleop/cloudxr/native/`. | **CHOSEN.** Everything needed is already on disk; Quest client is a browser page, not an APK. |
| (b) `isaaclab.devices.openxr` | **Deprecated** in this install: the package docstring says it "moved to `isaaclab_teleop.deprecated.openxr`". | Dead end — do not build on it. |
| lerobot source example (`examples/isaac_teleop_to_so101/`) | Would require cloning lerobot + `pip install -e ".[feetech,kinematics,dataset]"`, replacing the pip lerobot. | Not needed: we drive **our** Isaac envs directly; the SO101 retargeters it would use are already in the installed wheel. |

Data flow (all verified against the installed wheels):

```
Quest 3 browser (WebXR client)  --WebRTC video + controller poses-->
CloudXR runtime (Monado, bundled) + WSS proxy :48322 (TLS)
--> Isaac Sim Kit OpenXR session (--xr; auto-starts headless)
--> isaacsim.kit.xr.teleop.bridge -> OpenXRSessionHandles
--> TeleopSession retargeting pipeline:
      ControllersSource(right)
      -> ControllerTransform( world_T_anchor = base_link_T_world )   [rebase into robot base frame]
      -> SO101ClutchRetargeter   : absolute EE pose [x y z qx qy qz qw]   (clutched delta)
      -> SO101GripperRetargeter  : jaw closedness c in [0,1]              (analog trigger)
      -> TensorReorderer         : flat 8D action [x y z qx qy qz qw c]
--> script: DifferentialIKController(pose, absolute, dls) -> 5 arm joints
            c affine -> gripper joint [-0.1745 .. 1.7453 rad]
    => 6D absolute JointPositionAction  (env contract: preserve_order, scale=1, offset=0)
```

Right-controller controls:

| Input | Effect |
|---|---|
| Controller pose | EE target (delta around the origin captured when teleop starts / re-centers) |
| Trigger (analog) | Jaw closedness — half press = half closed |
| Grip (squeeze) | Re-center clutch at current controller pose (arm stays put) |
| A button | Toggle XR anchor rotation (built into `IsaacTeleopDevice`) |

Validate without hardware anytime:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device --steps 100
# headless GPU run of the real retargeting+IK pipeline with a scripted trajectory; prints PASS
```

---

## 0. One-time setup (do TONIGHT, not demo morning)

All on the laptop, from the repo root, venv active.

1. **Accept the CloudXR EULA once** (writes `~/.cloudxr/run/eula_accepted`;
   the embedded launcher cannot prompt interactively):

   ```bash
   source .venv/bin/activate
   CXR_INSTALL_DIR=~/.cloudxr python -c \
     "from isaacteleop.cloudxr.runtime import check_eula; check_eula(accept_eula=True); print('EULA marker OK')"
   ```

   Then smoke-test the bundled runtime end-to-end (verified working on this
   machine: ready in ~10 s):

   ```bash
   timeout 40 python -m isaacteleop.cloudxr
   # expect within ~15 s: readiness sentinel appears and the process keeps running;
   # timeout kills it — that is fine. Re-run anytime to re-verify.
   # If you see "failed to start within 30s", just retry once (stale-pid cleanup race).
   ```

2. **Pre-fetch the pinned web client for local hosting** (needs laptop
   internet; makes demo day fully LAN-independent). The origin is
   version-pinned to the installed isaacteleop (`v1.3.131` here):

   ```bash
   ORIGIN=$(python -c "from isaacteleop.cloudxr.oob_teleop_env import default_web_client_origin; print(default_web_client_origin())")
   # -> https://nvidia.github.io/IsaacTeleop/client/v1.3.131/
   mkdir -p ~/.cloudxr/static-client
   curl -fsSL -o ~/.cloudxr/static-client/index.html "$ORIGIN/index.html"
   curl -fsSL -o ~/.cloudxr/static-client/bundle.js   "$ORIGIN/bundle.js"
   ls -la ~/.cloudxr/static-client/    # expect index.html + bundle.js (~10 MB)
   ```

   (Verified working on this machine via curl; the wheel's internal Python
   downloader can hang behind some proxy setups — curl avoids that.)

3. **Find the laptop's LAN IP and free the ports** (Quest must reach the
   laptop over Wi-Fi):

   ```bash
   hostname -I        # note the first address, e.g. 192.168.1.42
   sudo ufw allow 48322/tcp   # WSS proxy + web client (skip if ufw inactive)
   sudo ufw allow 49100/tcp && sudo ufw allow 49100/udp   # CloudXR signaling/media
   ```

4. **Put the Quest on the SAME Wi-Fi network.** Hotspot mode on this laptop
   also works if the venue Wi-Fi blocks peer-to-peer traffic.

5. **Smoke-test the whole chain minus the headset**:

   ```bash
   nvidia-smi && free -g          # need ~4 GB GPU + RAM headroom (see test.md)
   OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device
   ```

---

## Demo-day sequence

### Laptop side (Isaac)

Terminal 1 — CloudXR runtime + WSS proxy serving the local web client:

```bash
cd /Storage/Projects/so101_mnri && source .venv/bin/activate
python -m isaacteleop.cloudxr --host-client
```

Ready when it prints (verified output):

```
CloudXR runtime:   running, log file: ~/.cloudxr/logs/cxr_server.<ts>.log
CloudXR WSS proxy: running, log file: ~/.cloudxr/logs/wss.<ts>.log
Keep this terminal open, Ctrl+C to terminate.
```

(`--host-client` serves the pinned client from `~/.cloudxr/static-client` at
`https://<laptop-ip>:48322/client/`; without it every GET returns only the
cert-acceptance page — verified. Flags checked against
`python -m isaacteleop.cloudxr --help`.)

Encouraging precedent: `~/.cloudxr/logs/cxr_server.2026-06-26T041912Z.log`
shows a **Quest 3 (Touch Plus controllers) already streamed video + controller
data to this laptop successfully** on June 26 — the chain is proven on this
exact hardware.

Terminal 2 — Isaac with the XR session:

```bash
cd /Storage/Projects/so101_mnri && source .venv/bin/activate
python scripts/isaac_teleop_vr.py --headless --cloudxr external
# --cloudxr external is REQUIRED here: terminal 1 already runs the runtime.
# options: --anchor-pos X Y Z  (world point shown at the headset origin;
#          default 0.35 0.0 0.95 ≈ table height in front of the arm)
# wait for "[vr] waiting for the Quest ...", then Kit logs
# "Acquired OpenXR handles from Kit XR bridge" once the headset connects
```

Keep heavy jobs off the GPU (~4 GB headroom; silent deaths otherwise).

### Quest side

1. Put on the headset. Open **Browser**.
2. Trust the self-signed cert (one time per headset):
   go to `https://<laptop-ip>:48322/` → click through the certificate warning
   (**Advanced → Proceed**) → you should see the page titled
   **"Certificate Accepted"**.
3. Open the client: `https://<laptop-ip>:48322/client/?serverIP=<laptop-ip>&port=48322`
   (`serverIP`/`port` are read by the client on page load).
4. Click **Enter VR** (grant the permission prompt), then click **Play /
   Start Teleop** in the client panel — this sends the `start teleop` command
   over the OpenXR control channel and flips the pipeline STOPPED→RUNNING.
5. You should now see the lab scene streamed in the headset; move the right
   controller and the SO-101 follows the delta.

### Expected first-frame behavior

- Before Play: stream renders, arm sits at home, **no motion** (pipeline
  holds last pose while PAUSED/STOPPED).
- On Play: the clutch latches the current right-controller pose as its origin
  and commands the EE to the *home* pose captured at env reset — **the arm
  does not jump**. Subsequent motion is `home + 1.0 × (controller − origin)`.
- Orientation maps 1:1 from grip to gripper with a fixed Rz(π) calibration
  offset; a 5-DOF arm cannot hit every orientation exactly — position tracks
  tightly, orientation is best-effort (DLS IK clamped to joint limits).
- Squeeze the grip to re-clutch if you run out of comfortable hand travel.
- Latency: expect ~50–100 ms motion-to-photon on LAN Wi-Fi (WebRTC).

### Shutdown

Ctrl-C terminal 2 (Isaac) first — the script tears down the TeleopSession —
then Ctrl-C terminal 1; the runtime module handles SIGINT/SIGTERM cleanly
(verified: process group exits, no stale locks).

---

## Fallback plan (if VR fails live)

1. **First 5 minutes of debugging**: see the troubleshooting table below.
   Most failures are Wi-Fi/firewall/cert issues, fixable in front of the audience.
2. **Kill VR, keep the demo**: switch to gamepad teleop.
   - Isaac-native (same envs, DS4, null-device-validated):
     `python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0`
     (delta-pose → `DifferentialIKController` → joint actions; see its
     docstring + `TELEOP_GUIDE.md` mode 2).
   - MuJoCo twin (verified in daily use):
     `MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single` —
     DS4 mapping documented in `scripts/README.md`.
3. **Last resort**: pre-rendered camera sweeps +
   `python -m envs.isaac.scripts.render_cameras`.

Time-box VR debugging to ~10 min before switching — the gamepad path has
zero setup.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `python -m isaacteleop.cloudxr` exits: "EULA was not accepted" | marker missing | run the `check_eula` one-liner from §0 step 1; verify `~/.cloudxr/run/eula_accepted` exists |
| Runtime fails to start ("failed to start within 30s") | stale pid/socket from a killed run, or transient GPU contention | just retry once (launcher auto-cleans the stale IPC socket); then check `~/.cloudxr/logs/runtime_stderr.log` and latest `cxr_server.*.log` |
| Script exits: "isaacteleop component missing" | wrong venv / partial install | `source .venv/bin/activate`; `pip show isaacteleop` (need ≥1.3 with `SO101ClutchRetargeter`) |
| Kit boots but no "waiting for the Quest" line | XR experience file not loaded | ensure you launched via the script (it passes `xr=True`); look for `isaaclab.python.xr.openxr.headless.kit` early in the log |
| Quest page won't load at all | different Wi-Fi / blocked port | same SSID; `sudo ufw allow 48322/tcp`; ping the laptop from another device |
| Cert warning loops / client says connection refused | cert never accepted | visit `https://<ip>:48322/` FIRST, accept, see "Certificate Accepted", then open `/client/` |
| Client loads but no video | serverIP wrong or media port blocked | use exact `hostname -I` value in `?serverIP=`; `sudo ufw allow 49100/tcp`,`49100/udp`; keep laptop & Quest on same subnet |
| Video OK, controllers tracked, but arm never moves | teleop not STARTED (state STOPPED) or no "Play" pressed | press Play/Start in the client panel; confirm log line `IsaacTeleop session started` |
| Arm moves by itself / drifts | anchor moved (A button toggles anchor rotation) | press A to toggle back, squeeze-grip re-clutch, or RESET in client |
| Arm jitters at extremes | target beyond joint limits | bring the controller back toward center; targets are clamped so this is cosmetic |
| Isaac dies silently (exit 0, log just stops) | GPU/RAM starvation (test.md) | `nvidia-smi`, `free -g`; close other sims; relaunch |
| `--cloudxr auto` used together with manual runtime | double launch / port busy | pick one: either let the script launch everything (`--cloudxr auto`, no terminal 1) or run terminal 1 yourself and pass `--cloudxr external` |

Port reference: WSS proxy + client = **48322/tcp** (`PROXY_PORT`),
CloudXR backend = **49100** (`--usb-local` mode uses `USB_UI_PORT`/8080-style
loopback instead — advanced, requires adb; see `isaacteleop.cloudxr --help`).
Logs: `~/.cloudxr/logs/{wss.*,cxr_server.*,runtime_stderr.log}`.

## Files touched for this feature

- `scripts/isaac_teleop_vr.py` — VR teleop script (`--null-device` dry run included)
- `VR_TELEOP_SETUP.md` — this document
- No changes to `envs/`, no new pip installs; everything else comes from the
  installed `isaacteleop` / `isaaclab_teleop` wheels.
