# IsaacTeleop VR SO-101 — Handoff (2026-09-04)

## Status: 90% Done — VR Sim Teleop Ready to Test

Everything is built, installed, and verified on `dl`. **Next agent just needs to start the VR teleop script and connect the Quest 3.**

---

## What Was Done

### 1. isaacteleop installed on `dl` ✅
- `isaacteleop 1.5+local` built from source in `~/IsaacTeleop/`
- Installed into conda env `isaac6` (Python 3.12) on `dl`
- SO-101 retargeters verified: `SO101ClutchRetargeter`, `SO101GripperRetargeter`
- `isaaclab_teleop 0.5.2` already existed in `isaac6`

### 2. SO-101 leader plugin built ✅
- `~/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin` built and tested
- Works in synthetic mode (no hardware needed)

### 3. Project synced to `dl` ✅
- `~/so101_mnri/` — full project synced from laptop (rsync, no .git/.venv/data)
- PYTHONPATH configured in conda activate.d: `~/so101_mnri` auto-added on `conda activate isaac6`
- All 10 SO-101 tasks registered and importable:
  - `SO101-CylReach-Single-v0` ← this is the one for VR
  - `SO101-CylReach-Dual-v0`
  - `SO101-CylGrasp-Dual-v0`
  - `SO101-PickLift-Single-v0`
  - `SO101-PickPlace-Single-v0`
  - Plus Play variants

### 4. Bug fix applied ✅
- `scripts/isaac_teleop_vr.py` line 437: added missing `from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg` inside `_init_single_arm()`. The import was only in `__init__` (local scope) and `_init_dual_arm`, but not in `_init_single_arm`.

### 5. Null-device test PASSED ✅
```
[null-device] PASS
```
Retargeting + IK pipeline works headless on `dl` (4× RTX 6000 Ada, 192 GB VRAM).

### 6. CloudXR runtime started ✅
- Runtime: running at `~/.cloudxr/logs/cxr_server.*.log`
- WSS proxy: running
- Device profile: Quest3
- Web client: https://nvidia.github.io/IsaacTeleop/client/release-1.5.x/
- Env file: `source ~/.cloudxr/run/cloudxr.env`
- Stop with: `python -m isaacteleop.cloudxr.service stop`

---

## What's Next (For Next Agent)

### Step 1: Open firewall ports on `dl`
```bash
ssh dl
sudo ufw allow 48322/tcp   # WSS proxy + web client
sudo ufw allow 49100/tcp && sudo ufw allow 49100/udp  # CloudXR signaling
```

### Step 2: Start the VR teleop (headless, no display needed)
```bash
ssh dl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac6
cd ~/so101_mnri

# CloudXR runtime should already be running (started above)
# If not: python -m isaacteleop.cloudxr.service start

# Start Isaac Lab with XR session (headless, connects to CloudXR):
python scripts/isaac_teleop_vr.py --headless --cloudxr external
```

Wait for: `[vr] waiting for the Quest ...` then `Acquired OpenXR handles from Kit XR bridge`

### Step 3: Connect Quest 3
1. Quest and `dl` must be on the same network (confirmed: routable, 128.101.125.152)
2. Quest Browser → `https://128.101.125.152:48322/client/`
3. Accept self-signed cert (Advanced → Proceed)
4. Enter VR → Click Play/Start Teleop
5. Right controller: squeeze grip = clutch, trigger = gripper

### Step 4 (later): Connect real SO-101 arm
```bash
python scripts/isaac_teleop_vr.py --headless --cloudxr external \
    --follower-port /dev/ttyACM0 --follower-id so101_follower_arm
```

---

## Key Files

| File | Location on `dl` | Purpose |
|------|-----------------|---------|
| VR teleop script | `~/so101_mnri/scripts/isaac_teleop_vr.py` | Main VR teleop (single + dual arm) |
| Project root | `~/so101_mnri/` | All SO-101 tasks, envs, scripts |
| IsaacTeleop source | `~/IsaacTeleop/` | Built from NVIDIA/IsaacTeleop |
| Leader plugin | `~/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin` | SO-101 leader arm plugin |
| CloudXR logs | `~/.cloudxr/logs/` | Runtime + WSS proxy logs |
| CloudXR env | `~/.cloudxr/run/cloudxr.env` | `source` this for CloudXR env vars |

## Quick Commands

```bash
# Null-device test (verify pipeline works)
ssh dl "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaac6 && cd ~/so101_mnri && OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device --steps 50"

# Check CloudXR status
ssh dl "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaac6 && python -m isaacteleop.cloudxr.service status"

# Stop CloudXR
ssh dl "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaac6 && python -m isaacteleop.cloudxr.service stop"

# Leader plugin test (synthetic, no hardware)
ssh dl "~/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin"
```

## Environment Info

| | Value |
|---|---|
| Server | `dl` (128.101.125.152) |
| GPU | 4× RTX 6000 Ada (48 GB each, ~156 GB free) |
| Conda env | `isaac6` (Python 3.12) |
| Isaac Sim | 6.0.1 |
| Isaac Lab | 3.0.0b2.post1 |
| isaacteleop | 1.5+local (built from source) |
| isaaclab_teleop | 0.5.2 |
| CloudXR | 6.3.0 (bundled in isaacteleop) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CloudXR not running | `python -m isaacteleop.cloudxr.service start` |
| Quest can't connect | Check `sudo ufw allow 48322/tcp 49100/tcp 49100/udp`; verify same network |
| `No module named 'envs'` | `cd ~/so101_mnri` (PYTHONPATH is set in conda activate.d) |
| `DifferentialIKControllerCfg not defined` | Already fixed in the synced script |
| Isaac Sim OOM | Won't happen — 48 GB per GPU is plenty |
| `Permission denied /dev/ttyACM*` | `sudo usermod -aG dialout $USER` + re-login |
| Swapchain error | Was a laptop 8GB VRAM issue — `dl` has 48 GB per GPU, should not occur |
