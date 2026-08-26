# VR Teleop Issue — Handoff for Next Agent

## Status
- Quest 3 connects via WebXR browser client ✓
- CloudXR runtime + OpenXR session starts ✓
- Controller tracking detected (`Quest Touch Plus Left/Right`) ✓
- `IsaacTeleop session started: SO101VRTeleop` ✓
- **Video stream fails** — swapchain creation error, Quest browser shows no Isaac scene

## Root Cause
The CloudXR Monado runtime cannot create the Vulkan swapchain for the stereo video stream to the headset. The error:

```
ERROR [swapchain_server_create] ipc_call_swapchain_create failed: XRT_ERROR_VULKAN
XR_ERROR_RUNTIME_FAILURE in xrCreateSwapchain: Failed to create swapchain
[Error] [XR] Failed to create color swapchain for display hmdLeft
```

This happens because:
1. Our `isaac_teleop_vr.py` disables the XR display pipeline after Kit boots (lines 480-493) to avoid GPU OOM from 4096×3584 per-eye textures on the 8GB laptop GPU
2. When the display is disabled, the OpenXR frame loop still tries to acquire swapchain images — but `swapchain == NULL`, causing `XR_ERROR_HANDLE_INVALID` spam
3. The CloudXR Monado runtime's `swapchain_server_create` fails because the Vulkan device doesn't have the required memory/format for the requested swapchain

## What Works
- **Controller tracking**: DeviceIO session, ControllerTracker initialized with left+right controllers
- **isaacteleop bridge**: TeleopSession, retargeting pipeline, IK controller all functional
- **Environment**: Isaac Lab env (SO101-CylReach-Single-v0) creates fine, physics runs

## What Doesn't Work
- **Stereo video streaming**: The swapchain that streams rendered frames to Quest fails
- **XR display composition**: Disabled to avoid OOM, but the frame loop still tries to use it

## Key Files
- `scripts/isaac_teleop_vr.py` — main VR teleop script
- `VR_TELEOP_SETUP.md` — setup guide for Quest 3 + CloudXR
- Isaac Sim 6.0.1 at `/home/thakk100/.local/share/ov/pkg/isaac_sim-2025.3.0`
- isaacteleop installed in Isaac Sim venv

## What Needs to Be Fixed

### Option A: Fix the swapchain creation (preferred)
The CloudXR Monado runtime creates swapchains via IPC. The error is in:
```
/builds/cloudxr/cloudxr-openxr-runtime/deps/monado/src/xrt/ipc/client/ipc_client_compositor.c:288
```

The issue might be:
- The Vulkan format requested by CloudXR isn't supported by the GPU driver
- The GPU doesn't have enough VRAM for the swapchain (8GB laptop GPU)
- The Monado runtime's Vulkan device selection picks the wrong GPU

**Approaches to try:**
1. Check CloudXR logs for what swapchain format/size it's requesting
2. Try running without the display disable — the OOM may have been a red herring (it was 4096×3584 per-eye but CloudXR might request a smaller resolution)
3. Set `/app/renderer/resolution/width` and `height` to smaller values (e.g., 1024×1024 per eye) instead of fully disabling display
4. Try `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` to get more detailed Vulkan error info

### Option B: Bypass CloudXR display, use controller-only mode
If the video stream can't work on this GPU, accept controller-only teleop:
- Remove the display disable (let Kit handle swapchain creation normally)
- Add a try/except around swapchain creation to gracefully degrade
- The Quest browser would need a different client that shows pre-recorded/static scene instead of live stream

### Option C: Reduce resolution aggressively
```python
# In isaac_teleop_vr.py, instead of disabling display entirely:
s.set("/app/renderer/resolution/width", 512)   # per-eye width
s.set("/app/renderer/resolution/height", 512)  # per-eye height
s.set("/xr/profile/display/enabled", True)      # keep display on
```
This might give CloudXR a small enough swapchain to fit in VRAM.

## Debugging Commands
```bash
# Check GPU memory
nvidia-smi

# Check Vulkan support
vulkaninfo --summary

# Run with verbose OpenXR logging
XR_LOG_LEVEL=DEBUG python scripts/isaac_teleop_vr.py --headless --cloudxr external 2>&1 | tee /tmp/vr_debug.log

# Check CloudXR runtime logs
ls -la ~/.local/share/cloudxr/
cat ~/.local/share/cloudxr/*.log

# Check if Monado/CloudXR picks the right GPU
VK_LOADER_DEBUG=all python -c "import ctypes; ctypes.CDLL('libvulkan.so.1')"
```

## Environment Info
- GPU: NVIDIA GeForce RTX (8GB VRAM) — laptop
- Driver: 580.86.17
- CUDA: 13.3
- Isaac Sim: 6.0.1.0 (isaacsim 6.0.1.0)
- isaacteleop: installed in Isaac Sim venv
- CloudXR: bundled with isaacteleop
- OS: Ubuntu 24.04

## What to Try First
1. Remove the display disable entirely and see if CloudXR can create the swapchain (OOM might not actually happen at CloudXR's resolution)
2. If OOM, try reducing resolution to 1024×1024 instead of disabling
3. Check CloudXR logs for the exact swapchain format/size being requested
4. If all else fails, accept controller-only mode and document the limitation
