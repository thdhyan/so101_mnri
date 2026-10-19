# SO-101 MNRI

> **Extension name:** `thdhyan.manip.so101_mnri`
> **Category:** Manipulation
> **Type:** training
> **Source:** `/Storage/Projects/so101_mnri`

## Description

MuJoCo and Isaac Sim dual-backend environments for the SO-101 robot arm (MNRI project)

## Notes

Shared policy code runs on both MuJoCo and Isaac Sim backends for sim2real transfer.

## Scripts / Workflows

- **Run Isaac**: `run_isaac.py`
- **Run MuJoCo**: `run_mujoco.py`

## Isaac Sim Dependencies

- `isaacsim.core.simulation_manager`
- `omni.kit.uiapp`

## Installation

```bash
# 1. Ensure /Storage/Projects is mounted in the container (see docker-compose.override.yml)
# 2. In Isaac Sim: Window → Extensions → gear icon → Add path → /workspace/user_exts
# 3. Enable: thdhyan.manip.so101_mnri

# OR standalone pip:
pip install -e /Storage/Projects/so101_mnri
```

## Keywords

`so-arm`, `so101`, `mujoco`, `isaac`, `manipulation`, `mnri`
