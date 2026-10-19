        # SO-101 MNRI

        **Extension name:** `thdhyan.manip.so101_mnri`
        **Category:** Manipulation
        **Source project:** `/Storage/Projects/so101_mnri`

        ## Description

        MuJoCo and Isaac Sim dual-backend environments for SO-101 arm sim2real

        ## Scripts

        | Action | Relative path |
        |--------|--------------|
        | **Run Isaac** | `run_isaac.py` |
| **Run MuJoCo** | `run_mujoco.py` |

        ## Usage

        ```bash
        # Enable via CLI (--ext-folder points to /Storage/Projects):
        isaacsim --ext-folder /Storage/Projects --enable thdhyan.manip.so101_mnri --headless

        # Or in Isaac Sim GUI:
        # Window → Extensions → gear → Add path → /Storage/Projects
        # Search for "thdhyan.manip.so101_mnri" → Enable
        ```

        ## Keywords

        so-arm, so101, mujoco, sim2real
