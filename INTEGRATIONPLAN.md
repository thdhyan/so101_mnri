# Integration plan: 4 task envs from `task-envs` worktree

Source: `.claude/worktrees/task-envs/EXPLAINATION.md` (written by the agent
that built these envs there, branch `worktree-task-envs`, still at commit
`8f5bf1a` — hasn't diverged in git history, all work is uncommitted).
Read that file in full before starting; this plan does not repeat its
design rationale, only the integration mechanics.

## What exists in the worktree (uncommitted, untracked)

```
.claude/worktrees/task-envs/
  EXPLAINATION.md
  so101_single_arm_pick_lift_env/     env.py, assets/{scene.xml, so101_follower.xml, meshes/*.stl}, requirements.txt
  so101_single_arm_pick_place_env/    same shape
  so101_dual_arm_cylinder_grasp_env/  same shape
  so101_dual_arm_cylinder_reach_env/  same shape
```

Each was cloned from the **old, pre-reorg** flat env pattern
(`so101_single_arm_env/`, `so101_dual_arm_env/` at repo root) — full robot
kinematics + duplicate mesh copies embedded per env, no shared `robots/`
dir, no aloha-style scene upgrades (no skybox/checker material, no
`overhead_cam`/`front_cam`, no `home` keyframe in at least the single-arm
task envs — confirmed missing during the interrupted integration attempt).

## What main looks like now (target layout)

```
envs/mujoco/so101_single_arm/assets/scene.xml
  <compiler angle="radian" meshdir="../../../../robots/so101/meshes" autolimits="true"/>
  <include file="../../../../robots/so101/so101_follower.xml"/>
  ... world: skybox/checker material, table, cube, goal_marker, cameras
      (outside_left, outside_right, overhead_cam, front_cam), keyframe "home"

envs/mujoco/so101_dual_arm/assets/scene.xml   — same pattern, so101_dual_follower.xml

robots/so101/
  meshes/*.stl                (13 files, single source, no per-env copies)
  so101_follower.xml          single-arm kinematics + wrist camera, NO meshdir override
  so101_dual_follower.xml     dual-arm kinematics + left_/right_wrist cameras, NO meshdir override
  so101.urdf, so101_dual.urdf
```

**Critical MuJoCo gotcha** (cost real time getting main's own two envs
right — read `robots/so101/README.md` "MuJoCo path resolution note"):
`<compiler meshdir=...>` resolves relative to the **top-level file being
loaded** (each env's `scene.xml`), never relative to a file it
`<include>`s. So `meshdir` must live ONLY in each env's `scene.xml`, never
inside `so101_follower.xml`/`so101_dual_follower.xml` — confirmed neither
shared file currently has a `meshdir` override (checked directly:
`grep meshdir robots/so101/so101_follower.xml robots/so101/so101_dual_follower.xml`
→ no matches). Keep it that way.

## Target state after integration

```
envs/mujoco/so101_single_arm_pick_lift/     env.py, assets/scene.xml, requirements.txt (+README)
envs/mujoco/so101_single_arm_pick_place/    same shape
envs/mujoco/so101_dual_arm_cylinder_grasp/  same shape
envs/mujoco/so101_dual_arm_cylinder_reach/  same shape
```

No `assets/meshes/`, no local `so101_follower.xml` copy in any of the 4 —
all reference `robots/so101/` exactly like the existing two envs do.

## Step-by-step

1. **Study the reference pair first.** Open
   `envs/mujoco/so101_single_arm/assets/scene.xml` and
   `envs/mujoco/so101_dual_arm/assets/scene.xml` side by side — every new
   scene.xml must follow their exact `<compiler>`/`<include>` header
   pattern (paths below).

2. **Create the 4 directories** under `envs/mujoco/`:
   `so101_single_arm_pick_lift/`, `so101_single_arm_pick_place/`,
   `so101_dual_arm_cylinder_grasp/`, `so101_dual_arm_cylinder_reach/`.

3. **Copy `env.py` and `requirements.txt` as-is** from each worktree env
   dir. Both single-arm and dual-arm `env.py` files use
   `ASSETS_DIR = Path(__file__).parent / "assets"` /
   `SCENE_XML = str(ASSETS_DIR / "scene.xml")` — this is location-agnostic,
   should need no path edits. Still: grep each `env.py` for any hardcoded
   `so101_single_arm_env` / `so101_dual_arm_env` module references (e.g. in
   docstrings or type-only imports) and fix to
   `envs.mujoco.so101_single_arm_pick_lift` etc.

4. **Rewrite each `scene.xml`.** For each of the 4:
   - Keep: table, cube/cylinder body + freejoint (or static cylinder for
     the reach env — no freejoint, per EXPLAINATION.md item 4), goal
     marker(s), any world-fixed cameras that are task-specific, and a
     `home` keyframe if practical to add (not present in the source — add
     one matching the env's default qpos + object init pose, following the
     pattern in `envs/mujoco/so101_single_arm/assets/scene.xml`'s
     keyframe block; optional polish, not a hard requirement — note this
     explicitly in the completion report either way).
   - Strip: the full robot `<body>` tree, `<default>` classes for the
     robot, mesh `<asset>` declarations, wrist camera declaration (already
     baked into the shared follower file).
   - Add at the top:
     `<compiler angle="radian" meshdir="../../../../robots/so101/meshes" autolimits="true"/>`
     and, right after the visual/asset block for world materials,
     `<include file="../../../../robots/so101/so101_follower.xml"/>`
     (single-arm envs) or `.../so101_dual_follower.xml` (dual-arm envs).
     Path depth check: `envs/mujoco/<name>/assets/scene.xml` is 4 levels
     below repo root — identical depth to the existing two envs, so these
     relative paths are copy-paste correct, no adjustment needed.
   - Double check body/joint/site names the new `env.py` expects
     (`left_gripperframe`, `cyl_end_a`, etc. per EXPLAINATION.md) still
     exist after switching to the shared include — the shared follower
     files use the same joint/site names as before (`shoulder_pan` etc.,
     `gripperframe` / `left_gripperframe` / `right_gripperframe`), so
     these should resolve unchanged; task-specific sites like `cyl_end_a`/
     `cyl_end_b` are defined on the cylinder body the new scene.xml keeps,
     not touched by the robot-file swap.

5. **Delete duplicated assets** from each new env dir once its scene.xml
   is repointed: no `assets/meshes/`, no local `so101_follower.xml` /
   `so101_dual_follower.xml`.

6. **Do not modify `robots/so101/`.** If the shared files turn out to be
   missing something a new env genuinely needs (e.g. a site the task
   requires on the robot body itself, not the object), stop and flag it
   rather than editing the shared file unilaterally — that risks breaking
   the two existing envs.

7. **Add a short README per new env**, modeled on
   `envs/mujoco/so101_single_arm/README.md`, with the corrected import
   path (`envs.mujoco.so101_single_arm_pick_lift.env`, etc.).

## Verification (must pass for each of the 4 envs before calling this done)

Use `/home/thakk100/miniconda3/envs/so101/bin/python`, `MUJOCO_GL=egl` for
headless rendering.

1. `mujoco.MjModel.from_xml_path('envs/mujoco/<name>/assets/scene.xml')`
   compiles without error.
2. If a keyframe exists, `mj_resetDataKeyframe` loads it without error.
3. Joint/actuator count: 6 for single-arm envs, 12 for dual-arm envs
   (`model.njnt` includes the freejoint on the manipulated object if
   present — check against the specific env's object count, don't assume
   a flat 6/12).
4. Import the module and call `make_env`/instantiate the env class
   directly, `reset()`, then one `step()` with a zero or near-home action
   — no exception.
5. `obs['images']` non-empty, every camera's image has `std() > 1.0`
   (rules out all-black/blank renders).
6. From repo root:
   `MUJOCO_GL=egl python -c "import sys; sys.path.insert(0,'.'); from envs.mujoco.so101_single_arm_pick_lift.env import make_env"`
   (and the equivalent for the other 3) — confirms the package import path
   resolves.
7. Run `scripts/verify_manual.py --headless-check` is only wired for
   `--env {single,dual}` today — either extend its `--env` choices to
   include the 4 new env names (small change, `load_env()` dispatch table)
   or note in the completion report that manual verification for these 4
   currently requires ad-hoc scripting, and file that gap in
   `AGENT_TASKS.md`.

## Constraints for whoever runs this

- Do not commit until integration + verification is fully green; commit
  message should describe all 4 envs in one commit (they're one logical
  unit of work) — see the existing commit style (`git log --oneline`) for
  tone.
- Do not touch `envs/mujoco/so101_single_arm/`, `envs/mujoco/so101_dual_arm/`,
  or `robots/so101/` — read-only references.
- Do not modify `.claude/worktrees/task-envs/` — read-only source. Do not
  run `git worktree remove` on it.
- After integration lands, update `AGENT_TASKS.md`: remove/close whatever
  line references this work if one exists, and add the 4 new envs to
  `README.md`'s environment table / `ENVS.md` if those docs should cover
  them (out of scope for the integration itself, but flag it).
