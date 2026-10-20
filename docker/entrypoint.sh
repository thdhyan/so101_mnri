#!/usr/bin/env bash
# =============================================================================
# SO-101 training container entrypoint.
#
# Runtime responsibilities (nothing about the workspace is baked in):
#   1. Clone this repo (with the third_party/skrl submodule = the custom skrl
#      fork), OR use a bind-mounted source tree when SO101_USE_MOUNT=1.
#   2. pip install -e third_party/skrl (the custom dataclass-API skrl).
#      Optionally mjlab (SO101_INSTALL_MJLAB=1).
#   3. Wire mounted output dirs (rl/runs, logs).
#   4. exec `python -m rl.train` with the passed CLI args.
#
# Env vars:
#   SO101_REPO_URL     default https://github.com/thdhyan/so101_mnri
#   SO101_GIT_BRANCH   default feature/mjlab
#   SO101_WORKDIR      default /workspace/so101_mnri
#   SO101_USE_MOUNT    1 = skip clone, use SO101_WORKDIR as-is (bind-mounted)
#   SO101_INSTALL_MJLAB  1 = also pip install -e third_party/mjlab
#   GITHUB_TOKEN       optional; enables private repo/submodule clone
#   WANDB_API_KEY      read by wandb (or WANDB_MODE=offline)
#   SO101_MOUNT_DIR    default /workspace/mounts — host bind target root
#
# Everything after the entrypoint is passed to rl.train verbatim, e.g.:
#   --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
# =============================================================================
set -euo pipefail

SO101_REPO_URL="${SO101_REPO_URL:-https://github.com/thdhyan/so101_mnri}"
SO101_GIT_BRANCH="${SO101_GIT_BRANCH:-feature/mjlab}"
SO101_WORKDIR="${SO101_WORKDIR:-/workspace/so101_mnri}"
SO101_MOUNT_DIR="${SO101_MOUNT_DIR:-/workspace/mounts}"

echo "[so101-entrypoint] repo=${SO101_REPO_URL} branch=${SO101_GIT_BRANCH}"

# ---------------------------------------------------------------- clone -----
if [ "${SO101_USE_MOUNT:-0}" != "1" ]; then
    if [ -d "${SO101_WORKDIR}/.git" ]; then
        echo "[so101-entrypoint] ${SO101_WORKDIR} already cloned — reusing"
    else
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            git config --global \
                url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf \
                "https://github.com/"
            echo "[so101-entrypoint] GITHUB_TOKEN set — private-clone mode"
        fi
        # submodules may be registered with SSH URLs — rewrite to HTTPS
        git config --global url."https://github.com/".insteadOf "git@github.com:"
        git clone --recurse-submodules --shallow-submodules --depth 1 \
            -b "${SO101_GIT_BRANCH}" "${SO101_REPO_URL}" "${SO101_WORKDIR}"
        cd "${SO101_WORKDIR}"
        git lfs pull || echo "[so101-entrypoint] WARN: git lfs pull failed"
        # git clone exits 0 even when submodules fail — fail loudly here
        MISSING=$(git submodule status | awk '$1 ~ /^-/ {print $2}')
        if [ -n "${MISSING}" ]; then
            echo "[so101-entrypoint] FATAL: submodules not fetched: ${MISSING}" >&2
            echo "[so101-entrypoint] hint: private repos need GITHUB_TOKEN in .env" >&2
            exit 1
        fi
    fi
else
    echo "[so101-entrypoint] SO101_USE_MOUNT=1 — using bind-mounted ${SO101_WORKDIR}"
fi
cd "${SO101_WORKDIR}"

# ------------------------------------------------- mount wiring -------------
for name in runs logs; do
    if [ -d "${SO101_MOUNT_DIR}/${name}" ] && [ ! -e "${SO101_WORKDIR}/${name}" ]; then
        ln -s "${SO101_MOUNT_DIR}/${name}" "${SO101_WORKDIR}/${name}"
        echo "[so101-entrypoint] linked ${name}/ -> ${SO101_MOUNT_DIR}/${name}"
    fi
done
# rl/train.py writes to rl/runs by default — link that path too
if [ -d "${SO101_MOUNT_DIR}/runs" ] && [ ! -e "${SO101_WORKDIR}/rl/runs" ]; then
    mkdir -p "${SO101_WORKDIR}/rl"
    ln -s "${SO101_MOUNT_DIR}/runs" "${SO101_WORKDIR}/rl/runs"
    echo "[so101-entrypoint] linked rl/runs -> ${SO101_MOUNT_DIR}/runs"
fi

# ------------------------------------------------- workspace packages -------
PY="$(cat /etc/so101_python 2>/dev/null || echo python3)"
echo "[so101-entrypoint] python: ${PY}"
# custom skrl fork — editable install preferred; on read-only rootfs (apptainer
# non-fakeroot) fall back to PYTHONPATH (skrl is pure-python, paths via __file__)
if ! $PY -m pip install --no-cache-dir -q -e third_party/skrl 2>/dev/null; then
    echo "[so101-entrypoint] pip -e failed (read-only rootfs?) — PYTHONPATH fallback"
    export PYTHONPATH="${SO101_WORKDIR}/third_party/skrl:${PYTHONPATH:-}"
fi
$PY -c "import skrl; print('[so101-entrypoint] custom skrl OK:', skrl.__version__, 'from', skrl.__file__)"
if [ "${SO101_INSTALL_MJLAB:-0}" = "1" ]; then
    if ! $PY -m pip install --no-cache-dir -q -e third_party/mjlab 2>/dev/null; then
        export PYTHONPATH="${SO101_WORKDIR}/third_party/mjlab/src:${PYTHONPATH:-}"
    fi
    # mjlab's deps may upgrade mujoco-warp past the known-good pin — re-pin
    $PY -m pip install --no-cache-dir -q "mujoco-warp==3.10.0.1" 2>/dev/null || true
    $PY -c "import mjlab; print('[so101-entrypoint] mjlab OK')"
fi

# ------------------------------------------------- run ----------------------
if [ "$#" -eq 0 ]; then
    set -- --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
fi
echo "[so101-entrypoint] exec: ${PY} -m rl.train $*"
exec "${PY}" -m rl.train "$@"
