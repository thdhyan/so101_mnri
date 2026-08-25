# =============================================================================
# STAGE 2 — thin runtime image on top of the pushed stage-1 base.
#
# The base (docker/Dockerfile.base) carries the parity Isaac stack + the
# entrypoint. This stage exists ONLY to retag a personal base for sharing.
# Workspace source (and the custom skrl submodule) are cloned at RUNTIME.
#
# Build (after `docker build -f docker/Dockerfile.base ...` + push):
#   docker build -f docker/Dockerfile \
#     --build-arg BASE_IMAGE=docker.io/<dockerhub-user>/so101-isaac-base:3.0.0b2 \
#     -t <dockerhub-user>/so101-isaac-train:3.0.0b2 .
#
# No-Docker-Hub workflow: skip this file — run/push Dockerfile.base directly.
# =============================================================================
ARG BASE_IMAGE=docker.io/thdhyan/so101-isaac-base:3.0.0b2
FROM ${BASE_IMAGE}

USER root

# Runtime defaults (override with docker run -e / apptainer --env)
ENV SO101_REPO_URL=https://github.com/thdhyan/so101_mnri \
    SO101_GIT_BRANCH=feature/mjlab \
    SO101_WORKDIR=/workspace/so101_mnri \
    SO101_MOUNT_DIR=/workspace/mounts

WORKDIR /workspace

# entrypoint already present in the base; re-COPY for standalone edits
COPY docker/entrypoint.sh /usr/local/bin/so101-entrypoint.sh
RUN chmod +x /usr/local/bin/so101-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/so101-entrypoint.sh"]
# Default: basic PPO training with the custom skrl fork
CMD ["--backend", "isaaclab", "--task", "SO101-PickLift-Single-v0", "--algo", "skrl", "--num-envs", "4096"]
