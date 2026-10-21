# Multi-stage Dockerfile for SO101 IL/RL Training
# Supports: SmolVLA, ACT, Diffusion Policy, GR00T N1.7, Pi0
# Target: dl, lambda, zz-bw, zz-dt machines (Ubuntu + NVIDIA GPU)

# ============================================================================
# Stage 1: Base with CUDA + Python
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git wget curl build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip --upgrade && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# ============================================================================
# Stage 2: PyTorch base
# ============================================================================
FROM base AS torch-base

RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# ============================================================================
# Stage 3: LeRobot (all backends)
# ============================================================================
FROM torch-base AS lerobot

# Install system deps for flash-attn (needed by some backends)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build && rm -rf /var/lib/apt/lists/*

# Clone LeRobot and install with all extras
RUN git clone https://github.com/huggingface/lerobot /workspace/lerobot
WORKDIR /workspace/lerobot

# Install LeRobot with all policy extras
RUN pip3 install --no-cache-dir -e ".[all]"

# Install flash-attn separately (needs special build)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || true

WORKDIR /workspace

# ============================================================================
# Stage 4: SO101 project code
# ============================================================================
FROM lerobot AS so101

# Copy project files
COPY rl/ /workspace/rl/
COPY vla/ /workspace/vla/
COPY envs/ /workspace/envs/
COPY scripts/ /workspace/scripts/
COPY robots/ /workspace/robots/
COPY IL_PLAN.md /workspace/
COPY README.md /workspace/
COPY ENVS.md /workspace/

# Create data directories
RUN mkdir -p /workspace/data /workspace/outputs /workspace/vla/runs

# Set Python path
ENV PYTHONPATH=/workspace

# Default entrypoint: run training
ENTRYPOINT ["python3.12", "-m", "vla.finetune"]
CMD ["--help"]
