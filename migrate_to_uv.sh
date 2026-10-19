#!/usr/bin/env bash
# Migration: so101 conda env → uv venv
# Run from: /home/thakk100/Projects/so101_mnri/
set -euo pipefail
LOG="$HOME/Projects/so101_mnri/migration.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== so101 migration $(date) ==="

cd /home/thakk100/Projects/so101_mnri

# Create venv
echo "[1/6] Creating .venv (Python 3.11)..."
uv venv .venv --python 3.11 --clear
source .venv/bin/activate

# Extract conda deps — exclude nvidia-* system CUDA runtime, editable installs
echo "[2/6] Extracting conda deps..."
conda run -n so101 pip freeze 2>/dev/null | \
  grep -v '^-e' | \
  grep -v '^#' | \
  grep -v '^nvidia-' | \
  grep -v 'nvidia_' > /tmp/so101_reqs.txt || true

echo "Package count: $(wc -l < /tmp/so101_reqs.txt)"

# Install torch first with correct CUDA index
echo "[3/6] Installing torch (cu121)..."
TORCH_LINE=$(grep -i '^torch==' /tmp/so101_reqs.txt || true)
if [ -n "$TORCH_LINE" ]; then
  # Install latest available torch for cu121 (conda may have had a newer/nonexistent version)
  uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
  # Remove torch lines from reqs to avoid conflict
  grep -v '^torch\b\|^torchvision\|^torchaudio' /tmp/so101_reqs.txt > /tmp/so101_reqs_notorch.txt
else
  cp /tmp/so101_reqs.txt /tmp/so101_reqs_notorch.txt
fi

# Install warp-lang
echo "[4/6] Installing warp-lang..."
uv pip install warp-lang || echo "WARN: warp-lang failed, check manually"

# Install remaining packages
echo "[5/6] Installing remaining packages..."
uv pip install -r /tmp/so101_reqs_notorch.txt || {
  echo "WARN: bulk install had issues, trying one by one..."
  while IFS= read -r pkg; do
    [[ "$pkg" =~ ^# ]] && continue
    [[ -z "$pkg" ]] && continue
    uv pip install "$pkg" || echo "SKIP: $pkg"
  done < /tmp/so101_reqs_notorch.txt
}

# Editable installs — mjlab and skrl submodules
echo "[6/6] Editable installs (mjlab, skrl)..."
for submod in third_party/mjlab third_party/skrl; do
  if [ -d "$submod" ]; then
    uv pip install -e "$submod" && echo "OK: $submod" || echo "FAIL: $submod"
  else
    echo "WARN: $submod not found — run: git submodule update --init"
  fi
done

# Self
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
  uv pip install -e . || true
fi

echo ""
echo "=== DONE $(date) ==="
echo "Activate: source /home/thakk100/Projects/so101_mnri/.venv/bin/activate"
