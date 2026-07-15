#!/usr/bin/env bash
# Phase 0 — create both pinned environments (bioconda CLI + uv Python).
# Documented in docs/PHASE0_environment_inventory_smoketest.md §4. Idempotent-ish.
set -euo pipefail

PROJECT=/home/shadeform/project
LOCALBIN=/home/shadeform/.local/bin

# ---------------------------------------------------------------- micromamba
export MAMBA_ROOT_PREFIX=/ephemeral/micromamba
mkdir -p "$MAMBA_ROOT_PREFIX"
if [ ! -x "$LOCALBIN/micromamba" ]; then
  echo "[*] Installing micromamba (user-space)…"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /home/shadeform/.local bin/micromamba
fi
"$LOCALBIN/micromamba" --version

echo "[*] Creating bioconda env 'cisgram' from environment.yml…"
"$LOCALBIN/micromamba" create -y -n cisgram -f "$PROJECT/environment.yml"
"$LOCALBIN/micromamba" run -n cisgram bcftools --version | head -1
"$LOCALBIN/micromamba" run -n cisgram tabix --version | head -1

# ---------------------------------------------------------------- uv Python
export UV_CACHE_DIR=/ephemeral/uv-cache
mkdir -p "$UV_CACHE_DIR"
cd "$PROJECT"
echo "[*] Creating uv venv (Python 3.11)…"
uv venv --python 3.11 .venv
echo "[*] Installing torch (CUDA 12.4 build)…"
uv pip install --python .venv/bin/python torch --torch-backend=cu124
echo "[*] Installing remaining Python deps…"
uv pip install --python .venv/bin/python -r requirements.txt
echo "[*] Registering Jupyter kernel…"
.venv/bin/python -m ipykernel install --user --name cisgram --display-name "Python (cisgram)"
echo "[*] Freezing lockfile…"
uv pip freeze --python .venv/bin/python > requirements.lock.txt

echo "[*] Verifying CUDA…"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0),
      "|", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
PY
echo "[✓] Environments ready."
