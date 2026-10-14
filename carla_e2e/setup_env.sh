#!/usr/bin/env bash
# ============================================================
# One-shot environment setup for the CARLA E2E project.
# Run from inside carla_e2e/:  bash setup_env.sh
# Requires internet. Creates a uv-managed py3.10 venv (no conda,
# no system changes) so the CARLA client wheel can be installed.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

echo "[setup] installing uv (single static binary) ..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] ensuring python 3.10 (uv downloads it if missing) ..."
uv python install 3.10

echo "[setup] creating venv + installing deps ..."
uv venv --python 3.10
uv pip install -r requirements.txt

echo "[setup] venv ready at .venv"
echo "        activate with:  source .venv/bin/activate"
