#!/usr/bin/env bash
# Environment setup.  Usage:
#   ./setup_env.sh                  # CPU env: tiers 1-3, tests, figures, gifs
#   ./setup_env.sh --with-backbones # + torch (CUDA if nvidia-smi is present)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt
pip install -e . >/dev/null

if [[ "${1:-}" == "--with-backbones" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "CUDA GPU detected -> installing torch (cu121 wheels)"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "No GPU detected -> installing CPU torch"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    echo "Next: server/setup_backbones.sh clones the backbone repos."
fi

echo
echo "Environment ready.  Quick verification:"
echo "  source .venv/bin/activate"
echo "  python experiments/run_all.py --fast"
