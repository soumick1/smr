#!/usr/bin/env bash
# Clone the frozen-backbone repos into third_party/ (SERVER side, after
# setup_env.sh --with-backbones).  VERIFY each URL before first use -- the
# adapters in src/smr/backbones/ deliberately fail loudly until wired.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party && cd third_party

# VGGT (Meta AI).  VERIFY: official repo at time of writing:
git clone https://github.com/facebookresearch/vggt.git || true

# Pi3 / pi-cubed -- official repo VERIFIED (yyfz/Pi3):
git clone https://github.com/yyfz/Pi3.git || true
pip install -r Pi3/requirements.txt || true
pip install -e Pi3 || echo "[Pi3] editable install failed; add $(pwd)/Pi3 to PYTHONPATH"

# VGGT-Omega.  VERIFY: fill in the official repository when confirmed:
echo "[VGGT-Omega] verify the official repository URL, then clone it here."

echo
echo "After cloning: follow each repo's install notes, download weights,"
echo "then complete the forward pass in src/smr/backbones/{vggt,pi3,vggt_omega}.py"
