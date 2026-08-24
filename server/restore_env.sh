#!/usr/bin/env bash
# Restore a WORKING torch + torchvision pair for this machine.
#
#   bash server/restore_env.sh            # diagnose only
#   bash server/restore_env.sh --fix      # install a driver-compatible pair
#
# WHY cu128 and not cu121:
#   * the driver here reports 12080 = CUDA 12.8, so any torch built for
#     CUDA 13.0 (what pip installed) cannot initialise;
#   * the cu121 index has NO torchvision wheel for Python 3.13 -- pip's
#     "from versions: 0.1.6, 0.2.0" means the venv had been running
#     torchvision 0.2.0 (2018).  That is the true cause of every
#     "old torchvision" failure we shimmed around;
#   * cu128 has cp313 wheels for a matched torch+torchvision pair and the
#     driver supports it.
set -uo pipefail
IDX="${SMR_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

show () {
python - <<'PY'
import importlib
for m in ("torch", "torchvision", "numpy", "lightning", "torchmetrics"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<12} NOT INSTALLED ({type(e).__name__})")
try:
    import torch
    print(f"  built for CUDA: {torch.version.cuda}")
    print(f"  cuda available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch unusable: {e}")
PY
}

echo "=== current ==="; show

if [ "${1:-}" != "--fix" ]; then
  echo
  echo "run with --fix to install a matched pair from $IDX"
  echo "(override the index with SMR_TORCH_INDEX=...)"
  exit 0
fi

echo
echo "=== installing matched torch + torchvision from $IDX ==="
# --index-url alone (not extra) so the broken pypi.ngc mirror is bypassed.
# No version pins: let the index serve the newest pair it has cp313 wheels
# for; anything on the cu128 index is driver-compatible here.
pip install --force-reinstall --index-url "$IDX" torch torchvision || {
  echo
  echo "FAILED. Try an older CUDA line, e.g.:"
  echo "  SMR_TORCH_INDEX=https://download.pytorch.org/whl/cu126 \\"
  echo "      bash server/restore_env.sh --fix"
  exit 1
}

echo
echo "=== verify ==="; show
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA STILL UNAVAILABLE -- try cu126"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
print(f"  gpu matmul OK {tuple(x.shape)} on {torch.cuda.get_device_name(0)}")
import torchvision.transforms.functional as F
print(f"  torchvision tensor center_crop OK "
      f"{tuple(F.center_crop(torch.zeros(3,8,8),[4,4]).shape)}")
PY
echo
echo "next:"
echo "  python scripts/check_backbones.py --frames lab_photos/room/images_8 \\"
echo "      --backbones vggt pi3 dust3r mast3r stream3r --max-views 4"
