# Update v4.3 -- correct restore target, and the real root cause found

## Two things your last output revealed

**1. The restore did NOT happen.**  torch 2.5.1 downloaded (780 MB) but
torchvision==0.20.1 was not found, so pip aborted the whole transaction.
You are still on torch 2.13.0+cu130, which is why CUDA is still dead.

**2. Your torchvision was 0.2.0 -- released in 2018.**
Read pip's message again: `from versions: 0.1.6, 0.2.0`.  The cu121 index
has NO torchvision wheel for Python 3.13, so when this venv was built pip
fell back to the last source-installable release, from 2018.  That single
fact explains EVERY "old torchvision" failure we shimmed around: the
missing VGG16_Weights enum, PIL-only center_crop, PIL-only resize, and
torchmetrics' import.  It was never a slightly-stale install.

## The right target is cu128, not cu121
  * your driver reports 12080 = CUDA 12.8, so cu130 torch cannot init;
  * cu121 has no cp313 torchvision at all -- that is the trap you were in;
  * cu128 has cp313 wheels for a MATCHED torch+torchvision pair, and 12.8
    is exactly what the driver supports.

    bash server/restore_env.sh --fix          # now targets cu128
    # if that fails:
    SMR_TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
        bash server/restore_env.sh --fix

No version pins: the index only serves driver-compatible builds, and
letting it pick avoids exactly the unsatisfiable pin that just failed.
The script verifies with a real GPU matmul AND a tensor center_crop, so it
proves both halves before you trust it.

## What changes once you are on a modern torchvision
The four compat shims become no-ops (they all self-check first), and
MUSt3R becomes viable again if you want it back.

## New: torch >= 2.6 checkpoint shim
The modern torch you will land on flips `torch.load(weights_only=True)` by
default, which is why dust3r failed with
    GLOBAL argparse.Namespace was not an allowed global
DUSt3R/MASt3R/MV-DUSt3R checkpoints pickle an argparse.Namespace of
training args.  `patch_torch_load_legacy_checkpoints()` allowlists exactly
that one class (rather than disabling the safety default), and is applied
before loading in the dust3r and mast3r adapters.  No-op on torch < 2.6.

## Order of operations
    bash server/restore_env.sh --fix
    pip install --no-deps lightning-bolts       # fast3r's pl_bolts import
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r stream3r fast3r --max-views 4

Drop mvdust3r: it needs pytorch3d (build-from-source) in its losses.py.
Six backbones is already more than the paper needs.

Tests: 90 passed, 8 GPU-tier skipped.
