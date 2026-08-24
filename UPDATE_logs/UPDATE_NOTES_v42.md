# Update v4.2 -- EMERGENCY: restore torch (my setup script broke it)

## What happened -- not your fault, and not the adapters
Two clues in your log say it together:
  * "In PyTorch 2.6, we changed the default value of weights_only"  -> torch
    is no longer 2.5.1
  * "The NVIDIA driver on your system is too old (found version 12080)"
    -> the new torch is built for a CUDA newer than this driver (12.8)

My setup_backbones.sh ran `pip install "lightning>=2.5.0" ...` WITHOUT
--no-deps.  Lightning depends on torch, pip upgraded it to a CUDA-13 build,
and every GPU backbone died at once.  The adapters are untouched; five of
them were passing minutes earlier.

## FIX (5 minutes)
    bash server/restore_env.sh            # shows what is installed now
    bash server/restore_env.sh --fix      # reinstalls the known-good pair

That pins **torch 2.5.1 + torchvision 0.20.1, both cu121** and verifies with
a real GPU matmul.  Note this also gives you a torchvision that MATCHES your
torch for the first time -- so the four compat shims (perceptual VGG16
weights, center_crop, resize, torchmetrics enums) become no-ops, and MUSt3R
becomes viable again.

If `--fix` cannot reach download.pytorch.org, fall back to:
    pip install --force-reinstall "torch==2.5.1" "torchvision==0.20.1"
and check `torch.version.cuda` is 12.1.

## Then re-run the five that were green
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r stream3r --max-views 4

## setup_backbones.sh hardened
* installs the Lightning/Hydra stack with `--no-deps`, always
* installs lightning-bolts (fast3r's `pl_bolts` import) and torchmetrics
* asserts `torch.cuda.is_available()` immediately after, and tells you to
  run restore_env.sh if it broke

## The two remaining backbone issues (AFTER the environment is back)
* fast3r: `No module named 'pl_bolts'` -> pip install --no-deps lightning-bolts
* mvdust3r: `No module named 'pytorch3d'` -> pytorch3d is a heavy,
  build-from-source dependency of their losses.py.  My recommendation is to
  DROP mvdust3r for the paper.  You already have six backbones spanning the
  whole family tree; pytorch3d is not worth a day of build pain 29 days
  before the abstract deadline.

## Standing advice, now with evidence
Never run a third-party requirements.txt in this venv.  Everything that has
gone wrong in the last hour came from dependency resolution, not from code.
