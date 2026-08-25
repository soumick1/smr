# Update v4.4 -- fast3r's last import (and that is the backbone work done)

The environment is back: vggt, pi3, dust3r, mast3r, stream3r all PASS.

## pl_bolts
fast3r imports it at module level for exactly one symbol,
`LinearWarmupCosineAnnealingLR`, referenced ONLY inside
configure_optimizers -- i.e. training.  Inference never touches it.  Line
24 of the same file also imports open3d (visualisation).

Two options:
  1. install it:   pip install --no-deps lightning-bolts
  2. do nothing:   the adapter now handles it.

`_import_lit_module()` tries the REAL import first, and only if the missing
module is on an explicit allowlist -- pl_bolts, open3d, wandb, all
inference-irrelevant -- installs a permissive stub and retries.  Anything
else re-raises: a genuinely required dependency is never silently faked.
It prints what it stubbed.

Permissive stubs mint dummy classes on attribute access (PEP 562 module
__getattr__) so `from pl_bolts.optimizers.lr_scheduler import X` succeeds.
One subtlety my own test caught: the stub must raise AttributeError for
DUNDERS, because importlib asks a package for __path__ and would otherwise
try to iterate a class.

## Run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r stream3r fast3r --max-views 4

Tests: 92 passed, 8 GPU-tier skipped.

## This is the end of the backbone work
Whatever fast3r does on this run, STOP HERE.  Six backbones spanning the
family tree -- VGGT (feed-forward multi-view), pi3, DUSt3R (pairwise +
global alignment), MASt3R (sparse GA), STream3R (causal/streaming), and
fast3r if it lands -- is more than Table 1 needs, and mvdust3r's pytorch3d
requirement is not worth the build.

29 days to the ICLR abstract.  Zero tables.  Next session is Pilot A:
chunk CO3D sequences, stitch by Sim(3) chaining vs the scaffold, report
ATE / RPE / peak memory.  Everything built in the last several sessions
exists to feed that experiment, and it is the one that decides ICLR vs
CVPR.
