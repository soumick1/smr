# Update v3.6 -- MUSt3R: collections.Iterable (Python 3.10 removal)

## The error
`AttributeError: module 'collections' has no attribute 'Iterable'`

Not in must3r, dust3r or mast3r source -- I grepped all three.  It comes
from a library in MUSt3R's import chain (it pulls dust3r.viz, open3d,
viser, roma) written before Python 3.10 deleted the `collections.X`
aliases that had been deprecated since 3.3 in favour of
`collections.abc.X`.  Your server runs Python 3.13, so they are gone.

## The fix
`patch_collections_abc_aliases()` re-binds the removed names to the
IDENTICAL objects in collections.abc -- it adds names back and changes
none, restoring exactly pre-3.10 behaviour.  Applied in MUSt3R's
`_import_inference()` before any import, idempotent, and verified here
under the real condition (this container is Python 3.12, which also lacks
them, so the test exercises the genuine failure rather than a simulation).

## Also: the checker now prints the raising frame on FAIL
A one-line `raised at: File "...", line N, in fn` accompanies every
failure, so the next unknown third-party incompatibility is identifiable
from the first run instead of needing --verbose-errors and a repeat.

## Re-run
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r must3r --max-views 4

MUSt3R has now cleared: missing package -> checkpoint path -> faiss ->
asmk -> collections alias.  Each was a genuine environment-era
incompatibility rather than an adapter bug, which is itself worth a
sentence in the paper: five backbones, five different dependency vintages,
one socket.  If the next failure is inside must3r_inference itself, that is
the return-shape question I flagged as unpinnable from the repo, and the
adapter will describe what it found.

Tests: 67 passed, 6 GPU-tier skipped.
