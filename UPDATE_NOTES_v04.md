# Update v0.4 -- the estimated-geometry milestone

Apply as usual (unzip over ~/smr, git diff, commit, push), then:

    pip install -U "numpy>=2.0,<2.8"        # silences the scipy/numpy
    pytest -q                               # mismatch warning; verify env
    python scripts/export_synthetic_frames.py    # room now CLOSED (4 walls),
                                                 # solid blobs
    python scripts/vggt_sanity.py                # new 4th panel: drift vs
                                                 # angular distance from view 0
    python experiments/run_tier3_vggt.py --backbone vggt   # THE milestone

## Reading your v0.3 sanity result (recap)
Dense frames confirmed the domain gap (18.8->5.2 deg, 0.18->0.065, depth
0.015).  The residual concentrates in views 150-270 deg from view 0:
reference-anchored drift -- VGGT expresses poses in view 0's frame and
error grows with distance from it.  Marginal SELF-consistency (0.053)
confirms genuine internal inconsistency, not our GT comparison.  This is
exactly the pathology the plan's "Settle as pose consistency" subsection
targets (and the reason pi^3 exists) -- the backbone-swap sanity has
reproduced the field's motivating problem on demand.  The new drift panel
makes it a curve: the "before" figure for the paper.

## What run_tier3_vggt.py measures (validated end-to-end on CPU with
## --backbone synthetic: V1 0.0017 rad, V3 9/9, V5 0.002)
V1  place->decode against VGGT's own poses: our machinery must stay
    sub-degree regardless of whose poses they are.
V3  relocalisation from corrupted descriptors.
V5a held-out view splat vs VGGT's own held-out depth (frame-consistent
    target, thresholded at 0.12).
V5b same splat vs TRUE GT after scale alignment (report-only): the honest
    inherited-error number = what the backbone costs us end to end.

## Also changed
* backbones/base.py -- descriptor rebuilt (block-mean colour+grey
  thumbnails, mean-removed, + per-channel histograms): 97.5% direct reloc
  at sigma 0.2 on the symmetric closed room (old point-sampled thumbnail:
  ~33%).  Placeholder until PCA-pooled backbone features (adapter v2).
  Original ladder unaffected: tier3 re-run all-pass (I3 now 8/8), 11/11
  unit tests.
* scene: closed / blob_n params (defaults preserve v0.1 exactly).
