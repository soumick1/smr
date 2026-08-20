# Update v0.3 -- diagnosing the VGGT pose WARNs

Apply as before: unzip over ~/smr, `git diff`, commit, push.

## Diagnosis of your sanity run
depth PASS (3.4%) + pose WARN (18.8 deg / 0.18) is the signature of an
INPUT DOMAIN GAP, not an adapter bug: a flipped pose convention gives
near-random rotations and O(1) centre errors, and the depth agreement
already requires the extrinsic inversion + scale normalisation to be
right.  The v0.2 exports were sparse point-splat speckle (visible in your
GT panel) -- depth survives (prior-driven) but pose needs correspondences,
and per-view aliased speckle provides none.  View 4 (back of the ring,
weakest overlap) worst, consistent with correspondence failure.

## Changed
* scripts/vggt_sanity.py -- SELF-consistency metric: cross-view depth
  reprojection using ONLY VGGT's own poses+depth.  Decision rule:
  self-inconsistent on speckle but consistent on dense frames => domain
  gap confirmed; self-consistent while mismatching GT => adapter bug
  (the metric is designed to catch our side too).  Validated here against
  exact geometry: 0.0043.
* scripts/export_synthetic_frames.py -- photo-like frames: 400k points,
  3x3 splat, EDT hole-fill (<=6 px), checker 10, per-point texture jitter
  0.5, 12 views (100% coverage verified).
* src/smr/scene/synthetic_scene.py -- checker / tex_jitter / splat params
  (v0.1 defaults preserved; all 11 unit tests still pass) + fill_holes().
* src/smr/backbones/vggt.py -- torch.amp.autocast deprecation fix.

## Run on the server
    python scripts/export_synthetic_frames.py         # overwrites frames
    python scripts/vggt_sanity.py
Expected: rotation and centre drop to low single digits / <3%, and
SELF-consistency < 0.05.  If rotation stays high WHILE self-consistency
is good, paste the output -- that pattern would point back at the
adapter's GT comparison and I will dig there.
