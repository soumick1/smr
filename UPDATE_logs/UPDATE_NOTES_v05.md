# Update v0.5 -- symmetry fix, GT-free real-photo path, instrumentation

Apply as usual, then on the server:
    python scripts/export_synthetic_frames.py     # distinct-walls room
    python scripts/vggt_sanity.py                 # + per-pair printout
    python experiments/run_tier3_vggt.py --backbone vggt
    # THE decisive run -- real photos, no GT needed:
    #   put 10-15 photos in some folder <dir>, then
    python scripts/vggt_sanity.py --frames <dir>
    python experiments/run_tier3_vggt.py --backbone vggt --frames <dir>

## What your last run established (keep these numbers)
* V5b = 0.0513 inherited error vs the true world DESPITE 28.9 deg mean
  global pose error: recall+splat depends only on LOCAL pose consistency,
  and reference-anchored drift is smooth, so it largely cancels in the
  backbone's own frame.  V1 = 0.0029 rad: the scaffold machinery is
  indifferent to whose poses it carries.  Paper-grade pair of numbers.
* The sanity regression (5.2 -> 28.9 deg) was MY closed wall creating a
  near-4-fold-symmetric checker box: global orientation ambiguous, depth
  easier (0.79%).  Root cause, not noise.

## Changed
* scene/exporter -- distinct-walls mode: per-wall checker frequencies +
  three unique saturated panels (1 cm proud of the wall, opaque), texture
  jitter halved.  Descriptor separation on the new scene: 94.5% direct
  reloc at sigma 0.2; V3 back to 9/9 on the synthetic path.
* backbones/base.py -- descriptor REVERTED to validated v2.  A depth
  channel + per-part normalisation was tried and rejected by measurement
  (64.5%, max inter-view cosine 0.952: depth/grey layouts are common
  across ring views and equal weighting injects a shared component).
  Rejection recorded in the code comment.  Real fix remains adapter v2
  (PCA-pooled backbone features).
* vggt_sanity.py -- per-pair self-consistency printout (to resolve the
  twice-identical 0.0528) + GT-free mode: folder of photos, reports
  self-consistency, saves depth + top-down trajectory figure.
* run_tier3_vggt.py -- GT-free frames dir (V1/V3/V5a run without GT;
  V5b skipped) + V3 pose-space median as a report line.

## Real-photo capture guidance
Walk a slow arc around a corner of the lab: 10-15 photos, consecutive
shots overlapping 30-50%, include textured objects (shelves, posters,
desks), avoid blank walls and motion blur.  Any resolution; .jpg fine.

## numpy note
vggt pins numpy<2 but ran fine under 2.5.2 -- leave it.  If vggt
internals ever break, revert with: pip install "numpy==1.26.4" "scipy<1.13"
