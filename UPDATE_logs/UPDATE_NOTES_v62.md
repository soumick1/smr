# Update v6.2 -- the stitcher was broken. Fixed, and pinned by a test.

## Do NOT run the other backbones on the previous harness.

## What the hydrant run actually told us
1. CONVENTION SETTLED. row_w2c_flip scores AUC 99.3 with median RRA 0.50
   deg.  That is the extraction confirmed against a real model.
2. The --validate line disagreeing (26.7) was a BAD SELF-CHECK, not a bad
   extraction: it sampled 8 frames spread ~44 degrees apart, a baseline
   VGGT fails at.  validate() now uses a CONTIGUOUS window, the way the
   pilot actually feeds the backbone (--wide-baseline keeps the old
   behaviour if you want the harder test).
3. THE PILOT ITSELF WAS BROKEN, and it was my bug.

## The bug
Chunk overlaps are a few CONSECUTIVE frames of a smooth orbit, so their
camera centres are nearly collinear (measured spread 0.009 on a 1 for a
well-spread cloud).  Umeyama on centres alone leaves the rotation about
that line unconstrained.  Measured, with 1% centre noise:

    overlap  spread   noise    pose-based   points-only
        4    0.009    0.01       0.000 deg    59.195 deg
        8    0.020    0.01       0.000 deg    79.209 deg
        8    0.100    0.01       0.000 deg     2.704 deg

Every chunk was being attached at a randomly wrong angle -- hence ATE 11.2
and AUC 7.1.  Nothing about SMR or the backbone was being measured.

## The fix
`sim3_from_poses()`: rotation by orthogonal Procrustes on the frame
ORIENTATIONS (well posed from a single pose), scale from centre spread,
translation from centroids.  Both stitchers use it, so the comparison stays
controlled.  Default overlap raised 4 -> 8, and the run prints the overlap
spread as a diagnostic.

Three new tests: the collinear case where points-only fails and pose-based
does not; exact recovery of a known similarity; and a guard that neither
stitcher can silently revert to points-only.

## Re-run, in this order
    python scripts/co3d_gt_poses.py --root ~/co3d_data --category hydrant \
        --out data/gt/hydrant.npz --convention row_w2c_flip \
        --validate --backbone vggt
      -> expect a HIGH AUC now (contiguous window).  If it is still low,
         stop and send it: that would mean something deeper than stitching.

    python experiments/pilot_a.py --gt data/gt/hydrant.npz \
        --backbone vggt --chunk 16 --overlap 8

      -> expect "backbone alone" ATE in the low hundredths and AUC well
         above 50.  If the baseline is still garbage, the stitcher is still
         wrong and no comparison means anything.

ONLY when the baseline row looks sane does the +SMR row carry information,
and only then is it worth spending an hour on the other eight backbones.

Tests: 128 passed, 10 skipped.
