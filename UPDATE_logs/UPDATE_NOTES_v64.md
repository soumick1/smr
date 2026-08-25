# Update v6.4 -- per-chunk probe: separating backbone failure from stitcher failure

## Read your last run again
    RPE-rot 55.97 degrees between ADJACENT keyframes whose true separation
    is 11.3 degrees, and both methods identical to three decimals.

RPE at delta=1 is dominated by pairs INSIDE a chunk, which stitching never
touches.  So the backbone's per-chunk output is already wrong, and both
stitchers are faithfully assembling garbage.  No amount of alignment work
fixes that, and no comparison between the two rows means anything yet.

I should have measured this before either stitcher was written.

## What shipped
`pilot_a.py` now probes every chunk on its own: each chunk's poses are
Sim(3)-aligned to GT for just those frames and scored (ATE, AUC@30, median
RRA).  Backbone quality and stitching quality can never be confused again.
If the worst chunk scores under AUC 50 the run says so in block capitals
and tells you the rows below are not a measurement of SMR.

Validated on the synthetic path: per-chunk AUC 98.8-99.5 when the chunks
are good, and the stitched result then lands where it should.

## Run this next -- it is a diagnosis, not a result
    python experiments/pilot_a.py --gt data/gt/hydrant.npz --backbone vggt \
        --keyframe-stride 6 --chunk 16 --overlap 8

and read ONLY the per-chunk table.  Then sweep the chunk geometry until it
is high:

    for ks in 2 3 4 6; do
      for ch in 8 12 16; do
        echo "=== keyframe-stride $ks chunk $ch ==="
        python experiments/pilot_a.py --gt data/gt/hydrant.npz \
            --backbone vggt --keyframe-stride $ks --chunk $ch \
            --overlap $((ch/2)) 2>/dev/null | sed -n '/per-chunk/,/^$/p'
      done
    done

We already know VGGT reaches AUC 99.3 on 10 frames spread over this
sequence, so a configuration where it works EXISTS.  The job is to find it.
My current suspicion: 16 keyframes at stride 6 spans ~170 degrees of the
orbit, so the first and last frames of a chunk see opposite sides of the
hydrant with no shared surface -- that is a hard multi-view problem, not a
degenerate one, but it may still be where VGGT breaks.  Smaller chunks
spanning 60-90 degrees are the first thing to try.

## Honest status
The pilot has now failed three times, each time for a different reason in
MY harness: collinear overlaps, degenerate contiguous baselines, and now
chunk geometry.  Each was caught rather than published, which is the system
working -- but it is three days of instrument-building, not results.

26 days to the ICLR abstract.  If the sweep above does not produce a
high-AUC configuration quickly, my recommendation is to move the target to
CVPR (mid-November) and do this properly: the experiment needs to be right
far more than it needs to be fast, and a rushed table with a broken harness
underneath is worse than no table at all.

Tests: 128 passed, 10 skipped.
