# Update v6.3 -- the chunking was wrong: contiguous frames are degenerate

## Your two numbers settled it
    wide-baseline  (10 frames spread over the orbit):  AUC 99.3, RRA 0.50 deg
    contiguous     (10 consecutive frames):            AUC 16.1, RTA 76 deg
Same convention, same model, same sequence.

Sixteen CONSECUTIVE video frames span only a few degrees of arc.  At that
baseline the relative translation direction is essentially unrecoverable --
which is what RTA 76 degrees (near-random) is telling us -- so every chunk
was already garbage before the stitcher touched it.  ATE 9.9 was measuring
my chunking, not SMR and not VGGT.

This also explains why these models publish AUC on RANDOMLY SAMPLED frames:
that is their operating regime, not consecutive video.

## Fix: --keyframe-stride (default 6)
The pilot now evaluates on every Nth frame.  With stride 6 on a 202-frame
orbit the keyframes sit ~10 degrees apart, chunks of 16 span a wide arc,
and the geometry is well conditioned.  Set --keyframe-stride 1 to reproduce
the old contiguous behaviour.

The run now prints the median rotation between adjacent keyframes.  Read it
before reading anything else: a few degrees means the chunk geometry is
degenerate and no comparison downstream is meaningful.

## Re-run
    python experiments/pilot_a.py --gt data/gt/hydrant.npz --backbone vggt \
        --keyframe-stride 6 --chunk 16 --overlap 8

Expect the median keyframe rotation around 10 degrees, and -- the gate --
"backbone alone" with a LOW ATE and a HIGH AUC.  Only then does the +SMR
row mean anything.

If the baseline row is still poor, try --keyframe-stride 4 and 8: the point
is to land in the regime where the backbone works, which is where a plug-in
claim has to be made anyway.  Claiming an advantage in a regime where the
backbone is broken would be worthless, and a reviewer would say so.

## Answering your question
Yes: every backbone has its own pose convention (vggt w2c, pi3 c2w, ...),
declared on its adapter class and applied by assemble(); that was settled
during the socket work and is covered by tests.  The CO3D convention is a
different thing -- it belongs to the ground-truth dataset, not to any
model -- so it had to be solved once, and it now applies to all nine
backbones identically.

Tests: 128 passed, 10 skipped.
