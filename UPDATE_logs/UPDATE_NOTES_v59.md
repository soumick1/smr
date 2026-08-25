# Update v5.9 -- Pilot A harness: it runs, and it already says something

## What shipped
* src/smr/eval/trajectory.py -- ATE, RPE, AUC@30, revisit drift, Umeyama
  Sim(3).  Pure numpy.  15 tests against analytically known cases (a
  similarity-transformed copy must score exactly zero; an injected 5-degree
  per-step error must read back as 5 degrees; AUC matched to a hand
  computation).  These decide the paper's headline table, so none of them
  is trusted on inspection.
* experiments/pilot_a.py -- chunk, infer once per chunk, stitch TWO ways
  from the same cached outputs, score both.  4 more tests.

## First result (synthetic, harness validation -- NOT a paper number)
40 frames, 5 chunks of 12, overlap 4, simulated per-chunk error:

  sigma    baseline ATE    +SMR ATE
  0.00        0.0000        0.0034
  0.01        0.0364        0.0520
  0.03        0.0550        0.0570
  0.06        0.3975        0.4508

SMR is NOT better here, and at sigma=0 it is worse by exactly 0.0034 --
our measured decode floor.  This is the prediction from EVALUATION_PLAN.md
section 0 landing on schedule: as a re-encoding of the backbone's own
poses, the scaffold is lossy, and on a 40-frame path there is not enough
accumulated drift for the lattice to pay for that loss.

Read it as a WORKING HARNESS, not as a verdict: this is a synthetic
trajectory whose "chunks" are perfect-plus-noise, with no revisits, no real
backbone drift, and only 40 frames.  It measures the plumbing, and the
plumbing is now proven end to end.

## A real fix found along the way
The first run gave SMR ATE 0.76 and RPE-rot 32 degrees -- broken, not
lossy.  Cause: a grid module reports position only MODULO its period, and
the existing decode unwraps inside a single +-lambda_max/2 window (~2
units).  A 40-frame path leaves that window, aliases, and the corrupted
targets rotate the whole Sim(3) fit.

Fixed with the decode grid cells actually use: unwrap each module's phase
to the branch nearest the PREVIOUS decoded position, then average modules.
Range becomes unbounded while consecutive motion stays under lambda_min/2 =
1.2 units.  A test pins that the continuity decode holds where the
single-window decode aliases.

This matters beyond the pilot: any long-sequence claim needs this decode.

## What the columns mean is in the message; next steps
1. GROUND TRUTH.  The pilot needs real GT poses.  CO3D ships them in
   frame_annotations.jgz -- that extractor is the next thing to build, and
   it is now the critical path.
2. Then a real backbone on a real sequence, 100-300 frames, one process
   per backbone.
3. The regimes where the mechanism should actually pay: sequences with
   REVISITS (the revisit column exists for this), and lengths where
   chunk-chaining drift exceeds the 0.003 decode floor by a wide margin.

Tests: 125 passed, 10 skipped.
